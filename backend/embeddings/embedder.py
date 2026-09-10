"""
Local embedding model wrapper (ONNX Runtime).

Uses the pre-exported fp32 ONNX graph published in the SAME official
Hugging Face repository as the torch weights (default all-MiniLM-L6-v2), run
through ONNX Runtime on CPU. No paid API and no PyTorch is required, which
keeps the serving process small enough for a memory-constrained host.

Because the ONNX graph carries the identical weights, the vectors produced here
occupy the SAME vector space as an index built with sentence-transformers, so an
existing FAISS index stays valid. The sentence-transformers behaviour is
reproduced exactly:

* truncation at the model's ``max_seq_length`` (256 for all-MiniLM-L6-v2),
* attention-mask-weighted MEAN pooling over the token embeddings,
* L2 normalization, so inner-product search in FAISS equals cosine similarity.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache

import numpy as np

from backend.batching import plan_length_batches
from backend.config import settings
from backend.runtime import MODEL_LOAD_LOCK, MODEL_LOCK, release_memory

# Files fetched from the model repo. The ONNX graph is the official fp32 export
# (NOT a quantized variant - quantization would shift the vectors and invalidate
# an existing FAISS index).
_ONNX_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_ST_CONFIG_FILE = "sentence_bert_config.json"
_POOLING_CONFIG_FILE = "1_Pooling/config.json"

# Used only if the repo omits the config files above.
_DEFAULT_MAX_SEQ_LENGTH = 256
_DEFAULT_DIMENSION = 384

# Upper bound on texts per batch. ``settings.EMBED_TOKEN_BUDGET`` usually binds
# first; this only stops a very large batch of very short texts.
_DEFAULT_BATCH_SIZE = 32
# Texts are tokenized this many at a time. Ingestion embeds the whole corpus in
# a single encode() call, and tokenizing every chunk up front would hold
# thousands of Encoding objects (ids, masks, offsets) in memory at once purely to
# plan batches. Windowing bounds that without affecting the output.
_TOKENIZE_WINDOW = 256


class _OnnxTextEncoder:
    """An ONNX Runtime session + tokenizer that mean-pools token embeddings."""

    def __init__(self, session, tokenizer, max_seq_length: int, dimension: int,
                 pad_id: int = 0):
        self.session = session
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dimension = dimension
        self.pad_id = pad_id
        # Only feed the inputs this particular graph declares (some exports omit
        # token_type_ids).
        self._input_names = [i.name for i in session.get_inputs()]

    def tokenize(self, texts: list[str]) -> tuple[list, list[int]]:
        """Tokenize without padding, returning the encodings and token lengths.

        Padding is applied per batch in :meth:`run_batch` instead of by the
        tokenizer, so batches can be planned from real token lengths and each is
        padded only to its own longest member.
        """
        encodings = self.tokenizer.encode_batch(texts)
        return encodings, [len(e.ids) for e in encodings]

    def run_batch(self, encodings, lengths: list[int], idxs: list[int]) -> np.ndarray:
        """Embed one planned batch: pad -> run -> mean-pool -> L2 normalize."""
        seq = max(lengths[i] for i in idxs)
        rows = len(idxs)

        input_ids = np.full((rows, seq), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((rows, seq), dtype=np.int64)
        types = (
            np.zeros((rows, seq), dtype=np.int64)
            if "token_type_ids" in self._input_names
            else None
        )
        for row, i in enumerate(idxs):
            enc = encodings[i]
            n = lengths[i]
            input_ids[row, :n] = enc.ids
            attention_mask[row, :n] = enc.attention_mask
            if types is not None:
                types[row, :n] = enc.type_ids

        feed = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = attention_mask
        if types is not None:
            feed["token_type_ids"] = types

        # (batch, seq_len, hidden) token embeddings. Inference is serialized
        # process-wide: each concurrent Run() would allocate its own set of
        # transformer activations, so unbounded parallelism here is what turns a
        # traffic spike into an out-of-memory restart. The session is pinned to
        # one thread anyway, so serializing costs no real throughput.
        with MODEL_LOCK:
            token_embeddings = self.session.run(None, feed)[0]

        # Attention-mask-weighted mean pooling: padding tokens must not
        # contribute, otherwise a padded batch yields different vectors than the
        # same texts encoded one at a time.
        mask = attention_mask.astype(np.float32)[..., None]
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        # Drop the (batch, seq_len, hidden) activation block as soon as it has
        # been pooled down to (batch, hidden); it is by far the largest array
        # allocated per call and holding it until the frame exits doubles peak.
        del token_embeddings, feed, summed, mask, input_ids, attention_mask, types

        # L2 normalize so inner product == cosine similarity.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-12, None)
        return np.asarray(pooled, dtype="float32")


# maxsize=1: the fp32 session costs ~108 MB resident, so caching a second one
# for a different model name would silently double the largest single allocation
# in the process. Only one embedding model can match the FAISS index anyway.
@lru_cache(maxsize=1)
def _load_model(model_name: str) -> _OnnxTextEncoder:
    """Download and open the ONNX session + tokenizer for ``model_name``.

    Imported lazily so that importing this module is cheap and does not require
    ONNX Runtime to be present until embeddings are actually needed. Cached, so
    the model is loaded at most once per process.
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    onnx_path = hf_hub_download(model_name, _ONNX_FILE)
    tokenizer_path = hf_hub_download(model_name, _TOKENIZER_FILE)

    # Read the model's own truncation length / dimension instead of hardcoding,
    # so the values always follow the model repo.
    max_seq_length = _DEFAULT_MAX_SEQ_LENGTH
    try:
        with open(hf_hub_download(model_name, _ST_CONFIG_FILE), encoding="utf-8") as fh:
            max_seq_length = int(json.load(fh).get("max_seq_length", max_seq_length))
    except Exception:  # pragma: no cover - repo without the ST config
        pass

    dimension = _DEFAULT_DIMENSION
    try:
        with open(hf_hub_download(model_name, _POOLING_CONFIG_FILE), encoding="utf-8") as fh:
            pooling = json.load(fh)
        dimension = int(pooling.get("word_embedding_dimension", dimension))
        if not pooling.get("pooling_mode_mean_tokens", True):
            print(
                f"[embedder] WARNING: {model_name} does not use mean pooling; "
                "this wrapper implements mean pooling only.",
                file=sys.stderr,
            )
    except Exception:  # pragma: no cover - repo without a pooling config
        pass

    # Low-memory CPU settings: a single thread and no arena/pattern caching keep
    # resident memory low on small instances. Spinning is disabled too - an
    # idle spinning thread holds its scratch buffers resident between requests.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in (
        ("session.intra_op.allow_spinning", "0"),
        ("session.inter_op.allow_spinning", "0"),
    ):
        try:
            opts.add_session_config_entry(key, value)
        except Exception:  # pragma: no cover - older onnxruntime builds
            pass

    session = ort.InferenceSession(
        onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    # Graph optimization builds a fused copy of the graph alongside the original
    # initializers. Trim once the session is live so those transient pages go
    # back to the OS instead of padding the process baseline forever.
    release_memory()

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=max_seq_length)
    # Padding is applied per batch (see _OnnxTextEncoder.run_batch), not by the
    # tokenizer, so batches can be sized against a real token-count budget.
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.no_padding()

    return _OnnxTextEncoder(
        session,
        tokenizer,
        max_seq_length,
        dimension,
        pad_id=pad_id if pad_id is not None else 0,
    )


class Embedder:
    """Thin wrapper around an ONNX Runtime sentence-embedding model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        # Double-checked locking: lru_cache memoizes the session but does not
        # prevent two threads from building one simultaneously, and each session
        # is ~108 MB. See backend.runtime.MODEL_LOAD_LOCK.
        if self._model is None:
            with MODEL_LOAD_LOCK:
                self._model = _load_model(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.dimension)

    def encode(
        self,
        texts: list[str],
        batch_size: int = _DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
        token_budget: int | None = None,
    ) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings.

        Memory is bounded in three ways, which matters because ingestion embeds
        the whole corpus in one call while the API embeds one question:

        * texts are tokenized a window at a time, so the tokenizer's output for
          thousands of chunks never exists all at once;
        * within a window, batches are planned against a ``batch * seq**2``
          budget, so peak attention memory is bounded regardless of chunk length;
        * results are scattered into one preallocated array instead of being
          collected and ``vstack``-ed, which would hold two full copies.

        Row order matches ``texts``, and vectors are unchanged by batching
        because mean pooling is attention-mask weighted.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        model = self.model
        budget = token_budget or settings.EMBED_TOKEN_BUDGET
        total = len(texts)
        out = np.zeros((total, self.dimension), dtype="float32")
        done = 0

        for window_start in range(0, total, _TOKENIZE_WINDOW):
            window = list(texts[window_start : window_start + _TOKENIZE_WINDOW])
            encodings, lengths = model.tokenize(window)
            for idxs in plan_length_batches(lengths, batch_size, budget):
                vectors = model.run_batch(encodings, lengths, idxs)
                for row, i in enumerate(idxs):
                    out[window_start + i] = vectors[row]
                del vectors
            del encodings, lengths
            done += len(window)
            if show_progress:
                print(f"  embedding {done}/{total}", end="\r", file=sys.stderr, flush=True)

        if show_progress:
            print(file=sys.stderr)
        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
