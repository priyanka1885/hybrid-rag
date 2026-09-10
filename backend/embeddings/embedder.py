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

from backend.config import settings

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


class _OnnxTextEncoder:
    """An ONNX Runtime session + tokenizer that mean-pools token embeddings."""

    def __init__(self, session, tokenizer, max_seq_length: int, dimension: int):
        self.session = session
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dimension = dimension
        # Only feed the inputs this particular graph declares (some exports omit
        # token_type_ids).
        self._input_names = [i.name for i in session.get_inputs()]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Embed one batch: tokenize -> run -> mean-pool -> L2 normalize."""
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.asarray(
                [e.type_ids for e in encodings], dtype=np.int64
            )

        # (batch, seq_len, hidden) token embeddings.
        token_embeddings = self.session.run(None, feed)[0]

        # Attention-mask-weighted mean pooling: padding tokens must not
        # contribute, otherwise a padded batch yields different vectors than the
        # same texts encoded one at a time.
        mask = attention_mask.astype(np.float32)[..., None]
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        # L2 normalize so inner product == cosine similarity.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-12, None)
        return np.asarray(pooled, dtype="float32")


@lru_cache(maxsize=2)
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
    # resident memory low on small instances.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=max_seq_length)
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(
        pad_id=pad_id if pad_id is not None else 0, pad_token="[PAD]"
    )

    return _OnnxTextEncoder(session, tokenizer, max_seq_length, dimension)


class Embedder:
    """Thin wrapper around an ONNX Runtime sentence-embedding model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.dimension)

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        model = self.model
        total = len(texts)
        out: list[np.ndarray] = []
        for start in range(0, total, batch_size):
            out.append(model.encode_batch(list(texts[start : start + batch_size])))
            if show_progress:
                done = min(start + batch_size, total)
                print(f"  embedding {done}/{total}", end="\r", file=sys.stderr, flush=True)
        if show_progress:
            print(file=sys.stderr)
        return np.vstack(out).astype("float32", copy=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
