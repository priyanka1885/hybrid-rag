"""
Cross-encoder reranking (ONNX Runtime).

    (Question, Candidate Chunk) -> Cross Encoder -> Relevance Score -> Sort

Unlike the bi-encoder used for dense retrieval (which embeds question and
passage separately), a cross-encoder jointly encodes the question and each
candidate passage, producing a much more precise relevance score. It is
expensive, so we only apply it to the small candidate set returned by hybrid
retrieval - never the whole corpus.

The model is the pre-exported fp32 ONNX graph from the SAME official Hugging
Face repository as the torch weights (default ms-marco-MiniLM-L-6-v2), executed
by ONNX Runtime on CPU so no PyTorch is needed in the serving process.

IMPORTANT: the score is the model's RAW LOGIT, with no sigmoid/softmax applied.
This matches the model's declared ``sbert_ce_default_activation_function``
(Identity) and therefore the existing answerability thresholds
(``MIN_RERANK_SCORE``) and the table boost, which are calibrated on logits.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from backend.batching import plan_length_batches
from backend.config import settings
from backend.runtime import MODEL_LOAD_LOCK, MODEL_LOCK, release_memory

# Official fp32 export (NOT a quantized variant, which would shift the logits
# the answerability thresholds are calibrated against).
_ONNX_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "tokenizer.json"

# Cross-encoders take the full BERT window; ms-marco-MiniLM-L-6-v2 uses 512.
_MAX_LENGTH = 512
# Upper bound on items per batch. The token budget below almost always binds
# first; this only stops a huge batch of very short rows.
_BATCH_SIZE = 16


class _OnnxCrossEncoder:
    """An ONNX Runtime session + tokenizer that scores (query, passage) pairs.

    Scoring the whole candidate pool is the single largest transient allocation
    in a request, so batches are planned against a ``batch * seq**2`` memory
    budget rather than a fixed size (see :mod:`backend.batching`). Pairs are
    tokenized once without padding and each batch is padded manually to its own
    longest member, so a single long passage no longer inflates every batch to
    the full 512-token window.
    """

    def __init__(self, session, tokenizer, pad_id: int = 0):
        self.session = session
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self._input_names = [i.name for i in session.get_inputs()]

    def predict(
        self,
        pairs: list[list[str]],
        batch_size: int | None = None,
        token_budget: int | None = None,
    ) -> np.ndarray:
        """Return one raw logit per (query, passage) pair.

        Scores are independent of how pairs are grouped - padding is excluded by
        the attention mask - so the batching here only affects memory and speed,
        never the logits the answerability thresholds are calibrated against. The
        caller's original ordering is restored before returning.
        """
        if not pairs:
            return np.zeros((0,), dtype="float32")

        batch_size = batch_size or settings.RERANK_BATCH_SIZE
        token_budget = token_budget or settings.RERANK_TOKEN_BUDGET

        # Tokenize once, unpadded, so each batch can be padded to its own longest
        # member and the real token lengths are known for batch planning.
        # tokenizers accepts (text, text_pair) tuples and builds the proper
        # [CLS] q [SEP] passage [SEP] pair encoding with 0/1 token types.
        encodings = self.tokenizer.encode_batch([(p[0], p[1]) for p in pairs])
        lengths = [len(e.ids) for e in encodings]
        scores = np.zeros((len(pairs),), dtype="float32")

        want_mask = "attention_mask" in self._input_names
        want_types = "token_type_ids" in self._input_names

        for idxs in plan_length_batches(lengths, batch_size, token_budget):
            seq = max(lengths[i] for i in idxs)
            rows = len(idxs)

            input_ids = np.full((rows, seq), self.pad_id, dtype=np.int64)
            mask = np.zeros((rows, seq), dtype=np.int64) if want_mask else None
            types = np.zeros((rows, seq), dtype=np.int64) if want_types else None
            for row, i in enumerate(idxs):
                enc = encodings[i]
                n = lengths[i]
                input_ids[row, :n] = enc.ids
                if mask is not None:
                    mask[row, :n] = enc.attention_mask
                if types is not None:
                    types[row, :n] = enc.type_ids

            feed = {}
            if "input_ids" in self._input_names:
                feed["input_ids"] = input_ids
            if mask is not None:
                feed["attention_mask"] = mask
            if types is not None:
                feed["token_type_ids"] = types

            # Serialized process-wide (see backend.runtime.MODEL_LOCK): the
            # cross-encoder is the heaviest allocation per request, so running
            # several concurrently is what pushes the instance over its limit.
            with MODEL_LOCK:
                logits = self.session.run(None, feed)[0]
            del feed, input_ids, mask, types

            # (batch, 1) for a single-label regression head -> one score per pair.
            # No activation function is applied (Identity), so these stay logits.
            batch_scores = np.asarray(logits, dtype="float32").reshape(rows, -1)[:, 0]
            for pos, i in enumerate(idxs):
                scores[i] = batch_scores[pos]
            del logits, batch_scores

        return scores


# maxsize=1: the fp32 cross-encoder session costs ~95 MB resident, so caching a
# second one for a different model name would double the second-largest
# allocation in the process for no benefit.
@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> _OnnxCrossEncoder:
    """Download and open the ONNX session + tokenizer for ``model_name``.

    Imported lazily so importing this module stays cheap, and cached so the
    cross-encoder is loaded at most once per process.
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    onnx_path = hf_hub_download(model_name, _ONNX_FILE)
    tokenizer_path = hf_hub_download(model_name, _TOKENIZER_FILE)

    # Low-memory CPU settings: single-threaded, no arena/pattern caching, and no
    # thread spinning (an idle spinning thread keeps its scratch space resident).
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
    # Release the transient pages graph optimization allocated alongside the
    # original initializers, so they do not sit in the process baseline.
    release_memory()

    tokenizer = Tokenizer.from_file(tokenizer_path)
    # "longest_first" is the HF default for pairs and matches how
    # sentence-transformers truncates cross-encoder inputs.
    tokenizer.enable_truncation(max_length=_MAX_LENGTH, strategy="longest_first")
    # Padding is applied per batch in predict() rather than by the tokenizer, so
    # batch composition can be planned from real token lengths and each batch is
    # padded only to its own longest member.
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.no_padding()
    return _OnnxCrossEncoder(session, tokenizer, pad_id=pad_id if pad_id is not None else 0)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        # Double-checked locking: lru_cache memoizes the session but does not
        # prevent two threads from building one simultaneously, and each session
        # is ~95 MB. See backend.runtime.MODEL_LOAD_LOCK.
        if self._model is None:
            with MODEL_LOAD_LOCK:
                self._model = _load_cross_encoder(self.model_name)
        return self._model

    def rerank(self, question: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        """Score each candidate against the question and return the top_k.

        Adds a ``rerank_score`` and a ``final_rank`` to each returned item.
        """
        top_k = top_k or settings.RERANK_TOP_K
        if not candidates:
            return []
        pairs = [[question, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        scored: list[dict] = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        top = scored[:top_k]
        for rank, item in enumerate(top, start=1):
            item["final_rank"] = rank
        return top
