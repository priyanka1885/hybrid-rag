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

from backend.config import settings

# Official fp32 export (NOT a quantized variant, which would shift the logits
# the answerability thresholds are calibrated against).
_ONNX_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "tokenizer.json"

# Cross-encoders take the full BERT window; ms-marco-MiniLM-L-6-v2 uses 512.
_MAX_LENGTH = 512
# Pairs are scored in small batches to bound peak activation memory.
_BATCH_SIZE = 16


class _OnnxCrossEncoder:
    """An ONNX Runtime session + tokenizer that scores (query, passage) pairs."""

    def __init__(self, session, tokenizer):
        self.session = session
        self.tokenizer = tokenizer
        self._input_names = [i.name for i in session.get_inputs()]

    def predict(self, pairs: list[list[str]], batch_size: int = _BATCH_SIZE) -> np.ndarray:
        """Return one raw logit per (query, passage) pair."""
        if not pairs:
            return np.zeros((0,), dtype="float32")
        scores: list[np.ndarray] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            # tokenizers accepts (text, text_pair) tuples and builds the proper
            # [CLS] q [SEP] passage [SEP] pair encoding with 0/1 token types.
            encodings = self.tokenizer.encode_batch(
                [(p[0], p[1]) for p in batch]
            )
            feed = {}
            if "input_ids" in self._input_names:
                feed["input_ids"] = np.asarray([e.ids for e in encodings], dtype=np.int64)
            if "attention_mask" in self._input_names:
                feed["attention_mask"] = np.asarray(
                    [e.attention_mask for e in encodings], dtype=np.int64
                )
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.asarray(
                    [e.type_ids for e in encodings], dtype=np.int64
                )

            logits = self.session.run(None, feed)[0]
            # (batch, 1) for a single-label regression head -> one score per pair.
            # No activation function is applied (Identity), so these stay logits.
            scores.append(np.asarray(logits, dtype="float32").reshape(len(batch), -1)[:, 0])
        return np.concatenate(scores)


@lru_cache(maxsize=2)
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

    # Low-memory CPU settings: single-threaded, no arena/pattern caching.
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
    # "longest_first" is the HF default for pairs and matches how
    # sentence-transformers truncates cross-encoder inputs.
    tokenizer.enable_truncation(max_length=_MAX_LENGTH, strategy="longest_first")
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(
        pad_id=pad_id if pad_id is not None else 0, pad_token="[PAD]"
    )
    return _OnnxCrossEncoder(session, tokenizer)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
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
