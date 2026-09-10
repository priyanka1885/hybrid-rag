"""
Memory-bounded batching for transformer inference.

A transformer's peak transient allocation is dominated by the per-layer
attention score matrix, whose size is

    batch_size * num_heads * seq_len**2 * 4 bytes

Note the SQUARE on sequence length. That makes a fixed batch size a poor way to
control memory: a batch of 8 padded to the full 512-token window allocates about
100 MB for a single attention tensor, while the same batch of 8 short table rows
padded to 128 tokens allocates about 6 MB. On a memory-limited instance the
fixed-batch version is what turns a handful of concurrent requests into an
out-of-memory restart, even though the average request is small.

:func:`plan_length_batches` fixes that by budgeting ``batch * seq**2`` instead of
``batch``. Candidates are grouped by length, so short passages still batch
widely (fast) while the occasional long passage runs in a small batch (bounded).
Peak memory becomes a function of the budget rather than of the longest
candidate that happened to be retrieved.

Batching is score-neutral: padding is excluded by the attention mask, so how
items are grouped does not affect the logits the answerability thresholds are
calibrated against.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence


def plan_length_batches(
    lengths: Sequence[int],
    max_batch_size: int,
    token_budget: int,
) -> Iterator[list[int]]:
    """Group indices into length-sorted batches under a ``batch * seq**2`` budget.

    Args:
        lengths: token count of each item, indexed the same as the caller's list.
        max_batch_size: hard cap on items per batch, whatever the budget allows.
        token_budget: maximum ``batch_size * padded_seq_len**2`` for one batch.
            Multiply by ``num_heads * 4`` to get the approximate attention-tensor
            size in bytes (e.g. 262144 * 12 * 4 ~ 12 MB for a 12-head model).

    Yields:
        Lists of indices, each safe to run as a single padded batch. Every index
        is yielded exactly once. An item that exceeds the budget on its own is
        still yielded alone rather than dropped, so no candidate is ever silently
        skipped for being long.
    """
    if not lengths:
        return
    max_batch_size = max(1, max_batch_size)
    token_budget = max(1, token_budget)

    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch: list[int] = []
    batch_seq = 0

    for i in order:
        candidate_seq = max(batch_seq, lengths[i])
        # A batch always keeps at least one item, so an over-budget single item
        # runs on its own instead of being discarded.
        fits = (
            len(batch) < max_batch_size
            and (len(batch) + 1) * candidate_seq * candidate_seq <= token_budget
        )
        if batch and not fits:
            yield batch
            batch, batch_seq = [i], lengths[i]
        else:
            batch.append(i)
            batch_seq = candidate_seq

    if batch:
        yield batch
