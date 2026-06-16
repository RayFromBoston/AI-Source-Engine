"""Core AL-1.0 Step 5 math helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Sequence


def merge_query_heads(alpha_per_head: Sequence[Sequence[float]]) -> list[float]:
    """
    Merge per-head attention with an arithmetic mean over heads.

    Implements:
        alpha_{i,j} = (1 / H) * sum_{h=1..H}(alpha_{h,i,j})
    with decode-time shape simplified to [H, T_k] for one generated token.
    """
    if not alpha_per_head:
        raise ValueError("alpha_per_head cannot be empty")

    key_len = len(alpha_per_head[0])
    if key_len == 0:
        raise ValueError("alpha_per_head must contain at least one key position")

    for head in alpha_per_head:
        if len(head) != key_len:
            raise ValueError("all heads must have the same key length")

    head_count = len(alpha_per_head)
    merged = [0.0] * key_len

    for head in alpha_per_head:
        for key_pos, value in enumerate(head):
            merged[key_pos] += value

    return [value / head_count for value in merged]


def source_bucket(
    alpha_ij: Sequence[float],
    source_idx_at_key_position: Sequence[int],
) -> Dict[int, float]:
    """
    Build one decode-step logging vector L_i(S).

    Implements:
        L_i(S) = sum_{j: S(j)=S}(alpha_{i,j})
    """
    if len(alpha_ij) != len(source_idx_at_key_position):
        raise ValueError("alpha_ij and source_idx_at_key_position must have equal length")

    buckets: Dict[int, float] = defaultdict(float)
    for key_pos, alpha_value in enumerate(alpha_ij):
        buckets[source_idx_at_key_position[key_pos]] += float(alpha_value)
    return dict(buckets)


def validate_probability_sum(distribution: Mapping[int, float], atol: float = 1e-6) -> bool:
    """Return True if distribution sums to ~1.0 within absolute tolerance."""
    total = sum(float(v) for v in distribution.values())
    return abs(total - 1.0) <= atol
