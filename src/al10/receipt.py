"""Receipt generation utilities for AL-1.0 style attribution output."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence

from .math import merge_query_heads, source_bucket

MODEL_OUTPUT_SOURCE_IDX = -2
PARAMETRIC_SOURCE_ID = "PARAMETRIC"


def aggregate_decode_step(
    alpha_per_head: Sequence[Sequence[float]],
    source_idx_at_key_position: Sequence[int],
) -> Dict[int, float]:
    """
    Compute L_i(S) for one generated token.

    - Merge heads with mean policy.
    - Group merged attention mass by source index.
    """
    merged = merge_query_heads(alpha_per_head)
    return source_bucket(merged, source_idx_at_key_position)


def response_ratio(per_step_buckets: Sequence[Mapping[int, float]]) -> Dict[int, float]:
    """
    Compute full-response source ratio:
        Ratio(S) = (1 / N) * sum_{i=1..N}(L_i(S))
    """
    if not per_step_buckets:
        raise ValueError("per_step_buckets cannot be empty")

    totals: Dict[int, float] = defaultdict(float)
    step_count = len(per_step_buckets)

    for buckets in per_step_buckets:
        for source_idx, mass in buckets.items():
            totals[int(source_idx)] += float(mass)

    ratios = {source_idx: mass / step_count for source_idx, mass in totals.items()}
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("computed source ratios have zero total mass")

    # Normalize to make the JSON ratio sum exactly to 1.0 (subject to float precision).
    return {source_idx: ratio / total_ratio for source_idx, ratio in ratios.items()}


def build_receipt(
    per_step_buckets: Sequence[Mapping[int, float]],
    idx_to_source_id: Mapping[int, str],
    *,
    model_id: str,
    registry_manifest_hash: str,
    training_manifest_hash: str,
    layer_policy: str = "last_block_self_attn_mean_heads",
    collapse_model_output: bool = True,
    min_ratio: float = 0.0,
    source_labels: Optional[Mapping[str, str]] = None,
) -> dict:
    """
    Build an AL-1.0-like attribution receipt JSON object.

    Args:
        per_step_buckets: One L_i(S) mapping per generated token.
        idx_to_source_id: Mapping from source_idx to stable source_id string.
        model_id: Model release identifier.
        registry_manifest_hash: Hash of source registry manifest.
        training_manifest_hash: Hash of training manifest.
        layer_policy: Logged attention layer/head policy string.
        collapse_model_output: If True, MODEL_OUTPUT source mass is moved to PARAMETRIC.
        min_ratio: Optional UI floor; sources below threshold are dropped then renormalized.
        source_labels: Optional map[source_id] -> human-readable label.
    """
    ratios = response_ratio(per_step_buckets)

    if collapse_model_output and MODEL_OUTPUT_SOURCE_IDX in ratios:
        model_output_mass = ratios.pop(MODEL_OUTPUT_SOURCE_IDX)
        parametric_idx = _parametric_idx(idx_to_source_id)
        ratios[parametric_idx] = ratios.get(parametric_idx, 0.0) + model_output_mass

    source_rows = []
    for source_idx, ratio in ratios.items():
        source_id = idx_to_source_id.get(source_idx, PARAMETRIC_SOURCE_ID)
        if ratio < min_ratio:
            continue
        row = {"source_id": source_id, "ratio": float(ratio)}
        if source_labels and source_id in source_labels:
            row["label"] = source_labels[source_id]
        source_rows.append(row)

    if not source_rows:
        raise ValueError("all sources were filtered out by min_ratio")

    denom = sum(row["ratio"] for row in source_rows)
    for row in source_rows:
        row["ratio"] = row["ratio"] / denom

    source_rows.sort(key=lambda row: row["ratio"], reverse=True)

    return {
        "receipt_spec": "AL-1.0",
        "model_id": model_id,
        "registry_manifest_hash": registry_manifest_hash,
        "training_manifest_hash": training_manifest_hash,
        "generated_token_count": len(per_step_buckets),
        "layer_policy": layer_policy,
        "sources": source_rows,
    }


def _parametric_idx(idx_to_source_id: Mapping[int, str]) -> int:
    for idx, source_id in idx_to_source_id.items():
        if source_id == PARAMETRIC_SOURCE_ID:
            return int(idx)

    # Reserve a deterministic fallback index if PARAMETRIC is not present in table.
    return -1
