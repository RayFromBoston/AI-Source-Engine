"""Runtime tracing helpers for decode-step AL-1.0 logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .receipt import aggregate_decode_step, build_receipt, response_ratio


class DecodeStepAggregator(Protocol):
    """Protocol for adapters that can aggregate one decode step."""

    def aggregate_step(
        self,
        alpha_per_head: Sequence[Sequence[float]],
        source_idx_at_key_position: Sequence[int],
    ) -> dict[int, float]:
        """Aggregate one decode step into a source bucket mapping."""


@dataclass(slots=True)
class SourceTagSidecar:
    """
    Sidecar that tracks source_idx for each key position in KV cache order.
    """

    source_idx: list[int] = field(default_factory=list)

    @classmethod
    def from_context(cls, source_idx: Sequence[int]) -> "SourceTagSidecar":
        return cls(source_idx=[int(value) for value in source_idx])

    def append(self, source_idx: int) -> None:
        self.source_idx.append(int(source_idx))

    def extend(self, source_idx: Sequence[int]) -> None:
        for value in source_idx:
            self.append(int(value))

    def validate_key_length(self, key_len: int) -> None:
        if len(self.source_idx) != key_len:
            raise ValueError(
                "source sidecar length must match key length "
                f"(got sidecar={len(self.source_idx)}, key_len={key_len})"
            )


@dataclass(slots=True)
class DecodeStepLogger:
    """
    Collects AL-1.0 logging vectors across decode steps and emits receipts.
    """

    per_step_buckets: list[dict[int, float]] = field(default_factory=list)

    def log_bucket(self, bucket: Mapping[int, float]) -> dict[int, float]:
        snapshot = {int(key): float(value) for key, value in bucket.items()}
        self.per_step_buckets.append(snapshot)
        return snapshot

    def log_decode_step(
        self,
        alpha_per_head: Sequence[Sequence[float]],
        source_idx_at_key_position: Sequence[int],
    ) -> dict[int, float]:
        bucket = aggregate_decode_step(alpha_per_head, source_idx_at_key_position)
        return self.log_bucket(bucket)

    def ratios(self) -> dict[int, float]:
        return response_ratio(self.per_step_buckets)

    def receipt(
        self,
        idx_to_source_id: Mapping[int, str],
        *,
        model_id: str,
        registry_manifest_hash: str,
        training_manifest_hash: str,
        layer_policy: str = "last_block_self_attn_mean_heads",
        collapse_model_output: bool = True,
        min_ratio: float = 0.0,
        source_labels: Mapping[str, str] | None = None,
    ) -> dict:
        return build_receipt(
            self.per_step_buckets,
            idx_to_source_id,
            model_id=model_id,
            registry_manifest_hash=registry_manifest_hash,
            training_manifest_hash=training_manifest_hash,
            layer_policy=layer_policy,
            collapse_model_output=collapse_model_output,
            min_ratio=min_ratio,
            source_labels=source_labels,
        )
