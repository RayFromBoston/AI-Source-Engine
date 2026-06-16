"""Base adapter primitives for plug-and-play AL-1.0 integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..errors import AdapterError
from ..receipt import MODEL_OUTPUT_SOURCE_IDX
from ..tracing import DecodeStepLogger, SourceTagSidecar


@dataclass(slots=True)
class BaseAL10Adapter:
    """
    Framework-neutral adapter contract for AL-1.0 decode logging.

    Usage:
    1) start_trace(prompt_source_idx)
    2) for each decode step: log_decode_step(alpha_per_head)
    3) finalize_receipt(...)
    """

    model_output_source_idx: int = MODEL_OUTPUT_SOURCE_IDX
    sidecar: SourceTagSidecar | None = None
    logger: DecodeStepLogger = field(default_factory=DecodeStepLogger)

    def start_trace(self, context_source_idx: Sequence[int]) -> None:
        self.sidecar = SourceTagSidecar.from_context(context_source_idx)
        self.logger = DecodeStepLogger()

    def _require_trace_started(self) -> SourceTagSidecar:
        if self.sidecar is None:
            raise AdapterError("trace not started; call start_trace() first")
        return self.sidecar

    def log_decode_step(self, alpha_per_head: Sequence[Sequence[float]]) -> dict[int, float]:
        sidecar = self._require_trace_started()
        key_len = len(alpha_per_head[0]) if alpha_per_head else 0
        sidecar.validate_key_length(key_len)
        bucket = self.logger.log_decode_step(alpha_per_head, sidecar.source_idx)
        # After emitting each token, add generated token placeholder to cache sidecar.
        sidecar.append(self.model_output_source_idx)
        return bucket

    def finalize_receipt(
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
        self._require_trace_started()
        if not self.logger.per_step_buckets:
            raise AdapterError("no decode steps logged; cannot finalize receipt")
        return self.logger.receipt(
            idx_to_source_id,
            model_id=model_id,
            registry_manifest_hash=registry_manifest_hash,
            training_manifest_hash=training_manifest_hash,
            layer_policy=layer_policy,
            collapse_model_output=collapse_model_output,
            min_ratio=min_ratio,
            source_labels=source_labels,
        )
