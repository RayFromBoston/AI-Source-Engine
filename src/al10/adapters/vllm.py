"""vLLM-oriented adapter helpers for AL-1.0 decode logging."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..errors import AdapterError
from .pytorch import PyTorchDecodeAdapter


class VLLMDecodeAdapter(PyTorchDecodeAdapter):
    """
    Serving-focused helper for vLLM-style generation loops.

    vLLM integration patterns vary by version and backend. This adapter keeps the
    AL-1.0 side stable by accepting per-step attention tensors in any shape
    handled by `PyTorchDecodeAdapter.log_decode_step_from_tensor`.
    """

    def log_decode_steps_from_attentions(self, per_step_attentions: Sequence[Any]) -> list[dict[int, float]]:
        """Log multiple decode steps from a sequence of attention tensors."""
        buckets: list[dict[int, float]] = []
        for attention in per_step_attentions:
            buckets.append(self.log_decode_step_from_tensor(attention))
        return buckets

    def log_decode_steps_from_layered_outputs(
        self,
        decode_outputs: Sequence[Any],
        *,
        layer_selector: int = -1,
    ) -> list[dict[int, float]]:
        """
        Log decode steps when each step contains a per-layer container.

        Example expected shape:
            decode_outputs = [
                [layer0_attn, layer1_attn, ...],  # step 1
                [layer0_attn, layer1_attn, ...],  # step 2
            ]
        """
        buckets: list[dict[int, float]] = []
        for step in decode_outputs:
            if not isinstance(step, (list, tuple)) or not step:
                raise AdapterError("each decode output must be a non-empty list/tuple of layers")
            layer_attention = step[layer_selector]
            buckets.append(self.log_decode_step_from_tensor(layer_attention))
        return buckets

    def finalize_generation_result(
        self,
        *,
        text: str,
        token_ids: Sequence[int] | None,
        idx_to_source_id: Mapping[int, str],
        model_id: str,
        registry_manifest_hash: str,
        training_manifest_hash: str,
        layer_policy: str = "last_block_self_attn_mean_heads",
    ) -> dict[str, Any]:
        """Finalize a vLLM-style text result with an AL-1.0 receipt."""
        receipt = self.finalize_receipt(
            idx_to_source_id,
            model_id=model_id,
            registry_manifest_hash=registry_manifest_hash,
            training_manifest_hash=training_manifest_hash,
            layer_policy=layer_policy,
        )
        return {"text": text, "token_ids": list(token_ids or []), "attribution_receipt": receipt}

    @staticmethod
    def from_prompt_source_idx(source_idx: Sequence[int]) -> "VLLMDecodeAdapter":
        adapter = VLLMDecodeAdapter()
        adapter.start_trace(source_idx)
        return adapter
