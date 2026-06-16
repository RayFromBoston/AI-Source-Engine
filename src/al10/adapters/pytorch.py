"""PyTorch-friendly adapter for AL-1.0 decode logging."""

from __future__ import annotations

from typing import Any, Sequence

from ..errors import AdapterError
from .base import BaseAL10Adapter


def _depth(value: Any) -> int:
    level = 0
    current = value
    while isinstance(current, list) and current:
        level += 1
        current = current[0]
    return level


def _to_list(value: Any) -> list:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, list):
        return value
    raise AdapterError("expected tensor-like object with tolist()/detach() or a nested list")


class PyTorchDecodeAdapter(BaseAL10Adapter):
    """
    Adapter for attention tensors from PyTorch-style decode paths.

    Supports last-layer attention with common shapes:
    - [B, H, T_q, T_k]
    - [H, T_q, T_k]
    - [H, T_k]
    """

    def log_decode_step_from_tensor(
        self,
        attention_tensor: Any,
        *,
        batch_index: int = 0,
        query_index: int = -1,
    ) -> dict[int, float]:
        payload = _to_list(attention_tensor)
        dims = _depth(payload)

        # Normalize to [H, T_k] for one decode token.
        if dims == 4:
            # [B, H, T_q, T_k]
            per_head = [head[query_index] for head in payload[batch_index]]
        elif dims == 3:
            # Could be [H, T_q, T_k] or [B, H, T_k].
            first = payload[0]
            if first and isinstance(first[0], list):
                # [H, T_q, T_k]
                per_head = [head[query_index] for head in payload]
            else:
                # [B, H, T_k]
                per_head = payload[batch_index]
        elif dims == 2:
            # Already [H, T_k]
            per_head = payload
        else:
            raise AdapterError(f"unsupported attention shape depth: {dims}")

        if not per_head or not isinstance(per_head[0], list):
            raise AdapterError("normalized attention must be shaped [H, T_k]")
        return self.log_decode_step(per_head)

    @staticmethod
    def from_prompt_source_idx(source_idx: Sequence[int]) -> "PyTorchDecodeAdapter":
        adapter = PyTorchDecodeAdapter()
        adapter.start_trace(source_idx)
        return adapter
