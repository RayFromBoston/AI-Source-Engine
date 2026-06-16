"""Hugging Face generate() integration helper for AL-1.0 receipts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..errors import AdapterError
from .pytorch import PyTorchDecodeAdapter


class HuggingFaceGenerateAdapter(PyTorchDecodeAdapter):
    """
    Small wrapper around `model.generate` for quick AL-1.0 experimentation.

    This adapter assumes a decoder-style model with attention outputs enabled.
    """

    def generate_with_receipt(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt: str,
        idx_to_source_id: Mapping[int, str],
        model_id: str,
        registry_manifest_hash: str,
        training_manifest_hash: str,
        prompt_source_idx: Sequence[int] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        layer_policy: str = "last_block_self_attn_mean_heads",
    ) -> dict[str, Any]:
        kwargs = dict(generation_kwargs or {})
        kwargs["return_dict_in_generate"] = True
        kwargs["output_attentions"] = True

        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_ids = inputs["input_ids"][0].tolist()
        if prompt_source_idx is None:
            prompt_source_idx = [-1] * len(prompt_ids)
        if len(prompt_source_idx) != len(prompt_ids):
            raise AdapterError("prompt_source_idx length must equal number of prompt tokens")

        self.start_trace(prompt_source_idx)
        output = model.generate(**inputs, **kwargs)

        attentions = getattr(output, "attentions", None) or getattr(output, "decoder_attentions", None)
        if not attentions:
            raise AdapterError(
                "generate() did not return attentions; ensure output_attentions=True "
                "and use a model/generate path that supports attention outputs"
            )

        for step in attentions:
            # generate() typically returns tuple(step_count) of tuple(layer_count) tensors.
            layer_attention = step[-1] if isinstance(step, (tuple, list)) else step
            self.log_decode_step_from_tensor(layer_attention)

        sequence = output.sequences[0]
        token_ids = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)

        receipt = self.finalize_receipt(
            idx_to_source_id,
            model_id=model_id,
            registry_manifest_hash=registry_manifest_hash,
            training_manifest_hash=training_manifest_hash,
            layer_policy=layer_policy,
        )
        return {"text": text, "token_ids": token_ids, "attribution_receipt": receipt}
