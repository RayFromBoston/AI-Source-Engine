"""Scaffolding helpers for plug-and-play integration snippets."""

from __future__ import annotations

from pathlib import Path


PYTORCH_TEMPLATE = """\
from al10.adapters import PyTorchDecodeAdapter

# 1) Build source_idx sidecar for prompt KV positions.
prompt_source_idx = [...]  # one int source idx per prompt token
adapter = PyTorchDecodeAdapter.from_prompt_source_idx(prompt_source_idx)

# 2) In decode loop, feed last-layer attention each step.
#    Expected shape examples: [B,H,Tq,Tk] or [H,Tk]
for attention_tensor in decode_step_attentions:
    adapter.log_decode_step_from_tensor(attention_tensor)

# 3) Finalize receipt when generation finishes.
receipt = adapter.finalize_receipt(
    idx_to_source_id=idx_to_source_id,
    model_id="org/model@release",
    registry_manifest_hash="sha256:...",
    training_manifest_hash="sha256:...",
)
print(receipt)
"""


HUGGINGFACE_TEMPLATE = """\
from transformers import AutoModelForCausalLM, AutoTokenizer
from al10.adapters import HuggingFaceGenerateAdapter

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

adapter = HuggingFaceGenerateAdapter()
result = adapter.generate_with_receipt(
    model=model,
    tokenizer=tokenizer,
    prompt="Explain AL-1.0 in one paragraph.",
    prompt_source_idx=None,  # defaults to PARAMETRIC for prompt tokens
    idx_to_source_id={-1: "PARAMETRIC", -2: "MODEL_OUTPUT"},
    model_id=f"{model_name}@local",
    registry_manifest_hash="sha256:...",
    training_manifest_hash="sha256:...",
    generation_kwargs={"max_new_tokens": 40},
)
print(result["text"])
print(result["attribution_receipt"])
"""


def write_scaffold(framework: str, output_path: str | None = None) -> Path:
    framework_key = framework.strip().lower()
    if framework_key == "pytorch":
        template = PYTORCH_TEMPLATE
    elif framework_key in {"hf", "huggingface"}:
        template = HUGGINGFACE_TEMPLATE
        framework_key = "huggingface"
    else:
        raise ValueError("framework must be one of: pytorch, hf, huggingface")

    path = Path(output_path or f"al10_plugin_{framework_key}.py")
    path.write_text(template, encoding="utf-8")
    return path
