# Integration guide

This project is designed around one adapter lifecycle:

1. **start_trace** with prompt/context `source_idx`
2. **log decode steps** with attention tensors
3. **finalize_receipt** with model/manifest metadata

## PyTorch loop integration

```python
from al10.adapters import PyTorchDecodeAdapter

adapter = PyTorchDecodeAdapter.from_prompt_source_idx(prompt_source_idx)

for attention_tensor in decode_step_attentions:
    adapter.log_decode_step_from_tensor(attention_tensor)

receipt = adapter.finalize_receipt(
    idx_to_source_id=idx_to_source_id,
    model_id="org/model@release",
    registry_manifest_hash="sha256:...",
    training_manifest_hash="sha256:...",
)
```

`log_decode_step_from_tensor` supports common shapes:

- `[B, H, T_q, T_k]`
- `[H, T_q, T_k]`
- `[H, T_k]`

## Hugging Face `generate()` wrapper

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from al10.adapters import HuggingFaceGenerateAdapter

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

adapter = HuggingFaceGenerateAdapter()
result = adapter.generate_with_receipt(
    model=model,
    tokenizer=tokenizer,
    prompt="Explain AL-1.0 briefly.",
    idx_to_source_id={-1: "PARAMETRIC", -2: "MODEL_OUTPUT"},
    model_id="gpt2@local",
    registry_manifest_hash="sha256:...",
    training_manifest_hash="sha256:...",
)
```

## vLLM-style integration helper

```python
from al10.adapters import VLLMDecodeAdapter

adapter = VLLMDecodeAdapter.from_prompt_source_idx(prompt_source_idx)
adapter.log_decode_steps_from_layered_outputs(decode_outputs, layer_selector=-1)
result = adapter.finalize_generation_result(
    text=generated_text,
    token_ids=generated_token_ids,
    idx_to_source_id=idx_to_source_id,
    model_id="org/model@release",
    registry_manifest_hash="sha256:...",
    training_manifest_hash="sha256:...",
)
```

## Dataset stamping utility

If your training rows already contain tokenized `input_ids`, use:

```bash
al10 stamp-dataset \
  --input raw.jsonl \
  --output stamped.jsonl \
  --default-source-idx 42
```

This ensures each row has `source_idx` and enforces:

`len(source_idx) == len(input_ids)`

## Receipt validation in pipelines

Use one of:

- CLI: `al10 validate-receipt receipt.json`
- Python: `from al10.validate import validate_receipt_dict`

This is useful both in CI and before returning user-facing outputs.

## HTTP API sandbox

For quick local integration tests without embedding Python directly:

```bash
python3 -m al10.cli serve-api --host 127.0.0.1 --port 8765 --api-key demo-key --rate-limit-per-minute 120
```

Then call:

- `GET /health`
- `GET /v1/demo`
- `POST /v1/receipt`
- `POST /v1/validate-receipt`
