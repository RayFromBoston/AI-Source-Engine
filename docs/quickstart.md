# Quickstart (10 minutes)

## 1) Install

```bash
python3 -m pip install -e .
```

For Hugging Face helper support:

```bash
python3 -m pip install -e ".[hf]"
```

## 2) Run demo receipt

```bash
al10 run-demo
```

This prints an AL-1.0 receipt JSON with normalized source ratios.

## 3) Create a source registry row

```bash
al10 init-registry \
  --output registry.jsonl \
  --source-id sha256:source-a \
  --content-hash sha256:content-a \
  --uri https://example.com/a \
  --rightsholder-id entity:a
```

The command writes `registry.jsonl` and returns a deterministic
`registry_manifest_hash`.

## 4) Validate a receipt

```bash
al10 validate-receipt path/to/receipt.json
```

Validation checks:

- required fields exist
- source rows are well-formed
- ratios sum to ~1.0
- top-level AL-1.0 metadata is present

## 5) Scaffold an integration snippet

```bash
al10 init-plugin --framework pytorch
```

or:

```bash
al10 init-plugin --framework hf
```

or:

```bash
al10 init-plugin --framework vllm
```

This creates starter code that follows the adapter lifecycle.

## 6) Run performance smoke benchmark

```bash
al10 bench-smoke --steps 1000 --heads 32 --key-len 1024
```

Use this as a quick local check, not a replacement for production profiling.

## 7) Build receipt from JSON payload

```bash
python3 -m al10.cli make-receipt --input tests/fixtures/golden_receipt_input.json
```

## 8) Run local HTTP API

```bash
python3 -m al10.cli serve-api --host 127.0.0.1 --port 8765 --api-key demo-key --rate-limit-per-minute 120
```

## 9) Run end-to-end app example

```bash
PYTHONPATH=src python3 examples/e2e_http_app.py
```

## 10) Run training ingest MVP flow

See:

- `docs/training-ingest-mvp.md`

for the `al10 train` end-to-end stamping/packing/validation command sequence.
For policy and scale controls, see `docs/production-hardening.md`.

## 11) Run config-driven training pipeline

```bash
python3 -m al10.cli train init-config --output train_run.json
python3 -m al10.cli train run --config train_run.json
```
