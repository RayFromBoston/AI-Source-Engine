# Start Guide (Production Use)

This guide is for the person who wants to use this project right now, from zero, without reverse-engineering the repo.

## What this project does

AI Source Engine gives you two things:

1. **Inference attribution receipts**
   - during generation, it reads attention behavior and outputs source influence ratios
2. **Training provenance alignment**
   - during training ingest, it carries `source_idx` with tokens so source identity survives data preparation

If you run this correctly, you end up with auditable artifacts instead of trust-only claims.

## What files matter first

If you are operating this system, these are the first files to care about:

- `src/al10/cli.py` — command entrypoints (`al10 ...`)
- `src/al10/receipt.py` — receipt construction logic
- `src/al10/train/pipeline.py` — training provenance pipeline
- `src/al10/server.py` — local HTTP API entrypoint
- `examples/` — runnable integration examples

## Setup

Install:

```bash
python3 -m pip install -e .
```

Optional Hugging Face extras:

```bash
python3 -m pip install -e ".[hf]"
```

## Part 1: Run inference attribution

### A) Fast demo

```bash
al10 run-demo
```

Expected result:

- command prints a JSON receipt with `sources` and `ratio` values
- ratios should sum to approximately `1.0`

### B) Validate a receipt

```bash
al10 validate-receipt path/to/receipt.json
```

Expected result:

- validation passes
- no missing required fields
- ratio sum invariant holds

### C) Integrate into generation loop

Use one of these example entrypoints:

- `examples/pytorch_adapter_loop.py`
- `examples/hf_generate_wrapper.py`
- `examples/vllm_adapter_loop.py`

What you should see:

- decode steps are logged
- a final receipt is produced at generation end

## Part 2: Run training provenance pipeline

Use this baseline flow:

```bash
python3 -m al10.cli train registry-index --registry registry.jsonl --output source_index_table.json
python3 -m al10.cli train stamp --input corpus.jsonl --output stamped.jsonl --index-table source_index_table.json
python3 -m al10.cli train pack --input stamped.jsonl --output packed.jsonl --seq-len 512
python3 -m al10.cli train validate --input packed.jsonl
python3 -m al10.cli train manifest-build --registry registry.jsonl --packed packed.jsonl --output training_manifest.json
```

Expected results:

- packed rows contain `input_ids` and `source_idx`
- invariant `len(input_ids) == len(source_idx)` passes on validation
- manifest is generated for audit/reproducibility

## Optional: run local API mode

```bash
python3 -m al10.cli serve-api --host 127.0.0.1 --port 8765
```

Expected result:

- `/health` responds
- `/v1/receipt` accepts payload and returns receipt JSON

## Signatory flow

To sign as a supporter:

1. add one row in `SIGNATORIES.md`
2. open a Pull Request

## Final operational check

Before shipping:

```bash
python3 -m unittest discover -s tests -v
al10 run-demo
```

If tests pass and demo emits a valid receipt, the baseline deployment path is healthy.
