# Quickstart

This file shows exactly how to run AI Source Engine fast, then points you to the full implementation guide.

## What this does

AI Source Engine has two technical jobs:

1. read attention numbers and produce source-ratio receipts at inference time
2. preserve per-token source identity (`source_idx`) through training data prep

If you want full detail, read:

- `ENGINEERING_GUIDE.md`

## 1) Install

```bash
python3 -m pip install -e .
```

Optional Hugging Face extras:

```bash
python3 -m pip install -e ".[hf]"
```

## 2) Generate a demo receipt (Part 1)

```bash
al10 run-demo
```

This prints an AL-1.0 receipt JSON with normalized source ratios.

## 3) Validate a receipt

```bash
al10 validate-receipt path/to/receipt.json
```

## 4) Integrate into a generation loop

Use one of these examples:

- `examples/pytorch_adapter_loop.py`
- `examples/hf_generate_wrapper.py`
- `examples/vllm_adapter_loop.py`

## 5) Run training provenance flow (Part 2)

```bash
python3 -m al10.cli train registry-index --registry registry.jsonl --output source_index_table.json
python3 -m al10.cli train stamp --input corpus.jsonl --output stamped.jsonl --index-table source_index_table.json
python3 -m al10.cli train pack --input stamped.jsonl --output packed.jsonl --seq-len 512
python3 -m al10.cli train validate --input packed.jsonl
```

## 6) Sign the open letter

- Read `README.md`
- Add your entry in `SIGNATORIES.md` by Pull Request
