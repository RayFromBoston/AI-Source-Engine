# Training ingest MVP

This guide covers the new `al10 train` command family for building
training-ready, provenance-aligned datasets.

## What it does

The pipeline converts corpus rows into packed training rows with:

- `input_ids`
- `source_idx` (same length as `input_ids`)

It also emits index tables, manifests, and validation reports so runs are
auditable.

## Command flow

### 1) Build source index table

```bash
python3 -m al10.cli train registry-index \
  --registry registry.jsonl \
  --output source_index_table.json \
  --ensure-source-id UNLICENSED_UNKNOWN
```

### 2) Stamp + tokenize corpus rows

Input JSONL rows should include at least `text`, and optionally `source_id`.

```bash
python3 -m al10.cli train stamp \
  --input corpus.jsonl \
  --output stamped.jsonl \
  --index-table source_index_table.json \
  --default-source-id UNLICENSED_UNKNOWN
```

Use model-native tokenization via Hugging Face backend:

```bash
python3 -m al10.cli train stamp \
  --input corpus.jsonl \
  --output stamped.jsonl \
  --index-table source_index_table.json \
  --tokenizer-backend hf \
  --tokenizer-name gpt2
```

### 3) Pack fixed-length sequences

```bash
python3 -m al10.cli train pack \
  --input stamped.jsonl \
  --output packed.jsonl \
  --sequence-length 2048 \
  --include-labels
```

For larger corpora, write multiple shard files:

```bash
python3 -m al10.cli train pack \
  --input stamped.jsonl \
  --output-dir train_shards \
  --sequence-length 2048 \
  --rows-per-shard 5000 \
  --shard-prefix train
```

### 4) Validate invariants

```bash
python3 -m al10.cli train validate \
  --input packed.jsonl \
  --report-output validate_report.json
```

### 5) Build run manifest with shard hashes

```bash
python3 -m al10.cli train manifest-build \
  --run-id run-001 \
  --registry-manifest-hash sha256:... \
  --shard packed.jsonl \
  --output training_manifest.json
```

### 6) Produce source distribution report

```bash
python3 -m al10.cli train report \
  --input packed.jsonl \
  --index-table source_index_table.json \
  --output source_report.json
```

## Notes

- The MVP tokenizer is deterministic and dependency-free (`simple-whitespace-v1`).
- Optional HF tokenizer backend is supported via `--tokenizer-backend hf --tokenizer-name ...`.
- Hard invariant: each row must satisfy `len(input_ids) == len(source_idx)`.
