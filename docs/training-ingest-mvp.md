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

### 3) Pack fixed-length sequences

```bash
python3 -m al10.cli train pack \
  --input stamped.jsonl \
  --output packed.jsonl \
  --sequence-length 2048 \
  --include-labels
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
- Teams can swap tokenization logic in future iterations with model-native tokenizers.
- Hard invariant: each row must satisfy `len(input_ids) == len(source_idx)`.
