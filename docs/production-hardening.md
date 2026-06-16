# Production hardening guide

This phase focuses on practical controls for running AL-1.0 training ingest in
larger and messier real-world environments.

## 1) Unknown-source policies

Use `al10 train stamp` with explicit policy:

- `error` (strict): halt on unknown source IDs
- `fallback` (audit-first): map unknowns to known fallback source ID
- `skip` (throughput-first): skip unknown rows

Example:

```bash
python3 -m al10.cli train stamp \
  --input corpus.jsonl \
  --output stamped.jsonl \
  --index-table source_index_table.json \
  --unknown-source-policy fallback \
  --fallback-source-id UNLICENSED_UNKNOWN
```

Recommended defaults:

- regulated/compliance environments: `error`
- broad web ingest pilots: `fallback` + explicit `UNLICENSED_UNKNOWN`

## 2) Streaming-friendly packing

`al10 train pack` now streams packed rows while writing output files.

For larger corpora:

```bash
python3 -m al10.cli train pack \
  --input stamped.jsonl \
  --output-dir train_shards \
  --rows-per-shard 5000 \
  --sequence-length 2048 \
  --shard-prefix train
```

This avoids building a giant packed output list in memory before writing.

## 3) Config-driven reproducibility

Use:

```bash
python3 -m al10.cli train init-config --output train_run.json
python3 -m al10.cli train run --config train_run.json
```

Keep the config file in version control for reproducible ingest runs.

## 4) Operational checkpoints

- Always store:
  - source index table
  - validation report
  - training manifest with shard hashes
- Ensure registry manifest hash is logged with each run
- Keep unknown-source policy explicit in configs (no implicit defaults)
