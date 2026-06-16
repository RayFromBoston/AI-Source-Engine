# Engineering Guide (Complete)

This is the complete technical guide for how AI Source Engine works and how to integrate it.

## System in one paragraph

AI Source Engine is a provenance layer for transformer models. It does two things:

1. at inference time, it reads decode-step attention and computes per-source influence ratios
2. at training ingest time, it carries source identity (`source_idx`) at token level so provenance survives data preparation

The output is auditable artifacts (receipts, manifests, validated aligned rows), not trust-only claims.

## Part 1: Inference attribution ratio engine

### Inputs

- per-step attention weights (`alpha`) over key positions
- source mapping per key position (`source_idx`)

### Math

Per head:

\[
\alpha_{h,i,j} = \mathrm{softmax}_j\left(\frac{q_{h,i}\cdot k_{h,j}}{\sqrt{d_k}}\right)
\]

Head merge:

\[
\alpha_{i,j} = \frac{1}{H}\sum_{h=1}^{H}\alpha_{h,i,j}
\]

Source bucket for generated token `i`:

\[
L_i(S) = \sum_{j:S(j)=S}\alpha_{i,j}
\]

Invariant:

\[
\sum_{S}L_i(S)\approx 1.0
\]

Full response ratio across `N` decode steps:

\[
\mathrm{Ratio}(S)=\frac{1}{N}\sum_{i=1}^{N}L_i(S)
\]

### Runtime flow

1. keep `source_idx` sidecar aligned to KV key positions
2. at decode (`T_q = 1`), read attention after softmax
3. merge heads and bucket by source
4. append per-step buckets
5. normalize and emit receipt

### Receipt output (minimum shape)

```json
{
  "receipt_spec": "AL-1.0",
  "model_id": "org/model@release",
  "registry_manifest_hash": "sha256:...",
  "training_manifest_hash": "sha256:...",
  "generated_token_count": 142,
  "layer_policy": "last_block_self_attn_mean_heads",
  "sources": [
    { "source_id": "sha256:abc...", "ratio": 0.412 },
    { "source_id": "sha256:def...", "ratio": 0.334 },
    { "source_id": "PARAMETRIC", "ratio": 0.254 }
  ]
}
```

Constraints:

- source ratios sum to 1.0 within float tolerance
- `source_idx -> source_id` mapping happens before output
- layer/policy stays stable and is documented per model release

### Tiny concrete example

Assume one decode step with 4 context positions:

- `context_source_idx = [10, 10, 21, 21]`
- merged attention over positions = `[0.2, 0.3, 0.1, 0.4]`

Bucket by source:

- source 10 gets `0.2 + 0.3 = 0.5`
- source 21 gets `0.1 + 0.4 = 0.5`

If later steps lean toward source 10, final normalized output might become:

- source 10: 0.68
- source 21: 0.32

## Part 2: Training source-vector pipeline

### Goal

Keep provenance attached to tokens through training ingest.

Each row must keep:

- `input_ids`
- `source_idx`

with strict invariant:

- `len(input_ids) == len(source_idx)`

### Flow

1. build source index table from registry
2. stamp corpus rows with source IDs
3. tokenize rows
4. expand source IDs to token level
5. pack fixed-length sequences
6. validate alignment invariant
7. emit manifests + hashes + reports

### CLI sequence

```bash
python3 -m al10.cli train registry-index --registry registry.jsonl --output source_index_table.json
python3 -m al10.cli train stamp --input corpus.jsonl --output stamped.jsonl --index-table source_index_table.json
python3 -m al10.cli train pack --input stamped.jsonl --output packed.jsonl --seq-len 512
python3 -m al10.cli train validate --input packed.jsonl
python3 -m al10.cli train manifest-build --registry registry.jsonl --packed packed.jsonl --output training_manifest.json
python3 -m al10.cli train report --input packed.jsonl --output source_report.json
```

## Core code paths

- Part 1 math/receipts:
  - `src/al10/math.py`
  - `src/al10/receipt.py`
  - `src/al10/tracing.py`
- Adapters:
  - `src/al10/adapters/pytorch.py`
  - `src/al10/adapters/huggingface.py`
  - `src/al10/adapters/vllm.py`
- Part 2 training pipeline:
  - `src/al10/train/pipeline.py`
  - `src/al10/train/tokenizer.py`
  - `src/al10/train/io.py`
  - `src/al10/train/integrations.py`
  - `src/al10/registry.py`

## What to run before shipping

```bash
python3 -m unittest discover -s tests -v
al10 run-demo
python3 -m al10.cli train validate --input packed.jsonl
```
