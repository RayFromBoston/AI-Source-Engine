# Engineering Guide (How It Works Internally)

This guide is for engineers who want to inspect, modify, or rebuild the system.

The goal here is not just to run commands. The goal is to understand the moving
parts deeply enough to change them safely.

## Mental model first

Treat the system as two engines that share source identity: 

1. **Inference engine**: turns live attention behavior into per-source ratios
2. **Training ingest engine**: keeps source identity aligned with tokens through
   preprocessing

If either engine is wrong, provenance trust breaks:

- inference wrong -> receipts are misleading
- training ingest wrong -> source identity is corrupted before training

## Part 1 internals: inference attribution ratio engine

### Data entering this engine

At each decode step:

- attention weights by head over key positions
- `source_idx` for each key position

Conceptually:

- `alpha_per_head[head][key_pos]`
- `context_source_idx[key_pos]`

### Core equations

Per-head attention:

\[
\alpha_{h,i,j} = \mathrm{softmax}_j\left(\frac{q_{h,i}\cdot k_{h,j}}{\sqrt{d_k}}\right)
\]

Merge heads (default policy):

\[
\alpha_{i,j} = \frac{1}{H}\sum_{h=1}^{H}\alpha_{h,i,j}
\]

Bucket by source for generated token `i`:

\[
L_i(S) = \sum_{j:S(j)=S}\alpha_{i,j}
\]

Per-step invariant:

\[
\sum_{S}L_i(S)\approx 1.0
\]

Aggregate across `N` decode steps:

\[
\mathrm{Ratio}(S)=\frac{1}{N}\sum_{i=1}^{N}L_i(S)
\]

### Runtime algorithm (practical)

For each generated token:

1. read per-head attention
2. merge heads into one key-position distribution
3. bucket mass by `source_idx`
4. append per-step source bucket vector

After generation ends:

5. sum buckets across steps
6. normalize to ratios
7. map `source_idx` back to source IDs
8. emit receipt JSON

### Tiny numeric example

One decode step:

- `context_source_idx = [10, 10, 21, 21]`
- merged attention = `[0.2, 0.3, 0.1, 0.4]`

Bucket result:

- source 10 = `0.2 + 0.3 = 0.5`
- source 21 = `0.1 + 0.4 = 0.5`

After many steps, normalized totals might be:

- source 10 = 0.68
- source 21 = 0.32

That final vector is what becomes receipt source ratios.

### Receipt shape (minimum)

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

Hard constraints:

- source ratios sum to approximately 1.0
- no missing source mapping at output time
- stable layer/policy declaration per release

## Part 2 internals: training source-vector ingest engine

### Problem this solves

Raw text preprocessing can destroy provenance if source identity is not carried
with tokens. Part 2 prevents that.

### Required output structure

Each training row must contain:

- `input_ids`
- `source_idx`

Hard invariant:

- `len(input_ids) == len(source_idx)` for every row

### Pipeline stages

1. build deterministic source index table from registry
2. stamp corpus rows with source IDs
3. tokenize text
4. expand source identity to token-level vector
5. pack fixed-length sequences
6. validate token/source length alignment
7. emit manifests/hashes/reports

### Why this matters

If token/source alignment breaks once, downstream attribution claims become
unreliable. This invariant is the integrity boundary.

### Baseline CLI sequence

```bash
python3 -m al10.cli train registry-index --registry registry.jsonl --output source_index_table.json
python3 -m al10.cli train stamp --input corpus.jsonl --output stamped.jsonl --index-table source_index_table.json
python3 -m al10.cli train pack --input stamped.jsonl --output packed.jsonl --seq-len 512
python3 -m al10.cli train validate --input packed.jsonl
python3 -m al10.cli train manifest-build --registry registry.jsonl --packed packed.jsonl --output training_manifest.json
python3 -m al10.cli train report --input packed.jsonl --output source_report.json
```

## Code map (where to edit what)

Inference math and receipts:

- `src/al10/math.py`
- `src/al10/receipt.py`
- `src/al10/tracing.py`

Framework adapters:

- `src/al10/adapters/pytorch.py`
- `src/al10/adapters/huggingface.py`
- `src/al10/adapters/vllm.py`

Training ingest:

- `src/al10/train/pipeline.py`
- `src/al10/train/tokenizer.py`
- `src/al10/train/io.py`
- `src/al10/train/integrations.py`
- `src/al10/registry.py`

## Engineering checks before modifying behavior

Before merging any algorithmic change:

1. run tests
2. run demo receipt
3. validate packed training rows
4. manually inspect one sample receipt for ratio sanity

Commands:

```bash
python3 -m unittest discover -s tests -v
al10 run-demo
python3 -m al10.cli train validate --input packed.jsonl
```

If these pass and invariants remain true, change is usually safe to review.
