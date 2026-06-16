# AL-1.0 Engineering Guide (Implementation Starter)

This document turns the paper into a software implementation target.

## Step 5: Math recap

Per head:

\[
\alpha_{h,i,j} = \mathrm{softmax}_j\left(\frac{q_{h,i}\cdot k_{h,j}}{\sqrt{d_k}}\right)
\]

Head merge (AL-1.0 default):

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

## Runtime integration points

1. Keep `source_idx` as a sidecar array aligned to key positions in KV cache.
2. During decode (`T_q = 1`), read attention weights after softmax in the selected layer.
3. Compute `L_i(S)` by:
   - averaging across heads
   - summing by `source_idx`
4. Append each `L_i` to `L_per_step`.
5. At generation end, compute `Ratio(S)` and emit JSON receipt.

## Receipt schema (minimum)

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

- source ratios must sum to `1.0` (within floating-point tolerance)
- map `source_idx -> source_id` before output
- keep policy stable between releases and state it on the model card

## Included reference module

The `src/al10` package includes:

- `merge_query_heads(...)`
- `source_bucket(...)`
- `aggregate_decode_step(...)`
- `response_ratio(...)`
- `build_receipt(...)`

This provides a drop-in starting point for serving stacks that can access
decode-step attention weights.
