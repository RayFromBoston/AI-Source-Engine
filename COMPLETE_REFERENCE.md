# COMPLETE REFERENCE: Full Explanation of the Two Technical Parts

If you only read one file in this repository, this is the file to read.
It is written to explain the system itself from first principles, in plain
English, without requiring you to jump between other docs.

The core technical claim is simple:

AI Source Engine has two real mechanisms.

The first mechanism reads model attention during generation and turns that into
source influence ratios.
The second mechanism carries source identity through training data preparation
so each token keeps provenance attached to it.

Everything else in this repository exists to support, test, package, or expose
those two mechanisms.

## What this project is actually for

Most model systems today can answer questions but cannot show a reliable,
machine-checkable "receipt" of where an answer came from. They can produce a
citation-looking response, but there is usually no grounded data structure that
proves source influence in the underlying generation process.

This repository exists to close that gap.

It does not pretend to solve every policy, social, or philosophical question
about AI. It solves a very specific engineering and auditability problem:
attach provenance to model behavior in a structured, reproducible way.

To do that, it implements two technical parts:

- Part 1 (inference-time attribution ratios)
- Part 2 (training-time source-vector alignment)

Together, those two parts create a practical provenance layer.

## Part 1 in plain language: read attention numbers, output source ratios

When a transformer model generates text, each new token is produced through a
decode step. At each step, attention heads distribute weight over prior
positions in context.

Part 1 takes those raw attention weights and answers a direct question:
"How much did each source influence this response?"

### What Part 1 receives

At minimum, Part 1 needs two things at decode time:

1. Attention weights over context positions (per head, per step).
2. A mapping from each context position to a source identity integer
   (`source_idx`).

That means each position is not just a token location. It is also tagged with
where it came from.

### What Part 1 computes

For every decode step, the pipeline is:

1. Merge attention heads into one distribution over positions.
2. Use `source_idx` to map each position to a source bucket.
3. Sum mass per source bucket for that step.

After all decode steps:

4. Add source buckets across steps.
5. Normalize total source mass into ratios.
6. Emit a receipt with source ratios and metadata.

The practical result is a numeric breakdown such as:

- source A: 0.62
- source B: 0.27
- source C: 0.11

Those numbers are not "citation strings." They are derived from model attention
flow and stored as a structured receipt object that can be validated and
compared over time.

### What the Part 1 output is

Part 1 produces a receipt JSON artifact. The exact schema is implemented in the
SDK, but conceptually it contains:

- model/run identity metadata
- contributing sources
- normalized ratio per source
- invariants that can be checked by validation tooling

This output is designed for machine workflows, not just human reading:
store it, hash it, validate it, track drift, and use it in governance reports.

### Where Part 1 lives in the code

The numerical logic is mainly in:

- `src/al10/math.py`
- `src/al10/receipt.py`
- `src/al10/tracing.py`

Framework wiring for actual model stacks is in:

- `src/al10/adapters/pytorch.py`
- `src/al10/adapters/huggingface.py`
- `src/al10/adapters/vllm.py`

CLI and API surfaces expose this for direct use:

- `al10 run-demo`
- `al10 make-receipt`
- `al10 serve-api`

## Part 2 in plain language: carry source vectors through training data

Part 2 addresses a different failure mode.

Even if you can attribute outputs at inference time, provenance often gets lost
during training-data preprocessing. Raw text is tokenized, packed, shuffled, and
rewritten into training format. If source identity is not carried in parallel
with tokens, provenance disappears before training even starts.

Part 2 prevents that loss.

### What "put source vectors onto training data" actually means

Every training row carries two aligned arrays:

- `input_ids`: the token IDs fed to the model
- `source_idx`: source identity per token

Alignment is strict: index-by-index, token-by-token.

If `input_ids[137]` is a token from source 21, then `source_idx[137]` must also
equal 21. This alignment is the non-negotiable invariant.

### What Part 2 takes in

Part 2 requires:

- a source registry (known sources and metadata),
- corpus rows with source context,
- tokenizer configuration,
- packing/sharding configuration.

### What Part 2 does end to end

The training provenance pipeline performs these transformations:

1. Build deterministic source index mapping.
2. Stamp corpus rows with source IDs.
3. Tokenize text.
4. Expand source identity to token granularity.
5. Pack fixed-length sequences.
6. Validate that `len(input_ids) == len(source_idx)` for each row.
7. Emit manifests and hashes for reproducibility/audit.

By the end, you do not just have packed training data. You have packed training
data with source lineage still attached at token level.

### Why Part 2 matters operationally

Without Part 2, provenance claims at training time are mostly trust statements.
With Part 2, provenance can be inspected and verified as part of data QA,
pipeline audits, or compliance review.

It also enables quantitative reporting like source distribution by shard, by
dataset slice, or by run.

### Where Part 2 lives in the code

Core implementation:

- `src/al10/train/pipeline.py`
- `src/al10/train/tokenizer.py`
- `src/al10/train/io.py`
- `src/al10/registry.py`

Trainer consumption helpers:

- `src/al10/train/integrations.py`

CLI surface for pipeline execution:

- `al10 train registry-index`
- `al10 train stamp`
- `al10 train pack`
- `al10 train validate`
- `al10 train manifest-build`
- `al10 train report`

## How the two parts work together in a real system

Part 2 and Part 1 are not duplicates. They operate at different phases.

Part 2 operates in the training data pipeline and preserves source lineage in
the data you feed into training.

Part 1 operates during generation and transforms live attention behavior into
source influence ratios for each response.

So the combined picture is:

- training data remains provenance-aware before/through training prep,
- generated outputs get source-ratio receipts at inference time,
- both sides produce artifacts that can be validated and audited.

That is the full technical model.

## How to actually use this, without reading other docs

You can use the system in three practical modes.

### Mode A: prove Part 1 works in minutes

Install and run demo:

```bash
python3 -m pip install -e .
al10 run-demo
```

This executes the inference attribution flow and outputs a sample receipt.

If you want to wire it into a generation loop, use one of:

- `examples/pytorch_adapter_loop.py`
- `examples/hf_generate_wrapper.py`
- `examples/vllm_adapter_loop.py`

### Mode B: prove Part 2 works on a small dataset

Run the training provenance pipeline with `al10 train` commands to produce
packed rows where `input_ids` and `source_idx` stay aligned.

Typical flow:

```bash
python3 -m al10.cli train registry-index --registry registry.jsonl --output index.json
python3 -m al10.cli train stamp --input corpus.jsonl --index index.json --output stamped.jsonl
python3 -m al10.cli train pack --input stamped.jsonl --output packed.jsonl --seq-len 512
python3 -m al10.cli train validate --input packed.jsonl
python3 -m al10.cli train manifest-build --registry registry.jsonl --packed packed.jsonl --output training_manifest.json
```

When validation passes, you have verified token/source alignment is intact.

### Mode C: expose Part 1 behind an API

Run local API server:

```bash
python3 -m al10.cli serve-api --host 127.0.0.1 --port 8765
```

Then submit payloads and receive receipts via HTTP.
Optional API key and rate limiting controls are included for hardening.

## What this repository includes beyond the two technical parts

The Open Letter and signatory flow are not the numerical core, but they are
deliberately included so policy demand and implementation evidence ship together.

Public/governance files:

- `README.md` (open letter landing page)
- `SIGNATORIES.md` (public sign workflow)
- `.github/pull_request_template.md` (sign-by-PR support)

Technical onboarding:

- `START_HERE.md`

Core context references on `main`:

- *The AGI Safety Bible*
- *We All Die in the Dark*

## One-sentence definition of the whole system

AI Source Engine is a two-part provenance system where training data keeps
per-token source identity and generated outputs produce source-ratio receipts,
so attribution can be audited instead of assumed.

## Appendix: Tiny concrete example (Part 1)

Assume one decode step with 4 context positions:

- `context_source_idx = [10, 10, 21, 21]`
- merged attention over positions = `[0.2, 0.3, 0.1, 0.4]`

Bucket by source:

- source 10 gets `0.2 + 0.3 = 0.5`
- source 21 gets `0.1 + 0.4 = 0.5`

If later decode steps lean more toward source 10, a final normalized receipt
could become:

- source 10: 0.68
- source 21: 0.32

That is exactly what "read attention numbers, then output source ratios" means
in practice.
