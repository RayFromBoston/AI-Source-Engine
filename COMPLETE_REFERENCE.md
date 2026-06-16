# COMPLETE REFERENCE: The Two Technical Parts, Exactly

This is the direct explanation of the system, without assuming the reader has
opened any other file.

You are correct: the technical system has two core parts.

1. **Part 1: Inference attribution engine**
   - reads attention numbers during generation
   - outputs source contribution ratios in a receipt
2. **Part 2: Training provenance pipeline**
   - carries source vectors (`source_idx`) through training data preparation
   - outputs training-ready data where every token still knows its source

Everything else in this repo is support around those two parts.

---

## What problem this solves

Most model stacks can generate text, but cannot prove where an answer came
from. This creates an attribution gap.

This project closes that gap by:

- producing source-ratio receipts at inference time (Part 1), and
- preserving per-token source identity in training data (Part 2).

So you get audit artifacts at both generation time and training-data time.

---

## PART 1: Read attention numbers and output source ratios

### What it takes in

For each decode step (each generated token), Part 1 needs:

- attention weights per head over prior positions
- a `source_idx` value for each prior position

So conceptually:

- `alpha_per_head[head][position] -> attention mass`
- `context_source_idx[position] -> source ID integer`

### What it does

For each decode step:

1. merge attention heads into one distribution over positions
2. map positions to sources using `context_source_idx`
3. sum attention mass into per-source buckets

Across the full response:

4. accumulate all per-step source buckets
5. normalize to ratios that sum to approximately 1.0
6. emit structured receipt JSON

### Tiny concrete example

Assume one decode step with 4 context positions:

- `context_source_idx = [10, 10, 21, 21]`
- merged attention over positions = `[0.2, 0.3, 0.1, 0.4]`

Bucket by source:

- source 10 gets `0.2 + 0.3 = 0.5`
- source 21 gets `0.1 + 0.4 = 0.5`

If later steps lean more toward source 10, final normalized output might become:

- source 10: 0.68
- source 21: 0.32

That is the ratio receipt.

### What it outputs

A receipt object with:

- run/model metadata
- list of contributing sources
- ratio for each source
- invariants (validated by tooling)

### Where this logic lives in code

- `src/al10/math.py` (head merge + source bucketing)
- `src/al10/receipt.py` (aggregate + normalize + receipt build)
- `src/al10/tracing.py` (decode-step logging lifecycle)
- `src/al10/adapters/*` (framework-specific wiring)

### How to use Part 1 quickly

Minimal flow:

1. start trace
2. log decode steps (attention tensors)
3. finalize receipt

Quick run:

```bash
python3 -m pip install -e .
al10 run-demo
```

Integration examples:

- `examples/pytorch_adapter_loop.py`
- `examples/hf_generate_wrapper.py`
- `examples/vllm_adapter_loop.py`

---

## PART 2: Put source vectors onto training data

### What "put vectors onto training data" means

Each token used for training gets a parallel source identity value.

So every training row carries two aligned arrays:

- `input_ids`: token IDs
- `source_idx`: source IDs per token, same length

If token 37 came from source 21, then `source_idx[37] == 21`.

### What it takes in

- source registry (source IDs and metadata)
- raw corpus rows (text + source identity)
- tokenizer configuration
- packing configuration (sequence length, sharding options)

### What it does

1. build deterministic source index table
2. stamp corpus rows with source IDs
3. tokenize rows
4. expand source labels to token level (1:1 alignment)
5. pack tokens into fixed-length training rows
6. validate invariant: `len(input_ids) == len(source_idx)` for every row
7. emit manifests/hashes for reproducibility

### Why this matters

Without this, provenance is lost during data preprocessing.
With this, source identity survives into the training-ready dataset.

### Where this logic lives in code

- `src/al10/train/pipeline.py`
- `src/al10/train/tokenizer.py`
- `src/al10/train/io.py`
- `src/al10/train/integrations.py`
- `src/al10/registry.py`

### How to use Part 2 quickly

High-level CLI flow:

```bash
python3 -m al10.cli train registry-index --registry registry.jsonl --output index.json
python3 -m al10.cli train stamp --input corpus.jsonl --index index.json --output stamped.jsonl
python3 -m al10.cli train pack --input stamped.jsonl --output packed.jsonl --seq-len 512
python3 -m al10.cli train validate --input packed.jsonl
python3 -m al10.cli train manifest-build --registry registry.jsonl --packed packed.jsonl --output training_manifest.json
```

Then load packed rows into training stacks with helpers in
`src/al10/train/integrations.py`.

---

## How the two parts fit together

Part 2 (training provenance) makes source identity available and durable in data
pipelines.

Part 1 (inference attribution) uses source identity and attention behavior to
produce output-time source influence receipts.

Combined outcome:

- training data keeps source lineage
- generated answers produce source-ratio receipts
- both can be validated and audited

That is the full technical system.

---

## What is not core (supporting layer)

These are important, but they are not the two core mechanisms:

- `README.md` open letter and petition positioning
- `SIGNATORIES.md` sign-by-PR workflow
- CI, release workflow, docs navigation files

They support adoption and governance. The two core technical parts are still:

1. inference ratio engine
2. training source-vector pipeline

---

## If someone wants to verify the system in 10 minutes

1. Run demo receipt (`al10 run-demo`) to see Part 1 output.
2. Run `al10 train ...` pipeline on tiny sample data to see Part 2 alignment.
3. Check that packed rows preserve `len(input_ids) == len(source_idx)`.
4. Inspect generated receipt and training manifest files.

If those checks pass, they have seen both parts working.
