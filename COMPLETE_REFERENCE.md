# COMPLETE REFERENCE: What This Is, How It Works, and How to Use It

This repository combines two things on purpose:

1. a **public demand** (the Open Letter), and
2. a **working implementation** (AI Source Engine / AL-1.0).

If you only read one paragraph, read this:

> The project exists to make model outputs auditable by producing a mathematical
> provenance receipt that shows which sources influenced a response, instead of
> asking people to trust a black box.

---

## 1) What is this, in plain English?

### Short version

AI Source Engine is a toolkit that logs model attention at generation time and
turns it into a structured receipt of source influence.

### Why that matters

Today, most models can generate fluent text but cannot prove where a response
came from. This system adds an attribution layer so teams can answer:

- "Which source(s) most influenced this answer?"
- "How much came from known sources vs parametric/model memory?"
- "Can we verify provenance claims with machine-readable artifacts?"

### What is the "Open Letter" part?

The Open Letter is the policy/advocacy side. It explains why attribution
logging should become a baseline requirement and lets supporters sign publicly.

### What is the "engineering" part?

The AL-1.0 engine is the implementation side. It provides code, CLI tools,
adapters, validators, and training-pipeline helpers to actually do attribution
logging in real workflows.

---

## 2) What are the major parts?

### Part A: Public accountability layer

- `README.md`: canonical Open Letter landing page (first thing visitors see)
- `SIGNATORIES.md`: public list of supporters
- `.github/pull_request_template.md`: supports "sign by pull request"
- `OPEN_LETTER.md`: pointer to README as canonical letter

Purpose: build public pressure and collect visible support.

### Part B: Technical attribution layer

- core SDK (`src/al10/`) for attribution math + receipt construction
- framework adapters (PyTorch, Hugging Face, vLLM-style)
- CLI and HTTP API for usage without custom app scaffolding
- training ingest pipeline for provenance-aligned training data
- tests/CI/validation for reliability and reproducibility

Purpose: make attribution operational, testable, and deployable.

---

## 3) How the inference attribution system works (step by step)

This is the core mechanism when generating model output.

### Inputs

- a generated token stream
- per-step attention weights (per head) over prior context positions
- a mapping from each context position to `source_idx`

### Pipeline

1. **Collect decode-step attention**
   For each generated token, capture per-head attention weights over prior
   positions.

2. **Merge heads**
   Reduce multi-head attention into one merged distribution per decode step
   (mean across heads in this implementation).

3. **Bucket by source**
   Map each context position to its `source_idx` and sum attention mass into
   per-source buckets.

4. **Accumulate across generation**
   Add per-step buckets across all generated tokens.

5. **Normalize**
   Convert accumulated source mass into ratios that sum to approximately 1.0.

6. **Build receipt**
   Emit JSON with metadata and ordered source contributions.

### Output

A receipt that includes:

- model/run metadata
- per-source ratios
- special handling for model-side/internal sources
- invariants validated by helper functions/tests

In other words: instead of "trust us," you get an artifact you can inspect,
store, validate, and compare.

---

## 4) How the training provenance pipeline works (step by step)

The training side makes source attribution survive data preparation.

1. **Create source registry**
   Build a registry of source IDs and metadata.

2. **Build source index table**
   Convert source IDs to deterministic integer IDs (`source_idx`) for efficient
   token-level storage.

3. **Stamp corpus**
   Attach source identity to raw corpus rows.

4. **Tokenize with alignment**
   Produce `input_ids` and matching `source_idx` arrays with 1:1 token-level
   alignment.

5. **Pack sequences**
   Pack tokenized rows into fixed sequence lengths for model training.

6. **Validate hard invariant**
   Ensure every row keeps `len(input_ids) == len(source_idx)`.

7. **Generate manifests + hashes**
   Produce deterministic manifests and content hashes for auditability.

8. **Report distribution**
   Output source distribution summaries for governance and QA review.

This gives teams not only attribution at inference time, but provenance-aware
training inputs that can be audited and reproduced.

---

## 5) How to use this repo (practical paths)

### Path 1: "I just want to see it work quickly"

1. Install package
2. Run demo receipt
3. Validate receipt

Commands:

```bash
python3 -m pip install -e .
al10 run-demo
al10 validate-receipt path/to/receipt.json
```

### Path 2: "I want to integrate it into generation"

1. choose adapter (PyTorch, HF, or vLLM-style)
2. start trace with source context
3. log decode steps during generation
4. finalize receipt at end of response

See:

- `examples/pytorch_adapter_loop.py`
- `examples/hf_generate_wrapper.py`
- `examples/vllm_adapter_loop.py`

### Path 3: "I want provenance-aware training data"

1. create registry
2. run `al10 train` pipeline (index -> stamp -> pack -> validate -> manifest)
3. load packed rows into trainer integrations

See:

- `docs/training-ingest-mvp.md`
- `docs/trainer-integrations.md`
- `examples/train_pytorch_integration.py`
- `examples/train_hf_trainer_integration.py`

### Path 4: "I need an API endpoint"

Run local API server and submit payloads to create receipts:

```bash
python3 -m al10.cli serve-api --host 127.0.0.1 --port 8765
```

Supports optional API key auth and rate limiting.

---

## 6) What problem each component solves

- **math + receipt modules**: convert raw attention into auditable attribution
- **registry + manifests**: create stable source identity and reproducible audit
  artifacts
- **adapters**: make integration into existing model stacks practical
- **CLI/API**: reduce integration friction and enable automation
- **training pipeline**: preserve provenance through data prep
- **tests + validation**: prove invariants and prevent silent regressions
- **open letter + signatories**: move policy and accountability in parallel with
  implementation

---

## 7) How the two halves connect (the point of this repo)

Without the public side, this is "just another SDK."
Without the technical side, the letter is "just a statement."

Together:

- the letter states the requirement (mandatory attribution logging),
- the engine shows it can be done now,
- receipts/manifests provide verifiable evidence,
- signatories create public legitimacy and pressure for adoption.

That combination is the whole point.

---

## 8) Canonical read order

1. `README.md` - Open Letter and call to action
2. `START_HERE.md` - technical onboarding
3. `COMPLETE_REFERENCE.md` - this full conceptual + practical explanation
4. `docs/quickstart.md` - quick commands
5. deeper docs by need:
   - inference integration: `docs/integrations.md`
   - training pipeline: `docs/training-ingest-mvp.md`
   - trainer loading: `docs/trainer-integrations.md`
   - ops hardening: `docs/production-hardening.md`

---

## 9) Core references on `main`

This project context also references:

- *The AGI Safety Bible*
- *We All Die in the Dark*
