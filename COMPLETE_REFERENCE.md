# COMPLETE REFERENCE: Open Letter + AI Source Engine

This document explains how the two parts of this project work together:

1. the public accountability layer (Open Letter + signatures), and
2. the technical implementation layer (AL-1.0 attribution tooling).

---

## 1) Part One: Public Accountability Layer

The public-facing side defines *why* attribution logging matters and how supporters join.

### Canonical files

- `README.md` - canonical Open Letter landing page (first page visitors see)
- `SIGNATORIES.md` - public signatory list and "sign by pull request" flow
- `.github/pull_request_template.md` - PR template with signatory submission fields
- `OPEN_LETTER.md` - canonical pointer to README letter

### How signatory collection works

#### Option A: Web form sign

People can sign through a hosted Name/Title/Email form linked from `README.md`.

#### Option B: Sign by Pull Request

Developers and researchers can sign on GitHub:

1. Fork repository.
2. Add one row to `SIGNATORIES.md`.
3. Open PR with signatory details.
4. Maintainers review and merge.

This route provides lightweight identity verification through public GitHub accounts.

### Maintainer moderation loop

1. Ensure one row was added in `SIGNATORIES.md`.
2. Confirm fields are complete and clearly non-duplicative.
3. Merge PR if valid.
4. Reject or request changes if incomplete/spam/duplicate.

---

## 2) Part Two: Technical Attribution Layer (AL-1.0)

The technical side defines *how* attribution logging is implemented in code.

### Core implementation modules

- `src/al10/math.py` - attention head merge + source bucketing math
- `src/al10/receipt.py` - decode aggregation and receipt construction
- `src/al10/models.py` - typed receipt and registry models
- `src/al10/tracing.py` - runtime decode-step tracing/logging helpers
- `src/al10/registry.py` - source registry + deterministic manifest hashes
- `src/al10/validate.py` - receipt and manifest validation

### Integration adapters

- `src/al10/adapters/pytorch.py` - PyTorch decode logging
- `src/al10/adapters/huggingface.py` - Hugging Face `generate()` wrapper
- `src/al10/adapters/vllm.py` - vLLM-style serving/decode integration

### CLI and API

- CLI entrypoint: `src/al10/cli.py`
- HTTP API server: `src/al10/server.py`

Representative commands:

- `al10 run-demo`
- `al10 make-receipt --input ...`
- `al10 validate-receipt ...`
- `al10 serve-api --host 127.0.0.1 --port 8765`

### Training-stack pipeline

Training data provenance pipeline lives under `src/al10/train/` and `al10 train ...` commands:

1. build source index from registry
2. stamp corpus rows with source identity
3. tokenize while preserving source alignment
4. pack fixed-length training rows
5. validate invariants (`len(input_ids) == len(source_idx)`)
6. build manifest + hashes
7. report source distribution

Key docs:

- `docs/training-ingest-mvp.md`
- `docs/trainer-integrations.md`
- `docs/production-hardening.md`

---

## 3) How Both Parts Work Together

These two parts are intentionally coupled:

- The Open Letter defines the governance and policy demand: mandatory attribution logging.
- The AL-1.0 engine demonstrates that attribution logging is implementable now.

In practical terms:

1. The letter and signatories build public and regulatory pressure.
2. The technical toolkit gives builders an immediate path to implementation.
3. Receipts, manifests, and validation create auditable artifacts rather than trust-only claims.

This is why the project ships both advocacy files and production-minded engineering code in one repo.

---

## 4) Recommended Read Order

1. `README.md` (Open Letter)
2. `START_HERE.md` (technical onboarding)
3. `COMPLETE_REFERENCE.md` (this full map)
4. `docs/quickstart.md`
5. `docs/training-ingest-mvp.md` and `docs/trainer-integrations.md`

---

## 5) Core references on `main`

The broader context for this repository also references:

- *The AGI Safety Bible*
- *We All Die in the Dark*
