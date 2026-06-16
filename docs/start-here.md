# Start Here: AI Source Engine (AL-1.0) Guide

This guide is the technical entry point for using the repository.

If you arrived from the Open Letter in `README.md`, start here.

## What this project is

AI Source Engine is a practical AL-1.0 starter kit for people who want to:

1. experiment with attribution receipts quickly,
2. integrate decode-step attribution into existing generation code, and
3. ship a stable JSON receipt format with invariants and validation checks.

It includes:

- a hardened core SDK (`al10`)
- framework adapters (base, PyTorch, Hugging Face wrapper, vLLM helper)
- CLI tools for registry/dataset/validation workflows
- tests and integration scaffolds

## Install

Base package:

```bash
python3 -m pip install -e .
```

Hugging Face helper extras:

```bash
python3 -m pip install -e ".[hf]"
```

## Quick start

Run the built-in demo:

```bash
al10 run-demo
```

Create a starter plugin scaffold:

```bash
al10 init-plugin --framework pytorch
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

## Core API surface

- `SourceRegistry`: source rows + deterministic manifest hash
- `SourceTagSidecar`: KV-aligned source index tracking
- `DecodeStepLogger`: decode-step bucket collection and receipt finalization
- `build_receipt(...)`: AL-1.0 JSON receipt generation
- `validate_receipt_file(...)`: invariant/schema checks

## Adapters

- `BaseAL10Adapter`: generic lifecycle
  - `start_trace(context_source_idx)`
  - `log_decode_step(alpha_per_head)`
  - `finalize_receipt(...)`
- `PyTorchDecodeAdapter`: logs from tensor-shaped attention outputs
- `HuggingFaceGenerateAdapter`: wraps `model.generate(...)` for quick experiments
- `VLLMDecodeAdapter`: serving-oriented helper for vLLM-style decode loops

## CLI commands

- `al10 init-registry`
- `al10 stamp-dataset`
- `al10 run-demo`
- `al10 validate-receipt`
- `al10 validate-manifests`
- `al10 init-plugin`
- `al10 make-receipt`
- `al10 serve-api`
- `al10 bench-smoke`
- `al10 train ...` (training ingest/stamping MVP commands)

## Documentation

- `docs/quickstart.md`
- `docs/integrations.md`
- `docs/engineering-guide.md`
- `docs/http-api.md`
- `docs/e2e-app.md`
- `docs/quality-and-ci.md`
- `docs/release-and-packaging.md`
- `docs/training-ingest-mvp.md`
- `docs/trainer-integrations.md`
- `docs/production-hardening.md`
- `docs/first-adopter-checklist.md`

## Signatories and open letter resources

- `README.md` (public Open Letter landing page)
- `OPEN_LETTER.md` (letter copy)
- `SIGNATORIES.md` (supports "sign by pull request")

## Repository layout

- `src/al10/` - SDK, adapters, CLI, validators, scaffolding
- `examples/` - runnable examples
- `tests/` - unit tests and command coverage
- `CHANGELOG.md` - release history
