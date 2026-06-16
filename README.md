# AI Source Engine (AL-1.0 Starter)

This repository now contains a practical, minimal implementation of the AL-1.0
attribution math described in *We All Die in the Dark*.

It focuses on the implementation core:

- per-head attention merge
- source-bucket logging vector `L`
- full-response source ratio computation
- JSON-ready attribution receipt output

## Install

```bash
python -m pip install -e .
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Quick example

```bash
python examples/minimal_receipt.py
```

## What's included

- `src/al10/math.py` - Step 5 math utilities
- `src/al10/receipt.py` - decode-step aggregation and final receipt builder
- `docs/engineering-guide.md` - implementation notes and integration guidance
- `tests/` - unit tests for invariants and receipt behavior

## Scope

This code is framework-agnostic and intentionally lightweight. It does **not**
replace your model architecture. It gives you a direct implementation target for
the post-softmax logging path and receipt emission pipeline.
