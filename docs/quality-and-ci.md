# Quality and CI

This starter includes multiple quality gates to make adoption safer:

## Test layers

- **Unit tests**: core math, receipts, validators, adapters, CLI
- **Golden tests**: fixture-based deterministic receipt output
- **Property tests**: randomized distributions for invariants

## CI workflow

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

- Python 3.10, 3.11, 3.12 matrix
- package install (`pip install -e .`)
- full unittest suite
- CLI smoke (`al10 run-demo`)
- example smoke (`examples/pytorch_adapter_loop.py`)

## Recommended local checks

```bash
python3 -m unittest discover -s tests -v
al10 bench-smoke --steps 1000 --heads 32 --key-len 1024
```

The benchmark command is a smoke benchmark, not a production profiler.
