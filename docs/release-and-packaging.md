# Release and packaging

This project is now packaging-ready with:

- source/wheel build via `python -m build`
- metadata checks via `python -m twine check dist/*`
- release workflow for tagged builds (`.github/workflows/release.yml`)

## Local packaging verification

```bash
python3 -m pip install --upgrade build twine
rm -rf dist build src/*.egg-info
python3 -m build
python3 -m twine check dist/*
```

Optional local wheel install check:

```bash
python3 -m pip install --user --force-reinstall dist/ai_source_engine-*.whl
python3 -m al10.cli --version
```

## Versioning policy

- package version lives in two places and is enforced by tests:
  - `pyproject.toml` (`project.version`)
  - `src/al10/version.py` (`__version__`)
- changelog must include a heading for current version:
  - `## [x.y.z]`

## Tag release flow

1. Update `CHANGELOG.md` and version fields.
2. Merge to `main`.
3. Create tag, e.g. `v0.2.0`.
4. Push tag.
5. GitHub release workflow builds and attaches artifacts.
