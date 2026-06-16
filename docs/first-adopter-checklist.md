# First adopter checklist

This checklist is for teams adopting AL-1.0 in a real training + serving stack.

Use it as a rollout playbook. Each section has practical go/no-go gates.

## 1) Team and ownership

- [ ] Identify one technical owner for data ingest/stamping.
- [ ] Identify one owner for training integration.
- [ ] Identify one owner for serving/inference integration.
- [ ] Define where manifests/registry/reports will be stored long-term.

**Go/No-Go gate:** every stage has an explicit owner.

## 2) Source registry readiness

- [ ] Create `registry.jsonl` and confirm each source has:
  - `source_id`
  - `content_hash`
  - `uri`
  - `rightsholder_id`
- [ ] Decide and document unknown source policy:
  - `error`, `fallback`, or `skip`
- [ ] If using fallback, reserve and register fallback source ID
      (e.g. `UNLICENSED_UNKNOWN`).

**Go/No-Go gate:** registry exists, hash is stable, policy is documented.

## 3) Ingest pipeline dry run

- [ ] Run `al10 train registry-index`.
- [ ] Run `al10 train stamp` on a sample corpus.
- [ ] Confirm output rows satisfy:
  - `len(input_ids) == len(source_idx)`
- [ ] Run `al10 train pack` and `al10 train validate`.
- [ ] Build manifest with `al10 train manifest-build`.
- [ ] Generate source distribution report with `al10 train report`.

**Go/No-Go gate:** validation passes and report output is sensible.

## 4) Config-driven reproducibility

- [ ] Create config template:
  - `al10 train init-config --output train_run.json`
- [ ] Check config into repo (or managed config store).
- [ ] Execute full run:
  - `al10 train run --config train_run.json`
- [ ] Store outputs:
  - source index table
  - stamped/packed shards
  - validation report
  - training manifest

**Go/No-Go gate:** one command can reliably reproduce same artifacts.

## 5) Trainer integration

- [ ] Load packed shards with `AL10TrainingDataset`.
- [ ] Use `build_torch_collate_fn()` (PyTorch) or `as_hf_trainer_dataset()` (HF).
- [ ] Verify batches carry `source_idx` in training loop.
- [ ] Confirm labels/padding behavior in one minibatch sanity check.

**Go/No-Go gate:** training loop consumes packed data without custom hacks.

## 6) Serving integration

- [ ] Integrate one adapter path:
  - PyTorch decode path, HF generate wrapper, or vLLM-style helper
- [ ] Emit receipt JSON for at least one response path.
- [ ] Validate receipts with `al10 validate-receipt`.
- [ ] Ensure model/version + manifest identifiers are attached.

**Go/No-Go gate:** serving returns text plus valid attribution receipt.

## 7) Quality and release checks

- [ ] Run full tests:
  - `python3 -m unittest discover -s tests -v`
- [ ] Run packaging checks:
  - `python3 -m build`
  - `python3 -m twine check dist/*`
- [ ] Verify version/changelog consistency.

**Go/No-Go gate:** CI and packaging checks are green.

## 8) Production operations

- [ ] Decide default API auth/rate-limit settings for local HTTP utility.
- [ ] Decide shard sizing policy for large corpus runs.
- [ ] Define retention policy for manifests and validation reports.
- [ ] Define incident workflow for unknown-source spikes or registry drift.

**Go/No-Go gate:** runbook exists and on-call can follow it.

## 9) Initial launch criteria

- [ ] At least one training run produced auditable shards/manifests.
- [ ] At least one model-serving path emits receipts.
- [ ] At least one downstream consumer validated receipts.
- [ ] Team signoff from data + training + serving owners.

**Launch when all four are true.**

---

## Suggested first rollout scope

Start with a bounded pilot:

1. one model family,
2. one curated subset of corpus,
3. one serving endpoint,
4. one reporting dashboard or notebook.

Then expand gradually after stable weekly runs.
