# Start Guide (From Zero to Working)

If this is your first time here, read this like instructions from one person to another.

You do not need to know the full codebase first.
You only need to know what this system is for and what to run.

## What this system does in plain language

AI Source Engine does two practical jobs:

1. **When a model generates output, it can produce a receipt**
   - the receipt says how much each source influenced the response
2. **When you prepare training data, it keeps source identity attached to tokens**
   - so provenance is not lost during tokenization/packing

If this works correctly, you can audit source influence with real artifacts instead of guesses.

## What success looks like

You are successful when you can do all three:

1. generate a receipt that shows source ratios,
2. validate that receipt passes checks,
3. produce packed training rows where `input_ids` and `source_idx` stay aligned.

## 1) Install

```bash
python3 -m pip install -e .
```

Optional Hugging Face extras:

```bash
python3 -m pip install -e ".[hf]"
```

If install fails, fix install first. Do not continue until this step works.

## 2) Run a receipt demo (inference side)

```bash
al10 run-demo
```

What you should see:

- JSON output in terminal
- a `sources` list
- `ratio` values that add up to about `1.0`

If you do not see that, stop and fix this before moving on.

## 3) Validate a receipt

```bash
al10 validate-receipt path/to/receipt.json
```

What this confirms:

- required fields exist
- source entries are shaped correctly
- ratio sum is valid

## 4) Run training provenance flow (training side)

Run the baseline pipeline:

```bash
python3 -m al10.cli train registry-index --registry registry.jsonl --output source_index_table.json
python3 -m al10.cli train stamp --input corpus.jsonl --output stamped.jsonl --index-table source_index_table.json
python3 -m al10.cli train pack --input stamped.jsonl --output packed.jsonl --seq-len 512
python3 -m al10.cli train validate --input packed.jsonl
python3 -m al10.cli train manifest-build --registry registry.jsonl --packed packed.jsonl --output training_manifest.json
```

What you should see:

- packed output rows containing `input_ids` and `source_idx`
- validation passes
- manifest file is generated

Most important invariant:

- `len(input_ids) == len(source_idx)` on every row

## 5) Integrate with your generation stack

Use the example that matches your stack:

- `examples/pytorch_adapter_loop.py`
- `examples/hf_generate_wrapper.py`
- `examples/vllm_adapter_loop.py`

What integration should do:

1. start trace with source-index context
2. log decode-step attention
3. finalize receipt at generation end

## 6) Optional API mode

Run local server:

```bash
python3 -m al10.cli serve-api --host 127.0.0.1 --port 8765
```

What you should confirm:

- health endpoint responds
- receipt endpoint returns JSON receipt payload

## 7) Final pre-ship check

```bash
python3 -m unittest discover -s tests -v
al10 run-demo
```

If tests pass and demo receipt is valid, baseline system is operational.

## If you are stuck

Keep the debug order simple:

1. install works
2. demo receipt works
3. receipt validation works
4. training validation works

Do not skip ahead when one step is broken.
