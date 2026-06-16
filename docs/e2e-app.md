# End-to-end app flow

This document shows a complete local flow from decode-step buckets to a served
AL-1.0 receipt.

## Option A: one-shot CLI path

```bash
python3 -m al10.cli make-receipt --input tests/fixtures/golden_receipt_input.json
```

## Option B: local HTTP API path

Start server:

```bash
python3 -m al10.cli serve-api --host 127.0.0.1 --port 8765 --api-key demo-key --rate-limit-per-minute 120
```

Then post a payload:

```bash
curl -s http://127.0.0.1:8765/v1/receipt \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer demo-key' \
  -d @tests/fixtures/golden_receipt_input.json
```

## Full scripted example

Run:

```bash
PYTHONPATH=src python3 examples/e2e_http_app.py
```

This script:

1. boots an in-process API server with auth + rate limit settings,
2. sends a `/v1/receipt` request,
3. prints the resulting `{ok, attribution_receipt}` JSON.
