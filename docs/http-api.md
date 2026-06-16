# Local HTTP API

The SDK now includes a lightweight standard-library HTTP server so people can
play with AL-1.0 receipts without wiring a full stack first.

Run:

```bash
python3 -m al10.cli serve-api --host 127.0.0.1 --port 8765
```

## Endpoints

### `GET /health`

Returns service health metadata.

### `GET /v1/demo`

Returns a deterministic demo payload:

```json
{
  "ok": true,
  "attribution_receipt": { "...": "..." }
}
```

### `POST /v1/receipt`

Build a receipt from request payload.

Request body:

```json
{
  "per_step_buckets": [
    { "1": 0.7, "2": 0.2, "-1": 0.1 },
    { "1": 0.5, "2": 0.3, "-1": 0.2 }
  ],
  "idx_to_source_id": {
    "1": "sha256:source-a",
    "2": "sha256:source-b",
    "-1": "PARAMETRIC",
    "-2": "MODEL_OUTPUT"
  },
  "model_id": "org/model@release",
  "registry_manifest_hash": "sha256:...",
  "training_manifest_hash": "sha256:..."
}
```

### `POST /v1/validate-receipt`

Validate an already-built receipt object. Returns `{ "ok": true, "validated": true }`
on success.

## cURL examples

```bash
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/v1/demo
```

```bash
curl -s http://127.0.0.1:8765/v1/receipt \
  -H 'content-type: application/json' \
  -d @tests/fixtures/golden_receipt_input.json
```
