"""Simple local HTTP API for AL-1.0 experimentation."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .receipt import aggregate_decode_step, build_receipt
from .validate import validate_receipt_dict


def create_demo_receipt() -> dict[str, Any]:
    """Build the same deterministic demo receipt used by CLI docs."""
    source_idx = [1, 1, 2, -1]
    step_1_alpha_heads = [
        [0.5, 0.2, 0.2, 0.1],
        [0.4, 0.3, 0.2, 0.1],
    ]
    step_2_alpha_heads = [
        [0.2, 0.1, 0.6, 0.1],
        [0.3, 0.1, 0.5, 0.1],
    ]

    per_step = [
        aggregate_decode_step(step_1_alpha_heads, source_idx),
        aggregate_decode_step(step_2_alpha_heads, source_idx),
    ]

    return build_receipt(
        per_step,
        {1: "sha256:source-a", 2: "sha256:source-b", -1: "PARAMETRIC"},
        model_id="demo/al10-http@v0",
        registry_manifest_hash="sha256:registry-demo",
        training_manifest_hash="sha256:training-demo",
    )


def build_receipt_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build receipt from request payload.

    Required fields:
      - per_step_buckets: list[dict[int|str, float]]
      - idx_to_source_id: dict[int|str, str]
      - model_id: str
      - registry_manifest_hash: str
      - training_manifest_hash: str
    Optional:
      - layer_policy, collapse_model_output, min_ratio, source_labels
    """
    required = [
        "per_step_buckets",
        "idx_to_source_id",
        "model_id",
        "registry_manifest_hash",
        "training_manifest_hash",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")

    per_step = [
        {int(source_idx): float(value) for source_idx, value in row.items()}
        for row in payload["per_step_buckets"]
    ]
    idx_to_source_id = {
        int(source_idx): str(source_id) for source_idx, source_id in payload["idx_to_source_id"].items()
    }

    receipt = build_receipt(
        per_step,
        idx_to_source_id,
        model_id=str(payload["model_id"]),
        registry_manifest_hash=str(payload["registry_manifest_hash"]),
        training_manifest_hash=str(payload["training_manifest_hash"]),
        layer_policy=str(payload.get("layer_policy", "last_block_self_attn_mean_heads")),
        collapse_model_output=bool(payload.get("collapse_model_output", True)),
        min_ratio=float(payload.get("min_ratio", 0.0)),
        source_labels=payload.get("source_labels"),
    )
    validate_receipt_dict(receipt)
    return receipt


class AL10RequestHandler(BaseHTTPRequestHandler):
    """Small JSON API around AL-1.0 demo and receipt generation."""

    server_version = "AL10HTTP/0.1"

    def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (HTTP handler naming)
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"ok": True, "service": "al10-http"})
            return
        if self.path == "/v1/demo":
            self._write_json(HTTPStatus.OK, {"ok": True, "attribution_receipt": create_demo_receipt()})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "route not found"})

    def do_POST(self) -> None:  # noqa: N802 (HTTP handler naming)
        if self.path not in {"/v1/receipt", "/v1/validate-receipt"}:
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "route not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except json.JSONDecodeError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid JSON body"})
            return

        try:
            if self.path == "/v1/receipt":
                receipt = build_receipt_from_payload(payload)
                self._write_json(HTTPStatus.OK, {"ok": True, "attribution_receipt": receipt})
            else:
                validate_receipt_dict(payload)
                self._write_json(HTTPStatus.OK, {"ok": True, "validated": True})
        except Exception as exc:  # noqa: BLE001 - convert to API error payload.
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Keep CLI output clean by suppressing per-request logs.
        _ = format, args


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run local AL-1.0 HTTP server."""
    with ThreadingHTTPServer((host, port), AL10RequestHandler) as server:
        print(json.dumps({"ok": True, "service": "al10-http", "host": host, "port": port}))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(json.dumps({"ok": True, "stopped": True}))
