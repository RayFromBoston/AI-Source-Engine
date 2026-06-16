"""Simple local HTTP API for AL-1.0 experimentation."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .receipt import aggregate_decode_step, build_receipt
from .validate import validate_receipt_dict


@dataclass(slots=True)
class ServerConfig:
    """Runtime configuration for local API server."""

    host: str = "127.0.0.1"
    port: int = 8765
    api_key: str | None = None
    rate_limit_per_minute: int = 0


class InMemoryRateLimiter:
    """Small in-memory fixed-window limiter keyed by client IP."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = max(0, int(limit_per_minute))
        self.window_seconds = 60.0
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit_per_minute <= 0:
            return True
        timestamp = float(now if now is not None else time.time())
        bucket = self._events.setdefault(key, deque())
        threshold = timestamp - self.window_seconds
        while bucket and bucket[0] < threshold:
            bucket.popleft()
        if len(bucket) >= self.limit_per_minute:
            return False
        bucket.append(timestamp)
        return True


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

    server_version = "AL10HTTP/0.2"
    api_key: str | None = None
    rate_limiter: InMemoryRateLimiter | None = None

    def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _is_authorized(self) -> bool:
        if not self.api_key:
            return True
        auth_header = self.headers.get("Authorization", "")
        provided_key = self.headers.get("X-API-Key", "")
        if auth_header.startswith("Bearer "):
            provided_key = auth_header.split(" ", 1)[1].strip()
        return provided_key == self.api_key

    def _check_guards(self) -> bool:
        # Keep /health open by default for liveness checks.
        if self.path == "/health":
            return True

        if not self._is_authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return False

        if self.rate_limiter is not None:
            client_ip = self.client_address[0] if self.client_address else "unknown"
            if not self.rate_limiter.allow(client_ip):
                self._write_json(HTTPStatus.TOO_MANY_REQUESTS, {"ok": False, "error": "rate limit exceeded"})
                return False
        return True

    def do_GET(self) -> None:  # noqa: N802 (HTTP handler naming)
        if not self._check_guards():
            return
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"ok": True, "service": "al10-http"})
            return
        if self.path == "/v1/demo":
            self._write_json(HTTPStatus.OK, {"ok": True, "attribution_receipt": create_demo_receipt()})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "route not found"})

    def do_POST(self) -> None:  # noqa: N802 (HTTP handler naming)
        if not self._check_guards():
            return
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


def build_handler(config: ServerConfig) -> type[AL10RequestHandler]:
    """Create configured request handler class for a specific server instance."""

    limiter = InMemoryRateLimiter(config.rate_limit_per_minute) if config.rate_limit_per_minute > 0 else None

    class ConfiguredHandler(AL10RequestHandler):
        api_key = config.api_key
        rate_limiter = limiter

    return ConfiguredHandler


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    api_key: str | None = None,
    rate_limit_per_minute: int = 0,
) -> None:
    """Run local AL-1.0 HTTP server."""
    config = ServerConfig(
        host=host,
        port=port,
        api_key=api_key,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    handler = build_handler(config)
    with ThreadingHTTPServer((config.host, config.port), handler) as server:
        print(
            json.dumps(
                {
                    "ok": True,
                    "service": "al10-http",
                    "host": config.host,
                    "port": config.port,
                    "auth": bool(config.api_key),
                    "rate_limit_per_minute": config.rate_limit_per_minute,
                }
            )
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(json.dumps({"ok": True, "stopped": True}))
