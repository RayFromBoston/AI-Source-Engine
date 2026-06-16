"""End-to-end local AL-1.0 app flow using the HTTP API."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from al10.server import ServerConfig, build_handler


def main() -> None:
    handler = build_handler(ServerConfig(api_key="demo-key", rate_limit_per_minute=20))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        payload = {
            "per_step_buckets": [
                {"1": 0.7, "2": 0.2, "-1": 0.1},
                {"1": 0.5, "2": 0.3, "-1": 0.2},
            ],
            "idx_to_source_id": {
                "1": "sha256:source-a",
                "2": "sha256:source-b",
                "-1": "PARAMETRIC",
                "-2": "MODEL_OUTPUT",
            },
            "model_id": "demo/e2e-http@v0",
            "registry_manifest_hash": "sha256:registry-demo",
            "training_manifest_hash": "sha256:training-demo",
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/receipt",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer demo-key"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
        print(json.dumps(body, indent=2, sort_keys=True))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
