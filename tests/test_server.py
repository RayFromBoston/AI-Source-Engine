import pathlib
import sys
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib import request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.server import AL10RequestHandler, build_receipt_from_payload, create_demo_receipt
from al10.validate import validate_receipt_dict


class TestServerHelpers(unittest.TestCase):
    def test_create_demo_receipt(self) -> None:
        receipt = create_demo_receipt()
        self.assertEqual(receipt["receipt_spec"], "AL-1.0")
        self.assertGreater(receipt["generated_token_count"], 0)
        validate_receipt_dict(receipt)

    def test_build_receipt_from_payload(self) -> None:
        payload = {
            "per_step_buckets": [
                {"1": 0.7, "2": 0.2, "-1": 0.1},
                {"1": 0.5, "2": 0.3, "-1": 0.2},
            ],
            "idx_to_source_id": {"1": "sha256:source-a", "2": "sha256:source-b", "-1": "PARAMETRIC"},
            "model_id": "test/model@r1",
            "registry_manifest_hash": "sha256:registry",
            "training_manifest_hash": "sha256:training",
        }
        receipt = build_receipt_from_payload(payload)
        self.assertEqual(receipt["receipt_spec"], "AL-1.0")
        self.assertEqual(receipt["model_id"], "test/model@r1")
        validate_receipt_dict(receipt)

    def test_build_receipt_from_payload_missing_field(self) -> None:
        with self.assertRaises(ValueError):
            build_receipt_from_payload({"model_id": "x"})

    def test_http_server_routes(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), AL10RequestHandler)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                body = response.read().decode("utf-8")
            self.assertIn('"ok": true', body)

            payload = {
                "per_step_buckets": [
                    {"1": 0.7, "2": 0.2, "-1": 0.1},
                    {"1": 0.5, "2": 0.3, "-1": 0.2},
                ],
                "idx_to_source_id": {"1": "sha256:source-a", "2": "sha256:source-b", "-1": "PARAMETRIC"},
                "model_id": "test/model@r1",
                "registry_manifest_hash": "sha256:registry",
                "training_manifest_hash": "sha256:training",
            }
            req = request.Request(
                f"http://127.0.0.1:{port}/v1/receipt",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=5) as response:
                post_body = response.read().decode("utf-8")
            self.assertIn('"attribution_receipt"', post_body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
