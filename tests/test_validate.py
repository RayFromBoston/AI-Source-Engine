import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.errors import ReceiptValidationError
from al10.registry import canonical_json, sha256_text
from al10.validate import validate_manifest_hash, validate_receipt_dict, validate_receipt_file


class TestValidate(unittest.TestCase):
    def test_validate_receipt_dict_success(self) -> None:
        payload = {
            "receipt_spec": "AL-1.0",
            "model_id": "org/model@r1",
            "registry_manifest_hash": "sha256:registry",
            "training_manifest_hash": "sha256:training",
            "generated_token_count": 1,
            "layer_policy": "last_block_self_attn_mean_heads",
            "sources": [
                {"source_id": "sha256:a", "ratio": 0.6},
                {"source_id": "PARAMETRIC", "ratio": 0.4},
            ],
        }
        validate_receipt_dict(payload)

    def test_validate_receipt_dict_rejects_bad_sum(self) -> None:
        payload = {
            "receipt_spec": "AL-1.0",
            "model_id": "org/model@r1",
            "registry_manifest_hash": "sha256:registry",
            "training_manifest_hash": "sha256:training",
            "generated_token_count": 1,
            "layer_policy": "last_block_self_attn_mean_heads",
            "sources": [{"source_id": "sha256:a", "ratio": 0.2}],
        }
        with self.assertRaises(ReceiptValidationError):
            validate_receipt_dict(payload)

    def test_validate_receipt_file(self) -> None:
        payload = {
            "receipt_spec": "AL-1.0",
            "model_id": "org/model@r1",
            "registry_manifest_hash": "sha256:registry",
            "training_manifest_hash": "sha256:training",
            "generated_token_count": 1,
            "layer_policy": "last_block_self_attn_mean_heads",
            "sources": [{"source_id": "PARAMETRIC", "ratio": 1.0}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "receipt.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            validate_receipt_file(path)

    def test_validate_manifest_hash(self) -> None:
        payload = {"run_id": "run-1", "registry_manifest_hash": "sha256:abc", "shards": ["a"]}
        expected = sha256_text(canonical_json(payload))
        payload["manifest_hash"] = expected
        validate_manifest_hash(payload, payload["manifest_hash"])


if __name__ == "__main__":
    unittest.main()
