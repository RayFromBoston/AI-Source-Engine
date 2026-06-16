import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.receipt import build_receipt


class TestGoldenReceipt(unittest.TestCase):
    def test_golden_receipt_fixture(self) -> None:
        fixtures = pathlib.Path(__file__).parent / "fixtures"
        input_payload = json.loads((fixtures / "golden_receipt_input.json").read_text(encoding="utf-8"))
        expected = json.loads((fixtures / "golden_receipt_output.json").read_text(encoding="utf-8"))

        per_step_buckets = [
            {int(source_idx): float(mass) for source_idx, mass in row.items()}
            for row in input_payload["per_step_buckets"]
        ]
        idx_to_source_id = {int(idx): source_id for idx, source_id in input_payload["idx_to_source_id"].items()}

        receipt = build_receipt(
            per_step_buckets,
            idx_to_source_id,
            model_id=input_payload["model_id"],
            registry_manifest_hash=input_payload["registry_manifest_hash"],
            training_manifest_hash=input_payload["training_manifest_hash"],
            layer_policy=input_payload["layer_policy"],
        )

        self.assertEqual(receipt["receipt_spec"], expected["receipt_spec"])
        self.assertEqual(receipt["model_id"], expected["model_id"])
        self.assertEqual(receipt["generated_token_count"], expected["generated_token_count"])
        self.assertEqual(len(receipt["sources"]), len(expected["sources"]))

        for got, want in zip(receipt["sources"], expected["sources"]):
            self.assertEqual(got["source_id"], want["source_id"])
            self.assertAlmostEqual(got["ratio"], want["ratio"], places=12)


if __name__ == "__main__":
    unittest.main()
