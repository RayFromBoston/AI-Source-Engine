import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.math import merge_query_heads, source_bucket, validate_probability_sum
from al10.receipt import (
    MODEL_OUTPUT_SOURCE_IDX,
    aggregate_decode_step,
    build_receipt,
    response_ratio,
)


class TestAL10Math(unittest.TestCase):
    def test_merge_query_heads_mean(self) -> None:
        alpha_heads = [
            [0.6, 0.4],
            [0.2, 0.8],
        ]
        merged = merge_query_heads(alpha_heads)
        self.assertAlmostEqual(merged[0], 0.4)
        self.assertAlmostEqual(merged[1], 0.6)

    def test_source_bucket(self) -> None:
        alpha_ij = [0.4, 0.1, 0.2, 0.3]
        source_idx = [1, 1, 2, -1]
        buckets = source_bucket(alpha_ij, source_idx)
        self.assertAlmostEqual(buckets[1], 0.5)
        self.assertAlmostEqual(buckets[2], 0.2)
        self.assertAlmostEqual(buckets[-1], 0.3)
        self.assertTrue(validate_probability_sum(buckets))


class TestAL10Receipt(unittest.TestCase):
    def test_response_ratio(self) -> None:
        per_step = [{1: 0.7, 2: 0.3}, {1: 0.5, 2: 0.5}]
        ratios = response_ratio(per_step)
        self.assertAlmostEqual(ratios[1], 0.6)
        self.assertAlmostEqual(ratios[2], 0.4)
        self.assertAlmostEqual(sum(ratios.values()), 1.0)

    def test_build_receipt_collapse_model_output(self) -> None:
        per_step = [{1: 0.6, MODEL_OUTPUT_SOURCE_IDX: 0.4}]
        idx_to_source_id = {1: "sha256:a", -1: "PARAMETRIC"}
        receipt = build_receipt(
            per_step,
            idx_to_source_id,
            model_id="org/model@r1",
            registry_manifest_hash="sha256:registry",
            training_manifest_hash="sha256:training",
            collapse_model_output=True,
        )
        self.assertEqual(receipt["receipt_spec"], "AL-1.0")
        self.assertEqual(receipt["generated_token_count"], 1)
        self.assertAlmostEqual(sum(row["ratio"] for row in receipt["sources"]), 1.0)
        self.assertTrue(any(row["source_id"] == "PARAMETRIC" for row in receipt["sources"]))

    def test_aggregate_decode_step(self) -> None:
        alpha_heads = [
            [0.4, 0.3, 0.2, 0.1],
            [0.2, 0.3, 0.3, 0.2],
        ]
        source_idx = [1, 1, 2, -1]
        buckets = aggregate_decode_step(alpha_heads, source_idx)
        self.assertAlmostEqual(sum(buckets.values()), 1.0)
        self.assertAlmostEqual(buckets[1], 0.6)
        self.assertAlmostEqual(buckets[2], 0.25)
        self.assertAlmostEqual(buckets[-1], 0.15)


if __name__ == "__main__":
    unittest.main()
