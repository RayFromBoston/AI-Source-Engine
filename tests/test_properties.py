import pathlib
import random
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.math import merge_query_heads, source_bucket, validate_probability_sum
from al10.receipt import response_ratio


def random_distribution(length: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    values = [rng.random() for _ in range(length)]
    total = sum(values) or 1.0
    return [value / total for value in values]


class TestProperties(unittest.TestCase):
    def test_merge_and_bucket_preserve_mass(self) -> None:
        for trial in range(100):
            key_len = 5 + (trial % 4)
            head_count = 2 + (trial % 3)
            alpha_heads = [random_distribution(key_len, seed=(trial * 97 + head)) for head in range(head_count)]
            merged = merge_query_heads(alpha_heads)
            self.assertAlmostEqual(sum(merged), 1.0, places=12)

            source_idx = [1 if i % 2 == 0 else 2 for i in range(key_len)]
            buckets = source_bucket(merged, source_idx)
            self.assertTrue(validate_probability_sum(buckets))

    def test_response_ratio_normalized(self) -> None:
        per_step = []
        for trial in range(40):
            dist = random_distribution(3, seed=trial * 13)
            per_step.append({1: dist[0], 2: dist[1], -1: dist[2]})

        ratios = response_ratio(per_step)
        self.assertAlmostEqual(sum(ratios.values()), 1.0, places=12)
        for value in ratios.values():
            self.assertGreaterEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
