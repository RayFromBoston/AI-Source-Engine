import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.train import (
    AL10TrainingDataset,
    as_hf_trainer_dataset,
    build_torch_collate_fn,
    pad_batch,
    write_jsonl,
)


class TestTrainIntegrations(unittest.TestCase):
    def test_dataset_from_rows_and_jsonl(self) -> None:
        rows = [
            {"input_ids": [1, 2, 3], "source_idx": [1, 1, 1]},
            {"input_ids": [4, 5], "source_idx": [2, 2], "labels": [4, 5]},
        ]
        dataset = AL10TrainingDataset.from_rows(rows)
        self.assertEqual(len(dataset), 2)
        self.assertIn("input_ids", dataset[0])
        self.assertIn("source_idx", dataset[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "packed.jsonl"
            write_jsonl(path, rows)
            loaded = AL10TrainingDataset.from_jsonl(path)
            self.assertEqual(len(loaded), 2)

    def test_pad_batch(self) -> None:
        batch = [
            {"input_ids": [1, 2, 3], "source_idx": [1, 1, 1]},
            {"input_ids": [4], "source_idx": [2]},
        ]
        padded = pad_batch(batch, pad_token_id=0, pad_source_idx=0, label_pad_id=-100)
        self.assertEqual(len(padded["input_ids"]), 2)
        self.assertEqual(len(padded["input_ids"][0]), len(padded["input_ids"][1]))
        self.assertIn("attention_mask", padded)

    def test_hf_dataset_wrapper(self) -> None:
        rows = [{"input_ids": [1, 2], "source_idx": [1, 1]}]
        dataset = as_hf_trainer_dataset(rows)
        self.assertEqual(len(dataset), 1)
        self.assertIn("input_ids", dataset[0])

    def test_torch_collate_fn_optional_dependency(self) -> None:
        collate = build_torch_collate_fn()
        batch = [{"input_ids": [1, 2], "source_idx": [1, 1]}]
        try:
            import torch  # noqa: F401
        except Exception:
            with self.assertRaises(RuntimeError):
                _ = collate(batch)
        else:
            out = collate(batch)
            self.assertIn("input_ids", out)
            self.assertEqual(len(out["input_ids"].shape), 2)


if __name__ == "__main__":
    unittest.main()
