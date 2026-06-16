import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.train import (
    build_training_manifest_with_hashes,
    invert_index_table,
    pack_tokenized_rows,
    stamp_corpus_rows,
    tokenize_stamped_rows,
    validate_training_rows,
    write_jsonl,
)


class TestTrainPipeline(unittest.TestCase):
    def test_stamp_and_tokenize_alignment(self) -> None:
        idx_to_source = {1: "sha256:source-a", 2: "UNLICENSED_UNKNOWN"}
        source_to_idx = invert_index_table(idx_to_source)

        rows = [
            {"text": "hello world", "source_id": "sha256:source-a"},
            {"text": "fallback source row"},
        ]
        stamped = stamp_corpus_rows(rows, default_source_id="UNLICENSED_UNKNOWN")
        tokenized = tokenize_stamped_rows(stamped, source_to_idx=source_to_idx)
        self.assertEqual(len(tokenized), 2)
        for row in tokenized:
            self.assertEqual(len(row["input_ids"]), len(row["source_idx"]))

    def test_pack_and_validate(self) -> None:
        rows = [
            {"row_id": 1, "source_id": "sha256:source-a", "input_ids": [1, 2, 3], "source_idx": [1, 1, 1]},
            {"row_id": 2, "source_id": "sha256:source-b", "input_ids": [4, 5], "source_idx": [2, 2]},
        ]
        packed = pack_tokenized_rows(rows, sequence_length=2, include_labels=True)
        self.assertGreaterEqual(len(packed), 2)
        for row in packed:
            self.assertEqual(len(row["input_ids"]), len(row["source_idx"]))
            self.assertEqual(row["labels"], row["input_ids"])

        report = validate_training_rows(packed)
        self.assertTrue(report["ok"])
        self.assertGreater(report["tokens"], 0)

    def test_manifest_with_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            shard_path = pathlib.Path(temp_dir) / "packed.jsonl"
            write_jsonl(shard_path, [{"input_ids": [1, 2], "source_idx": [1, 1], "token_count": 2}])
            manifest = build_training_manifest_with_hashes(
                run_id="run-1",
                registry_manifest_hash="sha256:registry",
                shard_paths=[str(shard_path)],
            )
            self.assertIn("manifest_hash", manifest)
            self.assertIn("shard_hashes", manifest)
            self.assertIn(str(shard_path), manifest["shard_hashes"])


if __name__ == "__main__":
    unittest.main()
