import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10 import cli


class TestCliTrain(unittest.TestCase):
    def test_train_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            registry_path = temp / "registry.jsonl"
            index_path = temp / "index.json"
            corpus_path = temp / "corpus.jsonl"
            stamped_path = temp / "stamped.jsonl"
            packed_path = temp / "packed.jsonl"
            report_path = temp / "validate_report.json"
            manifest_path = temp / "manifest.json"
            source_report_path = temp / "source_report.json"

            code = cli.main(
                [
                    "init-registry",
                    "--output",
                    str(registry_path),
                    "--source-id",
                    "sha256:source-a",
                    "--content-hash",
                    "sha256:content-a",
                    "--uri",
                    "https://example.com/a",
                    "--rightsholder-id",
                    "entity:a",
                ]
            )
            self.assertEqual(code, 0)

            code = cli.main(
                [
                    "train",
                    "registry-index",
                    "--registry",
                    str(registry_path),
                    "--output",
                    str(index_path),
                    "--ensure-source-id",
                    "UNLICENSED_UNKNOWN",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(index_path.exists())

            corpus_rows = [
                {"text": "hello world", "source_id": "sha256:source-a"},
                {"text": "row without explicit source"},
            ]
            corpus_path.write_text("\n".join(json.dumps(row) for row in corpus_rows) + "\n", encoding="utf-8")

            code = cli.main(
                [
                    "train",
                    "stamp",
                    "--input",
                    str(corpus_path),
                    "--output",
                    str(stamped_path),
                    "--index-table",
                    str(index_path),
                    "--default-source-id",
                    "UNLICENSED_UNKNOWN",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(stamped_path.exists())

            code = cli.main(
                [
                    "train",
                    "pack",
                    "--input",
                    str(stamped_path),
                    "--output",
                    str(packed_path),
                    "--sequence-length",
                    "2",
                    "--include-labels",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(packed_path.exists())

            code = cli.main(["train", "validate", "--input", str(packed_path), "--report-output", str(report_path)])
            self.assertEqual(code, 0)
            self.assertTrue(report_path.exists())

            code = cli.main(
                [
                    "train",
                    "manifest-build",
                    "--run-id",
                    "run-test",
                    "--registry-manifest-hash",
                    "sha256:registry",
                    "--shard",
                    str(packed_path),
                    "--output",
                    str(manifest_path),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(manifest_path.exists())

            code = cli.main(
                [
                    "train",
                    "report",
                    "--input",
                    str(packed_path),
                    "--index-table",
                    str(index_path),
                    "--output",
                    str(source_report_path),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(source_report_path.exists())

    def test_train_pack_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            tokenized_path = temp / "tokenized.jsonl"
            shard_dir = temp / "shards"

            rows = [
                {"input_ids": [1, 2], "source_idx": [1, 1], "token_count": 2},
                {"input_ids": [3, 4], "source_idx": [1, 1], "token_count": 2},
                {"input_ids": [5, 6], "source_idx": [2, 2], "token_count": 2},
            ]
            tokenized_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            code = cli.main(
                [
                    "train",
                    "pack",
                    "--input",
                    str(tokenized_path),
                    "--output-dir",
                    str(shard_dir),
                    "--sequence-length",
                    "2",
                    "--rows-per-shard",
                    "1",
                    "--shard-prefix",
                    "train",
                ]
            )
            self.assertEqual(code, 0)
            shards = sorted(shard_dir.glob("train-*.jsonl"))
            self.assertGreaterEqual(len(shards), 2)


if __name__ == "__main__":
    unittest.main()
