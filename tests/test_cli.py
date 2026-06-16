import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10 import cli


class TestCli(unittest.TestCase):
    def test_run_demo(self) -> None:
        code = cli.main(["run-demo"])
        self.assertEqual(code, 0)

    def test_init_registry_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = pathlib.Path(temp_dir) / "registry.jsonl"
            code = cli.main(
                [
                    "init-registry",
                    "--output",
                    str(registry_path),
                    "--source-id",
                    "sha256:src-a",
                    "--content-hash",
                    "sha256:content-a",
                    "--uri",
                    "https://example.com/a",
                    "--rightsholder-id",
                    "entity:a",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(registry_path.exists())

            code = cli.main(["validate-manifests", "--registry", str(registry_path)])
            self.assertEqual(code, 0)

    def test_stamp_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = pathlib.Path(temp_dir) / "input.jsonl"
            output_path = pathlib.Path(temp_dir) / "output.jsonl"
            input_path.write_text(json.dumps({"input_ids": [1, 2, 3]}) + "\n", encoding="utf-8")

            code = cli.main(
                [
                    "stamp-dataset",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--default-source-idx",
                    "7",
                ]
            )
            self.assertEqual(code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["source_idx"], [7, 7, 7])

    def test_validate_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            receipt_path = pathlib.Path(temp_dir) / "receipt.json"
            receipt = {
                "receipt_spec": "AL-1.0",
                "model_id": "org/model@r1",
                "registry_manifest_hash": "sha256:registry",
                "training_manifest_hash": "sha256:training",
                "generated_token_count": 1,
                "layer_policy": "last_block_self_attn_mean_heads",
                "sources": [{"source_id": "PARAMETRIC", "ratio": 1.0}],
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            code = cli.main(["validate-receipt", str(receipt_path)])
            self.assertEqual(code, 0)

    def test_init_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "plugin.py"
            code = cli.main(["init-plugin", "--framework", "pytorch", "--output", str(path)])
            self.assertEqual(code, 0)
            content = path.read_text(encoding="utf-8")
            self.assertIn("PyTorchDecodeAdapter", content)

    def test_init_plugin_hf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "plugin_hf.py"
            code = cli.main(["init-plugin", "--framework", "hf", "--output", str(path)])
            self.assertEqual(code, 0)
            content = path.read_text(encoding="utf-8")
            self.assertIn("HuggingFaceGenerateAdapter", content)

    def test_bench_smoke(self) -> None:
        code = cli.main(["bench-smoke", "--steps", "10", "--heads", "4", "--key-len", "16"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
