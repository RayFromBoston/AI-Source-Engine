import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.registry import SourceRegistry, build_training_manifest


class TestRegistry(unittest.TestCase):
    def test_registry_round_trip_and_hash(self) -> None:
        registry = SourceRegistry()
        registry.add(
            source_id="sha256:a",
            content_hash="sha256:content-a",
            uri="https://example.com/a",
            rightsholder_id="entity:a",
        )
        registry.add(
            source_id="sha256:b",
            content_hash="sha256:content-b",
            uri="https://example.com/b",
            rightsholder_id="entity:b",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "registry.jsonl"
            registry.to_jsonl(path)
            loaded = SourceRegistry.from_jsonl(path)
            self.assertEqual(len(loaded.entries), 2)
            self.assertEqual(registry.manifest_hash(), loaded.manifest_hash())

    def test_training_manifest_hash_exists(self) -> None:
        manifest = build_training_manifest(
            run_id="run-1",
            registry_manifest_hash="sha256:registry",
            shard_paths=["shard-1.jsonl", "shard-2.jsonl"],
        )
        self.assertIn("manifest_hash", manifest)
        self.assertTrue(str(manifest["manifest_hash"]).startswith("sha256:"))

    def test_index_table_includes_reserved_rows(self) -> None:
        registry = SourceRegistry()
        registry.add(
            source_id="sha256:a",
            content_hash="sha256:content-a",
            uri="https://example.com/a",
            rightsholder_id="entity:a",
        )
        table = registry.build_index_table()
        self.assertEqual(table[0], "SPECIAL")
        self.assertEqual(table[-1], "PARAMETRIC")
        self.assertEqual(table[-2], "MODEL_OUTPUT")


if __name__ == "__main__":
    unittest.main()
