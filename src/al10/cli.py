"""Command line tools for AL-1.0 starter workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from .receipt import aggregate_decode_step, build_receipt
from .registry import SourceRegistry, build_training_manifest
from .scaffold import write_scaffold
from .validate import validate_manifest_hash, validate_receipt_file, validate_registry_file


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_no} in {path}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def cmd_init_registry(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists() and args.append:
        registry = SourceRegistry.from_jsonl(output)
    else:
        registry = SourceRegistry()

    registry.add(
        source_id=args.source_id,
        content_hash=args.content_hash,
        uri=args.uri,
        rightsholder_id=args.rightsholder_id,
        license_class=args.license_class or "",
    )
    registry.to_jsonl(output)
    print(json.dumps({"registry_path": str(output), "registry_manifest_hash": registry.manifest_hash()}))
    return 0


def cmd_stamp_dataset(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = _read_jsonl(input_path)

    stamped_rows = []
    for row in rows:
        if "input_ids" not in row:
            raise ValueError("every row must include input_ids")
        input_ids = row["input_ids"]
        if not isinstance(input_ids, list):
            raise ValueError("input_ids must be a list")

        if "source_idx" in row:
            source_idx = row["source_idx"]
            if not isinstance(source_idx, list):
                raise ValueError("source_idx must be a list")
        elif args.default_source_idx is not None:
            source_idx = [args.default_source_idx] * len(input_ids)
        else:
            raise ValueError("row missing source_idx and no --default-source-idx provided")

        if len(source_idx) != len(input_ids):
            raise ValueError("len(source_idx) must equal len(input_ids)")

        stamped = dict(row)
        stamped["source_idx"] = source_idx
        stamped_rows.append(stamped)

    _write_jsonl(output_path, stamped_rows)

    result = {"rows": len(stamped_rows), "output": str(output_path)}
    if args.run_id and args.registry_manifest_hash and args.training_manifest_output:
        manifest = build_training_manifest(
            run_id=args.run_id,
            registry_manifest_hash=args.registry_manifest_hash,
            shard_paths=[str(output_path)],
        )
        manifest_path = Path(args.training_manifest_output)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        result["training_manifest"] = str(manifest_path)
        result["training_manifest_hash"] = manifest["manifest_hash"]

    print(json.dumps(result))
    return 0


def cmd_run_demo(_: argparse.Namespace) -> int:
    source_idx = [1, 1, 2, -1]
    step_1_alpha_heads = [
        [0.5, 0.2, 0.2, 0.1],
        [0.4, 0.3, 0.2, 0.1],
    ]
    step_2_alpha_heads = [
        [0.2, 0.1, 0.6, 0.1],
        [0.3, 0.1, 0.5, 0.1],
    ]

    per_step = [
        aggregate_decode_step(step_1_alpha_heads, source_idx),
        aggregate_decode_step(step_2_alpha_heads, source_idx),
    ]

    receipt = build_receipt(
        per_step,
        {1: "sha256:source-a", 2: "sha256:source-b", -1: "PARAMETRIC"},
        model_id="demo/al10-cli@v0",
        registry_manifest_hash="sha256:registry-demo",
        training_manifest_hash="sha256:training-demo",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def cmd_validate_receipt(args: argparse.Namespace) -> int:
    validate_receipt_file(args.path)
    print(json.dumps({"ok": True, "receipt": args.path}))
    return 0


def cmd_validate_manifests(args: argparse.Namespace) -> int:
    registry = validate_registry_file(args.registry)
    result = {"ok": True, "registry_manifest_hash": registry.manifest_hash()}

    if args.training_manifest:
        payload = json.loads(Path(args.training_manifest).read_text(encoding="utf-8"))
        expected_hash = str(payload.get("manifest_hash", ""))
        if not expected_hash:
            raise ValueError("training manifest is missing manifest_hash")
        validate_manifest_hash(payload, expected_hash)
        result["training_manifest_hash"] = expected_hash

    print(json.dumps(result))
    return 0


def cmd_init_plugin(args: argparse.Namespace) -> int:
    path = write_scaffold(args.framework, args.output)
    print(json.dumps({"ok": True, "path": str(path), "framework": args.framework}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="al10", description="AL-1.0 starter toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_registry = subparsers.add_parser("init-registry", help="Create/append a source registry")
    init_registry.add_argument("--output", required=True, help="Path to registry.jsonl")
    init_registry.add_argument("--source-id", required=True)
    init_registry.add_argument("--content-hash", required=True)
    init_registry.add_argument("--uri", required=True)
    init_registry.add_argument("--rightsholder-id", required=True)
    init_registry.add_argument("--license-class", default="")
    init_registry.add_argument("--append", action="store_true")
    init_registry.set_defaults(func=cmd_init_registry)

    stamp = subparsers.add_parser("stamp-dataset", help="Ensure source_idx aligns with input_ids")
    stamp.add_argument("--input", required=True, help="Input JSONL dataset path")
    stamp.add_argument("--output", required=True, help="Output JSONL dataset path")
    stamp.add_argument("--default-source-idx", type=int)
    stamp.add_argument("--run-id")
    stamp.add_argument("--registry-manifest-hash")
    stamp.add_argument("--training-manifest-output")
    stamp.set_defaults(func=cmd_stamp_dataset)

    run_demo = subparsers.add_parser("run-demo", help="Run a minimal AL-1.0 demo")
    run_demo.set_defaults(func=cmd_run_demo)

    validate_receipt = subparsers.add_parser("validate-receipt", help="Validate receipt JSON")
    validate_receipt.add_argument("path", help="Path to receipt JSON")
    validate_receipt.set_defaults(func=cmd_validate_receipt)

    validate_manifests = subparsers.add_parser(
        "validate-manifests", help="Validate registry and optional training manifest"
    )
    validate_manifests.add_argument("--registry", required=True, help="Path to registry JSONL")
    validate_manifests.add_argument("--training-manifest", help="Path to training manifest JSON")
    validate_manifests.set_defaults(func=cmd_validate_manifests)

    init_plugin = subparsers.add_parser("init-plugin", help="Generate framework integration scaffold")
    init_plugin.add_argument("--framework", required=True, choices=["pytorch", "hf", "huggingface"])
    init_plugin.add_argument("--output", help="Output file path")
    init_plugin.set_defaults(func=cmd_init_plugin)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - CLI should convert all to user-facing errors.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
