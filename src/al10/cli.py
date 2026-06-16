"""Command line tools for AL-1.0 starter workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from .receipt import aggregate_decode_step
from .registry import SourceRegistry, build_training_manifest
from .scaffold import write_scaffold
from .server import build_receipt_from_payload, create_demo_receipt, serve
from .train import (
    build_tokenizer,
    build_source_report,
    build_training_manifest_with_hashes,
    invert_index_table,
    load_index_table,
    pack_tokenized_rows,
    shard_rows,
    read_jsonl,
    save_index_table,
    stamp_corpus_rows,
    tokenize_stamped_rows,
    validate_training_rows,
    write_jsonl,
)
from .validate import validate_manifest_hash, validate_receipt_file, validate_registry_file
from .version import __version__


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
    receipt = create_demo_receipt()
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


def cmd_make_receipt(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    receipt = build_receipt_from_payload(payload)
    output = Path(args.output) if args.output else None
    if output:
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"ok": True, "output": str(output)}))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def cmd_serve_api(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("AL10_API_KEY")
    serve(args.host, args.port, api_key=api_key, rate_limit_per_minute=args.rate_limit_per_minute)
    return 0


def cmd_bench_smoke(args: argparse.Namespace) -> int:
    """
    Quick decode logging benchmark for local sanity checks.

    This is not a rigorous profiler, but it helps users verify that AL-1.0
    aggregation remains lightweight in their environment.
    """
    key_len = args.key_len
    heads = args.heads
    steps = args.steps

    source_idx = [1 if i % 3 == 0 else (2 if i % 3 == 1 else -1) for i in range(key_len)]
    alpha_per_head = [[1.0 / key_len for _ in range(key_len)] for _ in range(heads)]

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = aggregate_decode_step(alpha_per_head, source_idx)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(
        json.dumps(
            {
                "ok": True,
                "steps": steps,
                "heads": heads,
                "key_len": key_len,
                "total_ms": elapsed_ms,
                "avg_ms_per_step": elapsed_ms / steps if steps else 0.0,
            }
        )
    )
    return 0


def cmd_train_registry_index(args: argparse.Namespace) -> int:
    registry = SourceRegistry.from_jsonl(args.registry)
    idx_to_source = registry.build_index_table(
        include_special=not args.no_special,
        include_parametric=not args.no_parametric,
        include_model_output=not args.no_model_output,
    )

    ensure_source_ids = list(args.ensure_source_id or [])
    for source_id in ensure_source_ids:
        if source_id in idx_to_source.values():
            continue
        positive_indices = [idx for idx in idx_to_source.keys() if idx >= 0]
        next_idx = (max(positive_indices) + 1) if positive_indices else 1
        idx_to_source[next_idx] = str(source_id)

    save_index_table(args.output, idx_to_source)
    print(
        json.dumps(
            {
                "ok": True,
                "output": args.output,
                "rows": len(idx_to_source),
            }
        )
    )
    return 0


def cmd_train_stamp(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    idx_to_source = load_index_table(args.index_table)
    source_to_idx = invert_index_table(idx_to_source)

    stamped = stamp_corpus_rows(
        rows,
        text_field=args.text_field,
        source_id_field=args.source_id_field,
        default_source_id=args.default_source_id,
    )
    tokenizer = build_tokenizer(
        backend=args.tokenizer_backend,
        name_or_path=args.tokenizer_name,
        trust_remote_code=args.tokenizer_trust_remote_code,
        use_fast=not args.tokenizer_no_fast,
        add_special_tokens=args.tokenizer_add_special_tokens,
    )

    tokenized = tokenize_stamped_rows(
        stamped,
        source_to_idx=source_to_idx,
        tokenizer=tokenizer,
        drop_empty=not args.keep_empty,
    )
    write_jsonl(args.output, tokenized)

    total_tokens = sum(int(row["token_count"]) for row in tokenized)
    print(
        json.dumps(
            {
                "ok": True,
                "rows": len(tokenized),
                "tokens": total_tokens,
                "output": args.output,
                "tokenizer": tokenizer.name,
            }
        )
    )
    return 0


def cmd_train_pack(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    packed = pack_tokenized_rows(
        rows,
        sequence_length=args.sequence_length,
        drop_remainder=args.drop_remainder,
        include_labels=args.include_labels,
    )
    total_tokens = sum(int(row["token_count"]) for row in packed)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_groups = shard_rows(packed, rows_per_shard=args.rows_per_shard)
        shard_paths: list[str] = []
        for shard_idx, shard in enumerate(shard_groups):
            shard_path = output_dir / f"{args.shard_prefix}-{shard_idx:05d}.jsonl"
            write_jsonl(shard_path, shard)
            shard_paths.append(str(shard_path))
        print(
            json.dumps(
                {
                    "ok": True,
                    "rows": len(packed),
                    "tokens": total_tokens,
                    "output_dir": str(output_dir),
                    "shards": shard_paths,
                }
            )
        )
        return 0

    if not args.output:
        raise ValueError("either --output or --output-dir must be provided")

    write_jsonl(args.output, packed)
    print(json.dumps({"ok": True, "rows": len(packed), "tokens": total_tokens, "output": args.output}))
    return 0


def cmd_train_validate(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    report = validate_training_rows(rows)
    if args.report_output:
        Path(args.report_output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


def cmd_train_manifest_build(args: argparse.Namespace) -> int:
    manifest = build_training_manifest_with_hashes(
        run_id=args.run_id,
        registry_manifest_hash=args.registry_manifest_hash,
        shard_paths=list(args.shard),
    )
    Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "output": args.output, "manifest_hash": manifest["manifest_hash"]}))
    return 0


def cmd_train_report(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    summary = validate_training_rows(rows)
    idx_to_source = load_index_table(args.index_table) if args.index_table else None
    source_report = build_source_report(summary, idx_to_source)
    payload = {"ok": True, "summary": summary, "source_report": source_report}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


def cmd_train_init_config(args: argparse.Namespace) -> int:
    template = {
        "registry": {
            "path": "registry.jsonl",
            "index_output": "source_index_table.json",
            "ensure_source_id": ["UNLICENSED_UNKNOWN"],
            "no_special": False,
            "no_parametric": False,
            "no_model_output": False,
        },
        "stamp": {
            "input": "corpus.jsonl",
            "output": "stamped.jsonl",
            "text_field": "text",
            "source_id_field": "source_id",
            "default_source_id": "UNLICENSED_UNKNOWN",
            "keep_empty": False,
            "tokenizer": {
                "backend": "simple",
                "name": None,
                "trust_remote_code": False,
                "use_fast": True,
                "add_special_tokens": False,
            },
        },
        "pack": {
            "sequence_length": 2048,
            "drop_remainder": False,
            "include_labels": True,
            "output": "packed.jsonl",
            "output_dir": None,
            "rows_per_shard": 1000,
            "shard_prefix": "packed",
        },
        "validate": {"report_output": "validate_report.json"},
        "manifest": {
            "run_id": "run-001",
            "registry_manifest_hash": "auto",
            "output": "training_manifest.json",
            "shards": [],
        },
        "report": {"enabled": True, "output": "source_report.json"},
    }
    output = Path(args.output)
    output.write_text(json.dumps(template, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output)}))
    return 0


def _resolve_config_path(raw: str | None, *, base_dir: Path, default_name: str | None = None) -> Path:
    value = raw or default_name
    if not value:
        raise ValueError("path value is required")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def cmd_train_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config must be a JSON object")

    # Step 1: registry index table
    registry_cfg = cfg.get("registry", {})
    if not isinstance(registry_cfg, dict):
        raise ValueError("config.registry must be an object")
    registry_path = _resolve_config_path(registry_cfg.get("path"), base_dir=config_dir)
    index_output = _resolve_config_path(
        registry_cfg.get("index_output"),
        base_dir=config_dir,
        default_name="source_index_table.json",
    )
    registry = SourceRegistry.from_jsonl(registry_path)
    idx_to_source = registry.build_index_table(
        include_special=not bool(registry_cfg.get("no_special", False)),
        include_parametric=not bool(registry_cfg.get("no_parametric", False)),
        include_model_output=not bool(registry_cfg.get("no_model_output", False)),
    )
    for source_id in list(registry_cfg.get("ensure_source_id", [])):
        if source_id in idx_to_source.values():
            continue
        positive_indices = [idx for idx in idx_to_source.keys() if idx >= 0]
        next_idx = (max(positive_indices) + 1) if positive_indices else 1
        idx_to_source[next_idx] = str(source_id)
    save_index_table(index_output, idx_to_source)
    source_to_idx = invert_index_table(idx_to_source)
    registry_manifest_hash = registry.manifest_hash()

    # Step 2: stamp + tokenize
    stamp_cfg = cfg.get("stamp", {})
    if not isinstance(stamp_cfg, dict):
        raise ValueError("config.stamp must be an object")
    stamp_input = _resolve_config_path(stamp_cfg.get("input"), base_dir=config_dir)
    stamp_output = _resolve_config_path(stamp_cfg.get("output"), base_dir=config_dir, default_name="stamped.jsonl")
    rows = read_jsonl(stamp_input)
    stamped = stamp_corpus_rows(
        rows,
        text_field=str(stamp_cfg.get("text_field", "text")),
        source_id_field=str(stamp_cfg.get("source_id_field", "source_id")),
        default_source_id=stamp_cfg.get("default_source_id"),
    )
    tokenizer_cfg = stamp_cfg.get("tokenizer", {})
    if not isinstance(tokenizer_cfg, dict):
        raise ValueError("config.stamp.tokenizer must be an object")
    tokenizer = build_tokenizer(
        backend=str(tokenizer_cfg.get("backend", "simple")),
        name_or_path=tokenizer_cfg.get("name"),
        trust_remote_code=bool(tokenizer_cfg.get("trust_remote_code", False)),
        use_fast=bool(tokenizer_cfg.get("use_fast", True)),
        add_special_tokens=bool(tokenizer_cfg.get("add_special_tokens", False)),
    )
    tokenized = tokenize_stamped_rows(
        stamped,
        source_to_idx=source_to_idx,
        tokenizer=tokenizer,
        drop_empty=not bool(stamp_cfg.get("keep_empty", False)),
    )
    write_jsonl(stamp_output, tokenized)

    # Step 3: pack
    pack_cfg = cfg.get("pack", {})
    if not isinstance(pack_cfg, dict):
        raise ValueError("config.pack must be an object")
    packed = pack_tokenized_rows(
        tokenized,
        sequence_length=int(pack_cfg.get("sequence_length", 2048)),
        drop_remainder=bool(pack_cfg.get("drop_remainder", False)),
        include_labels=bool(pack_cfg.get("include_labels", False)),
    )
    shard_paths: list[str] = []
    output_dir_value = pack_cfg.get("output_dir")
    if output_dir_value:
        output_dir = _resolve_config_path(str(output_dir_value), base_dir=config_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        row_per_shard = int(pack_cfg.get("rows_per_shard", 1000))
        shard_prefix = str(pack_cfg.get("shard_prefix", "packed"))
        for shard_idx, shard in enumerate(shard_rows(packed, rows_per_shard=row_per_shard)):
            shard_path = output_dir / f"{shard_prefix}-{shard_idx:05d}.jsonl"
            write_jsonl(shard_path, shard)
            shard_paths.append(str(shard_path))
    else:
        packed_output = _resolve_config_path(pack_cfg.get("output"), base_dir=config_dir, default_name="packed.jsonl")
        write_jsonl(packed_output, packed)
        shard_paths.append(str(packed_output))

    # Step 4: validate
    validate_cfg = cfg.get("validate", {})
    if not isinstance(validate_cfg, dict):
        raise ValueError("config.validate must be an object")
    validation_report = validate_training_rows(packed)
    validate_output = validate_cfg.get("report_output")
    if validate_output:
        validate_report_path = _resolve_config_path(str(validate_output), base_dir=config_dir)
        validate_report_path.write_text(
            json.dumps(validation_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # Step 5: manifest
    manifest_cfg = cfg.get("manifest", {})
    if not isinstance(manifest_cfg, dict):
        raise ValueError("config.manifest must be an object")
    run_id = str(manifest_cfg.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("config.manifest.run_id is required")
    registry_hash_cfg = manifest_cfg.get("registry_manifest_hash", "auto")
    if registry_hash_cfg == "auto":
        manifest_registry_hash = registry_manifest_hash
    else:
        manifest_registry_hash = str(registry_hash_cfg)
    manifest_shards_cfg = list(manifest_cfg.get("shards", []))
    if manifest_shards_cfg:
        manifest_shards = [
            str(_resolve_config_path(str(path), base_dir=config_dir)) for path in manifest_shards_cfg
        ]
    else:
        manifest_shards = shard_paths
    manifest_output = _resolve_config_path(
        manifest_cfg.get("output"),
        base_dir=config_dir,
        default_name="training_manifest.json",
    )
    manifest = build_training_manifest_with_hashes(
        run_id=run_id,
        registry_manifest_hash=manifest_registry_hash,
        shard_paths=manifest_shards,
    )
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Step 6: source report (optional)
    report_cfg = cfg.get("report", {})
    if not isinstance(report_cfg, dict):
        raise ValueError("config.report must be an object")
    report_enabled = bool(report_cfg.get("enabled", True))
    source_report_payload: dict | None = None
    if report_enabled:
        source_report = build_source_report(validation_report, idx_to_source)
        source_report_payload = {"ok": True, "summary": validation_report, "source_report": source_report}
        source_report_output = report_cfg.get("output")
        if source_report_output:
            source_report_path = _resolve_config_path(str(source_report_output), base_dir=config_dir)
            source_report_path.write_text(json.dumps(source_report_payload, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "ok": True,
        "config": str(config_path),
        "tokenizer": tokenizer.name,
        "registry_manifest_hash": registry_manifest_hash,
        "index_table": str(index_output),
        "stamped_rows": len(tokenized),
        "packed_rows": len(packed),
        "shards": shard_paths,
        "manifest": str(manifest_output),
        "manifest_hash": manifest["manifest_hash"],
        "validation": validation_report,
        "source_report_written": bool(source_report_payload),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="al10", description="AL-1.0 starter toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
    init_plugin.add_argument("--framework", required=True, choices=["pytorch", "hf", "huggingface", "vllm"])
    init_plugin.add_argument("--output", help="Output file path")
    init_plugin.set_defaults(func=cmd_init_plugin)

    make_receipt = subparsers.add_parser("make-receipt", help="Build receipt from input JSON payload")
    make_receipt.add_argument("--input", required=True, help="Path to receipt input JSON")
    make_receipt.add_argument("--output", help="Optional output path for resulting receipt JSON")
    make_receipt.set_defaults(func=cmd_make_receipt)

    serve_api = subparsers.add_parser("serve-api", help="Run local AL-1.0 HTTP API server")
    serve_api.add_argument("--host", default="127.0.0.1")
    serve_api.add_argument("--port", type=int, default=8765)
    serve_api.add_argument("--api-key", help="Optional API key for Bearer/X-API-Key auth")
    serve_api.add_argument("--rate-limit-per-minute", type=int, default=0)
    serve_api.set_defaults(func=cmd_serve_api)

    bench = subparsers.add_parser("bench-smoke", help="Run decode-step benchmark smoke test")
    bench.add_argument("--steps", type=int, default=1000)
    bench.add_argument("--heads", type=int, default=32)
    bench.add_argument("--key-len", type=int, default=1024)
    bench.set_defaults(func=cmd_bench_smoke)

    train = subparsers.add_parser("train", help="Training ingest and provenance tooling")
    train_subparsers = train.add_subparsers(dest="train_command", required=True)

    train_registry_index = train_subparsers.add_parser(
        "registry-index", help="Export source_idx -> source_id table from registry"
    )
    train_registry_index.add_argument("--registry", required=True, help="Path to registry JSONL")
    train_registry_index.add_argument("--output", required=True, help="Path to index table JSON")
    train_registry_index.add_argument("--ensure-source-id", action="append", default=[])
    train_registry_index.add_argument("--no-special", action="store_true")
    train_registry_index.add_argument("--no-parametric", action="store_true")
    train_registry_index.add_argument("--no-model-output", action="store_true")
    train_registry_index.set_defaults(func=cmd_train_registry_index)

    train_stamp = train_subparsers.add_parser(
        "stamp", help="Stamp corpus rows with source_idx and tokenize to training rows"
    )
    train_stamp.add_argument("--input", required=True, help="Input JSONL with text/source fields")
    train_stamp.add_argument("--output", required=True, help="Output JSONL with input_ids/source_idx")
    train_stamp.add_argument("--index-table", required=True, help="Path to source index table JSON")
    train_stamp.add_argument("--text-field", default="text")
    train_stamp.add_argument("--source-id-field", default="source_id")
    train_stamp.add_argument("--default-source-id")
    train_stamp.add_argument("--tokenizer-backend", choices=["simple", "hf"], default="simple")
    train_stamp.add_argument("--tokenizer-name", help="Tokenizer model name/path for hf backend")
    train_stamp.add_argument("--tokenizer-trust-remote-code", action="store_true")
    train_stamp.add_argument("--tokenizer-no-fast", action="store_true")
    train_stamp.add_argument("--tokenizer-add-special-tokens", action="store_true")
    train_stamp.add_argument("--keep-empty", action="store_true")
    train_stamp.set_defaults(func=cmd_train_stamp)

    train_pack = train_subparsers.add_parser("pack", help="Pack tokenized rows to fixed sequence length")
    train_pack.add_argument("--input", required=True)
    output_group = train_pack.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output")
    output_group.add_argument("--output-dir")
    train_pack.add_argument("--sequence-length", type=int, required=True)
    train_pack.add_argument("--drop-remainder", action="store_true")
    train_pack.add_argument("--include-labels", action="store_true")
    train_pack.add_argument("--rows-per-shard", type=int, default=1000)
    train_pack.add_argument("--shard-prefix", default="packed")
    train_pack.set_defaults(func=cmd_train_pack)

    train_validate = train_subparsers.add_parser("validate", help="Validate training row invariants")
    train_validate.add_argument("--input", required=True)
    train_validate.add_argument("--report-output")
    train_validate.set_defaults(func=cmd_train_validate)

    train_manifest = train_subparsers.add_parser("manifest-build", help="Build training manifest with shard hashes")
    train_manifest.add_argument("--run-id", required=True)
    train_manifest.add_argument("--registry-manifest-hash", required=True)
    train_manifest.add_argument("--shard", required=True, action="append")
    train_manifest.add_argument("--output", required=True)
    train_manifest.set_defaults(func=cmd_train_manifest_build)

    train_report = train_subparsers.add_parser("report", help="Generate source distribution report")
    train_report.add_argument("--input", required=True)
    train_report.add_argument("--index-table")
    train_report.add_argument("--output")
    train_report.set_defaults(func=cmd_train_report)

    train_init_config = train_subparsers.add_parser("init-config", help="Write training run config template")
    train_init_config.add_argument("--output", required=True)
    train_init_config.set_defaults(func=cmd_train_init_config)

    train_run = train_subparsers.add_parser("run", help="Execute full training ingest pipeline from config JSON")
    train_run.add_argument("--config", required=True)
    train_run.set_defaults(func=cmd_train_run)

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
