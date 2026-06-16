"""Core pipeline logic for AL-1.0 training ingest MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..registry import build_training_manifest
from .io import file_sha256
from .tokenizer import SimpleWhitespaceTokenizer


def save_index_table(path: str | Path, idx_to_source_id: Mapping[int, str]) -> None:
    payload = {str(int(idx)): str(source_id) for idx, source_id in sorted(idx_to_source_id.items())}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_index_table(path: str | Path) -> dict[int, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("index table must be a JSON object")
    return {int(idx): str(source_id) for idx, source_id in payload.items()}


def invert_index_table(idx_to_source_id: Mapping[int, str]) -> dict[str, int]:
    source_to_idx: dict[str, int] = {}
    for idx, source_id in idx_to_source_id.items():
        source_to_idx[str(source_id)] = int(idx)
    return source_to_idx


def stamp_corpus_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    text_field: str = "text",
    source_id_field: str = "source_id",
    default_source_id: str | None = None,
) -> list[dict[str, Any]]:
    stamped: list[dict[str, Any]] = []
    for row_no, row in enumerate(rows, start=1):
        text = row.get(text_field)
        if not isinstance(text, str):
            raise ValueError(f"row {row_no} missing string field '{text_field}'")
        source_id = row.get(source_id_field, default_source_id)
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                f"row {row_no} missing source id field '{source_id_field}' and no valid default provided"
            )
        stamped.append(
            {
                "row_id": int(row.get("row_id", row_no)),
                "text": text,
                "source_id": source_id,
            }
        )
    return stamped


def tokenize_stamped_rows(
    stamped_rows: Iterable[Mapping[str, Any]],
    *,
    source_to_idx: Mapping[str, int],
    tokenizer: SimpleWhitespaceTokenizer | None = None,
    drop_empty: bool = True,
) -> list[dict[str, Any]]:
    tok = tokenizer or SimpleWhitespaceTokenizer()
    tokenized: list[dict[str, Any]] = []

    for row_no, row in enumerate(stamped_rows, start=1):
        source_id = str(row["source_id"])
        if source_id not in source_to_idx:
            raise ValueError(f"row {row_no} uses unknown source_id: {source_id}")
        source_idx = int(source_to_idx[source_id])

        input_ids = tok.encode(str(row["text"]))
        if not input_ids and drop_empty:
            continue
        source_idx_arr = [source_idx] * len(input_ids)
        tokenized.append(
            {
                "row_id": int(row["row_id"]),
                "source_id": source_id,
                "input_ids": input_ids,
                "source_idx": source_idx_arr,
                "token_count": len(input_ids),
            }
        )

    return tokenized


def pack_tokenized_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    sequence_length: int,
    drop_remainder: bool = False,
    include_labels: bool = False,
) -> list[dict[str, Any]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")

    packed: list[dict[str, Any]] = []
    buffer_ids: list[int] = []
    buffer_src: list[int] = []

    def flush_chunk(final: bool = False) -> None:
        if not buffer_ids:
            return
        if final and drop_remainder and len(buffer_ids) < sequence_length:
            return

        take = sequence_length if len(buffer_ids) >= sequence_length else len(buffer_ids)
        chunk_ids = buffer_ids[:take]
        chunk_src = buffer_src[:take]
        del buffer_ids[:take]
        del buffer_src[:take]

        chunk: dict[str, Any] = {
            "input_ids": chunk_ids,
            "source_idx": chunk_src,
            "token_count": len(chunk_ids),
        }
        if include_labels:
            chunk["labels"] = list(chunk_ids)
        packed.append(chunk)

    for row_no, row in enumerate(rows, start=1):
        input_ids = row.get("input_ids")
        source_idx = row.get("source_idx")
        if not isinstance(input_ids, list) or not isinstance(source_idx, list):
            raise ValueError(f"row {row_no} must include list input_ids and source_idx")
        if len(input_ids) != len(source_idx):
            raise ValueError(f"row {row_no} has len(input_ids) != len(source_idx)")

        buffer_ids.extend(int(token) for token in input_ids)
        buffer_src.extend(int(idx) for idx in source_idx)
        while len(buffer_ids) >= sequence_length:
            flush_chunk(final=False)

    flush_chunk(final=True)
    return packed


def validate_training_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_count = 0
    total_tokens = 0
    max_len = 0
    min_len: int | None = None
    source_counts: dict[int, int] = {}

    for row_no, row in enumerate(rows, start=1):
        row_count += 1
        input_ids = row.get("input_ids")
        source_idx = row.get("source_idx")
        if not isinstance(input_ids, list) or not isinstance(source_idx, list):
            raise ValueError(f"row {row_no} must include list input_ids and source_idx")
        if len(input_ids) != len(source_idx):
            raise ValueError(f"row {row_no} has len(input_ids) != len(source_idx)")

        length = len(input_ids)
        total_tokens += length
        max_len = max(max_len, length)
        min_len = length if min_len is None else min(min_len, length)

        for idx in source_idx:
            source_idx_int = int(idx)
            source_counts[source_idx_int] = source_counts.get(source_idx_int, 0) + 1

    if row_count == 0:
        raise ValueError("no rows found for validation")

    return {
        "ok": True,
        "rows": row_count,
        "tokens": total_tokens,
        "max_row_len": max_len,
        "min_row_len": (min_len or 0),
        "source_counts": {str(idx): count for idx, count in sorted(source_counts.items())},
    }


def build_source_report(
    validation_report: Mapping[str, Any],
    idx_to_source_id: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    source_counts = validation_report.get("source_counts", {})
    rows: list[dict[str, Any]] = []
    for source_idx_raw, count in source_counts.items():
        source_idx = int(source_idx_raw)
        source_id = idx_to_source_id.get(source_idx) if idx_to_source_id else None
        row = {"source_idx": source_idx, "token_count": int(count)}
        if source_id is not None:
            row["source_id"] = source_id
        rows.append(row)

    rows.sort(key=lambda row: row["token_count"], reverse=True)
    return {"ok": True, "sources": rows}


def build_training_manifest_with_hashes(
    *,
    run_id: str,
    registry_manifest_hash: str,
    shard_paths: list[str],
) -> dict[str, Any]:
    manifest = build_training_manifest(
        run_id=run_id,
        registry_manifest_hash=registry_manifest_hash,
        shard_paths=shard_paths,
    )
    manifest["shard_hashes"] = {path: file_sha256(path) for path in sorted(shard_paths)}
    return manifest
