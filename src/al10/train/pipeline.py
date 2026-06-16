"""Core pipeline logic for AL-1.0 training ingest MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..registry import build_training_manifest
from .io import file_sha256
from .tokenizer import SimpleWhitespaceTokenizer, TokenizerProtocol


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


def _resolve_source_idx(
    *,
    source_id: str,
    source_to_idx: Mapping[str, int],
    unknown_source_policy: str,
    fallback_source_id: str | None,
) -> tuple[int | None, str]:
    if source_id in source_to_idx:
        return int(source_to_idx[source_id]), source_id

    policy = unknown_source_policy.strip().lower()
    if policy == "error":
        raise ValueError(f"unknown source_id: {source_id}")
    if policy == "skip":
        return None, source_id
    if policy == "fallback":
        if not fallback_source_id:
            raise ValueError("fallback_source_id is required when unknown_source_policy=fallback")
        if fallback_source_id not in source_to_idx:
            raise ValueError(f"fallback_source_id not found in source index table: {fallback_source_id}")
        return int(source_to_idx[fallback_source_id]), fallback_source_id
    raise ValueError(f"unknown_source_policy must be one of error|skip|fallback, got: {unknown_source_policy}")


def iter_tokenized_stamped_rows(
    stamped_rows: Iterable[Mapping[str, Any]],
    *,
    source_to_idx: Mapping[str, int],
    tokenizer: TokenizerProtocol | None = None,
    drop_empty: bool = True,
    unknown_source_policy: str = "error",
    fallback_source_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    tok = tokenizer or SimpleWhitespaceTokenizer()

    for row_no, row in enumerate(stamped_rows, start=1):
        source_id_raw = str(row["source_id"])
        resolved = _resolve_source_idx(
            source_id=source_id_raw,
            source_to_idx=source_to_idx,
            unknown_source_policy=unknown_source_policy,
            fallback_source_id=fallback_source_id,
        )
        source_idx, resolved_source_id = resolved
        if source_idx is None:
            # unknown_source_policy=skip
            continue

        input_ids = tok.encode(str(row["text"]))
        if not input_ids and drop_empty:
            continue
        source_idx_arr = [source_idx] * len(input_ids)
        yield {
            "row_id": int(row.get("row_id", row_no)),
            "source_id": resolved_source_id,
            "input_ids": input_ids,
            "source_idx": source_idx_arr,
            "token_count": len(input_ids),
        }


def tokenize_stamped_rows(
    stamped_rows: Iterable[Mapping[str, Any]],
    *,
    source_to_idx: Mapping[str, int],
    tokenizer: TokenizerProtocol | None = None,
    drop_empty: bool = True,
    unknown_source_policy: str = "error",
    fallback_source_id: str | None = None,
) -> list[dict[str, Any]]:
    return list(
        iter_tokenized_stamped_rows(
            stamped_rows,
            source_to_idx=source_to_idx,
            tokenizer=tokenizer,
            drop_empty=drop_empty,
            unknown_source_policy=unknown_source_policy,
            fallback_source_id=fallback_source_id,
        )
    )


def iter_packed_tokenized_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    sequence_length: int,
    drop_remainder: bool = False,
    include_labels: bool = False,
) -> Iterator[dict[str, Any]]:
    """Stream packed rows to avoid holding full outputs in memory."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")

    buffer_ids: list[int] = []
    buffer_src: list[int] = []

    def make_chunk(take: int) -> dict[str, Any]:
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
        return chunk

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
            yield make_chunk(sequence_length)

    if buffer_ids and not (drop_remainder and len(buffer_ids) < sequence_length):
        yield make_chunk(len(buffer_ids))


def pack_tokenized_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    sequence_length: int,
    drop_remainder: bool = False,
    include_labels: bool = False,
) -> list[dict[str, Any]]:
    return list(
        iter_packed_tokenized_rows(
            rows,
            sequence_length=sequence_length,
            drop_remainder=drop_remainder,
            include_labels=include_labels,
        )
    )


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


def shard_rows(rows: list[Mapping[str, Any]], rows_per_shard: int) -> list[list[Mapping[str, Any]]]:
    """Split rows into deterministic shard chunks."""
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be > 0")
    return [rows[i : i + rows_per_shard] for i in range(0, len(rows), rows_per_shard)]
