"""I/O helpers for AL-1.0 training data pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for payload in iter_jsonl(path):
        rows.append(payload)
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    source = Path(path)
    for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_no} in {source}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"row {line_no} in {source} must be a JSON object")
        yield payload


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_jsonl_stream(path: str | Path, rows: Iterable[dict]) -> tuple[int, int]:
    """
    Stream rows to JSONL file and return (row_count, token_count).
    """
    output = Path(path)
    row_count = 0
    token_count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            row_count += 1
            if "token_count" in row:
                token_count += int(row["token_count"])
    return row_count, token_count


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
