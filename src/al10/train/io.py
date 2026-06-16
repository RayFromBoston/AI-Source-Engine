"""I/O helpers for AL-1.0 training data pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
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
        rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
