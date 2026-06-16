"""Source registry and manifest utilities for AL-1.0 workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from .errors import RegistryError
from .models import SourceRegistryEntry, utc_now_iso


def sha256_text(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class SourceRegistry:
    """Append-oriented source registry with deterministic manifest hash."""

    def __init__(self, entries: Optional[Iterable[SourceRegistryEntry]] = None) -> None:
        self._entries: list[SourceRegistryEntry] = []
        self._by_id: dict[str, SourceRegistryEntry] = {}
        if entries:
            for entry in entries:
                self.add_entry(entry)

    @property
    def entries(self) -> list[SourceRegistryEntry]:
        return list(self._entries)

    def add_entry(self, entry: SourceRegistryEntry) -> None:
        entry.validate()
        if entry.source_id in self._by_id:
            raise RegistryError(f"duplicate source_id: {entry.source_id}")
        self._entries.append(entry)
        self._by_id[entry.source_id] = entry

    def add(
        self,
        *,
        source_id: str,
        content_hash: str,
        uri: str,
        rightsholder_id: str,
        license_class: str = "",
        ingested_at: Optional[str] = None,
    ) -> SourceRegistryEntry:
        entry = SourceRegistryEntry(
            source_id=source_id,
            content_hash=content_hash,
            uri=uri,
            rightsholder_id=rightsholder_id,
            license_class=license_class,
            ingested_at=ingested_at or utc_now_iso(),
        )
        self.add_entry(entry)
        return entry

    def revoke(self, source_id: str, revoked_at: Optional[str] = None) -> None:
        if source_id not in self._by_id:
            raise RegistryError(f"cannot revoke unknown source_id: {source_id}")
        self._by_id[source_id].revoked_at = revoked_at or utc_now_iso()

    def to_jsonl(self, path: str | Path) -> None:
        output = Path(path)
        lines = [canonical_json(entry.to_dict()) for entry in self._entries]
        output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "SourceRegistry":
        source_path = Path(path)
        if not source_path.exists():
            raise RegistryError(f"registry file not found: {source_path}")
        entries: list[SourceRegistryEntry] = []
        for line_no, raw in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegistryError(f"invalid JSON on line {line_no}") from exc
            entry = SourceRegistryEntry.from_dict(payload)
            entries.append(entry)
        return cls(entries)

    def manifest_hash(self) -> str:
        """
        Compute deterministic manifest hash over canonicalized source rows.

        Hash is stable regardless of insertion order.
        """
        sorted_rows = sorted((entry.to_dict() for entry in self._entries), key=lambda row: row["source_id"])
        return sha256_text(canonical_json(sorted_rows))

    def build_index_table(
        self,
        *,
        include_special: bool = True,
        include_parametric: bool = True,
        include_model_output: bool = True,
        special_source_id: str = "SPECIAL",
        parametric_source_id: str = "PARAMETRIC",
        model_output_source_id: str = "MODEL_OUTPUT",
    ) -> dict[int, str]:
        """
        Build a stable source_idx -> source_id table for training/inference use.
        """
        idx_to_source: dict[int, str] = {}
        if include_special:
            idx_to_source[0] = special_source_id

        current = 1
        for entry in sorted(self._entries, key=lambda value: value.source_id):
            idx_to_source[current] = entry.source_id
            current += 1

        if include_parametric:
            idx_to_source[-1] = parametric_source_id
        if include_model_output:
            idx_to_source[-2] = model_output_source_id
        return idx_to_source


def build_training_manifest(
    *,
    run_id: str,
    registry_manifest_hash: str,
    shard_paths: Iterable[str],
) -> dict[str, object]:
    """
    Build a training manifest payload and include deterministic top-level hash.
    """
    shards = sorted(str(path) for path in shard_paths)
    manifest: dict[str, object] = {
        "run_id": run_id,
        "registry_manifest_hash": registry_manifest_hash,
        "shards": shards,
    }
    manifest["manifest_hash"] = sha256_text(canonical_json(manifest))
    return manifest
