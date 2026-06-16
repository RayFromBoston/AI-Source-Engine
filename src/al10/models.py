"""Typed models for AL-1.0 registry and receipt data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .errors import AL10ValidationError


def utc_now_iso() -> str:
    """Return current UTC time in RFC3339-like format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class SourceRegistryEntry:
    """One source row in the AL-1.0 source registry."""

    source_id: str
    content_hash: str
    uri: str
    rightsholder_id: str
    license_class: str = ""
    ingested_at: str = field(default_factory=utc_now_iso)
    revoked_at: Optional[str] = None

    def validate(self) -> None:
        if not self.source_id:
            raise AL10ValidationError("source_id must be non-empty")
        if not self.content_hash:
            raise AL10ValidationError("content_hash must be non-empty")
        if not self.uri:
            raise AL10ValidationError("uri must be non-empty")
        if not self.rightsholder_id:
            raise AL10ValidationError("rightsholder_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "source_id": self.source_id,
            "content_hash": self.content_hash,
            "uri": self.uri,
            "rightsholder_id": self.rightsholder_id,
            "license_class": self.license_class,
            "ingested_at": self.ingested_at,
        }
        if self.revoked_at is not None:
            data["revoked_at"] = self.revoked_at
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceRegistryEntry":
        return cls(
            source_id=str(payload["source_id"]),
            content_hash=str(payload["content_hash"]),
            uri=str(payload["uri"]),
            rightsholder_id=str(payload["rightsholder_id"]),
            license_class=str(payload.get("license_class", "")),
            ingested_at=str(payload.get("ingested_at", utc_now_iso())),
            revoked_at=payload.get("revoked_at"),
        )


@dataclass(slots=True)
class ReceiptSource:
    """A source contribution entry in an attribution receipt."""

    source_id: str
    ratio: float
    label: Optional[str] = None

    def validate(self) -> None:
        if not self.source_id:
            raise AL10ValidationError("receipt source_id must be non-empty")
        if self.ratio < 0:
            raise AL10ValidationError("receipt ratio must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"source_id": self.source_id, "ratio": float(self.ratio)}
        if self.label is not None:
            row["label"] = self.label
        return row


@dataclass(slots=True)
class AttributionReceipt:
    """Top-level AL-1.0 attribution receipt."""

    receipt_spec: str
    model_id: str
    registry_manifest_hash: str
    training_manifest_hash: str
    generated_token_count: int
    layer_policy: str
    sources: list[ReceiptSource]

    def validate(self, atol: float = 1e-6) -> None:
        if self.receipt_spec != "AL-1.0":
            raise AL10ValidationError("receipt_spec must be AL-1.0")
        if self.generated_token_count < 0:
            raise AL10ValidationError("generated_token_count must be >= 0")
        if not self.model_id:
            raise AL10ValidationError("model_id must be non-empty")
        if not self.registry_manifest_hash:
            raise AL10ValidationError("registry_manifest_hash must be non-empty")
        if not self.training_manifest_hash:
            raise AL10ValidationError("training_manifest_hash must be non-empty")
        if not self.layer_policy:
            raise AL10ValidationError("layer_policy must be non-empty")
        if not self.sources:
            raise AL10ValidationError("receipt must contain at least one source")
        total = 0.0
        for source in self.sources:
            source.validate()
            total += source.ratio
        if abs(total - 1.0) > atol:
            raise AL10ValidationError(f"source ratios must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_spec": self.receipt_spec,
            "model_id": self.model_id,
            "registry_manifest_hash": self.registry_manifest_hash,
            "training_manifest_hash": self.training_manifest_hash,
            "generated_token_count": self.generated_token_count,
            "layer_policy": self.layer_policy,
            "sources": [source.to_dict() for source in self.sources],
        }
