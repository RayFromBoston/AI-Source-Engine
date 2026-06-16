"""Validation helpers for AL-1.0 receipts and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .errors import AL10ValidationError, ReceiptValidationError
from .models import AttributionReceipt, ReceiptSource
from .registry import SourceRegistry, canonical_json, sha256_text


def validate_receipt_dict(payload: Mapping[str, object], *, atol: float = 1e-6) -> None:
    """Validate a JSON-like receipt dict."""
    required = {
        "receipt_spec",
        "model_id",
        "registry_manifest_hash",
        "training_manifest_hash",
        "generated_token_count",
        "layer_policy",
        "sources",
    }
    missing = [field for field in required if field not in payload]
    if missing:
        raise ReceiptValidationError(f"missing receipt fields: {', '.join(sorted(missing))}")

    sources_payload = payload["sources"]
    if not isinstance(sources_payload, Sequence) or isinstance(sources_payload, (str, bytes)):
        raise ReceiptValidationError("sources must be a list")

    sources: list[ReceiptSource] = []
    for row in sources_payload:
        if not isinstance(row, Mapping):
            raise ReceiptValidationError("each source row must be an object")
        try:
            source = ReceiptSource(
                source_id=str(row["source_id"]),
                ratio=float(row["ratio"]),
                label=(str(row["label"]) if "label" in row else None),
            )
        except KeyError as exc:
            raise ReceiptValidationError("source rows require source_id and ratio") from exc
        sources.append(source)

    receipt = AttributionReceipt(
        receipt_spec=str(payload["receipt_spec"]),
        model_id=str(payload["model_id"]),
        registry_manifest_hash=str(payload["registry_manifest_hash"]),
        training_manifest_hash=str(payload["training_manifest_hash"]),
        generated_token_count=int(payload["generated_token_count"]),
        layer_policy=str(payload["layer_policy"]),
        sources=sources,
    )

    try:
        receipt.validate(atol=atol)
    except AL10ValidationError as exc:
        raise ReceiptValidationError(str(exc)) from exc


def validate_receipt_file(path: str | Path, *, atol: float = 1e-6) -> None:
    receipt_path = Path(path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_receipt_dict(payload, atol=atol)


def validate_registry_file(path: str | Path) -> SourceRegistry:
    registry = SourceRegistry.from_jsonl(path)
    _ = registry.manifest_hash()
    return registry


def validate_manifest_hash(payload: Mapping[str, object], expected_hash: str) -> None:
    """
    Validate deterministic hash of a manifest payload.

    The `manifest_hash` field is excluded from the hash input.
    """
    clean_payload = {key: value for key, value in payload.items() if key != "manifest_hash"}
    actual_hash = sha256_text(canonical_json(clean_payload))
    if actual_hash != expected_hash:
        raise AL10ValidationError(
            f"manifest hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
