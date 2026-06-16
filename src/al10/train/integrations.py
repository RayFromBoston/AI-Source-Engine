"""Trainer integration helpers for AL-1.0 training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import read_jsonl


def _normalize_row(row: Mapping[str, Any], row_no: int) -> dict[str, Any]:
    input_ids = row.get("input_ids")
    source_idx = row.get("source_idx")
    labels = row.get("labels")

    if not isinstance(input_ids, list):
        raise ValueError(f"row {row_no} missing list input_ids")
    if not isinstance(source_idx, list):
        raise ValueError(f"row {row_no} missing list source_idx")
    if len(input_ids) != len(source_idx):
        raise ValueError(f"row {row_no} has len(input_ids) != len(source_idx)")
    if labels is not None:
        if not isinstance(labels, list):
            raise ValueError(f"row {row_no} has non-list labels")
        if len(labels) != len(input_ids):
            raise ValueError(f"row {row_no} has len(labels) != len(input_ids)")

    normalized = {
        "input_ids": [int(token) for token in input_ids],
        "source_idx": [int(source) for source in source_idx],
    }
    if labels is not None:
        normalized["labels"] = [int(token) for token in labels]
    return normalized


@dataclass(slots=True)
class AL10TrainingDataset:
    """
    Minimal dataset wrapper for training frameworks (PyTorch/HF Trainer).

    This class intentionally follows the common `__len__` + `__getitem__`
    protocol so it can be consumed by PyTorch DataLoader and by HF Trainer.
    """

    rows: list[dict[str, Any]]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AL10TrainingDataset":
        payload = read_jsonl(path)
        rows = [_normalize_row(row, row_no) for row_no, row in enumerate(payload, start=1)]
        return cls(rows=rows)

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> "AL10TrainingDataset":
        normalized = [_normalize_row(row, row_no) for row_no, row in enumerate(rows, start=1)]
        return cls(rows=normalized)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        out = {
            "input_ids": list(row["input_ids"]),
            "source_idx": list(row["source_idx"]),
        }
        if "labels" in row:
            out["labels"] = list(row["labels"])
        return out

    @property
    def column_names(self) -> list[str]:
        if not self.rows:
            return ["input_ids", "source_idx"]
        names = list(self.rows[0].keys())
        names.sort()
        return names


def pad_batch(
    batch: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int = 0,
    pad_source_idx: int = 0,
    label_pad_id: int = -100,
) -> dict[str, list[list[int]] | list[int]]:
    """Pad variable-length rows for minibatch training."""
    if not batch:
        raise ValueError("batch cannot be empty")

    normalized = [_normalize_row(row, row_no) for row_no, row in enumerate(batch, start=1)]
    max_len = max(len(row["input_ids"]) for row in normalized)

    out_input_ids: list[list[int]] = []
    out_source_idx: list[list[int]] = []
    out_attention_mask: list[list[int]] = []
    out_labels: list[list[int]] | None = [] if any("labels" in row for row in normalized) else None

    for row in normalized:
        length = len(row["input_ids"])
        pad_len = max_len - length
        out_input_ids.append(row["input_ids"] + [pad_token_id] * pad_len)
        out_source_idx.append(row["source_idx"] + [pad_source_idx] * pad_len)
        out_attention_mask.append([1] * length + [0] * pad_len)
        if out_labels is not None:
            labels = row.get("labels")
            if labels is None:
                labels = list(row["input_ids"])
            out_labels.append(list(labels) + [label_pad_id] * pad_len)

    payload: dict[str, list[list[int]] | list[int]] = {
        "input_ids": out_input_ids,
        "source_idx": out_source_idx,
        "attention_mask": out_attention_mask,
    }
    if out_labels is not None:
        payload["labels"] = out_labels
    return payload


def build_torch_collate_fn(
    *,
    pad_token_id: int = 0,
    pad_source_idx: int = 0,
    label_pad_id: int = -100,
):
    """
    Return a PyTorch collate_fn that pads and converts to tensors.

    Raises RuntimeError when torch is unavailable.
    """

    def _collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        payload = pad_batch(
            batch,
            pad_token_id=pad_token_id,
            pad_source_idx=pad_source_idx,
            label_pad_id=label_pad_id,
        )
        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PyTorch is required for build_torch_collate_fn. "
                "Install torch to use this collate function."
            ) from exc

        tensor_payload: dict[str, Any] = {}
        for key, value in payload.items():
            tensor_payload[key] = torch.tensor(value, dtype=torch.long)
        return tensor_payload

    return _collate


def as_hf_trainer_dataset(rows_or_path: Sequence[Mapping[str, Any]] | str | Path) -> AL10TrainingDataset:
    """
    Return a dataset object compatible with Hugging Face Trainer expectations.

    Trainer can consume dataset-like objects implementing `__len__` and
    `__getitem__`, which this wrapper provides.
    """
    if isinstance(rows_or_path, (str, Path)):
        return AL10TrainingDataset.from_jsonl(rows_or_path)
    return AL10TrainingDataset.from_rows(rows_or_path)
