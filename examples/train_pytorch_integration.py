"""Example: consume packed AL-1.0 shards in a PyTorch loop."""

from __future__ import annotations

from pathlib import Path

from al10.train import AL10TrainingDataset, build_torch_collate_fn


def main() -> None:
    shard_path = Path("packed.jsonl")
    if not shard_path.exists():
        print("Create packed.jsonl first using `al10 train pack ...`.")
        return

    dataset = AL10TrainingDataset.from_jsonl(shard_path)
    collate_fn = build_torch_collate_fn()

    try:
        from torch.utils.data import DataLoader  # type: ignore
    except Exception:
        print("PyTorch is not installed. Install torch to run this example.")
        return

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    print({key: tuple(value.shape) for key, value in batch.items()})


if __name__ == "__main__":
    main()
