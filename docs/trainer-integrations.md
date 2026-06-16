# Trainer integrations

This phase provides direct helpers for using AL-1.0 packed training rows in
training loops.

## What is included

- `AL10TrainingDataset`
  - dataset wrapper with `__len__` / `__getitem__`
  - load from JSONL or in-memory rows
- `pad_batch(...)`
  - pure-Python padding utility
- `build_torch_collate_fn(...)`
  - collate function that pads and converts to torch tensors
- `as_hf_trainer_dataset(...)`
  - returns dataset object directly consumable by Hugging Face Trainer

## PyTorch usage

```python
from al10.train import AL10TrainingDataset, build_torch_collate_fn
from torch.utils.data import DataLoader

dataset = AL10TrainingDataset.from_jsonl("packed.jsonl")
loader = DataLoader(dataset, batch_size=8, collate_fn=build_torch_collate_fn())
for batch in loader:
    # batch contains input_ids, source_idx, attention_mask, and labels (if available)
    pass
```

## Hugging Face Trainer usage

```python
from al10.train import as_hf_trainer_dataset

train_dataset = as_hf_trainer_dataset("packed.jsonl")
# pass train_dataset directly to transformers.Trainer(...)
```

## Examples

- `examples/train_pytorch_integration.py`
- `examples/train_hf_trainer_integration.py`

## Notes

- `source_idx` is preserved in batches for provenance-aware training logic.
- Batch helpers keep strict row invariant assumptions from ingest phase:
  `len(input_ids) == len(source_idx)`.
