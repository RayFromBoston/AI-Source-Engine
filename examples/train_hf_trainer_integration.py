"""Example: use AL-1.0 dataset with Hugging Face Trainer."""

from __future__ import annotations

from pathlib import Path

from al10.train import as_hf_trainer_dataset


def main() -> None:
    shard_path = Path("packed.jsonl")
    if not shard_path.exists():
        print("Create packed.jsonl first using `al10 train pack ...`.")
        return

    dataset = as_hf_trainer_dataset(shard_path)
    print({"rows": len(dataset), "columns": dataset.column_names})

    # Optional full Trainer run (requires transformers + torch + a model).
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
    except Exception:
        print("Transformers/Torch not installed. Install with `pip install -e \".[hf]\"` to run Trainer.")
        return

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    args = TrainingArguments(output_dir="./trainer-output", per_device_train_batch_size=2, max_steps=1)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    print("Trainer object ready:", bool(trainer))


if __name__ == "__main__":
    main()
