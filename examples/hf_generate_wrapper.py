"""Example: wrap Hugging Face generate() with AL-1.0 receipt logging."""

from al10.adapters import HuggingFaceGenerateAdapter


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    adapter = HuggingFaceGenerateAdapter()
    result = adapter.generate_with_receipt(
        model=model,
        tokenizer=tokenizer,
        prompt="Summarize AL-1.0 in two sentences.",
        idx_to_source_id={-1: "PARAMETRIC", -2: "MODEL_OUTPUT"},
        model_id=f"{model_name}@local",
        registry_manifest_hash="sha256:registry-demo",
        training_manifest_hash="sha256:training-demo",
        generation_kwargs={"max_new_tokens": 40},
    )

    print(result["text"])
    print(result["attribution_receipt"])


if __name__ == "__main__":
    main()
