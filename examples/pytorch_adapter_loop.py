"""Example: integrate AL-1.0 logging in a PyTorch-style decode loop."""

from al10.adapters import PyTorchDecodeAdapter


def main() -> None:
    # Prompt/context key positions are attributed to these source indices.
    adapter = PyTorchDecodeAdapter.from_prompt_source_idx([1, 1, 2, -1])

    # Mock last-layer decode-step attentions in [H, T_k] format.
    decode_attentions = [
        [[0.5, 0.2, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
        [[0.2, 0.1, 0.5, 0.1, 0.1], [0.3, 0.1, 0.4, 0.1, 0.1]],
    ]
    for step_attention in decode_attentions:
        adapter.log_decode_step_from_tensor(step_attention)

    receipt = adapter.finalize_receipt(
        idx_to_source_id={
            1: "sha256:source-a",
            2: "sha256:source-b",
            -1: "PARAMETRIC",
            -2: "MODEL_OUTPUT",
        },
        model_id="demo/pytorch-loop@v0",
        registry_manifest_hash="sha256:registry-demo",
        training_manifest_hash="sha256:training-demo",
    )
    print(receipt)


if __name__ == "__main__":
    main()
