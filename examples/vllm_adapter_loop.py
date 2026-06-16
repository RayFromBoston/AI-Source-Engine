"""Example: vLLM-style decode loop with AL-1.0 adapter."""

from al10.adapters import VLLMDecodeAdapter


def main() -> None:
    adapter = VLLMDecodeAdapter.from_prompt_source_idx([1, 1, -1])

    # Simulated per-step layered outputs where we consume the last layer.
    layered_outputs = [
        [[[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]]],  # step 1 -> [layers][H][T_k]
        [[[0.2, 0.3, 0.2, 0.3], [0.3, 0.3, 0.2, 0.2]]],  # step 2
    ]
    adapter.log_decode_steps_from_layered_outputs(layered_outputs, layer_selector=-1)

    result = adapter.finalize_generation_result(
        text="Generated response text",
        token_ids=[101, 102, 103],
        idx_to_source_id={1: "sha256:source-a", -1: "PARAMETRIC", -2: "MODEL_OUTPUT"},
        model_id="demo/vllm-loop@v0",
        registry_manifest_hash="sha256:registry-demo",
        training_manifest_hash="sha256:training-demo",
    )
    print(result)


if __name__ == "__main__":
    main()
