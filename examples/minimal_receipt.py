"""Minimal usage example for AL-1.0 reference utilities."""

from al10 import aggregate_decode_step, build_receipt


def main() -> None:
    # Two decode steps, two heads, four key positions.
    # source_idx sidecar per key position:
    #   key 0 -> source 1
    #   key 1 -> source 1
    #   key 2 -> source 2
    #   key 3 -> PARAMETRIC (-1)
    source_idx = [1, 1, 2, -1]

    step_1_alpha_heads = [
        [0.5, 0.2, 0.2, 0.1],
        [0.4, 0.3, 0.2, 0.1],
    ]
    step_2_alpha_heads = [
        [0.2, 0.1, 0.6, 0.1],
        [0.3, 0.1, 0.5, 0.1],
    ]

    per_step = [
        aggregate_decode_step(step_1_alpha_heads, source_idx),
        aggregate_decode_step(step_2_alpha_heads, source_idx),
    ]

    idx_to_source_id = {
        1: "sha256:source-a",
        2: "sha256:source-b",
        -1: "PARAMETRIC",
    }

    receipt = build_receipt(
        per_step,
        idx_to_source_id,
        model_id="demo/al10-reference@v0",
        registry_manifest_hash="sha256:registry-demo",
        training_manifest_hash="sha256:training-demo",
    )

    print(receipt)


if __name__ == "__main__":
    main()
