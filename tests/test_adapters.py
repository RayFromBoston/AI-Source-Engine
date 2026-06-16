import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10.adapters import BaseAL10Adapter, HuggingFaceGenerateAdapter, PyTorchDecodeAdapter
from al10.errors import AdapterError


class FakeTensor:
    def __init__(self, payload):
        self._payload = payload

    def __getitem__(self, index):
        value = self._payload[index]
        if isinstance(value, list):
            return FakeTensor(value)
        return value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._payload


class FakeTokenizer:
    def __call__(self, prompt, return_tensors="pt"):
        _ = prompt, return_tensors
        return {"input_ids": FakeTensor([[10, 11, 12]])}

    def decode(self, token_ids, skip_special_tokens=True):
        _ = skip_special_tokens
        return "decoded:" + ",".join(str(token) for token in token_ids)


class FakeGenerateOutput:
    def __init__(self):
        # two decode steps; each step has one layer tensor [B,H,T_q,T_k]
        self.attentions = (
            (FakeTensor([[[[0.7, 0.2, 0.1]], [[0.6, 0.3, 0.1]]]]),),
            (FakeTensor([[[[0.2, 0.3, 0.4, 0.1]], [[0.1, 0.4, 0.4, 0.1]]]]),),
        )
        self.sequences = [FakeTensor([10, 11, 12, 13, 14])]


class FakeModel:
    def generate(self, **kwargs):
        _ = kwargs
        return FakeGenerateOutput()


class TestAdapters(unittest.TestCase):
    def test_base_adapter_lifecycle(self) -> None:
        adapter = BaseAL10Adapter()
        adapter.start_trace([1, 1, -1])
        bucket = adapter.log_decode_step([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]])
        self.assertAlmostEqual(sum(bucket.values()), 1.0)
        receipt = adapter.finalize_receipt(
            {1: "sha256:a", -1: "PARAMETRIC", -2: "MODEL_OUTPUT"},
            model_id="org/model@r1",
            registry_manifest_hash="sha256:registry",
            training_manifest_hash="sha256:training",
        )
        self.assertEqual(receipt["receipt_spec"], "AL-1.0")

    def test_finalize_without_steps_fails(self) -> None:
        adapter = BaseAL10Adapter()
        adapter.start_trace([1])
        with self.assertRaises(AdapterError):
            adapter.finalize_receipt(
                {1: "sha256:a"},
                model_id="org/model@r1",
                registry_manifest_hash="sha256:registry",
                training_manifest_hash="sha256:training",
            )

    def test_pytorch_adapter_tensor_shape(self) -> None:
        adapter = PyTorchDecodeAdapter.from_prompt_source_idx([1, 1, -1])
        tensor = FakeTensor([[[[0.4, 0.3, 0.3]], [[0.6, 0.2, 0.2]]]])
        bucket = adapter.log_decode_step_from_tensor(tensor)
        self.assertAlmostEqual(sum(bucket.values()), 1.0)

    def test_hf_generate_adapter(self) -> None:
        adapter = HuggingFaceGenerateAdapter()
        result = adapter.generate_with_receipt(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            prompt="hello",
            idx_to_source_id={1: "sha256:a", 2: "sha256:b", -1: "PARAMETRIC", -2: "MODEL_OUTPUT"},
            model_id="fake/model@r1",
            registry_manifest_hash="sha256:registry",
            training_manifest_hash="sha256:training",
            prompt_source_idx=[1, 1, -1],
            generation_kwargs={"max_new_tokens": 2},
        )
        self.assertIn("attribution_receipt", result)
        self.assertEqual(result["attribution_receipt"]["receipt_spec"], "AL-1.0")


if __name__ == "__main__":
    unittest.main()
