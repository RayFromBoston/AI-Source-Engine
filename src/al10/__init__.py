"""AL-1.0 attribution logging utilities."""

from .adapters import BaseAL10Adapter, HuggingFaceGenerateAdapter, PyTorchDecodeAdapter
from .errors import AL10Error, AL10ValidationError, AdapterError, ReceiptValidationError, RegistryError
from .math import merge_query_heads, source_bucket, validate_probability_sum
from .models import AttributionReceipt, ReceiptSource, SourceRegistryEntry
from .registry import SourceRegistry, build_training_manifest
from .receipt import (
    MODEL_OUTPUT_SOURCE_IDX,
    PARAMETRIC_SOURCE_ID,
    aggregate_decode_step,
    build_receipt,
    build_receipt_model,
    response_ratio,
)
from .tracing import DecodeStepLogger, SourceTagSidecar
from .validate import validate_manifest_hash, validate_receipt_dict, validate_receipt_file

__all__ = [
    "AL10Error",
    "AL10ValidationError",
    "AdapterError",
    "AttributionReceipt",
    "BaseAL10Adapter",
    "DecodeStepLogger",
    "HuggingFaceGenerateAdapter",
    "MODEL_OUTPUT_SOURCE_IDX",
    "PARAMETRIC_SOURCE_ID",
    "PyTorchDecodeAdapter",
    "ReceiptSource",
    "ReceiptValidationError",
    "RegistryError",
    "SourceRegistry",
    "SourceRegistryEntry",
    "SourceTagSidecar",
    "aggregate_decode_step",
    "build_receipt_model",
    "build_receipt",
    "build_training_manifest",
    "merge_query_heads",
    "response_ratio",
    "source_bucket",
    "validate_manifest_hash",
    "validate_probability_sum",
    "validate_receipt_dict",
    "validate_receipt_file",
]
