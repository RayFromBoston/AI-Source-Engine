"""AL-1.0 attribution logging utilities."""

from .adapters import BaseAL10Adapter, HuggingFaceGenerateAdapter, PyTorchDecodeAdapter, VLLMDecodeAdapter
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
from .server import (
    AL10RequestHandler,
    InMemoryRateLimiter,
    ServerConfig,
    build_handler,
    build_receipt_from_payload,
    create_demo_receipt,
    serve,
)
from .train import (
    HuggingFaceTokenizerAdapter,
    SimpleWhitespaceTokenizer,
    build_tokenizer,
    build_source_report,
    build_training_manifest_with_hashes,
    invert_index_table,
    load_index_table,
    pack_tokenized_rows,
    save_index_table,
    shard_rows,
    stamp_corpus_rows,
    tokenize_stamped_rows,
    validate_training_rows,
)
from .tracing import DecodeStepLogger, SourceTagSidecar
from .validate import validate_manifest_hash, validate_receipt_dict, validate_receipt_file
from .version import __version__

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
    "VLLMDecodeAdapter",
    "ReceiptSource",
    "ReceiptValidationError",
    "RegistryError",
    "SourceRegistry",
    "SourceRegistryEntry",
    "SourceTagSidecar",
    "AL10RequestHandler",
    "InMemoryRateLimiter",
    "ServerConfig",
    "HuggingFaceTokenizerAdapter",
    "SimpleWhitespaceTokenizer",
    "aggregate_decode_step",
    "build_tokenizer",
    "build_source_report",
    "build_handler",
    "build_receipt_model",
    "build_receipt",
    "build_receipt_from_payload",
    "build_training_manifest",
    "build_training_manifest_with_hashes",
    "create_demo_receipt",
    "invert_index_table",
    "load_index_table",
    "merge_query_heads",
    "pack_tokenized_rows",
    "response_ratio",
    "save_index_table",
    "shard_rows",
    "serve",
    "source_bucket",
    "stamp_corpus_rows",
    "tokenize_stamped_rows",
    "__version__",
    "validate_training_rows",
    "validate_manifest_hash",
    "validate_probability_sum",
    "validate_receipt_dict",
    "validate_receipt_file",
]
