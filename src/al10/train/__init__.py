"""AL-1.0 training ingest MVP helpers."""

from .io import file_sha256, read_jsonl, write_jsonl
from .pipeline import (
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
from .tokenizer import (
    HuggingFaceTokenizerAdapter,
    SimpleWhitespaceTokenizer,
    TokenizerProtocol,
    build_tokenizer,
)

__all__ = [
    "SimpleWhitespaceTokenizer",
    "HuggingFaceTokenizerAdapter",
    "TokenizerProtocol",
    "build_source_report",
    "build_tokenizer",
    "build_training_manifest_with_hashes",
    "file_sha256",
    "invert_index_table",
    "load_index_table",
    "pack_tokenized_rows",
    "read_jsonl",
    "save_index_table",
    "shard_rows",
    "stamp_corpus_rows",
    "tokenize_stamped_rows",
    "validate_training_rows",
    "write_jsonl",
]
