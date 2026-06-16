"""AL-1.0 training ingest MVP helpers."""

from .io import file_sha256, iter_jsonl, read_jsonl, write_jsonl, write_jsonl_stream
from .integrations import (
    AL10TrainingDataset,
    as_hf_trainer_dataset,
    build_torch_collate_fn,
    pad_batch,
)
from .pipeline import (
    build_source_report,
    build_training_manifest_with_hashes,
    invert_index_table,
    iter_packed_tokenized_rows,
    iter_tokenized_stamped_rows,
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
    "AL10TrainingDataset",
    "as_hf_trainer_dataset",
    "build_torch_collate_fn",
    "SimpleWhitespaceTokenizer",
    "HuggingFaceTokenizerAdapter",
    "TokenizerProtocol",
    "build_source_report",
    "build_tokenizer",
    "build_training_manifest_with_hashes",
    "file_sha256",
    "iter_jsonl",
    "iter_packed_tokenized_rows",
    "iter_tokenized_stamped_rows",
    "invert_index_table",
    "load_index_table",
    "pack_tokenized_rows",
    "read_jsonl",
    "save_index_table",
    "shard_rows",
    "stamp_corpus_rows",
    "tokenize_stamped_rows",
    "pad_batch",
    "validate_training_rows",
    "write_jsonl",
    "write_jsonl_stream",
]
