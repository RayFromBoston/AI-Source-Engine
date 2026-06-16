"""Tokenizer helpers for AL-1.0 training ingest MVP."""

from __future__ import annotations

import hashlib
import re


_TOKEN_RE = re.compile(r"\S+")


class SimpleWhitespaceTokenizer:
    """
    Deterministic, dependency-free tokenizer for MVP data stamping.

    This is intentionally simple so teams can run the pipeline without requiring
    framework-specific tokenizer installations. Production stacks can swap this
    with model-native tokenizers.
    """

    name = "simple-whitespace-v1"

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        tokens = _TOKEN_RE.findall(text)
        if not tokens:
            return []
        return [self._token_to_id(token) for token in tokens]

    @staticmethod
    def _token_to_id(token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # Keep ids positive and deterministic.
        return int.from_bytes(digest[:4], "big", signed=False) or 1
