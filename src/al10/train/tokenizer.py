"""Tokenizer helpers for AL-1.0 training ingest MVP."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol


_TOKEN_RE = re.compile(r"\S+")


class TokenizerProtocol(Protocol):
    """Minimal tokenizer protocol expected by training pipeline."""

    name: str

    def encode(self, text: str) -> list[int]:
        """Return deterministic token ids for text."""


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


class HuggingFaceTokenizerAdapter:
    """Tokenizer adapter around `transformers.AutoTokenizer`."""

    def __init__(
        self,
        *,
        name_or_path: str,
        trust_remote_code: bool = False,
        use_fast: bool = True,
        add_special_tokens: bool = False,
    ) -> None:
        if not name_or_path:
            raise ValueError("name_or_path is required for hf tokenizer backend")
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "transformers is required for hf tokenizer backend. "
                "Install with: python3 -m pip install -e \".[hf]\""
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
        self._add_special_tokens = bool(add_special_tokens)
        self.name = f"hf:{name_or_path}"

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return list(
            self._tokenizer.encode(
                text,
                add_special_tokens=self._add_special_tokens,
            )
        )


def build_tokenizer(
    *,
    backend: str,
    name_or_path: str | None = None,
    trust_remote_code: bool = False,
    use_fast: bool = True,
    add_special_tokens: bool = False,
) -> TokenizerProtocol:
    """Construct tokenizer backend for training ingest pipeline."""
    backend_key = backend.strip().lower()
    if backend_key == "simple":
        return SimpleWhitespaceTokenizer()
    if backend_key == "hf":
        if not name_or_path:
            raise ValueError("--tokenizer-name is required when --tokenizer-backend hf")
        return HuggingFaceTokenizerAdapter(
            name_or_path=name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            add_special_tokens=add_special_tokens,
        )
    raise ValueError(f"unsupported tokenizer backend: {backend}")
