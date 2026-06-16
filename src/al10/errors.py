"""Error types for AL-1.0 helper package."""

from __future__ import annotations


class AL10Error(Exception):
    """Base package exception."""


class AL10ValidationError(AL10Error):
    """Raised when data does not satisfy AL-1.0 invariants."""


class ReceiptValidationError(AL10ValidationError):
    """Raised when an attribution receipt is malformed."""


class RegistryError(AL10Error):
    """Raised for source registry operations."""


class AdapterError(AL10Error):
    """Raised by adapter integrations."""
