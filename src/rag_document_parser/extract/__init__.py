from __future__ import annotations

from .backend import DocumentBackend, ParsedDocument
from .registry import default_backends

__all__ = ["DocumentBackend", "ParsedDocument", "default_backends"]
