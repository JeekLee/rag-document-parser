from __future__ import annotations

from .extract.backend import DocumentBackend, ParsedDocument
from .extract.formats.markdown import MarkdownBackend
from .extract.registry import default_backends
from .models import Evidence, EvidenceUnit, PendingAsset, SourceEvidence

__all__ = [
    "DocumentBackend",
    "Evidence",
    "EvidenceUnit",
    "MarkdownBackend",
    "PendingAsset",
    "ParsedDocument",
    "SourceEvidence",
    "default_backends",
]
