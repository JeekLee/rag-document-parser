from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


def _plain(value: Any) -> Any:
    if isinstance(value, RdpModel):
        return value.to_dict()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python", by_alias=True)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


class RdpModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, field in self.__class__.model_fields.items():
            key = field.alias or name
            payload[key] = _plain(getattr(self, name))
        return payload

    def _field_name(self, key: str) -> str:
        if key in self.__class__.model_fields:
            return key
        for name, field in self.__class__.model_fields.items():
            if field.alias == key:
                return name
        return key

    def __getitem__(self, key: str) -> Any:
        field_name = self._field_name(key)
        if field_name in self.__class__.model_fields:
            return getattr(self, field_name)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Keep legacy mapping writes while validating declared fields only."""
        field_name = self._field_name(key)
        if field_name not in self.__class__.model_fields:
            raise KeyError(key)
        setattr(self, field_name, value)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.to_dict() == _plain(other)
        return super().__eq__(other)


Mapping.register(RdpModel)


EvidenceType = Literal["text", "table", "image", "diagram", "asset"]
SourceEvidenceKind = Literal["text", "table", "image", "diagram", "chunk"]
WarningSeverity = Literal["low", "medium", "high"]


class QualityWarning(RdpModel):
    """Typed warning envelope with a compatibility-preserving detail payload."""

    type: str = Field(min_length=1)
    severity: WarningSeverity | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def collect_details(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        known = {"type", "severity", "message", "details"}
        details = dict(value.get("details", {}))
        details.update({key: item for key, item in value.items() if key not in known})
        return {
            "type": value.get("type"),
            "severity": value.get("severity"),
            "message": value.get("message"),
            "details": details,
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.severity is not None:
            payload["severity"] = self.severity
        if self.message is not None:
            payload["message"] = self.message
        payload.update(_plain(self.details))
        return payload

    def __getitem__(self, key: str) -> Any:
        if key in self.details:
            return self.details[key]
        return super().__getitem__(key)


class AssetRefContent(RdpModel):
    asset_id: str = Field(min_length=1)
    caption: str | None = None
    uri: str | None = Field(default=None, min_length=1)
    mime: str | None = Field(default=None, min_length=1)
    ext: str | None = Field(default=None, min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_resolved_asset(self) -> Self:
        resolved = (self.uri, self.mime, self.ext, self.sha256, self.bytes)
        if any(value is not None for value in resolved) and not all(
            value is not None for value in resolved
        ):
            raise ValueError("resolved asset references require uri, mime, ext, sha256, and bytes")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        for field_name in ("uri", "mime", "ext", "sha256", "bytes"):
            if getattr(self, field_name) is None:
                payload.pop(field_name, None)
        return payload


class CommonMetadata(RdpModel):
    chunk_kind: str = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)
    display_format: str = Field(min_length=1)


class CommonMetadataPayload(RdpModel):
    common: CommonMetadata


class TableColumn(RdpModel):
    id: str = Field(min_length=1)
    text: str


class TableCell(RdpModel):
    column_id: str = Field(min_length=1)
    text: str = ""
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    children: list[EvidenceChild] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    row_addr: int | None = Field(default=None, ge=0)
    col_addr: int | None = Field(default=None, ge=0)
    address: str | None = None
    formula: str | None = None
    number_format: str | None = None
    hyperlink: str | None = None
    hidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if not self.metadata:
            payload.pop("metadata", None)
        for field_name in (
            "row_addr",
            "col_addr",
            "address",
            "formula",
            "number_format",
            "hyperlink",
        ):
            if getattr(self, field_name) is None:
                payload.pop(field_name, None)
        if not self.hidden:
            payload.pop("hidden", None)
        return payload


class TableRow(RdpModel):
    index: int = Field(ge=1)
    cells: list[TableCell] = Field(default_factory=list)
    source_row: int | None = Field(default=None, ge=1)
    hidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.source_row is None:
            payload.pop("source_row", None)
        if not self.hidden:
            payload.pop("hidden", None)
        return payload


class StructuredTableContent(RdpModel):
    caption: str | None = None
    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)
    header_rows: list[TableRow] = Field(default_factory=list)
    compact: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.header_rows == []:
            payload.pop("header_rows", None)
        if self.compact is None:
            payload.pop("compact", None)
        return payload


Number = int | float


class BoundingBox(RdpModel):
    x: Number
    y: Number
    width: Number = Field(ge=0)
    height: Number = Field(ge=0)
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.unit is None:
            payload.pop("unit", None)
        return payload


class DiagramPoint(RdpModel):
    x: Number
    y: Number


class DiagramNode(RdpModel):
    id: str = Field(min_length=1)
    shape_type: str = Field(min_length=1)
    text: str
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagramEdge(RdpModel):
    from_: str = Field(alias="from", min_length=1)
    to: str = Field(min_length=1)
    type: str = "line"
    label: str = ""
    confidence: str = ""
    connector_id: str = ""


class DiagramConnector(RdpModel):
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    bbox: BoundingBox | None = None
    points: list[DiagramPoint] = Field(default_factory=list)
    arrow: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredDiagramContent(RdpModel):
    caption: str | None = None
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    connectors: list[DiagramConnector] = Field(default_factory=list)
    mermaid: str | None = None
    asset_id: str | None = None
    confidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.asset_id is None:
            payload.pop("asset_id", None)
        if self.confidence is None:
            payload.pop("confidence", None)
        return payload


EvidenceContent = str | AssetRefContent | StructuredTableContent | StructuredDiagramContent


class EvidenceChild(RdpModel):
    type: EvidenceType = Field(validation_alias=AliasChoices("type", "kind"))
    format: str = Field(min_length=1)
    content: EvidenceContent
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content_variant(self) -> Self:
        _validate_evidence_variant(self.type, self.format, self.content)
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if not self.metadata:
            payload.pop("metadata", None)
        return payload


class EvidenceItem(RdpModel):
    type: EvidenceType
    content: EvidenceContent
    format: str | None = Field(default=None, min_length=1)
    source_unit_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content_variant(self) -> Self:
        if self.format is not None:
            _validate_evidence_variant(self.type, self.format, self.content)
        elif self.type != "text" or not isinstance(self.content, str):
            raise ValueError("non-text evidence items require a format")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.format is None:
            payload.pop("format", None)
        return payload


class Evidence(RdpModel):
    entries: list[EvidenceItem] = Field(default_factory=list, alias="items")

    @property
    def items(self) -> list[EvidenceItem]:
        return self.entries

    @items.setter
    def items(self, value: list[EvidenceItem]) -> None:
        self.entries = value


class SourceInfo(RdpModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suffix: str = Field(pattern=r"^\.[^.]+$")
    bytes: int = Field(ge=0)
    id: str | None = None
    name: str | None = None
    url: str | None = None


class PendingAsset(RdpModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    data: bytes
    mime: str = Field(min_length=1)
    ext: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentAsset(RdpModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    mime: str = Field(min_length=1)
    ext: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceEvidence(RdpModel):
    kind: SourceEvidenceKind
    text: str


class EvidenceUnit(RdpModel):
    id: str = Field(min_length=1)
    type: EvidenceType
    format: str = Field(min_length=1)
    source: SourceEvidence
    content: EvidenceContent
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content_variant(self) -> Self:
        _validate_evidence_variant(self.type, self.format, self.content)
        if self.source.kind != self.type:
            raise ValueError(
                f"source kind {self.source.kind!r} does not match evidence type {self.type!r}"
            )
        return self


class RagChunk(RdpModel):
    id: str = Field(min_length=1)
    source: SourceEvidence
    evidence: Evidence
    summary: str
    keywords: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResult(RdpModel):
    source: SourceInfo
    units: list[EvidenceUnit]
    assets: list[DocumentAsset] = Field(default_factory=list)
    quality_warnings: list[QualityWarning] = Field(default_factory=list)


class ParsedDocument(RdpModel):
    units: list[EvidenceUnit]
    assets: list[PendingAsset] = Field(default_factory=list)
    quality_warnings: list[QualityWarning] = Field(default_factory=list)


def _validate_evidence_variant(
    evidence_type: EvidenceType,
    evidence_format: str,
    content: EvidenceContent,
) -> None:
    expected_content: type[str] | type[RdpModel]
    expected_format: str | None
    if evidence_type == "text":
        expected_content = str
        expected_format = "plain"
    elif evidence_type == "table":
        expected_content = StructuredTableContent
        expected_format = "structured_table"
    elif evidence_type == "diagram":
        expected_content = StructuredDiagramContent
        expected_format = "structured_diagram"
    else:
        expected_content = AssetRefContent
        expected_format = None

    if not isinstance(content, expected_content):
        raise ValueError(
            f"{evidence_type!r} evidence requires {expected_content.__name__} content"
        )
    if expected_format is not None and evidence_format != expected_format:
        raise ValueError(
            f"{evidence_type!r} evidence requires format {expected_format!r}"
        )


TableCell.model_rebuild()
TableRow.model_rebuild()
StructuredTableContent.model_rebuild()
EvidenceChild.model_rebuild()
EvidenceItem.model_rebuild()
EvidenceUnit.model_rebuild()
RagChunk.model_rebuild()
ParsedDocument.model_rebuild()
