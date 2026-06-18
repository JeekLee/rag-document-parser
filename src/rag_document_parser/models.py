from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, field in self.__class__.model_fields.items():
            key = field.alias or name
            payload[key] = _plain(getattr(self, name))
        if self.model_extra:
            for key, value in self.model_extra.items():
                payload[key] = _plain(value)
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
        if self.model_extra and key in self.model_extra:
            return self.model_extra[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, self._field_name(key), value)

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


class AssetRefContent(RdpModel):
    asset_id: str
    caption: str | None = None


class CommonMetadata(RdpModel):
    chunk_kind: str
    section_path: list[str] = Field(default_factory=list)
    display_format: str


class CommonMetadataPayload(RdpModel):
    common: CommonMetadata


class TableColumn(RdpModel):
    id: str
    text: str


class TableCell(RdpModel):
    column_id: str
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    children: list[EvidenceChild] = Field(default_factory=list)


class TableRow(RdpModel):
    index: int
    cells: list[TableCell] = Field(default_factory=list)


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
    width: Number
    height: Number


class DiagramPoint(RdpModel):
    x: Number
    y: Number


class DiagramNode(RdpModel):
    id: str
    shape_type: str
    text: str
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagramEdge(RdpModel):
    from_: str = Field(alias="from")
    to: str
    type: str = "line"
    label: str = ""
    confidence: str = ""
    connector_id: str = ""


class DiagramConnector(RdpModel):
    id: str
    type: str
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
    type: str
    format: str
    content: EvidenceContent
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if not self.metadata:
            payload.pop("metadata", None)
        return payload


class EvidenceItem(RdpModel):
    type: str
    content: EvidenceContent
    format: str | None = None
    source_unit_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

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
    sha256: str
    suffix: str
    bytes: int
    id: str | None = None
    name: str | None = None
    url: str | None = None


class PendingAsset(RdpModel):
    id: str
    kind: str
    data: bytes
    mime: str
    ext: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentAsset(RdpModel):
    id: str
    kind: str
    uri: str
    mime: str
    ext: str
    sha256: str
    bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceEvidence(RdpModel):
    kind: str
    text: str


class EvidenceUnit(RdpModel):
    id: str
    type: str
    format: str
    source: SourceEvidence
    content: EvidenceContent
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagChunk(RdpModel):
    id: str
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
    quality_warnings: list[dict[str, Any]] = Field(default_factory=list)


class ParsedDocument(RdpModel):
    units: list[EvidenceUnit]
    assets: list[PendingAsset] = Field(default_factory=list)
    quality_warnings: list[dict[str, Any]] = Field(default_factory=list)


TableCell.model_rebuild()
TableRow.model_rebuild()
StructuredTableContent.model_rebuild()
EvidenceChild.model_rebuild()
EvidenceItem.model_rebuild()
EvidenceUnit.model_rebuild()
RagChunk.model_rebuild()
ParsedDocument.model_rebuild()
