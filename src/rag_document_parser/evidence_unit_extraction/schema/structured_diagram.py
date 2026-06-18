from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...models import (
    BoundingBox,
    DiagramConnector,
    DiagramEdge,
    DiagramNode,
    DiagramPoint,
    StructuredDiagramContent,
)


def diagram_node(
    node_id: str,
    shape_type: str,
    text: str,
    *,
    bbox: BoundingBox | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DiagramNode:
    return DiagramNode(
        id=node_id,
        shape_type=shape_type,
        text=text,
        bbox=bbox,
        metadata=dict(metadata or {}),
    )


def diagram_connector(
    connector_id: str,
    connector_type: str,
    *,
    bbox: BoundingBox | None = None,
    points: Iterable[DiagramPoint] | None = None,
    arrow: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DiagramConnector:
    return DiagramConnector(
        id=connector_id,
        type=connector_type,
        bbox=bbox,
        points=list(points or []),
        arrow=arrow,
        metadata=dict(metadata or {}),
    )


def diagram_edge(
    from_id: str,
    to_id: str,
    *,
    edge_type: str = "line",
    label: str = "",
    confidence: str = "",
    connector_id: str = "",
) -> DiagramEdge:
    return DiagramEdge(
        **{
            "from": from_id,
            "to": to_id,
            "type": edge_type,
            "label": label,
            "confidence": confidence,
            "connector_id": connector_id,
        }
    )


def structured_diagram(
    *,
    nodes: Iterable[DiagramNode | Mapping[str, Any]],
    edges: Iterable[DiagramEdge | Mapping[str, Any]] | None = None,
    connectors: Iterable[DiagramConnector | Mapping[str, Any]] | None = None,
    caption: str | None = None,
    mermaid: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> StructuredDiagramContent:
    payload: dict[str, Any] = {
        "caption": caption,
        "nodes": list(nodes),
        "edges": list(edges or []),
        "connectors": list(connectors or []),
        "mermaid": mermaid,
    }
    if extra:
        payload.update(extra)
    return StructuredDiagramContent(**payload)
