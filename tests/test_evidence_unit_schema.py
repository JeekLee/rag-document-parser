from __future__ import annotations


def test_common_metadata_schema_helper_keeps_canonical_shape():
    from rag_document_parser.evidence_unit_extraction.schema import common_metadata
    from rag_document_parser.models import CommonMetadataPayload

    metadata = common_metadata("table", "structured_table", section_path=["고시"])

    assert isinstance(metadata, CommonMetadataPayload)
    assert metadata.to_dict() == {
        "common": {
            "chunk_kind": "table",
            "section_path": ["고시"],
            "display_format": "structured_table",
        }
    }
    assert common_metadata("image", "image").to_dict() == {
        "common": {
            "chunk_kind": "image",
            "section_path": [],
            "display_format": "image",
        }
    }


def test_evidence_unit_schema_reexports_canonical_envelope_types():
    from rag_document_parser import EvidenceItem, EvidenceUnit, SourceEvidence
    from rag_document_parser.evidence_unit_extraction.schema import (
        EvidenceItem as SchemaEvidenceItem,
    )
    from rag_document_parser.evidence_unit_extraction.schema import (
        EvidenceUnit as SchemaEvidenceUnit,
    )
    from rag_document_parser.evidence_unit_extraction.schema import (
        SourceEvidence as SchemaSourceEvidence,
    )

    assert SchemaEvidenceItem is EvidenceItem
    assert SchemaEvidenceUnit is EvidenceUnit
    assert SchemaSourceEvidence is SourceEvidence


def test_asset_ref_schema_helper_keeps_canonical_shape():
    from rag_document_parser.evidence_unit_extraction.schema import asset_ref_content
    from rag_document_parser.models import AssetRefContent

    content = asset_ref_content("img-0001")

    assert isinstance(content, AssetRefContent)
    assert not isinstance(content, dict)
    assert content.to_dict() == {
        "asset_id": "img-0001",
        "caption": None,
    }
    assert asset_ref_content("img-0002", caption="도표").to_dict() == {
        "asset_id": "img-0002",
        "caption": "도표",
    }


def test_structured_table_schema_helpers_keep_canonical_shape():
    from rag_document_parser.evidence_unit_extraction.schema import (
        asset_ref_content,
        structured_table,
        table_cell,
        table_column,
        table_row,
    )
    from rag_document_parser.models import StructuredTableContent

    child = {
        "type": "image",
        "format": "asset_ref",
        "content": asset_ref_content("img-0001"),
    }
    header_row = table_row(
        1,
        [
            table_cell("c1", "구분", colspan=2),
        ],
    )
    row = table_row(
        1,
        [
            table_cell("c1", "본인부담", rowspan=2, children=[child]),
            table_cell("c2", "기재형식 예시"),
        ],
    )

    table = structured_table(
        columns=[table_column("c1", "구분"), table_column("c2", "세부")],
        rows=[row],
        header_rows=[header_row],
    )

    assert isinstance(table, StructuredTableContent)
    assert not isinstance(table, dict)
    assert table.columns[0].text == "구분"
    assert table["columns"][0]["text"] == "구분"
    assert table.to_dict() == {
        "caption": None,
        "columns": [
            {"id": "c1", "text": "구분"},
            {"id": "c2", "text": "세부"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    {
                        "column_id": "c1",
                        "text": "본인부담",
                        "rowspan": 2,
                        "colspan": 1,
                        "children": [child],
                    },
                    {
                        "column_id": "c2",
                        "text": "기재형식 예시",
                        "rowspan": 1,
                        "colspan": 1,
                        "children": [],
                    },
                ],
            }
        ],
        "header_rows": [
            {
                "index": 1,
                "cells": [
                    {
                        "column_id": "c1",
                        "text": "구분",
                        "rowspan": 1,
                        "colspan": 2,
                        "children": [],
                    }
                ],
            }
        ],
    }


def test_structured_diagram_schema_helpers_keep_canonical_shape():
    from rag_document_parser.evidence_unit_extraction.schema import (
        diagram_connector,
        diagram_edge,
        diagram_node,
        structured_diagram,
    )
    from rag_document_parser.models import BoundingBox, DiagramPoint, StructuredDiagramContent

    bbox = {"x": 100, "y": 120, "width": 200, "height": 80, "unit": "hwpx"}
    start = {"x": 300, "y": 160}
    end = {"x": 500, "y": 160}

    diagram = structured_diagram(
        nodes=[
            diagram_node("n1", "rect", "수급권자", bbox=bbox, metadata={"source": "test"}),
            diagram_node("n2", "rect", "심사평가원"),
        ],
        edges=[
            diagram_edge(
                "n1",
                "n2",
                edge_type="arrow",
                label="청구",
                confidence="manual",
                connector_id="c1",
            )
        ],
        connectors=[
            diagram_connector(
                "c1",
                "line",
                bbox=bbox,
                points=[start, end],
                arrow=True,
                metadata={"source": "test_line"},
            )
        ],
        mermaid="graph LR",
    )

    assert isinstance(diagram, StructuredDiagramContent)
    assert not isinstance(diagram, dict)
    assert isinstance(diagram.nodes[0].bbox, BoundingBox)
    assert isinstance(diagram.connectors[0].points[0], DiagramPoint)
    assert diagram.nodes[0].text == "수급권자"
    assert diagram["nodes"][0]["text"] == "수급권자"
    assert diagram.to_dict() == {
        "caption": None,
        "nodes": [
            {
                "id": "n1",
                "shape_type": "rect",
                "text": "수급권자",
                "bbox": bbox,
                "metadata": {"source": "test"},
            },
            {
                "id": "n2",
                "shape_type": "rect",
                "text": "심사평가원",
                "bbox": None,
                "metadata": {},
            },
        ],
        "edges": [
            {
                "from": "n1",
                "to": "n2",
                "type": "arrow",
                "label": "청구",
                "confidence": "manual",
                "connector_id": "c1",
            }
        ],
        "connectors": [
            {
                "id": "c1",
                "type": "line",
                "bbox": bbox,
                "points": [start, end],
                "arrow": True,
                "metadata": {"source": "test_line"},
            }
        ],
        "mermaid": "graph LR",
    }
