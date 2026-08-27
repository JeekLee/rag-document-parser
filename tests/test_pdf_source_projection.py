from __future__ import annotations

from rag_document_parser.evidence_unit_extraction.formats.pdf.source_projection import (
    PdfTableSourceProjector,
)


def test_pdf_table_source_projector_uses_semantic_column_labels():
    table = {
        "columns": [
            {"id": "c1", "text": "구분"},
            {"id": "c2", "text": "금액"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    {
                        "column_id": "c1",
                        "text": "외래",
                        "rowspan": 1,
                        "colspan": 1,
                        "children": [],
                    },
                    {
                        "column_id": "c2",
                        "text": "1000",
                        "rowspan": 1,
                        "colspan": 1,
                        "children": [],
                    },
                ],
            }
        ],
    }

    source = PdfTableSourceProjector().project(table)

    assert source == "table: 2 columns\nrow 1: 구분: 외래; 금액: 1000"
