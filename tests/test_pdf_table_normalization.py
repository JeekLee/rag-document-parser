from __future__ import annotations

from rag_document_parser.evidence_unit_extraction.formats.pdf.table_normalization import (
    PdfTableNormalizer,
    _simple_cell,
)


def test_pdf_table_normalizer_applies_semantic_repairs_through_one_boundary():
    table = {
        "columns": [
            {"id": "c1", "text": "분류"},
            {"id": "c2", "text": "코드"},
            {"id": "c3", "text": "행위명"},
        ],
        "rows": [
            {
                "index": 1,
                "cells": [
                    _simple_cell("c1", "(Ⅰ)"),
                    _simple_cell("c2", "M6850 / C8040"),
                    _simple_cell("c3", "낭종흡인요법 / 흉막천자"),
                ],
            }
        ],
    }

    PdfTableNormalizer().normalize(table)

    assert [
        [cell["text"] for cell in row["cells"]]
        for row in table["rows"]
    ] == [
        ["(Ⅰ)", "M6850", "낭종흡인요법"],
        ["", "C8040", "흉막천자"],
    ]
