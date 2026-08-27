from __future__ import annotations

import io
import warnings as python_warnings
from dataclasses import dataclass
from typing import Any

from ....models import EvidenceUnit, ParsedDocument
from .package_validation import XlsxPackageValidator
from .worksheet_parser import XlsxParseState, XlsxWorksheetParser

_DEFAULT_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_ARCHIVE_MEMBERS = 10_000
_DEFAULT_MAX_CELLS = 1_000_000
_DEFAULT_MAX_WORKSHEETS = 256


@dataclass
class XlsxBackend:
    """Extract modern Excel workbooks without evaluating their formulas."""

    supported_suffixes = (".xlsx",)

    include_hidden_sheets: bool = True
    max_uncompressed_bytes: int = _DEFAULT_MAX_UNCOMPRESSED_BYTES
    max_archive_members: int = _DEFAULT_MAX_ARCHIVE_MEMBERS
    max_cells: int = _DEFAULT_MAX_CELLS
    max_worksheets: int = _DEFAULT_MAX_WORKSHEETS

    def parse(self, data: bytes, suffix: str) -> ParsedDocument:
        try:
            from openpyxl import load_workbook
        except (ImportError, ModuleNotFoundError) as exc:
            raise NotImplementedError(
                "XLSX extraction requires the optional 'openpyxl' dependency. "
                "Install rag-document-parser with the 'xlsx' extra."
            ) from exc

        XlsxPackageValidator(
            max_members=self.max_archive_members,
            max_uncompressed_bytes=self.max_uncompressed_bytes,
            max_cells=self.max_cells,
            max_worksheets=self.max_worksheets,
        ).validate(data)

        formula_workbook: Any | None = None
        value_workbook: Any | None = None
        reader_warnings: list[python_warnings.WarningMessage] = []
        try:
            with python_warnings.catch_warnings(record=True) as caught:
                python_warnings.simplefilter("always")
                formula_workbook = load_workbook(
                    io.BytesIO(data),
                    data_only=False,
                    read_only=False,
                    keep_links=False,
                    rich_text=True,
                )
                value_workbook = load_workbook(
                    io.BytesIO(data),
                    data_only=True,
                    read_only=False,
                    keep_links=False,
                    rich_text=True,
                )
                reader_warnings.extend(caught)
        except (MemoryError, RecursionError):
            raise
        except Exception as exc:
            raise ValueError("Invalid or unsupported XLSX workbook") from exc

        state = XlsxParseState()
        state.add_reader_warnings(reader_warnings)
        worksheet_parser = XlsxWorksheetParser(
            max_cells=self.max_cells,
            max_tables=self.max_archive_members,
        )
        units: list[EvidenceUnit] = []
        try:
            for chartsheet in formula_workbook.chartsheets:
                charts = list(getattr(chartsheet, "_charts", []))
                state.quality_warnings.append(
                    {
                        "type": "xlsx_charts_unsupported",
                        "severity": "medium",
                        "message": (
                            "Chart sheets are not extracted by the XLSX backend."
                        ),
                        "sheet_name": chartsheet.title,
                        "count": max(1, len(charts)),
                        "chart_sheet": True,
                    }
                )
            for worksheet in formula_workbook.worksheets:
                if (
                    worksheet.sheet_state != "visible"
                    and not self.include_hidden_sheets
                ):
                    state.quality_warnings.append(
                        {
                            "type": "xlsx_hidden_sheet_skipped",
                            "severity": "low",
                            "message": "A hidden worksheet was skipped by configuration.",
                            "sheet_name": worksheet.title,
                            "sheet_state": worksheet.sheet_state,
                        }
                    )
                    continue

                value_worksheet = value_workbook[worksheet.title]
                worksheet_parser.parse(
                    worksheet,
                    value_worksheet,
                    units,
                    state,
                )
        finally:
            if formula_workbook is not None:
                formula_workbook.close()
            if value_workbook is not None:
                value_workbook.close()

        return ParsedDocument(
            units=units,
            quality_warnings=state.quality_warnings,
        )
