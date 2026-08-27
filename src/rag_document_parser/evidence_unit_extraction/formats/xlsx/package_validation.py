from __future__ import annotations

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass


@dataclass(frozen=True)
class XlsxPackageValidator:
    """Validates XLSX package structure and resource limits before loading."""

    max_members: int
    max_uncompressed_bytes: int
    max_cells: int
    max_worksheets: int

    def validate(self, data: bytes) -> None:
        _validate_xlsx_archive(
            data,
            max_members=self.max_members,
            max_uncompressed_bytes=self.max_uncompressed_bytes,
            max_cells=self.max_cells,
            max_worksheets=self.max_worksheets,
        )


EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384
_WORKSHEET_MAIN_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_WORKSHEET_SHEET_DATA_TAG = f"{{{_WORKSHEET_MAIN_NAMESPACE}}}sheetData"
_WORKSHEET_ROW_TAG = f"{{{_WORKSHEET_MAIN_NAMESPACE}}}row"
_WORKSHEET_CELL_TAG = f"{{{_WORKSHEET_MAIN_NAMESPACE}}}c"
_RELATIONSHIP_ID_NAMES = (
    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
    "{http://purl.oclc.org/ooxml/officeDocument/relationships}id",
)
_XLSX_WORKBOOK_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
)
_WORKBOOK_CONTENT_TYPES = {
    _XLSX_WORKBOOK_CONTENT_TYPE,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml",
    "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
    "application/vnd.ms-excel.template.macroEnabled.main+xml",
}


@dataclass(frozen=True)
class _PackageRelationship:
    relationship_id: str
    relationship_type: str
    target: str
    target_mode: str


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _validate_xlsx_archive(
    data: bytes,
    *,
    max_members: int,
    max_uncompressed_bytes: int,
    max_cells: int,
    max_worksheets: int,
) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > max_members:
                raise ValueError(
                    f"XLSX archive has too many members: {len(members)} > {max_members}"
                )
            total_size = sum(member.file_size for member in members)
            if total_size > max_uncompressed_bytes:
                raise ValueError(
                    "XLSX archive is too large after decompression: "
                    f"{total_size} > {max_uncompressed_bytes} bytes"
                )
            if any(member.flag_bits & 0x1 for member in members):
                raise ValueError("Encrypted XLSX archives are not supported")
            names = {member.filename for member in members}
            required = {
                "[Content_Types].xml",
                "xl/workbook.xml",
                "xl/_rels/workbook.xml.rels",
            }
            if not required.issubset(names):
                raise ValueError("Input is not an XLSX workbook archive")
            _validate_standard_workbook_part(
                archive,
                max_entries=max_members,
            )
            _preflight_worksheet_structure(
                archive,
                members,
                max_cells=max_cells,
                max_relationships=max_members,
                max_worksheets=max_worksheets,
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("Input is not a valid XLSX archive") from exc


def _validate_standard_workbook_part(
    archive: zipfile.ZipFile,
    *,
    max_entries: int,
) -> None:
    try:
        from defusedxml.ElementTree import iterparse
    except (ImportError, ModuleNotFoundError) as exc:
        raise NotImplementedError(
            "XLSX extraction requires the optional 'defusedxml' dependency. "
            "Install rag-document-parser with the 'xlsx' extra."
        ) from exc

    workbook_overrides: list[tuple[str, str]] = []
    workbook_defaults: list[str] = []
    element_stack: list[str] = []
    entry_count = 0
    # openpyxl descriptors accept child elements that can override attributes.
    # Keep package selection unambiguous before the normal reader sees the XML.
    try:
        with archive.open("[Content_Types].xml") as content_types_xml:
            for event, element in iterparse(
                content_types_xml,
                events=("start", "end"),
            ):
                element_tag = str(element.tag)
                local_name = _local_name(element_tag)
                if event == "end":
                    element_stack.pop()
                    element.clear()
                    continue

                depth = len(element_stack)
                element_stack.append(element_tag)
                if depth == 0:
                    if local_name != "Types":
                        raise ValueError("Invalid XLSX content-types manifest root")
                    continue
                if depth != 1:
                    raise ValueError(
                        "XLSX content-types entries must not contain child elements"
                    )
                if local_name not in {"Default", "Override"}:
                    raise ValueError(
                        "XLSX content-types manifest contains an unexpected element"
                    )
                entry_count += 1
                if entry_count > max_entries:
                    raise ValueError(
                        "XLSX content-types manifest has too many entries: "
                        f"{entry_count} > {max_entries}"
                    )

                content_type = str(element.attrib.get("ContentType", ""))
                if content_type not in _WORKBOOK_CONTENT_TYPES:
                    continue
                if local_name == "Default":
                    workbook_defaults.append(content_type)
                else:
                    workbook_overrides.append(
                        (str(element.attrib.get("PartName", "")), content_type)
                    )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Invalid XLSX content-types manifest") from exc

    expected = [("/xl/workbook.xml", _XLSX_WORKBOOK_CONTENT_TYPE)]
    if workbook_defaults or workbook_overrides != expected:
        raise ValueError("XLSX archive must declare exactly one standard workbook part")


def _preflight_worksheet_structure(
    archive: zipfile.ZipFile,
    members: list[zipfile.ZipInfo],
    *,
    max_cells: int,
    max_relationships: int,
    max_worksheets: int,
) -> None:
    try:
        from defusedxml.ElementTree import iterparse
    except (ImportError, ModuleNotFoundError) as exc:
        raise NotImplementedError(
            "XLSX extraction requires the optional 'defusedxml' dependency. "
            "Install rag-document-parser with the 'xlsx' extra."
        ) from exc

    physical_cells = 0
    merged_footprint = 0
    hyperlink_footprint = 0
    comment_cells = 0
    members_by_name = {member.filename: member for member in members}
    worksheet_names = _worksheet_part_names(
        archive,
        max_relationships=max_relationships,
        max_worksheets=max_worksheets,
    )
    worksheet_members = [members_by_name[name] for name in worksheet_names]
    comment_part_names = _comment_part_names(
        archive,
        worksheet_names,
        max_relationships=max_relationships,
    )
    try:
        for member in worksheet_members:
            with archive.open(member) as worksheet_xml:
                element_stack: list[str] = []
                row_counter = 0
                column_counter = 0
                for event, element in iterparse(
                    worksheet_xml,
                    events=("start", "end"),
                ):
                    element_tag = str(element.tag)
                    if event == "end":
                        element_stack.pop()
                        element.clear()
                        continue

                    parent_tag = element_stack[-1] if element_stack else None
                    element_stack.append(element_tag)
                    local_name = _local_name(element_tag)
                    parent_local_name = (
                        _local_name(parent_tag) if parent_tag is not None else None
                    )
                    if element_tag == _WORKSHEET_ROW_TAG:
                        if parent_tag != _WORKSHEET_SHEET_DATA_TAG:
                            raise ValueError(
                                "Worksheet rows must be direct children of sheetData"
                            )
                        row_reference = str(element.attrib.get("r", ""))
                        if row_reference:
                            _validate_row_number(row_reference)
                            row_counter = int(row_reference)
                        else:
                            row_counter += 1
                            if row_counter > EXCEL_MAX_ROWS:
                                raise ValueError(
                                    "Inferred worksheet row is outside Excel bounds: "
                                    f"{row_counter}"
                                )
                        column_counter = 0
                    elif parent_tag == _WORKSHEET_ROW_TAG:
                        if element_tag != _WORKSHEET_CELL_TAG:
                            raise ValueError(
                                "Worksheet row contains an unexpected element: "
                                f"{element_tag!r}"
                            )
                        cell_reference = str(element.attrib.get("r", ""))
                        if cell_reference:
                            _validate_cell_coordinate(cell_reference)
                            match = re.fullmatch(
                                r"\$?([A-Za-z]{1,3})\$?(\d+)",
                                cell_reference.strip(),
                            )
                            if match is None:  # pragma: no cover - validated above
                                raise ValueError(
                                    f"Invalid worksheet cell reference: {cell_reference!r}"
                                )
                            column_counter = column_number(match.group(1))
                        else:
                            column_counter += 1
                            if column_counter > EXCEL_MAX_COLUMNS:
                                raise ValueError(
                                    "Inferred worksheet cell is outside Excel bounds: "
                                    f"row {row_counter}, column {column_counter}"
                                )
                        physical_cells += 1
                    elif element_tag == _WORKSHEET_CELL_TAG:
                        raise ValueError(
                            "Worksheet cells must be direct children of rows"
                        )
                    elif parent_local_name == "hyperlink":
                        # A child named ref would override the bounded attribute.
                        raise ValueError("Worksheet hyperlink entries must be empty")
                    elif local_name == "mergeCell":
                        merged_footprint += _cell_range_area(
                            str(element.attrib.get("ref", ""))
                        )
                    elif local_name == "hyperlink":
                        hyperlink_footprint += _cell_range_area(
                            str(element.attrib.get("ref", "")),
                            reference_kind="worksheet hyperlink",
                        )
                    elif local_name == "col":
                        _validate_column_range(
                            str(element.attrib.get("min", "")),
                            str(element.attrib.get("max", "")),
                        )
                    represented = (
                        physical_cells + merged_footprint + hyperlink_footprint
                    )
                    if represented > max_cells:
                        raise ValueError(
                            "XLSX workbook contains too many cells before loading: "
                            f"{represented} > {max_cells}"
                        )
        for name in comment_part_names:
            with archive.open(name) as comments_xml:
                element_stack = []
                comment_list_depth: int | None = None
                saw_comment_list = False
                for event, element in iterparse(
                    comments_xml,
                    events=("start", "end"),
                ):
                    element_tag = str(element.tag)
                    local_name = _local_name(element_tag)
                    if event == "end":
                        current_depth = len(element_stack) - 1
                        if (
                            comment_list_depth == current_depth
                            and local_name == "commentList"
                        ):
                            comment_list_depth = None
                        element_stack.pop()
                        element.clear()
                        continue

                    depth = len(element_stack)
                    parent_local_name = (
                        _local_name(element_stack[-1]) if element_stack else None
                    )
                    element_stack.append(element_tag)
                    if depth == 0:
                        if local_name != "comments":
                            raise ValueError("Invalid worksheet comments root")
                        continue
                    if comment_list_depth is None:
                        if depth == 1 and local_name == "commentList":
                            if saw_comment_list:
                                raise ValueError(
                                    "Worksheet comments contain multiple comment lists"
                                )
                            saw_comment_list = True
                            comment_list_depth = depth
                        continue
                    if depth == comment_list_depth + 1:
                        # NestedSequence coerces every direct child to a comment.
                        if local_name != "comment":
                            raise ValueError(
                                "Worksheet comment list contains an unexpected element"
                            )
                        reference = str(element.attrib.get("ref", ""))
                        if not reference:
                            raise ValueError("Worksheet comment has no cell reference")
                        _validate_cell_coordinate(reference)
                        comment_cells += 1
                    elif (
                        depth == comment_list_depth + 2
                        and parent_local_name == "comment"
                        and local_name == "ref"
                    ):
                        raise ValueError(
                            "Worksheet comment cell references must be attributes"
                        )
                    represented = (
                        physical_cells
                        + merged_footprint
                        + hyperlink_footprint
                        + comment_cells
                    )
                    if represented > max_cells:
                        raise ValueError(
                            "XLSX workbook contains too many cells before loading: "
                            f"{represented} > {max_cells}"
                        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Invalid worksheet XML in XLSX archive") from exc


def _read_relationships(
    archive: zipfile.ZipFile,
    part_name: str,
    *,
    max_relationships: int,
    scope: str,
) -> list[_PackageRelationship]:
    try:
        from defusedxml.ElementTree import iterparse
    except (ImportError, ModuleNotFoundError) as exc:
        raise NotImplementedError(
            "XLSX extraction requires the optional 'defusedxml' dependency. "
            "Install rag-document-parser with the 'xlsx' extra."
        ) from exc

    relationships: list[_PackageRelationship] = []
    seen_ids: set[str] = set()
    element_stack: list[str] = []
    # RelationshipList treats every root child as a relationship, regardless of
    # its tag, and descriptor children can override Id/Type/Target attributes.
    try:
        with archive.open(part_name) as relationships_xml:
            for event, element in iterparse(
                relationships_xml,
                events=("start", "end"),
            ):
                element_tag = str(element.tag)
                local_name = _local_name(element_tag)
                if event == "end":
                    element_stack.pop()
                    element.clear()
                    continue

                depth = len(element_stack)
                element_stack.append(element_tag)
                if depth == 0:
                    if local_name != "Relationships":
                        raise ValueError(f"Invalid {scope} relationships root")
                    continue
                if depth != 1:
                    raise ValueError(f"{scope} relationship entries must be empty")
                if local_name != "Relationship":
                    raise ValueError(
                        f"{scope} relationships contain an unexpected element"
                    )

                count = len(relationships) + 1
                if count > max_relationships:
                    verb = "have" if scope.endswith("s") else "has"
                    raise ValueError(
                        f"{scope} {verb} too many relationships: "
                        f"{count} > {max_relationships}"
                    )
                relationship_id = str(element.attrib.get("Id", ""))
                relationship_type = str(element.attrib.get("Type", ""))
                target = str(element.attrib.get("Target", ""))
                if not relationship_id or not relationship_type or not target:
                    raise ValueError(
                        f"{scope} relationship is missing a required attribute"
                    )
                if relationship_id in seen_ids:
                    raise ValueError(
                        f"{scope} contains duplicate relationship identifiers"
                    )
                seen_ids.add(relationship_id)
                relationships.append(
                    _PackageRelationship(
                        relationship_id=relationship_id,
                        relationship_type=relationship_type,
                        target=target,
                        target_mode=str(element.attrib.get("TargetMode", "")),
                    )
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid {scope} relationships in XLSX archive") from exc
    return relationships


def _resolve_internal_target(source_part: str, target: str) -> str:
    if "\\" in target:
        raise ValueError("XLSX relationship target contains a backslash")
    if target.startswith("/"):
        resolved = target[1:]
    else:
        resolved = posixpath.normpath(
            posixpath.join(posixpath.dirname(source_part), target)
        )
    if (
        not resolved
        or resolved in {".", ".."}
        or resolved.startswith("/")
        or resolved.startswith("../")
        or posixpath.normpath(resolved) != resolved
    ):
        raise ValueError(
            f"XLSX relationship target is not a canonical package path: {target!r}"
        )
    return resolved


def _workbook_sheet_relationship_ids(
    archive: zipfile.ZipFile,
    *,
    max_worksheets: int,
) -> set[str]:
    try:
        from defusedxml.ElementTree import iterparse
    except (ImportError, ModuleNotFoundError) as exc:
        raise NotImplementedError(
            "XLSX extraction requires the optional 'defusedxml' dependency. "
            "Install rag-document-parser with the 'xlsx' extra."
        ) from exc

    relationship_ids: set[str] = set()
    element_stack: list[str] = []
    sheets_depth: int | None = None
    saw_sheets = False
    # Workbook sheets use a NestedSequence: any direct child would otherwise be
    # coerced to a sheet, and child descriptors could replace its r:id.
    try:
        with archive.open("xl/workbook.xml") as workbook_xml:
            for event, element in iterparse(
                workbook_xml,
                events=("start", "end"),
            ):
                element_tag = str(element.tag)
                local_name = _local_name(element_tag)
                if event == "end":
                    current_depth = len(element_stack) - 1
                    if sheets_depth == current_depth and local_name == "sheets":
                        sheets_depth = None
                    element_stack.pop()
                    element.clear()
                    continue

                depth = len(element_stack)
                element_stack.append(element_tag)
                if depth == 0:
                    if local_name != "workbook":
                        raise ValueError("Invalid XLSX workbook root")
                    continue
                if sheets_depth is None:
                    if depth == 1 and local_name == "sheets":
                        if saw_sheets:
                            raise ValueError(
                                "XLSX workbook contains multiple sheets containers"
                            )
                        saw_sheets = True
                        sheets_depth = depth
                    continue
                if depth != sheets_depth + 1:
                    raise ValueError("XLSX workbook sheet entries must be empty")
                if local_name != "sheet":
                    raise ValueError(
                        "XLSX workbook sheets contain an unexpected element"
                    )
                if len(relationship_ids) + 1 > max_worksheets:
                    raise ValueError(
                        "XLSX workbook contains too many sheets: "
                        f"{len(relationship_ids) + 1} > {max_worksheets}"
                    )
                namespaced_ids = {
                    str(element.attrib[name])
                    for name in _RELATIONSHIP_ID_NAMES
                    if element.attrib.get(name)
                }
                if not namespaced_ids:
                    raise ValueError(
                        "XLSX workbook sheet has no relationship identifier"
                    )
                if len(namespaced_ids) != 1:
                    raise ValueError(
                        "XLSX workbook sheet has conflicting relationship identifiers"
                    )
                relationship_id = namespaced_ids.pop()
                if relationship_id in relationship_ids:
                    raise ValueError(
                        "Multiple workbook sheets use the same relationship identifier"
                    )
                relationship_ids.add(relationship_id)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Invalid workbook XML in XLSX archive") from exc
    if not saw_sheets or not relationship_ids:
        raise ValueError("XLSX workbook contains no sheets")
    return relationship_ids


def _comment_part_names(
    archive: zipfile.ZipFile,
    worksheet_names: list[str],
    *,
    max_relationships: int,
) -> list[str]:
    archive_names = set(archive.namelist())
    comment_parts: list[str] = []
    relationship_count = 0
    for worksheet_name in worksheet_names:
        directory, filename = posixpath.split(worksheet_name)
        relationships_name = posixpath.join(
            directory,
            "_rels",
            f"{filename}.rels",
        )
        if relationships_name not in archive_names:
            continue
        relationships = _read_relationships(
            archive,
            relationships_name,
            max_relationships=max_relationships - relationship_count,
            scope="XLSX worksheets",
        )
        relationship_count += len(relationships)
        for relationship in relationships:
            if relationship.target_mode == "External":
                resolved = None
            else:
                resolved = _resolve_internal_target(
                    worksheet_name,
                    relationship.target,
                )
            if not relationship.relationship_type.rstrip("/").endswith("/comments"):
                continue
            if resolved is None:
                raise ValueError("External worksheet comments are not supported")
            if resolved not in archive_names:
                raise ValueError(
                    "Worksheet comments part is missing from XLSX archive: "
                    f"{resolved!r}"
                )
            comment_parts.append(resolved)
    return comment_parts


def _worksheet_part_names(
    archive: zipfile.ZipFile,
    *,
    max_relationships: int,
    max_worksheets: int,
) -> list[str]:
    sheet_relationship_ids = _workbook_sheet_relationship_ids(
        archive,
        max_worksheets=max_worksheets,
    )
    relationships = _read_relationships(
        archive,
        "xl/_rels/workbook.xml.rels",
        max_relationships=max_relationships,
        scope="XLSX workbook",
    )

    names: list[str] = []
    target_set: set[str] = set()
    archive_names = set(archive.namelist())
    resolved_sheet_ids: set[str] = set()
    for relationship in relationships:
        if relationship.target_mode == "External":
            resolved = None
        else:
            resolved = _resolve_internal_target(
                "xl/workbook.xml",
                relationship.target,
            )
        if relationship.relationship_id not in sheet_relationship_ids:
            continue
        resolved_sheet_ids.add(relationship.relationship_id)
        if resolved is None:
            raise ValueError("External workbook sheet parts are not supported")
        if resolved not in archive_names:
            raise ValueError(
                f"Workbook sheet part is missing from XLSX archive: {resolved!r}"
            )
        if resolved in target_set:
            raise ValueError("Multiple workbook sheets reference the same package part")
        target_set.add(resolved)

        relationship_type = relationship.relationship_type.rstrip("/")
        if relationship_type.endswith("/chartsheet"):
            continue
        if not relationship_type.endswith("/worksheet"):
            raise ValueError(
                "Unsupported workbook sheet relationship type: "
                f"{relationship.relationship_type!r}"
            )
        names.append(resolved)

    unresolved = sheet_relationship_ids - resolved_sheet_ids
    if unresolved:
        raise ValueError("Workbook sheet relationship is missing from XLSX archive")
    return names


def _cell_range_area(
    reference: str,
    *,
    reference_kind: str = "merged-cell",
) -> int:
    match = re.fullmatch(
        r"\$?([A-Za-z]{1,3})\$?(\d+)(?::\$?([A-Za-z]{1,3})\$?(\d+))?",
        reference.strip(),
    )
    if match is None:
        raise ValueError(f"Invalid {reference_kind} reference: {reference!r}")
    min_col = column_number(match.group(1))
    min_row = int(match.group(2))
    max_col = column_number(match.group(3) or match.group(1))
    max_row = int(match.group(4) or match.group(2))
    if (
        min_col <= 0
        or min_row <= 0
        or max_col < min_col
        or max_row < min_row
        or max_col > EXCEL_MAX_COLUMNS
        or max_row > EXCEL_MAX_ROWS
    ):
        raise ValueError(
            f"{reference_kind.capitalize()} reference is outside Excel bounds: "
            f"{reference!r}"
        )
    return (max_col - min_col + 1) * (max_row - min_row + 1)


def _validate_cell_coordinate(reference: str) -> None:
    if not reference:
        return
    match = re.fullmatch(r"\$?([A-Za-z]{1,3})\$?(\d+)", reference.strip())
    if match is None:
        raise ValueError(f"Invalid worksheet cell reference: {reference!r}")
    column = column_number(match.group(1))
    row = int(match.group(2))
    if not (1 <= column <= EXCEL_MAX_COLUMNS and 1 <= row <= EXCEL_MAX_ROWS):
        raise ValueError(f"Worksheet cell is outside Excel bounds: {reference!r}")


def _validate_row_number(value: str) -> None:
    if not value:
        return
    try:
        row = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid worksheet row number: {value!r}") from exc
    if not 1 <= row <= EXCEL_MAX_ROWS:
        raise ValueError(f"Worksheet row is outside Excel bounds: {row}")


def _validate_column_range(min_value: str, max_value: str) -> None:
    try:
        min_column = int(min_value)
        max_column = int(max_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid worksheet column range: {min_value!r}:{max_value!r}"
        ) from exc
    if not (1 <= min_column <= max_column <= EXCEL_MAX_COLUMNS):
        raise ValueError(
            f"Worksheet column range is outside Excel bounds: {min_column}:{max_column}"
        )


def column_number(column: str) -> int:
    result = 0
    for character in column.strip().upper():
        if not "A" <= character <= "Z":
            continue
        result = result * 26 + ord(character) - ord("A") + 1
    return result
