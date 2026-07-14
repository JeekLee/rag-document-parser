# XLSX Backend Design

## Context

An Excel worksheet is neither a plain-text document nor necessarily one table.
It can contain prose blocks, title/date/count rows, multiple rectangular data
regions, merged multi-row headers, formulas, hidden content, and a large
style-only tail outside the real data.

The reference workbook supplied for this work contains five visible sheets:
one guide sheet with two prose blocks and four data sheets with tables. It also
contains cached formulas, hierarchical headers, body merges, rich strings, and
style-only rows and columns. Those characteristics define the initial quality
bar.

## Scope

The MVP adds built-in `.xlsx` support with these goals:

1. Emit prose as `EvidenceUnit(type="text", format="plain")`.
2. Emit detected data grids as `EvidenceUnit(type="table",
   format="structured_table")`.
3. Preserve workbook order, sheet name/state, source ranges, physical row/cell
   coordinates, merged spans, formulas, hyperlinks, notes, and non-General
   number formats.
4. Use a formula's stored cached value without executing the formula. Fall back
   to its expression when no cache exists.
5. Ignore style-only dimensions when finding content, preflight physical cells
   and merge footprints before workbook loading, and bound logical output ranges
   again after loading.

The following are intentionally out of scope:

- legacy `.xls`, binary `.xlsb`, and macro-enabled `.xlsm`;
- formula recalculation, VBA, external connections, or office-suite invocation;
- exact rendering of every locale-specific Excel number format;
- chart, drawing, shape, image, pivot, slicer, and form-control extraction;
- print-area or page-layout inference.

Charts and images that `openpyxl` exposes produce quality warnings rather than
being silently represented as evidence. Cell comments/notes are preserved as
nested plain-text children because their text and owning cell are available
without visual inference.

## Package Boundary

The backend lives at:

```text
src/rag_document_parser/evidence_unit_extraction/formats/xlsx/
    __init__.py
    backend.py
```

`XlsxBackend.supported_suffixes` is `(".xlsx",)`. The default registry maps
`.xlsx` to this backend, and the package root exports `XlsxBackend`.
`RagDocumentParser` remains responsible for suffix normalization and routing.

Spreadsheet support remains optional:

```toml
xlsx = [
    "openpyxl>=3.1,<4",
    "defusedxml>=0.7",
    "Pillow>=10.0",
]
```

`openpyxl` is imported inside `parse()`. Importing the package or using another
backend therefore does not require the XLSX extra. If parsing is requested
without it, the backend raises `NotImplementedError` with an installation hint.
Pillow is included so `openpyxl` exposes embedded images to the backend, which
can then report their unsupported status instead of silently dropping them.

## Safe Loading And Formula Policy

Before `openpyxl` runs, the backend verifies that the input is an unencrypted
ZIP package containing `[Content_Types].xml` and `xl/workbook.xml`. The default
limits are:

| Limit | Default |
| --- | ---: |
| archive members | 10,000 |
| declared uncompressed bytes | 256 MiB |
| represented worksheet cells | 1,000,000 |
| workbook sheets | 256 |
| workbook or worksheet relationship nodes | 10,000 |

Formula expressions and cached values require two workbook views. The backend
loads the same in-memory bytes with `data_only=False` and `data_only=True`, both
with `keep_links=False`, normal worksheet mode, and rich-text support. Formula
code is never evaluated and no network call is made.

Content types must declare exactly one XLSX workbook at
`/xl/workbook.xml`. Workbook and relationship XML are streamed first. Sheet and
relationship records are structurally validated before `openpyxl` can apply its
descriptor coercions; only namespaced relationship IDs from workbook sheets are
resolved. IDs and package targets are unique, internal targets must resolve to
the same canonical member name that `openpyxl` will load, and both sheets and
relationship nodes are capped.

Worksheet XML is then streamed with the `sheetData → row → cell` hierarchy and
implicit row/column counters validated against Excel bounds. Physical cells,
merged-range footprints, and effective hyperlink ranges count against
`max_cells`. Worksheet relationships are followed to comments parts, where
comment-list structure is validated, every comment must name one cell, and each
comment is charged to the same budget. This prevents a cell-heavy,
relationship-heavy, huge-merge, descriptor-override, or range-expansion package
from reaching normal-mode loading. A second, post-load check counts sparse
region bounding boxes, including blank cells inside a retained table.

Shared strings, styles, and drawing/chart parts are bounded by the ZIP byte
limit rather than separate XML-node limits. A public service accepting hostile
workbooks should lower the configurable byte/cell limits for its workload and
run parsing in a process with memory and execution-time limits; ZIP preflight is
defense in depth, not a replacement for process isolation.

For each formula cell:

- if the data-only view has a cached value, that value becomes cell text;
- if the cached value is `None`, the original expression (including `=`) becomes
  cell text and the expression is also retained in the cell's `formula` extra;
- the missing-cache case emits exactly one warning per coordinate:

```json
{
  "type": "xlsx_formula_cache_missing",
  "severity": "medium",
  "message": "...",
  "sheet_name": "Formula",
  "cell": "C2",
  "formula": "=B2*2"
}
```

## Logical Values

A cell is meaningful when it has a nonblank logical value, comment, or
hyperlink. Style alone does not make it meaningful. Rich-text runs are joined
through `openpyxl`'s rich-text value representation.

Cell text uses a deterministic, deliberately small formatting subset:

- booleans: `TRUE` / `FALSE`;
- dates: `YYYY-MM-DD`;
- datetimes: ISO 8601, reduced to a date when time is midnight;
- times: `HH:MM:SS`;
- integer and ordinary finite numeric values: stable base-10 text;
- percentage patterns such as `0.00%`: multiply by 100 and retain the requested
  decimal precision;
- all-zero integer patterns such as `0000`: retain leading zero padding;
- strings and formulas: normalize line endings, remove illegal control
  characters, preserve internal whitespace/newlines, and strip outer
  whitespace.

Unsupported custom formats fall back to the logical value. Every non-General
format is still stored on its cell as `number_format`, so downstream consumers
can apply a richer display layer later.

## Region Detection

The backend builds a sparse set of meaningful coordinates. A nonblank merged
anchor contributes the full merged footprint for adjacency, while covered
followers never duplicate the value.

Declared Excel Table ranges are emitted first as independent tables, even when
two table ranges touch vertically or horizontally. Their complete rectangular
footprints are bounded by `max_cells` and indexed once; this also detects
overlapping declarations without pairwise table comparisons. Their coordinates
are then removed from the sparse grid. This treats an explicit workbook
structure as stronger evidence than whitespace heuristics.

Declared ranges also act as geometric barriers. If surrounding cells wrap
around a declared table, inferred regions are partitioned at the table's row
edges and blank table-column interval instead of creating a bounding rectangle
through it. Consequently a declared cell is never emitted again through an
inferred table, while prose or side data surrounding it is retained.

Remaining logical regions are found in source order:

1. Split occupied rows wherever a completely blank row occurs.
2. Within each row band, split columns wherever a completely blank column
   occurs.
3. Bound each resulting component by its actual logical coordinates.

This makes style-only cells such as `Z100` irrelevant to a real `A1:B2` table,
and splits the guide sheet's `A1:A37` and `A39` blocks at blank row 38.

A remaining component becomes a table only when it has at least two rows and
columns and a defensible header followed by body data. Header selection uses
this priority:

1. the starting row of an intersecting auto-filter range, after verifying that
   the row contains at least two textual labels;
2. within the first twelve candidate rows, a textual row followed by a body-like
   type/style transition (for example, bold labels followed by nonbold values,
   or labels followed by numbers/formulas);
3. the first row with at least two textual labels and a following populated row.

When a selected header row is preceded by merged parent/group cells, that row is
included. A single merge spanning the entire component width remains a prose
title because it is more commonly a caption than a useful column group. This
conservative rule produces hierarchical headers such as:

```text
변경 전 / 요법코드
변경 전 / 항암화학요법
변경 후 / 요법코드
변경 후 / 항암화학요법
```

Rows before the selected header remain evidence rather than disappearing: each
populated source row becomes a text unit whose anchor values are joined by
` | `. A one-column, one-row, or otherwise ambiguous component is also emitted
as text rather than guessed into a lossy grid.

## Evidence Mapping

Every unit has:

```python
metadata["common"]["section_path"] == [worksheet.title]
```

All worksheets are included by default, including `hidden` and `veryHidden`
sheets. `include_hidden_sheets=False` is available when a caller explicitly
wants to skip them. Sheet state is always present in spreadsheet metadata.

### Text Units

Text metadata contains `sheet_name`, `sheet_state`, the actual `cell_range`, and
a `cells` list with each meaningful anchor address. Formula, number-format,
hyperlink, hidden-column, and comment details are retained on those descriptors
when present; formula summaries and nonempty hidden row/column lists are
optional. Text content/source also flatten hyperlinks and notes into readable
text. A vertical merge such as `A1:A37` produces one unit and keeps that full
range.

Text source follows the normal section convention:

```text
section: 안내
<cell text>
```

### Table Units

One `TableColumn` is created per physical column (`c1`, `c2`, ...). Header
lineage is flattened top-to-bottom with ` / `; a blank lineage falls back to
the Excel column letter.

For a declared Excel Table with `headerRowCount=0`, `tableColumns` names become
the logical column labels, `header_rows` is empty, `header_range` is `None`, and
the first physical row remains a body row. Declared tables with headers retain
their physical header cells in `header_rows`.

Header rows live in `StructuredTableContent.header_rows`; physical body rows
live in `rows` and have logical indices starting at 1. Each row adds
`source_row`. Each emitted cell adds `address` and conditionally adds `formula`,
`number_format`, `hyperlink`, or `hidden`.

A merged range emits only its top-left cell with `rowspan` and `colspan`.
Covered followers are omitted. Blank, unmerged cells inside the retained grid
remain explicit cells so column alignment is stable.

Baseline table provenance is:

```python
metadata["spreadsheet"] == {
    "sheet_name": "Summary",
    "sheet_state": "visible",
    "cell_range": "A4:C6",
    "header_range": "A4:C4",
    "data_range": "A5:C6",
}
```

Nonempty `hidden_rows` and `hidden_columns` are optional additions. Standard
table metadata remains `{table_id, headers, row_count}`. No shared model change
is required because row/cell models already allow source-specific extras.

Table source contains the sheet, physical range, flattened columns, header
rows, logical body-row index, original sheet row, and nonblank `label=value`
pairs. Header and body hyperlink/note text are included in source text while
remaining structured extras/children in content.

## Quality Warnings

The backend currently emits these stable warning types:

- `xlsx_formula_cache_missing`: a formula had no saved result;
- `xlsx_reader_warning`: `openpyxl` reported a degraded workbook feature;
- `xlsx_images_unsupported`: a sheet contains ignored images;
- `xlsx_charts_unsupported`: a sheet contains ignored charts;
- `xlsx_hidden_sheet_skipped`: hidden content was explicitly disabled.

Malformed/non-XLSX input, encryption, archive limits, and cell limits fail with
`ValueError`; they do not return a silently partial document.

## Reference Workbook Verification

The supplied workbook parses into 13 units: nine text units (guide content plus
title/date/count preambles) and four structured tables. The exact table results
are:

| Sheet | Header range | Data range | Body rows |
| --- | --- | --- | ---: |
| `검토중인 허가초과 항암요법` | `A4:C4` | `A5:C16` | 12 |
| `인정되고 있는 허가초과 항암요법(용법용량포함)` | `B2:K2` | `B3:K713` | 711 |
| `불승인 요법` | `A3:E3` | `A4:E678` | 675 |
| `허가초과 항암요법 변경대비표` | `B3:O4` | `B5:O30` | 26 |

Verification also checks that:

- style-only rows through 719 do not extend the 711-row table;
- cached `TODAY`, `COUNTA`, and `ROW` results are used;
- the last table has two header rows and correct flattened group labels;
- `불승인 요법!D521:D523` becomes one cell with `rowspan=3`;
- the guide sheet produces `A1:A37` and `A39:A39` text units;
- no warnings are produced for the supported contents of this file.

The `/tmp` workbook is used for local acceptance verification and is not copied
into the repository. Unit tests build compact workbooks in memory so the test
suite remains self-contained.

## Test Coverage

Focused tests cover:

1. public export, default registry entry, and `.XLSX` parser routing;
2. prose/table separation and source order;
3. single- and multi-row headers plus flattened merged groups;
4. merged header/body spans and omission of covered followers;
5. missing formula caches and warning shape;
6. dates, booleans, percentages, leading-zero formats, hyperlinks, addresses,
   formula extras, and source rows;
7. style-only range trimming and hidden-sheet metadata;
8. invalid archives and configured cell limits;
9. canonical workbook/worksheet relationships, sheet/row structure, custom
   worksheet/comment targets, and pre-load hyperlink/comment range validation;
10. adjacent, surrounding, and headerless declared Excel Tables;
11. the existing full project test suite and JSON serialization boundaries.
