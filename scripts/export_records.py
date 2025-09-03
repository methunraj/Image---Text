#!/usr/bin/env python3
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Dict


def ensure_records(obj: Any) -> List[Dict[str, Any]]:
    """Normalize arbitrary JSON-like data to a list of row dicts.

    - list[dict] -> as-is
    - list[scalar/other] -> wrap as {"value": ...}
    - dict -> [dict]
    - str -> [{"text": str}]
    - other -> [{"value": obj}]
    """
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            return obj  # type: ignore[return-value]
        return [{"value": x} for x in obj]
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, str):
        return [{"text": obj}]
    return [{"value": obj}]


def all_columns(records: Iterable[Dict[str, Any]]) -> List[str]:
    cols: List[str] = []
    for r in records:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    return cols


def _norm_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    # JSON-encode nested structures
    return json.dumps(v, ensure_ascii=False)


def to_json_bytes(records: List[Dict[str, Any]]) -> bytes:
    return json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")


def to_markdown_bytes(records: List[Dict[str, Any]], cols: List[str]) -> bytes:
    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", "<br>")

    lines: List[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in records:
        row = [esc(str(_norm_cell(r.get(c, "")))) for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines).encode("utf-8")


def to_xlsx_bytes(records: List[Dict[str, Any]], cols: List[str]) -> bytes:
    from openpyxl import Workbook  # requires dependency

    wb = Workbook()
    ws = wb.active
    ws.title = "Export"
    ws.append(cols)
    for r in records:
        ws.append([_norm_cell(r.get(c, "")) for c in cols])
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def to_docx_bytes(records: List[Dict[str, Any]], cols: List[str]) -> bytes:
    """Create a simple DOCX table for records.

    - First row as headers
    - One row per record; non-primitive values are JSON-encoded
    """
    from docx import Document  # requires python-docx

    doc = Document()
    doc.add_heading("Export", level=1)

    # Create table with header
    table = doc.add_table(rows=1, cols=len(cols))
    hdr_cells = table.rows[0].cells
    for i, c in enumerate(cols):
        hdr_cells[i].text = str(c)

    for r in records:
        row_cells = table.add_row().cells
        for i, c in enumerate(cols):
            row_cells[i].text = str(_norm_cell(r.get(c, "")))

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def to_docx_from_text_bytes(text: str, title: str | None = None) -> bytes:
    """Create a DOCX document from plain text (best-effort for Markdown)."""
    from docx import Document

    doc = Document()
    if title:
        doc.add_heading(title, level=1)
    # Keep simple: one paragraph block; avoid complex markdown parsing
    for line in str(text or "").splitlines():
        doc.add_paragraph(line)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Export records (JSON) to JSON/Markdown/XLSX")
    p.add_argument("input", help="Path to JSON file (array or object)")
    p.add_argument("--out", default="export", help="Output base name without extension")
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    recs = ensure_records(data)
    cols = all_columns(recs)

    Path(args.out + ".json").write_bytes(to_json_bytes(recs))
    Path(args.out + ".md").write_bytes(to_markdown_bytes(recs, cols))
    Path(args.out + ".xlsx").write_bytes(to_xlsx_bytes(recs, cols))
    print(f"Wrote {args.out}.json {args.out}.md {args.out}.xlsx")


if __name__ == "__main__":
    _cli()
