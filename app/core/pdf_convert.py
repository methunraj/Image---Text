from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional


def _import_pdfium():
    try:
        import pypdfium2 as pdfium  # type: ignore
        return pdfium
    except Exception:
        return None


def is_available() -> bool:
    return _import_pdfium() is not None


def parse_page_ranges(spec: str, total_pages: int) -> List[int]:
    """Parse a page range string like "1-5,8,10-12" into 0-based indices.
    Invalid tokens are ignored; indices are clamped to [0, total_pages).
    """
    out: List[int] = []
    if not spec:
        return out
    try:
        parts = [p.strip() for p in spec.split(',') if p.strip()]
        for part in parts:
            if '-' in part:
                a, b = part.split('-', 1)
                try:
                    start = max(1, int(a))
                    end = min(total_pages, int(b))
                    if start <= end:
                        out.extend([i - 1 for i in range(start, end + 1)])
                except Exception:
                    continue
            else:
                try:
                    page = int(part)
                    if 1 <= page <= total_pages:
                        out.append(page - 1)
                except Exception:
                    continue
    except Exception:
        return []
    # Deduplicate while preserving order
    seen = set()
    ordered: List[int] = []
    for i in out:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def convert_pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "PNG",
    overwrite: bool = False,
    page_spec: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Render a PDF to per-page image files.

    Returns: (created_paths, skipped_paths, errors)
    """
    pdfium = _import_pdfium()
    if pdfium is None:
        raise RuntimeError("pypdfium2 is not installed. Please install requirements and retry.")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    created: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    doc = pdfium.PdfDocument(str(pdf_path))
    total = len(doc)
    if page_spec:
        indices = parse_page_ranges(page_spec, total)
        if not indices:
            indices = list(range(total))
    else:
        indices = list(range(total))

    # Render each page individually for broader compatibility
    scale = dpi / 72.0  # PDF user space is 72 DPI
    for idx in indices:
        try:
            page = doc[idx]
            try:
                bmp = page.render(scale=scale)
                pil = bmp.to_pil()
            finally:
                try:
                    page.close()
                except Exception:
                    pass

            filename = f"{stem}_p{idx+1:04d}.{fmt.lower()}"
            target = out_dir / filename
            if target.exists() and not overwrite:
                skipped.append(str(target))
                continue
            if fmt.upper() == "JPEG":
                # Convert to RGB for JPEG compatibility
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                pil.save(str(target), format="JPEG", quality=90, optimize=True)
            else:
                pil.save(str(target), format=fmt.upper())
            created.append(str(target))
        except Exception as e:
            errors.append(f"page {idx+1}: {e}")
    try:
        doc.close()
    except Exception:
        pass
    return created, skipped, errors
