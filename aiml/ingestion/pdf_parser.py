"""
Extract per-page text from a PDF. This is intentionally simple (text-only)
for Phase 1. Phase 2 swaps this for a layout-aware parser (YOLO region
detection -> separate text/figure/table crops).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PageText:
    doc_id: str
    page_num: int
    text: str


def _clean(text: str) -> str:
    text = re.sub(r"-\n", "", text)          # de-hyphenate line breaks
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def parse_pdf(pdf_path: str | Path) -> list[PageText]:
    """Return one PageText per page, with a doc_id derived from the filename."""
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem
    reader = PdfReader(str(pdf_path))

    pages: list[PageText] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = _clean(raw)
        if cleaned:
            pages.append(PageText(doc_id=doc_id, page_num=i + 1, text=cleaned))
    return pages


def parse_pdf_dir(pdf_dir: str | Path) -> list[PageText]:
    """Parse every .pdf in a directory."""
    pdf_dir = Path(pdf_dir)
    all_pages: list[PageText] = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        all_pages.extend(parse_pdf(pdf_file))
    return all_pages


if __name__ == "__main__":
    import sys

    pages = parse_pdf_dir(sys.argv[1] if len(sys.argv) > 1 else "data/papers")
    print(f"Parsed {len(pages)} pages.")
    if pages:
        print(f"Sample page ({pages[0].doc_id}, p{pages[0].page_num}):")
        print(pages[0].text[:300], "...")
