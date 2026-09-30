"""
Render PDF pages to raster images so a layout detector can run on them.
Uses PyMuPDF (fitz) — no external poppler binary needed, which keeps the
whole pipeline pip-installable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class RenderedPage:
    doc_id: str
    page_num: int  # 1-indexed, matches pdf_parser.PageText.page_num
    image: Image.Image
    dpi: int


def render_pdf_pages(pdf_path: str | Path, dpi: int = 200) -> list[RenderedPage]:
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem
    doc = fitz.open(str(pdf_path))

    zoom = dpi / 72  # fitz's default render is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)

    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(RenderedPage(doc_id=doc_id, page_num=i + 1, image=img, dpi=dpi))
    doc.close()
    return pages


def render_pdf_dir(pdf_dir: str | Path, dpi: int = 200) -> list[RenderedPage]:
    pdf_dir = Path(pdf_dir)
    all_pages: list[RenderedPage] = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        all_pages.extend(render_pdf_pages(pdf_file, dpi=dpi))
    return all_pages
