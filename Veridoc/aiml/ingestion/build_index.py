"""
CLI: parse every PDF in a directory, build BOTH indexes -- fully offline,
no API key, no model downloads:
  - text index:  chunked page text -> TF-IDF + SVD -> vector_store.py
  - image index: OpenCV-detected figure/table regions -> cropped and saved,
                 associated with their page (no image embedding model)

Usage:
    python -m aiml.ingestion.build_index --pdf_dir data/papers --out_dir data/index
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from aiml.ingestion.chunker import chunk_pages
from aiml.ingestion.layout_detector import crop_region, detect_regions
from aiml.ingestion.pdf_parser import parse_pdf_dir
from aiml.ingestion.pdf_render import render_pdf_dir
from aiml.retrieval.image_store import FigureRecord, ImageStore
from aiml.retrieval.vector_store import VectorStore


def build_text_index(pdf_dir: str, out_dir: str, chunk_size: int, overlap: int) -> None:
    print(f"[text] Parsing PDFs from {pdf_dir} ...")
    pages = parse_pdf_dir(pdf_dir)
    print(f"[text]   -> {len(pages)} pages")

    print("[text] Chunking ...")
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    print(f"[text]   -> {len(chunks)} chunks")

    if not chunks:
        print("[text] No text extracted -- check that data/papers contains readable PDFs.")
        return

    print("[text] Fitting TF-IDF + SVD ...")
    store = VectorStore(n_components=128)
    store.add(chunks)
    store.save(out_dir)
    print(f"[text] Saved ({len(chunks)} chunks indexed).")


def build_image_index(pdf_dir: str, out_dir: str, dpi: int) -> None:
    print(f"[vision] Rendering PDF pages at {dpi} DPI ...")
    rendered_pages = render_pdf_dir(pdf_dir, dpi=dpi)
    print(f"[vision]   -> {len(rendered_pages)} page images")

    crops_dir = Path(out_dir) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for page in tqdm(rendered_pages, desc="[vision] detecting regions"):
        regions = detect_regions(page.image)

        for i, region in enumerate(regions):
            crop = crop_region(page.image, region)
            if crop.width < 40 or crop.height < 40:
                continue  # skip slivers, usually detector noise

            figure_id = f"{page.doc_id}_p{page.page_num}_fig{i}"
            crop_path = crops_dir / f"{figure_id}.png"
            crop.save(crop_path)

            all_records.append(
                FigureRecord(
                    figure_id=figure_id,
                    doc_id=page.doc_id,
                    page_num=page.page_num,
                    label=region.label,
                    crop_path=str(crop_path.relative_to(out_dir)),
                    metadata={
                        "source": f"{page.doc_id} (p.{page.page_num})",
                        "detector": region.source,
                        "confidence": region.confidence,
                    },
                )
            )

    image_store = ImageStore()
    image_store.add(all_records)
    image_store.save(out_dir)
    print(f"[vision] Saved ({len(all_records)} figure/table regions).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="data/papers")
    parser.add_argument("--out_dir", default="data/index")
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--skip_vision", action="store_true", help="Text index only (faster).")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created empty directory '{pdf_dir}' -- drop your PDFs in there and re-run this command.")
        return

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(
            f"No .pdf files found in '{pdf_dir}'. Add at least one PDF there, "
            f"then re-run:\n  python -m aiml.ingestion.build_index --pdf_dir {pdf_dir} --out_dir {args.out_dir}"
        )
        return

    build_text_index(args.pdf_dir, args.out_dir, args.chunk_size, args.overlap)

    if not args.skip_vision:
        build_image_index(args.pdf_dir, args.out_dir, args.dpi)
    else:
        print("[vision] Skipped (--skip_vision).")


if __name__ == "__main__":
    main()
