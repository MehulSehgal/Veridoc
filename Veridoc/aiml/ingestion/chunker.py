"""
Sliding-window chunker over words, keeping citation metadata (doc_id, page_num)
attached to every chunk so the agent can cite sources precisely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pdf_parser import PageText


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_num: int
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    """
    Word-count sliding window per page. Small enough to keep citations
    page-accurate; overlap prevents losing context at chunk boundaries.
    """
    chunks: list[Chunk] = []
    for page in pages:
        words = page.text.split()
        if not words:
            continue

        start = 0
        idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_id = f"{page.doc_id}_p{page.page_num}_c{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    page_num=page.page_num,
                    text=chunk_text,
                    metadata={"source": f"{page.doc_id} (p.{page.page_num})"},
                )
            )
            idx += 1
            if end == len(words):
                break
            start = end - overlap  # step forward, keep overlap words

    return chunks
