"""
Retriever agent. Wraps vector-store search as a callable "tool" with a
uniform interface, so the ReAct loop can log it as a tool call
(this is the tool-use / function-calling piece of the pipeline).

Multimodal, without any image embedding model: figure/table retrieval
piggybacks on the text search above. It finds which pages' text best
matched the question and returns the figures/tables detected on those
pages -- no CLIP, no API call, no download.
"""
from __future__ import annotations

from dataclasses import dataclass

from aiml.retrieval.image_store import FigureRecord, ImageStore
from aiml.retrieval.vector_store import VectorStore


@dataclass
class RetrievedEvidence:
    text: str
    source: str
    score: float


@dataclass
class RetrievedFigure:
    crop_path: str
    label: str
    source: str
    score: float


TOOL_SPEC = {
    "name": "retrieve_evidence",
    "description": "Search the paper corpus for passages relevant to a sub-query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "k": {"type": "integer", "description": "Number of passages to return."},
        },
        "required": ["query"],
    },
}


class RetrieverAgent:
    def __init__(self, store: VectorStore, image_store: ImageStore | None = None):
        self.store = store
        self.image_store = image_store

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedEvidence]:
        results = self.store.search(query, k=k)
        return [
            RetrievedEvidence(text=chunk.text, source=chunk.metadata["source"], score=score)
            for chunk, score in results
        ]

    def retrieve_many(self, queries: list[str], k_each: int = 3) -> list[RetrievedEvidence]:
        """Run retrieval for each sub-query and dedupe by source+text."""
        seen: set[str] = set()
        evidence: list[RetrievedEvidence] = []
        for q in queries:
            for ev in self.retrieve(q, k=k_each):
                key = ev.source + ev.text[:50]
                if key not in seen:
                    seen.add(key)
                    evidence.append(ev)
        evidence.sort(key=lambda e: e.score, reverse=True)
        return evidence

    def retrieve_figures(self, query: str, k: int = 2, pages_to_check: int = 5) -> list[RetrievedFigure]:
        """
        No image embedding model: find the pages whose TEXT best matches the
        query (reusing the same TF-IDF store), then return whichever
        figures/tables were detected on those pages, best page first.
        """
        if self.image_store is None:
            return []

        top_chunks = self.store.search(query, k=pages_to_check)
        if not top_chunks:
            return []

        # order pages by their best matching chunk's score, dedupe by (doc, page)
        page_scores: dict[tuple[str, int], float] = {}
        for chunk, score in top_chunks:
            key = (chunk.doc_id, chunk.page_num)
            page_scores[key] = max(page_scores.get(key, 0.0), score)
        ordered_pages = sorted(page_scores, key=lambda k: -page_scores[k])

        records = self.image_store.figures_on_pages(ordered_pages, k=k)
        return [
            RetrievedFigure(
                crop_path=rec.crop_path,
                label=rec.label,
                source=rec.metadata.get("source", f"{rec.doc_id} p.{rec.page_num}"),
                score=page_scores.get((rec.doc_id, rec.page_num), 0.0),
            )
            for rec in records
        ]
