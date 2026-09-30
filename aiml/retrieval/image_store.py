"""
Figure/table store — no CLIP, no image embedding model. Figures are
indexed by which page they came from; retrieval piggybacks on the TEXT
store instead of visual similarity: we find which page's text best
matches the question (via retrieval/vector_store.py) and surface the
figures/tables that live on that page.

This is a deliberate trade: it gives up true "find me a diagram that looks
like X" search in exchange for zero model downloads. In practice it works
well for paper QA, since a figure relevant to a question is almost always
on (or right next to) the page whose text discusses it.
"""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FigureRecord:
    figure_id: str
    doc_id: str
    page_num: int
    label: str  # "figure" or "table"
    crop_path: str  # path to saved crop image, relative to the index dir
    metadata: dict = field(default_factory=dict)


class ImageStore:
    def __init__(self):
        self.records: list[FigureRecord] = []

    def add(self, records: list[FigureRecord]) -> None:
        self.records.extend(records)

    def figures_on_pages(self, doc_page_pairs: list[tuple[str, int]], k: int = 2) -> list[FigureRecord]:
        """
        doc_page_pairs should be ordered by relevance (best page first).
        Takes at most one figure per page first (for variety across the
        best-matching pages), then fills any remaining slots with extra
        figures from those same pages if needed.
        """
        ordered_pairs = {pair: i for i, pair in enumerate(doc_page_pairs)}
        candidates = [r for r in self.records if (r.doc_id, r.page_num) in ordered_pairs]
        candidates.sort(key=lambda r: ordered_pairs[(r.doc_id, r.page_num)])

        picked: list[FigureRecord] = []
        seen_pages: set[tuple[str, int]] = set()
        for r in candidates:
            if (r.doc_id, r.page_num) not in seen_pages:
                picked.append(r)
                seen_pages.add((r.doc_id, r.page_num))
            if len(picked) == k:
                return picked

        for r in candidates:
            if r not in picked:
                picked.append(r)
            if len(picked) == k:
                break
        return picked

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "figures.pkl", "wb") as f:
            pickle.dump([asdict(r) for r in self.records], f)

    @classmethod
    def load(cls, in_dir: str | Path) -> "ImageStore":
        in_dir = Path(in_dir)
        figures_path = in_dir / "figures.pkl"
        if not figures_path.exists():
            raise FileNotFoundError(
                f"No figure index found at '{figures_path}'. Either it wasn't built yet "
                f"(re-run build_index.py without --skip_vision), or the PDFs had no "
                f"detectable figures/tables."
            )
        with open(figures_path, "rb") as f:
            raw = pickle.load(f)
        store = cls()
        store.records = [FigureRecord(**r) for r in raw]
        return store

    @classmethod
    def exists(cls, in_dir: str | Path) -> bool:
        return (Path(in_dir) / "figures.pkl").exists()
