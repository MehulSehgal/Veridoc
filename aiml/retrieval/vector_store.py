"""
Text vector store — no API key, no downloaded model. Retrieval is TF-IDF
(term weighting) reduced with SVD (a.k.a. LSA), fit directly on your own
corpus with scikit-learn. Search is still literally a dot product between
L2-normalized vectors (cosine similarity) -- same math as an embedding-model
version, just with classical term weighting instead of a learned encoder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from aiml.ingestion.chunker import Chunk


class VectorStore:
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None  # (N, n_components), L2-normalized rows

    def add(self, chunks: list[Chunk]) -> None:
        """
        Fits the vectorizer + SVD on this corpus. Call once with the FULL
        chunk set (TF-IDF needs the whole vocabulary to weight terms
        sensibly) -- this isn't an incremental "add one doc at a time" API.
        """
        self.chunks = chunks
        texts = [c.text for c in chunks]

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
        tfidf = self.vectorizer.fit_transform(texts)

        n_comp = max(2, min(self.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        reduced = self.svd.fit_transform(tfidf)
        self.matrix = normalize(reduced).astype("float32")

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if self.vectorizer is None or self.matrix is None or not self.chunks:
            return []
        q_tfidf = self.vectorizer.transform([query])
        q_vec = normalize(self.svd.transform(q_tfidf)).astype("float32")[0]

        scores = self.matrix @ q_vec  # cosine similarity via dot product (unit vectors)
        top_idx = np.argsort(-scores)[: min(k, len(self.chunks))]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "text_store.pkl", "wb") as f:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "vectorizer": self.vectorizer,
                    "svd": self.svd,
                    "chunks": self.chunks,
                    "matrix": self.matrix,
                },
                f,
            )

    @classmethod
    def exists(cls, in_dir: str | Path) -> bool:
        return (Path(in_dir) / "text_store.pkl").exists()

    @classmethod
    def load(cls, in_dir: str | Path) -> "VectorStore":
        in_dir = Path(in_dir)
        store_path = in_dir / "text_store.pkl"
        if not store_path.exists():
            raise FileNotFoundError(
                f"No text index found at '{store_path}'. "
                f"Build one first with:\n"
                f"  python -m aiml.ingestion.build_index --pdf_dir data/papers --out_dir {in_dir}"
            )
        with open(store_path, "rb") as f:
            data = pickle.load(f)
        store = cls(n_components=data["n_components"])
        store.vectorizer = data["vectorizer"]
        store.svd = data["svd"]
        store.chunks = data["chunks"]
        store.matrix = data["matrix"]
        return store
