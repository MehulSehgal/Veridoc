"""
Answerer + Critic — no LLM call, no API key.

Answerer: extractive. Splits every retrieved passage into sentences, scores
each sentence against the question with TF-IDF cosine similarity (the exact
same dot-product-on-unit-vectors math used for retrieval), and returns the
top-scoring sentences, each tagged with its source. Answers are therefore
verbatim excerpts from the papers, not generated prose -- an explicit,
honest trade-off for running with zero external model calls.

Critic: rule-based. Confidence combines (a) how well the retrieved evidence
matched the question in the first place, and (b) whether the extracted
answer actually contains the question's key terms. Below threshold, it
proposes a reformulated query built from the missing keywords, which is
what drives the ReAct loop's retry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aiml.agents.retriever_agent import RetrievedEvidence

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "in", "on", "for", "to", "and", "or",
    "what", "why", "how", "does", "do", "did", "was", "were", "this", "that",
    "it", "its", "with", "as", "be", "which",
}
CONFIDENCE_ACCEPT = 0.6


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def _keywords(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z']+", text) if w.lower() not in _STOPWORDS}


def answer(question: str, evidence: list[RetrievedEvidence], max_sentences: int = 4) -> str:
    if not evidence:
        return "No relevant evidence was found in the indexed papers for this question."

    sentences: list[str] = []
    sources: list[str] = []
    for ev in evidence:
        for s in _split_sentences(ev.text):
            sentences.append(s)
            sources.append(ev.source)

    if not sentences:
        return "Evidence was retrieved but contained no cleanly extractable sentences."

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(sentences + [question])
    except ValueError:
        return " ".join(f"{s} [source: {src}]" for s, src in zip(sentences[:max_sentences], sources))

    q_vec = matrix[-1]
    sent_vecs = matrix[:-1]
    sims = cosine_similarity(sent_vecs, q_vec).ravel()

    ranked = sims.argsort()[::-1][:max_sentences]
    ranked = sorted(i for i in ranked if sims[i] > 0)  # keep reading order

    if not ranked:
        return "Evidence was retrieved but none of it matched the question closely enough to extract an answer."

    return " ".join(f"{sentences[i]} [source: {sources[i]}]" for i in ranked)


@dataclass
class CriticVerdict:
    confidence: float
    faithful: bool
    reason: str
    reformulated_query: str | None


def critique(question: str, draft_answer: str, evidence: list[RetrievedEvidence]) -> CriticVerdict:
    if not evidence:
        return CriticVerdict(0.0, False, "No evidence was retrieved.", question)

    avg_score = sum(e.score for e in evidence) / len(evidence)
    # LSA cosine scores tend to sit in a narrower band than raw TF-IDF cosine;
    # a gentle rescale keeps "confidence" intuitive on a 0-1 scale.
    confidence = max(0.0, min(1.0, avg_score * 1.6))

    q_keywords = _keywords(question)
    answer_lower = draft_answer.lower()
    missing = [kw for kw in q_keywords if kw not in answer_lower]
    faithful = len(missing) <= max(1, len(q_keywords) // 2)

    if confidence >= CONFIDENCE_ACCEPT and faithful:
        return CriticVerdict(confidence, True, "Evidence match and keyword coverage look sufficient.", None)

    if confidence < CONFIDENCE_ACCEPT:
        reason = f"Retrieved evidence only weakly matched the question (avg similarity {avg_score:.2f})."
    else:
        reason = f"Answer is missing question terms: {', '.join(missing[:4])}."

    reformulated = " ".join(missing) if missing else None
    return CriticVerdict(confidence, faithful, reason, reformulated)
