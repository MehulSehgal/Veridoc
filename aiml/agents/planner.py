"""
Planner agent — no LLM call, no API key. Decomposes a question into
sub-queries using simple, deterministic linguistic heuristics: splitting on
conjunctions, and falling back to a keyword-only query for longer questions
so a broader match is also tried.

The three "strategy" modes are kept for interface/eval-harness compatibility
with the original design (and because comparing them is still a legitimate,
demoable ablation) -- without an LLM they differ in how aggressively they
decompose the question rather than in prompting style:
  - zero_shot: use the question exactly as asked, no decomposition
  - few_shot:  split only on explicit conjunctions ("and"/"or"/","/";")
  - cot:       split on conjunctions AND add a keyword-only sub-query,
               approximating "think about what's really being asked"
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "in", "on", "for", "to", "and", "or",
    "what", "why", "how", "does", "do", "did", "was", "were", "this", "that",
    "it", "its", "with", "as", "be", "which",
}
_SPLIT_PATTERN = re.compile(r"\band\b|\bor\b|[,;]|\bbut\b", flags=re.IGNORECASE)


def plan(question: str, strategy: str = "cot") -> list[str]:
    question = question.strip()
    if not question:
        return [question]

    if strategy == "zero_shot":
        return [question]

    parts = [p.strip() for p in _SPLIT_PATTERN.split(question) if p.strip()]
    if not parts:
        parts = [question]

    if strategy == "cot" and len(question.split()) > 8:
        keywords = [w for w in re.findall(r"[A-Za-z']+", question) if w.lower() not in _STOPWORDS]
        if keywords:
            parts.append(" ".join(keywords))

    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out[:3]
