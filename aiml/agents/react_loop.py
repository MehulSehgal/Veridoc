"""
Orchestrates the multi-agent ReAct loop:

    Thought (Planner decomposes question)
    Action  (Retriever fetches evidence)
    Observation (evidence returned)
    Thought (Answerer drafts an answer)
    Action  (Critic scores it)
    Observation (confidence + reformulated query, if any)
    -> loop again with reformulated query if confidence < threshold
    -> else return final answer + full trace

This IS the "agentic workflow / improvement loop" made concrete: it is a
plain Python loop with explicit stop conditions, not a black-box library call.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from aiml.agents.critic import answer, critique, CriticVerdict
from aiml.agents.planner import plan
from aiml.agents.retriever_agent import RetrievedFigure, RetrieverAgent, RetrievedEvidence
from aiml.retrieval.image_store import ImageStore
from aiml.retrieval.vector_store import VectorStore

CONFIDENCE_THRESHOLD = 0.6
MAX_ITERATIONS = 3


@dataclass
class TraceStep:
    label: str
    content: str


@dataclass
class ReActResult:
    final_answer: str
    trace: list[TraceStep] = field(default_factory=list)
    iterations_used: int = 0
    final_confidence: float = 0.0
    figures: list[RetrievedFigure] = field(default_factory=list)


def run_react_loop(
    question: str,
    retriever: RetrieverAgent,
    strategy: str = "cot",
) -> ReActResult:
    trace: list[TraceStep] = []
    all_evidence: list[RetrievedEvidence] = []
    current_query = question
    draft_answer = ""
    verdict: CriticVerdict | None = None

    for i in range(1, MAX_ITERATIONS + 1):
        # --- Thought: Planner decomposes the (possibly reformulated) query ---
        sub_queries = plan(current_query, strategy=strategy)
        trace.append(TraceStep("Thought (Planner)", f"Sub-queries: {sub_queries}"))

        # --- Action: Retriever fetches evidence ---
        new_evidence = retriever.retrieve_many(sub_queries, k_each=3)
        all_evidence = _merge_evidence(all_evidence, new_evidence)
        trace.append(
            TraceStep(
                "Action (Retriever)",
                f"Retrieved {len(new_evidence)} new passages "
                f"({len(all_evidence)} total unique).",
            )
        )

        # --- Observation: show what was retrieved ---
        preview = "\n".join(f"- [{e.source}] {e.text[:100]}..." for e in all_evidence[:5])
        trace.append(TraceStep("Observation", preview))

        # --- Thought: Answerer drafts an answer from all evidence so far ---
        draft_answer = answer(question, all_evidence)
        trace.append(TraceStep("Thought (Answerer)", draft_answer))

        # --- Action: Critic scores the draft ---
        verdict = critique(question, draft_answer, all_evidence)
        trace.append(
            TraceStep(
                "Action (Critic)",
                f"confidence={verdict.confidence:.2f} faithful={verdict.faithful} "
                f"reason={verdict.reason}",
            )
        )

        if verdict.confidence >= CONFIDENCE_THRESHOLD or not verdict.reformulated_query:
            break

        trace.append(
            TraceStep(
                "Observation",
                f"Confidence below threshold. Reformulating query -> "
                f"'{verdict.reformulated_query}'",
            )
        )
        current_query = verdict.reformulated_query

    # Visual evidence: a lightweight extra retrieval pass over the figure/table
    # index, using the original question. Kept separate from the main text
    # loop above so a slow/missing image index never blocks the text answer.
    figures: list[RetrievedFigure] = []
    if retriever.image_store is not None:
        figures = retriever.retrieve_figures(question, k=2)
        if figures:
            trace.append(
                TraceStep(
                    "Observation (Visual)",
                    f"Found {len(figures)} relevant figure(s)/table(s): "
                    + ", ".join(f"{fig.label} from {fig.source}" for fig in figures),
                )
            )

    return ReActResult(
        final_answer=draft_answer,
        trace=trace,
        iterations_used=i,
        final_confidence=verdict.confidence if verdict else 0.0,
        figures=figures,
    )


def _merge_evidence(
    existing: list[RetrievedEvidence], new: list[RetrievedEvidence]
) -> list[RetrievedEvidence]:
    seen = {e.source + e.text[:50] for e in existing}
    merged = list(existing)
    for e in new:
        key = e.source + e.text[:50]
        if key not in seen:
            seen.add(key)
            merged.append(e)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="data/index")
    parser.add_argument("--question", required=True)
    parser.add_argument("--strategy", default="cot", choices=["zero_shot", "few_shot", "cot"])
    args = parser.parse_args()

    try:
        store = VectorStore.load(args.index_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    image_store = ImageStore.load(args.index_dir) if ImageStore.exists(args.index_dir) else None
    retriever = RetrieverAgent(store, image_store=image_store)
    result = run_react_loop(args.question, retriever, strategy=args.strategy)

    print("\n===== TRACE =====")
    for step in result.trace:
        print(f"\n[{step.label}]\n{step.content}")

    print("\n===== FINAL ANSWER =====")
    print(result.final_answer)
    print(f"\n(iterations={result.iterations_used}, confidence={result.final_confidence:.2f})")

    if result.figures:
        print("\n===== VISUAL EVIDENCE =====")
        for fig in result.figures:
            print(f"- {fig.label} from {fig.source} (score={fig.score:.2f}) -> {fig.crop_path}")


if __name__ == "__main__":
    main()
