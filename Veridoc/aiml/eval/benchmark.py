"""
Evaluation harness. Runs the benchmark question set through:
  - three prompting strategies for the Planner (zero_shot / few_shot / cot)
  - with and without the Critic improvement loop

...and reports a metrics table. This is the artifact that proves you
*measured* your design choices instead of just asserting they help.

Metrics (kept simple and dependency-light so it's easy to extend):
  - keyword_coverage: fraction of `expected_keywords` present in the final
    answer (a cheap proxy for faithfulness/completeness without needing a
    human rater or a judge-LLM call for every question)
  - avg_confidence: mean Critic confidence score
  - avg_iterations: mean number of ReAct loop iterations used

Usage:
    python -m eval.benchmark --index_dir data/index --questions eval/questions.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from aiml.agents.react_loop import MAX_ITERATIONS, run_react_loop
from aiml.agents.retriever_agent import RetrieverAgent
from aiml.retrieval.vector_store import VectorStore


@dataclass
class RunMetrics:
    strategy: str
    critic_enabled: bool
    keyword_coverage: float
    avg_confidence: float
    avg_iterations: float


def _keyword_coverage(answer_text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def run_condition(
    questions: list[dict],
    retriever: RetrieverAgent,
    strategy: str,
    critic_enabled: bool,
) -> RunMetrics:
    coverages, confidences, iterations = [], [], []

    for q in questions:
        # to isolate the critic's effect, force a single iteration when disabled
        max_iter_override = 1 if not critic_enabled else MAX_ITERATIONS
        import aiml.agents.react_loop as loop_module

        original_max = loop_module.MAX_ITERATIONS
        loop_module.MAX_ITERATIONS = max_iter_override
        try:
            result = run_react_loop(q["question"], retriever, strategy=strategy)
        finally:
            loop_module.MAX_ITERATIONS = original_max

        coverages.append(_keyword_coverage(result.final_answer, q.get("expected_keywords", [])))
        confidences.append(result.final_confidence)
        iterations.append(result.iterations_used)

    n = len(questions)
    return RunMetrics(
        strategy=strategy,
        critic_enabled=critic_enabled,
        keyword_coverage=sum(coverages) / n,
        avg_confidence=sum(confidences) / n,
        avg_iterations=sum(iterations) / n,
    )


def print_table(rows: list[RunMetrics]) -> None:
    header = f"{'Strategy':<10} | {'Critic':<7} | {'KeywordCov':<11} | {'AvgConf':<8} | {'AvgIters':<8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.strategy:<10} | {str(r.critic_enabled):<7} | "
            f"{r.keyword_coverage:<11.2f} | {r.avg_confidence:<8.2f} | {r.avg_iterations:<8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="data/index")
    parser.add_argument("--questions", default="eval/questions.json")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)

    store = VectorStore.load(args.index_dir)
    retriever = RetrieverAgent(store)

    rows = []
    for strategy in ["zero_shot", "few_shot", "cot"]:
        for critic_enabled in [False, True]:
            print(f"Running strategy={strategy}, critic_enabled={critic_enabled} ...")
            rows.append(run_condition(questions, retriever, strategy, critic_enabled))

    print("\n===== ABLATION RESULTS =====")
    print_table(rows)


if __name__ == "__main__":
    main()
