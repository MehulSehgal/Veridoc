# Veridoc

Veridoc is a local research assistant for PDFs. It reads paper content, builds a searchable index, finds the most relevant passages, and answers questions with evidence from the document itself.

## Project organization

The project is now organized into three main areas:

- `aiml/` — document ingestion, retrieval, models, evaluation, and agent logic
- `backend/` — FastAPI API server and static assets for the web app
- `frontend/` — browser UI files and Streamlit interface code

The project is designed to feel like a lightweight research workflow rather than a black-box chatbot. It brings together retrieval, document structure analysis, a small reasoning loop, and a simple review step that checks whether the answer is actually supported by the evidence.

This version runs fully offline. There is no API key, no login, and no model download at runtime. The system relies on classical tools such as TF-IDF, SVD, OpenCV, and a rule-based critic.

## What it does

The app can:

- parse PDFs and extract text page by page
- split the text into overlapping chunks
- build a retrieval index from those chunks
- detect figures and tables in rendered pages
- answer questions using the most relevant evidence
- show the reasoning steps as a trace
- re-run the search if the confidence is low

In other words, it acts like a small research desk for paper analysis. You can point it at a folder of PDFs, ask a question, and it will try to ground the answer in the actual paper text and visual elements.

## How it works

The workflow is simple and explicit:

1. A PDF is ingested.
2. Text is extracted and chunked.
3. A TF-IDF + SVD index is built.
4. The planner breaks the user question into smaller search questions.
5. The retriever finds relevant passages and related figures.
6. The answerer drafts a response from the retrieved evidence.
7. The critic checks confidence and reformulates the query if needed.
8. The final answer is returned along with the trace.

This is a practical version of a ReAct-style loop: think, act, observe, improve.

## AI and ML in the project

Veridoc deliberately mixes two different kinds of "intelligence," and it's
worth being precise about which is which:

- **ML (machine learning)** — statistical/mathematical techniques that
  operate on data: vectorizing text, reducing dimensions, measuring
  similarity, detecting regions in an image. Nothing here is a neural
  network trained on external data; it's fit directly on your own corpus
  at index time.
- **AI (agentic logic)** — the orchestration layer on top: deciding what to
  search for, when to retry, when to trust an answer. This is structured,
  rule-based decision logic that mimics the *shape* of an LLM agent
  (plan → act → observe → improve) without an LLM underneath it.

### Machine learning components — what's used, where

| Component | Technique / library | Where it lives | What it does |
|---|---|---|---|
| Text retrieval | TF-IDF (term weighting) + Truncated SVD / LSA — `scikit-learn` | `aiml/retrieval/vector_store.py` | Converts chunk text into vectors, fit on your own corpus; search is a cosine-similarity dot product between query and chunk vectors |
| Extractive answering | TF-IDF sentence scoring — `scikit-learn` | `aiml/agents/critic.py` | Scores every candidate sentence against the question, keeps the top matches as the "answer" |
| Figure/table detection | Classical computer vision — adaptive thresholding, morphological dilation, contour detection — `OpenCV` | `aiml/ingestion/layout_detector.py` | Finds figure/table-shaped regions on a rendered page without a trained detector model |
| Cross-modal figure matching | Page co-location (reuses the TF-IDF retrieval scores above) | `aiml/agents/retriever_agent.py` | Surfaces figures from whichever pages best matched the question textually |
| Self-attention (educational) | Raw matrix ops — `Q·Kᵀ`, softmax, weighted sum — `PyTorch` | `aiml/models/attention_from_scratch.py` | Standalone demo of the mechanism behind transformer attention; not wired into the main answer pipeline |

### AI / agentic components — what's used, where

| Component | Approach | Where it lives | What it does |
|---|---|---|---|
| Planner | Rule-based question decomposition (regex conjunction-splitting + keyword extraction) | `aiml/agents/planner.py` | Breaks one question into 1–3 sub-queries; three modes (`zero_shot` / `few_shot` / `cot`) vary how aggressively it decomposes |
| Retriever agent | Tool-use wrapper around the ML retrieval store | `aiml/agents/retriever_agent.py` | Exposes retrieval as a callable "tool" the loop can invoke and log |
| Critic | Rule-based confidence scoring (retrieval-score strength + question/answer keyword overlap) | `aiml/agents/critic.py` | Decides whether an answer is good enough, or proposes a reformulated query if not |
| ReAct loop | Explicit Thought → Action → Observation orchestration with a retry cap | `aiml/agents/react_loop.py` | Ties Planner → Retriever → Answerer → Critic together, re-running the search when confidence is low |

No component in either table calls an external API or downloads model
weights at runtime — every number above is computed locally, either fit on
your corpus at index time (ML) or evaluated by fixed rules at query time (AI).

## Why this is useful

This project is useful because it demonstrates a practical, low-cost way to build a research assistant without depending on external APIs or expensive model infrastructure.

It is especially relevant when you want:

- local document search
- question answering over research PDFs
- evidence-based answers
- explainable retrieval and reasoning
- a demo that works without cloud services

It is not trying to be a fully generative AI assistant. Instead, it is a grounded and transparent prototype: it shows where the answer came from and how confident it is.

## Running it

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the index from a folder of PDFs:

```bash
python -m aiml.ingestion.build_index --pdf_dir data/papers --out_dir data/index
```

Run the question-answer loop:

```bash
python -m aiml.agents.react_loop --index_dir data/index --question "What loss function does the paper use and why?"
```

Start the browser demo:

```bash
uvicorn backend.server:app --reload --port 8000
```

Then open http://localhost:8000 in a browser.

## Notes

The main tradeoff is that the answers are extractive rather than fully generated prose. That is intentional. This project prefers grounded, source-based answers over fluent but unverified text.

It is a good prototype for understanding how retrieval, document processing, and lightweight agent loops fit together in practice.

### Folder layout

```text
Veridoc/
├── aiml/
│   ├── agents/
│   ├── eval/
│   ├── ingestion/
│   ├── models/
│   └── retrieval/
├── backend/
│   ├── server.py
│   └── static/
├── frontend/
│   ├── app.py
│   ├── styles.py
│   └── assets/
├── data/
├── README.md
├── requirements.txt
└── ...
```
