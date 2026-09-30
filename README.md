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

### Retrieval
The main retrieval system uses TF-IDF and SVD (latent semantic analysis). It is implemented in the vector store module used for indexing and similarity search.

This is classic ML for document search. The model is trained on the corpus itself, and similarity is computed using cosine similarity, which is the same idea as embedding-based retrieval in a more lightweight form.

### Computer vision
The figure/table detection step uses OpenCV to process rendered PDF pages, find candidate regions, and crop them. This is implemented in the ingestion pipeline and is a good example of classical CV instead of a downloaded detector model.

### Agent-style reasoning
The reasoning loop is a simplified agent workflow. It does not rely on a large language model. Instead, it explicitly plans, retrieves, answers, and criticizes.

### Critic / confidence check
The critic is intentionally rule-based rather than LLM-based. It checks whether the answer is well supported by the evidence and whether the retrieval signal is strong enough. If it is not, it reformulates the question and tries again.

### Optional learning demo
There is also a PyTorch example in the attention demo module that shows how self-attention works without using a library shortcut.

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
