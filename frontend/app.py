"""
ArgusRAG demo UI. Ask a question, watch the investigation unfold: a
Planner/Retriever/Answerer/Critic trace on the left, the evidence that
was actually pulled (text passages and figures) on the right, and a
verdict card with a confidence ring at the end.

Run with: streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow `aiml.*` imports

from aiml.agents.react_loop import run_react_loop
from aiml.agents.retriever_agent import RetrieverAgent
from aiml.retrieval.image_store import ImageStore
from aiml.retrieval.vector_store import VectorStore
from frontend.styles import CSS, confidence_ring_svg

st.set_page_config(page_title="Veridoc — Case Board", layout="wide", page_icon="🗂️")
st.markdown(CSS, unsafe_allow_html=True)

# ---------- sidebar: the "case file" controls ----------
with st.sidebar:
    st.markdown('<div class="sidebar-label">Case file</div>', unsafe_allow_html=True)
    index_dir = st.text_input("Index directory", value="data/index", label_visibility="visible")
    strategy = st.selectbox(
        "Planner reasoning mode",
        ["cot", "few_shot", "zero_shot"],
        format_func=lambda s: {"cot": "Chain of thought", "few_shot": "Few-shot", "zero_shot": "Zero-shot"}[s],
    )
    load_clicked = st.button("Open case file", use_container_width=True)

    st.divider()
    if "store" in st.session_state:
        n_chunks = len(st.session_state.store.chunks)
        n_figs = len(st.session_state.image_store.records) if st.session_state.get("image_store") else 0
        st.caption(f"{n_chunks} passages indexed")
        st.caption(f"{n_figs} figures/tables indexed")
    else:
        st.caption("No case file open yet.")

if load_clicked:
    try:
        with st.spinner("Reading the case file..."):
            st.session_state.store = VectorStore.load(index_dir)
            st.session_state.image_store = (
                ImageStore.load(index_dir) if ImageStore.exists(index_dir) else None
            )
        st.sidebar.success("Case file loaded.")
    except FileNotFoundError as e:
        st.sidebar.error(str(e))

# ---------- header ----------
st.markdown('<div class="case-title">The investigation desk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="case-subtitle">Ask a question about the papers in this case file. '
    "The Planner breaks it down, the Retriever pulls evidence — text and figures — "
    "and the Critic checks the answer before signing off, reopening the search if it isn't convinced.</div>",
    unsafe_allow_html=True,
)

question = st.text_input(
    "Question",
    placeholder="e.g. What loss function does the paper use, and why?",
    label_visibility="collapsed",
)
run_clicked = st.button("Open investigation")

if run_clicked and question:
    if "store" not in st.session_state:
        st.error("Open a case file first, in the sidebar.")
    else:
        retriever = RetrieverAgent(st.session_state.store, image_store=st.session_state.get("image_store"))
        with st.spinner("Following the trail..."):
            result = run_react_loop(question, retriever, strategy=strategy)
        st.session_state.result = result

# ---------- results ----------
if "result" in st.session_state and run_clicked:
    result = st.session_state.result
    trace_col, evidence_col = st.columns([3, 2], gap="large")

    with trace_col:
        st.markdown('<div class="panel-heading">Investigation trace</div>', unsafe_allow_html=True)
        html = ['<div class="trace-list">']
        for step in result.trace:
            retry_class = " retry" if "Confidence below threshold" in step.content else ""
            html.append(
                f'<div class="trace-step{retry_class}">'
                f'<div class="trace-label">{step.label}</div>'
                f'<div class="trace-content">{step.content}</div>'
                f"</div>"
            )
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)

    with evidence_col:
        st.markdown('<div class="panel-heading">Evidence pulled</div>', unsafe_allow_html=True)

        if result.figures:
            for fig in result.figures:
                fig_path = Path(index_dir) / fig.crop_path
                st.markdown('<div class="figure-card">', unsafe_allow_html=True)
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                st.markdown(
                    f'<div class="figure-caption">{fig.label} — {fig.source} '
                    f"(match {fig.score:.2f})</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No figures matched this question.")

    st.markdown("<br>", unsafe_allow_html=True)
    verdict_col1, verdict_col2 = st.columns([1, 6])
    with verdict_col1:
        st.markdown(confidence_ring_svg(result.final_confidence), unsafe_allow_html=True)
    with verdict_col2:
        st.markdown(
            f'<div class="verdict-card" style="border:none; padding:0;">'
            f'<div><div class="verdict-text">{result.final_answer}</div>'
            f'<div class="verdict-meta">{result.iterations_used} investigation round'
            f'{"s" if result.iterations_used != 1 else ""} · '
            f'confidence {result.final_confidence:.0%}</div></div></div>',
            unsafe_allow_html=True,
        )

elif "result" not in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Open a case file in the sidebar, then ask a question to start an investigation.")
