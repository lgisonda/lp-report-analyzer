"""
LP Report Analyzer — Streamlit frontend.

RAG-powered Q&A and side-by-side metric extraction over fund SEC filings.
Backend: pdfplumber + sentence-transformers + Chroma + Anthropic Claude.
"""

import streamlit as st

from backend.extract import extract_pages
from backend.chunk import chunk_pages
from backend.embed import embed_texts, get_model
from backend.store import add_chunks, list_docs, doc_exists
from backend.retrieve import retrieve
from backend.chat import answer_with_claude
from backend.metrics import (
    CANONICAL_METRICS,
    extract_all_metrics,
    save_metrics,
    load_metrics,
    metrics_exist,
)

st.set_page_config(
    page_title="LP Report Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual polish — custom CSS overrides Streamlit's defaults for a more
# finance-tool aesthetic. Hides the default "Made with Streamlit" chrome,
# tightens spacing, and styles the metric cards.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit's default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tighter top padding so the hero sits closer to the top */
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}

    /* Hero title */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f3a5f;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #6b6b6b;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    /* Section header styling */
    h2, h3 {color: #1f3a5f !important;}

    /* Metric cards (st.metric) — soften the default border */
    [data-testid="stMetric"] {
        background-color: #f6f4ee;
        border: 1px solid #e5e1d6;
        border-radius: 8px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #6b6b6b;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #1a1a1a;
        font-weight: 600;
    }

    /* Comparison row borders */
    .metric-row {
        border-bottom: 1px solid #ececec;
        padding: 0.6rem 0;
    }
    .metric-row:last-child {border-bottom: none;}

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {background-color: #faf8f3;}
    [data-testid="stSidebar"] h2 {
        font-size: 1.1rem;
        color: #1f3a5f;
        margin-bottom: 0.5rem;
    }

    /* Doc-name badge in sidebar */
    .doc-badge {
        display: inline-block;
        font-size: 0.7rem;
        padding: 1px 6px;
        border-radius: 3px;
        margin-left: 4px;
    }
    .doc-badge-ok {background: #d8e8d8; color: #2d5a2d;}
    .doc-badge-pending {background: #f0e0c0; color: #6b4d10;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~400MB)...")
def _warm_model():
    return get_model()


def _short_label(doc_name: str) -> str:
    """Pull a short fund label from a doc filename, e.g. 'BREIT 12.31.2025 10K.pdf' -> 'BREIT'."""
    return doc_name.split(" ", 1)[0] if " " in doc_name else doc_name.rsplit(".", 1)[0]


_warm_model()


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-title">LP Report Analyzer</div>
    <p class="hero-subtitle">
      RAG-powered Q&A and side-by-side metric extraction over fund SEC filings.
      Built on Anthropic Claude.
    </p>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📁 Indexed documents")
    docs = list_docs()
    if docs:
        for d in docs:
            if metrics_exist(d):
                badge = '<span class="doc-badge doc-badge-ok">metrics ✓</span>'
            else:
                badge = '<span class="doc-badge doc-badge-pending">no metrics</span>'
            st.markdown(
                f"<div style='font-size:0.85rem; margin-bottom:6px;'>"
                f"<code>{d}</code> {badge}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("_No documents indexed yet — upload one to get started._")

    st.caption(f"Vector store: `./.chroma/`")

    docs_missing_metrics = [d for d in docs if not metrics_exist(d)]
    if docs_missing_metrics:
        st.markdown("---")
        st.markdown(
            f"_{len(docs_missing_metrics)} doc(s) missing metrics_"
        )
        if st.button("⚙️ Generate metrics", use_container_width=True):
            for d in docs_missing_metrics:
                with st.spinner(f"Extracting metrics for {d}..."):
                    progress = st.empty()

                    def _cb(i: int, total: int, label: str, _d=d) -> None:
                        progress.write(f"  → {_d} · {i}/{total}: {label}")

                    payload = extract_all_metrics(d, progress_cb=_cb)
                    save_metrics(payload)
                    progress.write(f"  → {d}: done")
            st.success("Metrics extracted.")
            st.rerun()


# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------

st.markdown("## 1 · Ingest a report")
st.caption("Upload a fund 10-K or 10-Q. Text and tables are extracted, embedded, and indexed.")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

if uploaded is not None:
    doc_name = uploaded.name

    if doc_exists(doc_name):
        st.info(
            f"`{doc_name}` is already indexed. "
            f"To re-ingest, delete `.chroma/` and restart."
        )
    else:
        with st.status(f"Ingesting {doc_name}...", expanded=True) as status:
            st.write("Extracting text and tables per page...")
            pages = list(extract_pages(uploaded))
            st.write(f"  → {len(pages)} pages parsed")

            st.write("Chunking...")
            chunks = chunk_pages(pages, doc_name=doc_name)
            st.write(f"  → {len(chunks)} chunks")

            st.write("Embedding chunks...")
            embeddings = embed_texts([c.text for c in chunks])
            st.write(f"  → {len(embeddings)} vectors (dim {len(embeddings[0])})")

            st.write("Storing in Chroma...")
            add_chunks(chunks, embeddings)
            st.write("  → done")

            st.write("Extracting canonical metrics...")
            metrics_progress = st.empty()

            def _progress_cb(i: int, total: int, label: str) -> None:
                metrics_progress.write(f"  → {i}/{total}: {label}")

            payload = extract_all_metrics(doc_name, progress_cb=_progress_cb)
            save_metrics(payload)
            metrics_progress.write(f"  → {len(payload['metrics'])} metrics cached")

            status.update(label=f"✓ Ingested {doc_name}", state="complete")

        st.success(f"Indexed **{doc_name}**.")


# ---------------------------------------------------------------------------
# 2. Key metrics
# ---------------------------------------------------------------------------

st.markdown("## 2 · Key metrics")

all_docs = list_docs()
if not all_docs:
    st.info("Ingest at least one document above to see the dashboard.")
    st.stop()

docs_with_metrics = [(d, load_metrics(d)) for d in all_docs if metrics_exist(d)]

if not docs_with_metrics:
    st.info(
        "No cached metrics yet. Click **Generate metrics** in the sidebar, "
        "or re-ingest a document."
    )

elif len(docs_with_metrics) == 1:
    # Single-doc dashboard: 3-column grid of st.metric cards.
    doc_name, payload = docs_with_metrics[0]
    st.caption(
        f"`{_short_label(doc_name)}` · {doc_name} · extracted {payload['extracted_at']}"
    )
    metrics = payload["metrics"]
    cols = st.columns(3)
    for i, m in enumerate(CANONICAL_METRICS):
        cell = metrics.get(m["key"], {})
        value = cell.get("value") or "—"
        page = cell.get("page")
        cols[i % 3].metric(
            label=m["label"],
            value=value,
            help=f"Source: p. {page}" if page else "Not found in document",
        )

else:
    # Multi-doc dashboard: card-row layout with prominent values and small
    # gray page citations underneath. One row per metric, one column per fund.
    fund_count = len(docs_with_metrics)
    header_cols = st.columns([2] + [3] * fund_count)
    header_cols[0].markdown("**Metric**")
    for i, (doc, _) in enumerate(docs_with_metrics, start=1):
        header_cols[i].markdown(f"**{_short_label(doc)}**")

    st.markdown("<hr style='margin: 0.4rem 0; border: 0; border-top: 1px solid #ddd;'>", unsafe_allow_html=True)

    for m in CANONICAL_METRICS:
        cols = st.columns([2] + [3] * fund_count)
        cols[0].markdown(
            f"<div style='padding-top:6px;'><b>{m['label']}</b></div>",
            unsafe_allow_html=True,
        )
        for i, (doc, payload) in enumerate(docs_with_metrics, start=1):
            cell = payload["metrics"].get(m["key"], {})
            value = cell.get("value") or "—"
            page = cell.get("page")
            if value in ("NOT FOUND", "—") or page is None:
                cols[i].markdown(
                    f"<div style='color:#bbb; padding-top:6px;'>{value}</div>",
                    unsafe_allow_html=True,
                )
            else:
                cols[i].markdown(
                    f"<div style='padding-top:6px;'>{value}"
                    f"<br><span style='color:#888; font-size:0.8rem;'>p. {page}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            "<hr style='margin: 0.3rem 0; border: 0; border-top: 1px solid #f0f0f0;'>",
            unsafe_allow_html=True,
        )

    st.caption("— means the metric was not extractable from the filing.")


# ---------------------------------------------------------------------------
# 3. Q&A
# ---------------------------------------------------------------------------

st.markdown("## 3 · Ask a question")
st.caption(
    "Free-form Q&A over the indexed filings. Answers are grounded in retrieved "
    "excerpts and include inline page citations."
)

col_q, col_filter = st.columns([3, 1])
with col_q:
    question = st.text_input(
        "Question",
        placeholder="e.g. What were total revenues in 2025?",
        label_visibility="collapsed",
    )
with col_filter:
    doc_filter = st.selectbox(
        "Limit to document",
        options=["All"] + all_docs,
        label_visibility="collapsed",
    )

if not question:
    st.stop()

filter_name = None if doc_filter == "All" else doc_filter
results = retrieve(question, n_results=10, doc_name=filter_name)

if not results:
    st.warning("No relevant chunks found.")
    st.stop()

with st.spinner("Claude is reading the excerpts..."):
    answer = answer_with_claude(question, results)

st.markdown("#### Answer")
with st.container(border=True):
    st.markdown(answer)

with st.expander(f"Show the {len(results)} retrieved chunks", expanded=False):
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        st.markdown(
            f"**#{i}** · `{meta['doc_name']}` · page **{meta['page']}** · "
            f"distance `{r['distance']:.3f}`"
        )
        st.write(r["text"])
        st.markdown("---")
