"""Retrieval — embed a query, hit Chroma, return ranked chunks.

Three-layer strategy:

1. **Hybrid (table-aware) per-doc retrieval.** For each doc considered, run two
   vector searches and merge: an unrestricted pass plus a table-only pass that
   filters to chunks containing "[TABLE". Dense embeddings of markdown tables
   are weak against prose — a chunk that mentions "industrial" 50 times in
   narrative will out-rank a sparse table row like "Industrial | 2,940". The
   table-only pass guarantees a semantically relevant table makes it into the
   context window.

2. **Per-doc split when comparing.** If the user has loaded multiple docs and
   does NOT pick a filter (the "All" option), naive vector search will surface
   chunks from whichever doc has more semantic density on the topic — the
   other doc gets shut out, and any "compare X and Y" question fails. To fix
   this we run the hybrid retrieval ONCE PER DOC and merge, so each doc is
   guaranteed representation in the context.

3. **Single-doc when filtered.** If the filter picks a specific doc, behave
   like a normal hybrid retriever scoped to that doc.

Claude's prompt is what tells it to (a) prefer tables for numeric questions
and (b) structure cross-doc comparisons with per-doc citations.
"""

from backend.embed import embed_texts
from backend.store import query as store_query, list_docs


def _flatten(results: dict) -> list[dict]:
    """Flatten Chroma's nested lists into a list of result dicts."""
    if not results or not results.get("documents") or not results["documents"][0]:
        return []
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def _hybrid_query(
    query_embedding: list[float],
    n_results: int,
    doc_name: str | None,
    n_tables: int,
) -> list[dict]:
    """Hybrid pass: unrestricted top-N + table-only top-M, scoped to one doc.

    Pass `doc_name=None` to scope across all docs (single-doc store, or before
    we know multi-doc handling is needed).
    """
    main = _flatten(
        store_query(query_embedding, n_results=n_results, doc_name=doc_name)
    )
    tables = _flatten(
        store_query(
            query_embedding,
            n_results=n_tables,
            doc_name=doc_name,
            content_contains="[TABLE",
        )
    )

    seen: set[tuple] = set()
    merged: list[dict] = []
    for r in main + tables:
        key = (r["metadata"]["doc_name"], r["metadata"]["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)
    return merged


def retrieve(
    query_text: str,
    n_results: int = 10,
    doc_name: str | None = None,
    n_tables: int = 5,
) -> list[dict]:
    """Hybrid retrieval, with per-doc split when no filter is set.

    Behavior:
      - Filter set (single doc): hybrid retrieval scoped to that doc only.
      - Filter unset, only one doc indexed: identical to filter set on that doc.
      - Filter unset, multiple docs indexed: hybrid retrieval per doc with a
        split budget, so EVERY doc is represented in the merged result set.

    The per-doc budget split is what makes "compare BREIT and ARCC"-style
    questions actually work — naive top-N retrieval almost always picks one
    doc to dominate. Slightly enlarges the context window when multi-doc, but
    Claude can handle it.
    """
    query_embedding = embed_texts([query_text])[0]

    # Single-doc filter — straightforward hybrid retrieval.
    if doc_name is not None:
        return _hybrid_query(query_embedding, n_results, doc_name, n_tables)

    # Multi-doc store, no filter — split budget per doc so every doc appears.
    docs = list_docs()
    if len(docs) <= 1:
        return _hybrid_query(query_embedding, n_results, None, n_tables)

    # Give each doc the FULL hybrid budget. Splitting it (e.g. n_results // N)
    # keeps token counts down but starves the per-doc retrieval — the right
    # page often sits at rank 6+ within its doc, and gets cut off if we only
    # take 5 chunks per doc. Haiku's context window swallows the extra easily.
    seen: set[tuple] = set()
    merged: list[dict] = []
    for d in docs:
        for r in _hybrid_query(query_embedding, n_results, d, n_tables):
            key = (r["metadata"]["doc_name"], r["metadata"]["chunk_index"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
    return merged
