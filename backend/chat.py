"""Claude-powered cited Q&A over retrieved chunks."""

import anthropic
import streamlit as st

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 800


def _format_excerpts(retrieved: list[dict]) -> str:
    """Render retrieved chunks as `[p. N] ...` blocks for the prompt."""
    blocks = []
    for r in retrieved:
        page = r["metadata"]["page"]
        doc = r["metadata"]["doc_name"]
        blocks.append(f"[{doc} · p. {page}]\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def answer_with_claude(question: str, retrieved: list[dict]) -> str:
    """Ask Claude to answer the question using ONLY the retrieved excerpts."""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    excerpts = _format_excerpts(retrieved)

    prompt = f"""You are a fund analyst. Answer the question using ONLY the excerpts below.

The excerpts may include both prose and structured tables. Tables are marked
with a `[TABLE from page N]` header and formatted as markdown pipe tables.
**When a question is numeric, prefer the table data over narrative prose** — the
tables are the cleanest, most reliable source for numbers.

Each excerpt is prefixed with a header like `[<doc_name> · p. N]`. When the
excerpts come from MULTIPLE documents (different doc names), you are
answering across funds. In that case:
- Cite using a doc-aware format like `[<short fund label> p. N]` — for example
  `[BREIT p. 137]` or `[ARCC p. 19]`. Use the shortest unambiguous label
  derived from the doc name (the fund ticker / acronym is usually obvious).
- For comparison questions, structure the answer to address each fund
  explicitly. Don't merge funds into a single number.
- If the excerpts contain data for one fund but not the other, state both
  what you found and what's missing.

Format rules — follow ALL of them:
1. Give the final answer directly. Do NOT narrate reasoning, do NOT reconcile contradictions in the text, do NOT write phrases like "however, actually..." or "based on the data provided". Just state the answer.
2. One to three sentences for single-fund questions. Two to four sentences for cross-fund comparisons. Prefer the shorter end.
3. Cite every factual claim inline using the page numbers (and doc labels when multi-fund) shown in the excerpt headers.
4. If multiple excerpts discuss different metrics, choose the one that DIRECTLY answers the specific question asked. A question about "property count" must use property-count data, not asset value or revenue data — even if the asset-value chunk is more prominent.
5. If the excerpts do not actually contain the answer, reply exactly: "The provided excerpts do not contain this information." Do not guess.
6. Do not repeat the question back.

Question: {question}

Excerpts:

{excerpts}

Answer:"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
