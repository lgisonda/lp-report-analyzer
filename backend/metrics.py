"""Canonical metrics extraction.

The cross-fund comparison failure mode (REIT "distribution rate" vs BDC
"dividend yield", different table structures, etc.) isn't fixable inside a
pure RAG loop — it's a vocabulary problem that gets worse with more docs.

The fix is structural: extract a fixed set of canonical metrics ONCE per doc
at ingest time, cache them as JSON, and answer comparison questions from the
structured cache instead of from RAG. Free-form Q&A still uses RAG; only
side-by-side comparisons pull from the structured layer.

This module owns:
- CANONICAL_METRICS: the list of metrics we extract for every fund.
- extract_metric / extract_all_metrics: per-doc extraction (Steps 2-4).
- save_metrics / load_metrics: JSON persistence next to .chroma/.

"""

import json
from datetime import datetime
from pathlib import Path

import anthropic
import streamlit as st

from backend.retrieve import retrieve

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300  # Per-metric extraction is short — value + page only.
METRICS_DIR = Path(".chroma")  # Cache lives next to the vector store.

# Each metric has:
#   key       — snake_case identifier; JSON key in the cache.
#   label     — display label for the dashboard UI.
#   question  — the retrieval query. This is what gets embedded against the
#               chunks; phrase it like a real analyst question so the
#               embedding lands on the right chunks.
#   guidance  — extra instruction added to Claude's extraction prompt.
#               Use this to bridge REIT vs BDC vocabulary, point at which
#               table to use, and call out common pitfalls.

CANONICAL_METRICS: list[dict] = [
    {
        "key": "total_assets",
        "label": "Total Assets",
        "question": (
            "What is the fund's total assets at the most recent fiscal year-end? "
            "Look for 'Total assets' on the consolidated balance sheet, the "
            "consolidated statement of assets and liabilities, or the segment "
            "reporting note."
        ),
        "guidance": (
            "Find the largest 'Total assets' figure for the most recent fiscal "
            "year-end. Acceptable sources, in order of preference: "
            "(1) the 'Total assets' line on the consolidated balance sheet (REITs) "
            "or consolidated statement of assets and liabilities (BDCs); "
            "(2) the 'Total assets' line in the segment reporting footnote (label "
            "the answer with '(segment-reporting basis)' so the reader knows). "
            "DO NOT report a single segment's total assets — only the company-wide "
            "or fully-consolidated total. "
            "DO NOT report a subsidiary or VIE's total assets — only the parent. "
            "Sanity check: for a major fund the number should be in the tens of "
            "billions; if your answer is far smaller you've probably grabbed a "
            "subsidiary or a sub-portfolio — re-read and find the consolidated total. "
            "Report value with units, e.g. '$24.5 billion' or '$24,500 million'."
        ),
    },
    {
        "key": "nav_per_share",
        "label": "NAV / Share",
        "question": "What is the most recent net asset value (NAV) per share?",
        "guidance": (
            "For non-traded REITs with multiple share classes, use the Class I NAV per share "
            "as the primary value (other classes are typically within a few cents). "
            "For BDCs, use 'Net asset value per share' from the balance sheet. "
            "Report a single dollar value, e.g. '$13.45'."
        ),
    },
    {
        "key": "total_revenue",
        "label": "Total Revenue / Investment Income",
        "question": "What was total revenue or total investment income for the most recent fiscal year?",
        "guidance": (
            "Use the most recent annual figure (full fiscal year, not interim quarter). "
            "For REITs this is 'Total revenues' on the income statement. "
            "For BDCs this is 'Total investment income'. "
            "Report value with units."
        ),
    },
    {
        "key": "net_income",
        "label": "Net Income / NII",
        "question": "What was net income or net investment income for the most recent fiscal year?",
        "guidance": (
            "For REITs report 'Net income attributable to [the fund]' for the full fiscal year. "
            "For BDCs report 'Net investment income' (the income-statement subtotal BEFORE "
            "realized/unrealized gains and losses). Do NOT report 'Net increase in net assets "
            "resulting from operations' — that includes mark-to-market gains. "
            "Report value with units."
        ),
    },
    {
        "key": "mgmt_fee_rate",
        "label": "Management Fee Rate",
        "question": "What is the management fee rate?",
        "guidance": (
            "Report the fee as a percentage with its base — e.g. '1.25% of NAV' or "
            "'1.5% of assets up to 1.0x leverage, 1.0% above'. Tiered structures should "
            "be reported in full. Do NOT report dollar amounts of fees paid — only the rate "
            "and the asset base it's applied to."
        ),
    },
    {
        "key": "distribution_rate",
        "label": "Distribution Rate / Yield",
        "question": (
            "What is the most recent annualized distribution rate, dividend rate, "
            "or dividend yield disclosed in the filing?"
        ),
        "guidance": (
            "For non-traded REITs, look for 'annualized distribution rate' or 'distribution "
            "rate' explicitly stated as a percentage of NAV. "
            "For BDCs, look for an explicitly stated 'dividend yield' percentage or "
            "'dividends declared per share' for the year (which can be reported as-is). "
            "**You MUST cite an explicit page where the percentage or per-share amount "
            "appears. DO NOT compute a yield from stock price and dividends — return "
            "NOT FOUND if the filing doesn't state a yield or per-share total directly.** "
            "Report the value with its base, e.g. '4.0% of NAV' or '$1.92 per share' "
            "or '8.5% dividend yield'."
        ),
    },
    {
        "key": "leverage_ratio",
        "label": "Leverage",
        "question": (
            "What is the fund's leverage measure — debt-to-equity ratio, "
            "loan-to-value (LTV), or asset coverage ratio?"
        ),
        "guidance": (
            "Different fund types report leverage differently — accept and label whichever "
            "the filing actually discloses: "
            "REITs typically report **loan-to-value (LTV)** as a percentage (e.g. '47% LTV'). "
            "BDCs typically report **debt-to-equity ratio** (e.g. '1.05x debt-to-equity') "
            "and/or **asset coverage ratio** (e.g. '195% asset coverage' — required by the "
            "Investment Company Act). "
            "Report the value WITH ITS LABEL so the reader knows which measure it is — "
            "e.g. '47% LTV' or '1.05x debt-to-equity' or '195% asset coverage'. "
            "If only total debt and total equity are stated separately and no ratio is "
            "disclosed, you may compute and report a debt-to-equity ratio, but mark it as "
            "computed (e.g. '~1.0x debt-to-equity (computed)'). "
            "If no leverage data is disclosed at all, respond NOT FOUND."
        ),
    },
    {
        "key": "largest_sector",
        "label": "Largest Sector / Industry",
        "question": "What is the largest sector or industry concentration in the portfolio by fair value?",
        "guidance": (
            "For REITs this is the largest property sector (Rental Housing, Industrial, etc.) by "
            "asset value or property count — use the portfolio summary table. "
            "For BDCs this is the largest industry concentration (Software, Healthcare, etc.) by "
            "fair value of investments — use the industry concentration table. "
            "Report sector name AND percentage, e.g. 'Industrial — 27% of portfolio'."
        ),
    },
    {
        "key": "portfolio_count",
        "label": "Portfolio Count",
        "question": (
            "How many properties does the REIT own in total, or how many active "
            "portfolio companies does the BDC hold investments in, as of the "
            "most recent fiscal year-end?"
        ),
        "guidance": (
            "Report the **total active holdings** as of the most recent fiscal "
            "year-end — NOT the number of new investments made during the year, "
            "NOT the number of investments exited, NOT the count of portfolio "
            "companies in any single industry. "
            "For REITs: use the total property count from the portfolio summary "
            "table — the 'Total' row that sums across all sectors. Don't report "
            "a single sector. "
            "For BDCs: use the total count of distinct portfolio companies as of "
            "year-end. This is usually in the 10-K business overview, phrased like "
            "'investments in X portfolio companies' or 'X different portfolio "
            "companies' or shown at the bottom of the Schedule of Investments. "
            "Pay attention to wording — 'X new investments during 2025' or 'X "
            "transactions completed' is NOT the same thing as total active count. "
            "If unclear, prefer the larger number that is explicitly described as "
            "the count of holdings at period end. "
            "Report a single integer."
        ),
    },
]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _format_excerpts(retrieved: list[dict]) -> str:
    """Render retrieved chunks the same way chat.py does."""
    blocks = []
    for r in retrieved:
        page = r["metadata"]["page"]
        doc = r["metadata"]["doc_name"]
        blocks.append(f"[{doc} · p. {page}]\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def _parse_response(text: str) -> dict:
    """Pull VALUE / PAGE out of Claude's response.

    Tolerant to extra whitespace, leading/trailing prose, and either order.
    Returns {value, page} where either may be None on failure.
    """
    value: str | None = None
    page: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.upper().startswith("VALUE:"):
            value = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PAGE:"):
            raw_page = line.split(":", 1)[1].strip()
            # Tolerate "p. 137", "137", "-", etc.
            digits = "".join(ch for ch in raw_page if ch.isdigit())
            page = int(digits) if digits else None
    if value and value.upper() == "NOT FOUND":
        value = "NOT FOUND"
        page = None
    # Enforce: a value without a page citation is treated as NOT FOUND.
    # This kills hallucinations where Claude infers a number without source.
    elif value and page is None:
        value = "NOT FOUND"
    return {"value": value, "page": page}


def extract_metric(metric: dict, doc_name: str) -> dict:
    """Extract one canonical metric from one indexed doc.

    Uses the existing hybrid retriever scoped to `doc_name`, then asks Claude
    for a strict VALUE / PAGE response. Returns:
        {
          "key": ...,
          "label": ...,
          "value": "$24.5 billion" | "NOT FOUND" | None,
          "page":  137 | None,
          "raw":   <Claude's raw response, for debugging>,
        }
    """
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    chunks = retrieve(metric["question"], n_results=10, doc_name=doc_name, n_tables=5)
    excerpts = _format_excerpts(chunks) if chunks else ""

    prompt = f"""You are extracting a specific financial metric from a fund filing.

Metric: {metric['label']}
Question: {metric['question']}

Guidance:
{metric['guidance']}

Read the excerpts below and extract the metric. The excerpts may include both
prose and structured tables (marked with [TABLE from page N] headers). Prefer
table data when extracting numeric values.

Strict rules:
- The VALUE must be supported by an explicit number that appears in one of the
  excerpts. Do NOT guess, infer, average, or compute (unless the guidance
  explicitly permits computation).
- The PAGE must be a real page number from one of the excerpt headers shown
  below. If you cannot identify a single page that supports the value, return
  NOT FOUND.
- If both VALUE and PAGE cannot be supplied with confidence, return NOT FOUND
  for VALUE and `-` for PAGE.

Excerpts:

{excerpts if excerpts else '(no excerpts retrieved)'}

Respond in EXACTLY this format and nothing else:
VALUE: <the value with units, or NOT FOUND>
PAGE: <the page number from an excerpt header where the value appears, or - if NOT FOUND>"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    parsed = _parse_response(raw)

    return {
        "key": metric["key"],
        "label": metric["label"],
        "value": parsed["value"],
        "page": parsed["page"],
        "raw": raw,
    }


def extract_all_metrics(doc_name: str, progress_cb=None) -> dict:
    """Run every canonical metric against one doc; return a results dict.

    `progress_cb(i, total, label)` is optional; the Streamlit ingest flow
    passes a callback that updates a status line.
    """
    results: dict[str, dict] = {}
    total = len(CANONICAL_METRICS)
    for i, metric in enumerate(CANONICAL_METRICS, start=1):
        if progress_cb:
            progress_cb(i, total, metric["label"])
        results[metric["key"]] = extract_metric(metric, doc_name)
    return {
        "doc_name": doc_name,
        "extracted_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": results,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _metrics_path(doc_name: str) -> Path:
    """Filename for cached metrics JSON. Mirrors doc_name to keep things obvious."""
    safe = doc_name.replace("/", "_").replace("\\", "_")
    return METRICS_DIR / f"metrics_{safe}.json"


def save_metrics(payload: dict) -> None:
    """Persist a metrics payload to disk."""
    METRICS_DIR.mkdir(exist_ok=True)
    path = _metrics_path(payload["doc_name"])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metrics(doc_name: str) -> dict | None:
    """Read cached metrics for a doc, or None if not present."""
    path = _metrics_path(doc_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_exist(doc_name: str) -> bool:
    """Quick existence check, used to skip re-extraction."""
    return _metrics_path(doc_name).exists()
