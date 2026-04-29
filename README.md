# LP Report Analyzer

A Streamlit app that ingests fund quarterly report PDFs, does RAG over them, and lets you chat with the documents via Claude. Every answer cites the specific page it came from.

Work in progress. See the project plan in the knowledge wiki.

## Stack

Python · Streamlit · pdfplumber · Chroma · sentence-transformers · Anthropic Claude API

## Running locally

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Seed data

Two public SEC EDGAR filings, expected in the `data/` folder:

- `breit_q4_2024.pdf` — Blackstone Real Estate Income Trust 10-Q (Q4 2024)
- `arcc_q3_2024.pdf` — Ares Capital Corporation 10-Q (Q3 2024)

Both are text-based (not scanned) so extraction is clean.
