"""Chunking — split per-page content into retrieval-sized chunks.

Text and tables are chunked differently:
- Text: split with the recursive character splitter (keeps chunks ~1000 chars).
- Tables: each table becomes ONE chunk, rendered as a clean markdown table so
  Claude can read numeric data without wrestling with flattened whitespace.
"""

import re
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    page: int
    doc_name: str
    chunk_index: int  # 0-indexed, within the document


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a pdfplumber-extracted table as a markdown pipe table."""
    if not table:
        return ""
    # Normalize: replace None, strip whitespace, collapse internal newlines
    cleaned = [
        [(cell or "").strip().replace("\n", " ") for cell in row]
        for row in table
    ]
    # Drop rows that are fully empty
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""

    header = cleaned[0]
    body = cleaned[1:]
    width = len(header)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        # Pad/truncate to header width so every row has the same number of cells
        padded = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


def _extract_row_labels(table: list[list[str | None]], max_labels: int = 20) -> list[str]:
    """Pull the first-column text labels out of a pdfplumber table.

    Row labels (e.g. sector names like 'Industrial', 'Rental Housing') carry
    most of the semantic signal in a financial table. Surfacing them in the
    chunk text lets embedding search match queries like 'how many industrial
    properties' against the right table.
    """
    if not table:
        return []
    labels: list[str] = []
    # Scan every row — pdfplumber's "header" row in financial filings is often
    # actually a data row (multi-row headers get flattened oddly). Including
    # row 0 sometimes picks up the real first label; at worst it picks up a
    # real header like "Property Sector", which is still a useful keyword.
    for row in table:
        if not row:
            continue
        # First non-empty cell in the row is usually the row label
        for cell in row:
            cell = (cell or "").strip().replace("\n", " ")
            if cell:
                # Filter out pure numbers / artifacts
                if len(cell) > 1 and not cell.replace(",", "").replace(".", "").replace("$", "").replace("%", "").strip().isdigit():
                    labels.append(cell)
                break
        if len(labels) >= max_labels:
            break
    return labels


def _find_table_caption(page_text: str) -> str:
    """Best-effort natural-language caption for tables on a page.

    Tables embed as bags of numbers; questions are natural language. Prepending
    a caption like "The following table provides a summary of our portfolio by
    property sector..." pulls the table's embedding toward the kinds of
    questions users actually ask.
    """
    if not page_text:
        return ""
    # SEC filings almost always introduce tables with this phrase.
    m = re.search(
        r"(?i)(the following table[^.\n]{0,240}[.\n])",
        page_text,
    )
    if m:
        return m.group(1).strip()
    # Looser: "summary of ..."
    m = re.search(r"(?i)(summary of [^.\n]{0,240}[.\n])", page_text)
    if m:
        return m.group(1).strip()
    # Fallback: first substantial prose line on the page.
    for line in page_text.split("\n"):
        line = line.strip()
        if len(line) > 40 and not line.lower().startswith(("page ", "http", "©")):
            return line[:240]
    return ""


def chunk_pages(
    pages: list[tuple[int, str, list]],
    doc_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_table_chars: int = 60,
) -> list[Chunk]:
    """Turn (page_num, text, tables) tuples into a flat list of chunks.

    Tables get emitted as single chunks with a [TABLE ...] header so the model
    knows it's reading structured data, not prose.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Chunk] = []
    global_idx = 0

    for page_num, page_text, page_tables in pages:
        # 1) Tables first — each gets its own chunk
        caption = _find_table_caption(page_text)
        for table in page_tables:
            md = _table_to_markdown(table)
            if len(md) < min_table_chars:
                continue  # skip tiny / empty tables (often artifacts)
            header = f"[TABLE from page {page_num}]"
            labels = _extract_row_labels(table)
            label_line = f"Row labels: {', '.join(labels)}" if labels else ""

            parts = [header]
            if caption:
                parts.append(caption)
            if label_line:
                parts.append(label_line)
            parts.append("")  # blank line before markdown table
            parts.append(md)

            chunks.append(
                Chunk(
                    text="\n".join(parts),
                    page=page_num,
                    doc_name=doc_name,
                    chunk_index=global_idx,
                )
            )
            global_idx += 1

        # 2) Then prose
        if not page_text.strip():
            continue
        for piece in splitter.split_text(page_text):
            chunks.append(
                Chunk(
                    text=piece,
                    page=page_num,
                    doc_name=doc_name,
                    chunk_index=global_idx,
                )
            )
            global_idx += 1

    return chunks
