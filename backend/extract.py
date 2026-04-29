"""PDF extraction — yields per-page text AND per-page structured tables."""

from typing import Iterator
import pdfplumber


def extract_pages(pdf_file) -> Iterator[tuple[int, str, list]]:
    """Yield (page_number, text, tables) tuples for every page in a PDF.

    `tables` is a list of tables on the page; each table is a list of rows; each
    row is a list of cell strings. Pages with no tables yield an empty list.
    """
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            yield i, text, tables
