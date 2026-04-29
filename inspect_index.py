"""Dump what's actually in the Chroma index for page 137.

Usage:
    # Make sure Streamlit is stopped first (releases the DB lock)
    py inspect_index.py
"""
import chromadb
from pathlib import Path

CHROMA_DIR = "./.chroma"
PAGE_TO_INSPECT = 137

client = chromadb.PersistentClient(path=CHROMA_DIR)

for col in client.list_collections():
    c = client.get_collection(col.name)
    all_data = c.get()
    total = len(all_data["ids"])
    print(f"Collection: {col.name}  (total chunks: {total})")

    table_chunks = [
        (i, d) for i, d in enumerate(all_data["documents"])
        if d.startswith("[TABLE")
    ]
    row_label_chunks = [
        i for i, d in enumerate(all_data["documents"])
        if "Row labels:" in d
    ]
    print(f"  [TABLE chunks:  {len(table_chunks)}")
    print(f"  'Row labels:' chunks: {len(row_label_chunks)}")

    # Show all chunks on the target page
    print(f"\n  Chunks on page {PAGE_TO_INSPECT}:")
    for i, meta in enumerate(all_data["metadatas"]):
        if meta.get("page") == PAGE_TO_INSPECT:
            doc = all_data["documents"][i]
            print(f"\n  --- chunk #{i} (doc_name={meta['doc_name']}) ---")
            print("  " + doc[:600].replace("\n", "\n  "))
    print()
