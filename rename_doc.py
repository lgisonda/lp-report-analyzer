"""One-shot rename of a doc inside the Chroma index, without re-ingesting.

Updates three things:
1. All chunk metadata (doc_name field) for the doc
2. The chunk IDs (which are namespaced as f"{doc_name}__{chunk_index}")
3. The cached metrics JSON file (filename + internal doc_name)

Usage:
    Edit OLD_NAME / NEW_NAME below and run:
        py rename_doc.py
"""

import json
from pathlib import Path

import chromadb

# --- edit these two ---------------------------------------------------------
OLD_NAME = "SREIT-10K-2025_1231_Q425_FINAL-3.20.2026.pdf"
NEW_NAME = "SREIT 12.31.2025 10K.pdf"
# ----------------------------------------------------------------------------

CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "lp_reports"


def main() -> None:
    if OLD_NAME == NEW_NAME:
        print("OLD_NAME and NEW_NAME are the same — nothing to do.")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)

    # Pull every chunk for the old doc, including embeddings
    res = col.get(
        where={"doc_name": OLD_NAME},
        include=["metadatas", "documents", "embeddings"],
    )
    n = len(res["ids"])
    if n == 0:
        print(f"No chunks found with doc_name={OLD_NAME!r}. Aborting.")
        return

    print(f"Found {n} chunks for {OLD_NAME!r}")

    # Build the new IDs / metadatas
    new_ids: list[str] = []
    new_metas: list[dict] = []
    for old_id, meta in zip(res["ids"], res["metadatas"]):
        new_id = f"{NEW_NAME}__{meta['chunk_index']}"
        new_ids.append(new_id)
        new_meta = dict(meta)
        new_meta["doc_name"] = NEW_NAME
        new_metas.append(new_meta)

    # Delete the old, add the new (Chroma can't rename IDs in place)
    col.delete(ids=res["ids"])
    col.add(
        ids=new_ids,
        documents=res["documents"],
        embeddings=res["embeddings"],
        metadatas=new_metas,
    )
    print(f"Renamed {n} chunks: {OLD_NAME!r} -> {NEW_NAME!r}")

    # Rename the metrics JSON file and update its internal doc_name
    safe_old = OLD_NAME.replace("/", "_").replace("\\", "_")
    safe_new = NEW_NAME.replace("/", "_").replace("\\", "_")
    old_metrics = CHROMA_DIR / f"metrics_{safe_old}.json"
    new_metrics = CHROMA_DIR / f"metrics_{safe_new}.json"

    if old_metrics.exists():
        payload = json.loads(old_metrics.read_text(encoding="utf-8"))
        payload["doc_name"] = NEW_NAME
        new_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        old_metrics.unlink()
        print(f"Renamed metrics JSON: {old_metrics.name} -> {new_metrics.name}")
    else:
        print(f"WARNING: No metrics JSON found at {old_metrics}")

    print("Done. Restart Streamlit (or wait for auto-reload) to see the change.")


if __name__ == "__main__":
    main()
