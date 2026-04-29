"""Chroma vector store — persists to .chroma/ in the project root."""

from pathlib import Path
import chromadb

from backend.chunk import Chunk

PERSIST_DIR = Path(".chroma")
COLLECTION_NAME = "lp_reports"
_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    """Return a singleton persistent Chroma client."""
    global _client
    if _client is None:
        PERSIST_DIR.mkdir(exist_ok=True)
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def get_collection():
    """Return (or create) the single collection we use for all docs."""
    return get_client().get_or_create_collection(name=COLLECTION_NAME)


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Insert chunks with their embeddings. IDs are namespaced by doc name."""
    collection = get_collection()
    ids = [f"{c.doc_name}__{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {"doc_name": c.doc_name, "page": c.page, "chunk_index": c.chunk_index}
        for c in chunks
    ]
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query(
    query_embedding: list[float],
    n_results: int = 5,
    doc_name: str | None = None,
    content_contains: str | None = None,
) -> dict:
    """Return top-N most similar chunks.

    Optionally filter by doc_name (metadata) and/or content_contains (restrict
    to chunks whose document text contains the given substring). The substring
    filter is how we do table-only retrieval — pass content_contains="[TABLE".
    """
    where = {"doc_name": doc_name} if doc_name else None
    where_document = {"$contains": content_contains} if content_contains else None
    return get_collection().query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        where_document=where_document,
    )


def list_docs() -> list[str]:
    """Return sorted list of doc_names currently in the collection."""
    result = get_collection().get()
    if not result or not result.get("metadatas"):
        return []
    doc_names = {m["doc_name"] for m in result["metadatas"] if m and "doc_name" in m}
    return sorted(doc_names)


def doc_exists(doc_name: str) -> bool:
    """Quick check whether a doc has already been ingested."""
    return doc_name in list_docs()
