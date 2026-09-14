"""
embed_and_store.py — embeds tagged chunks and stores them in ChromaDB.

Run this file directly to ingest everything in data/raw_pdfs/ into the
persistent Chroma collection at chroma_db/.

Usage:
    python -m ingestion.embed_and_store
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .loader import load_all_pdfs
from .chunker import chunk_pages
from .metadata_tagger import tag_chunks
from .source_registry import validate_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "fiqh_corpus"

# Local, free embedding model — no API key needed. Swap for an
# OpenAI/Anthropic-hosted embedding model later if you want higher quality.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest_all() -> None:
    validate_registry()

    all_pages = load_all_pdfs(RAW_PDF_DIR)
    if not all_pages:
        print("Nothing to ingest. Add PDFs to data/raw_pdfs/ first.")
        return

    collection = get_chroma_collection()

    total_chunks = 0
    for filename, pages in all_pages.items():
        chunks = chunk_pages(pages)
        tagged = tag_chunks(chunks)  # will raise if filename isn't in the registry

        if not tagged:
            print(f"[WARNING] No chunks produced for {filename} — skipping.")
            continue

        collection.upsert(
            ids=[t.id for t in tagged],
            documents=[t.text for t in tagged],
            metadatas=[t.to_chroma_metadata() for t in tagged],
        )
        total_chunks += len(tagged)
        madhab = tagged[0].madhab
        print(f"Ingested {filename}: {len(tagged)} chunks tagged as madhab={madhab}")

    print(f"\nDone. {total_chunks} total chunks in collection '{COLLECTION_NAME}'.")
    print("Per-madhab counts:")
    for m in ("Hanafi", "Shafi'i", "Maliki", "Hanbali"):
        count = collection.get(where={"madhab": m}, include=[])
        print(f"  {m}: {len(count['ids'])}")


if __name__ == "__main__":
    ingest_all()
