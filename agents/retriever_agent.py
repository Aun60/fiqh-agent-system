"""
retriever_agent.py — queries ChromaDB, with HARD metadata filtering by
madhab when scope == "single".

Important: madhab filtering happens at the database query level (Chroma's
`where` clause), not by asking the LLM to "please only use Hanafi sources."
Filtering at retrieval time means chunks from other madhabs are never even
seen by the synthesizer — there's nothing for it to mix up.
"""

from dataclasses import dataclass

from ingestion.embed_and_store import get_chroma_collection
from .router_agent import RouteDecision, VALID_MADHABS

TOP_K_SINGLE = 10          # chunks to retrieve when scoped to one madhab
TOP_K_PER_MADHAB_COMPARATIVE = 5  # chunks per madhab when comparative


@dataclass
class RetrievedChunk:
    text: str
    madhab: str
    scholar: str
    source_text: str
    page_number: int
    filename: str


def _rows_to_chunks(result: dict) -> list[RetrievedChunk]:
    chunks = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        chunks.append(
            RetrievedChunk(
                text=doc,
                madhab=meta["madhab"],
                scholar=meta["scholar"],
                source_text=meta["source_text"],
                page_number=meta["page_number"],
                filename=meta["filename"],
            )
        )
    return chunks


def retrieve(query: str, route: RouteDecision) -> dict[str, list[RetrievedChunk]]:
    """
    Returns {madhab: [chunks]} — even for "single" scope, keyed dict keeps
    a consistent shape for downstream code and makes the isolation explicit
    at every call site (you can always see exactly which madhab(s) chunks
    came from).
    """
    collection = get_chroma_collection()

    if route.scope == "single" and route.madhab:
        result = collection.query(
            query_texts=[query],
            n_results=TOP_K_SINGLE,
            where={"madhab": route.madhab},  # HARD FILTER — other madhabs excluded at the DB level
        )
        chunks = _rows_to_chunks(result)
        # Defensive re-check: never trust filtering blindly downstream.
        chunks = [c for c in chunks if c.madhab == route.madhab]
        print(f"[DEBUG] Retrieved {len(chunks)} chunks for madhab={route.madhab}, query={query!r}")
        for c in chunks[:3]:
            print(f"  - [{c.source_text} p{c.page_number}]: {c.text[:120]}...")
        return {route.madhab: chunks}

    # comparative (or off_topic won't reach here — caller should short-circuit)
    out: dict[str, list[RetrievedChunk]] = {}
    for madhab in VALID_MADHABS:
        result = collection.query(
            query_texts=[query],
            n_results=TOP_K_PER_MADHAB_COMPARATIVE,
            where={"madhab": madhab},
        )
        chunks = [c for c in _rows_to_chunks(result) if c.madhab == madhab]
        out[madhab] = chunks
    return out
