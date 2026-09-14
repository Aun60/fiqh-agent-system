"""
chunker.py — splits page text into overlapping chunks suitable for embedding.

Uses a paragraph-aware sliding window: splits on blank lines first (natural
paragraph/ruling boundaries in fiqh texts), then packs paragraphs into
chunks up to a target size, with overlap so a ruling split across a chunk
boundary isn't lost.
"""

from dataclasses import dataclass

from .loader import PageText

CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200


@dataclass
class Chunk:
    filename: str
    page_number: int
    chunk_index: int
    text: str


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_pages(pages: list[PageText]) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    buffer = ""
    buffer_start_page = None

    def flush():
        nonlocal buffer, buffer_start_page, chunk_index
        if buffer.strip():
            chunks.append(
                Chunk(
                    filename=pages[0].filename if pages else "",
                    page_number=buffer_start_page or 1,
                    chunk_index=chunk_index,
                    text=buffer.strip(),
                )
            )
            chunk_index += 1
        buffer = ""
        buffer_start_page = None

    for page in pages:
        if not page.text:
            continue
        for para in _split_paragraphs(page.text):
            if buffer_start_page is None:
                buffer_start_page = page.page_number
            if len(buffer) + len(para) > CHUNK_SIZE_CHARS and buffer:
                flush()
                # carry overlap forward
                buffer = buffer[-CHUNK_OVERLAP_CHARS:] if buffer else ""
                buffer_start_page = page.page_number
            buffer += ("\n\n" if buffer else "") + para
    flush()

    return chunks
