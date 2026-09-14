"""
metadata_tagger.py — attaches madhab/scholar/text metadata to each chunk.

This is the safety-critical step: every chunk gets stamped with EXACTLY the
madhab of the PDF it came from, taken from source_registry.py. Nothing here
infers or guesses a madhab from content — it only trusts the registry.
This is deliberate: guessing from text is how mix-ups happen.
"""

from dataclasses import dataclass, asdict

from .chunker import Chunk
from .source_registry import get_source_by_filename


@dataclass
class TaggedChunk:
    id: str
    text: str
    madhab: str
    scholar: str
    source_text: str
    era: str
    filename: str
    page_number: int

    def to_chroma_metadata(self) -> dict:
        d = asdict(self)
        d.pop("id")
        d.pop("text")
        return d


def tag_chunks(chunks: list[Chunk]) -> list[TaggedChunk]:
    tagged = []
    for c in chunks:
        source = get_source_by_filename(c.filename)  # raises if unregistered — good, fail loud
        chunk_id = f"{c.filename}::p{c.page_number}::c{c.chunk_index}"
        tagged.append(
            TaggedChunk(
                id=chunk_id,
                text=c.text,
                madhab=source.madhab,
                scholar=source.scholar,
                source_text=source.text_title,
                era=source.era,
                filename=c.filename,
                page_number=c.page_number,
            )
        )
    return tagged
