"""
synthesizer_agent.py — writes the final answer from retrieved chunks.

Non-negotiable output rules, enforced via the prompt AND via post-generation
validation (validator.py checks the output actually complies):
  1. Every ruling stated must be attributed: "[Madhab], per [Scholar]'s
     [Text]: ..."
  2. In comparative mode, each madhab's position is presented as its OWN
     clearly headed section — never blended into one paragraph.
  3. If a madhab's retrieved chunks don't actually answer the question,
     the agent says so for that madhab rather than inventing a position.
  4. Never state a ruling not grounded in the retrieved chunks.
"""

from .router_agent import RouteDecision
from .retriever_agent import RetrievedChunk
from .llm_client import call_llm

SYNTHESIZER_SYSTEM_PROMPT = """You are a fiqh answer synthesizer for an Islamic jurisprudence Q&A system.

You will be given retrieved source passages, each labeled with its madhab, scholar, and source text. Write an answer using ONLY the information in these passages. Follow these rules strictly:

1. ATTRIBUTION: Every ruling or position you state MUST be attributed in this form:
   "According to the [Madhab] school, per [Scholar]'s [Source Text]: <the ruling>"

2. NEVER blend madhabs together. If passages from multiple madhabs are provided, give each madhab its own clearly labeled section (use a heading with the madhab name). Do not summarize them into one merged position.

3. GROUNDING: Only state what the passages actually support. If the passages for a given madhab don't clearly address the question, say plainly: "The retrieved [Madhab] sources don't directly address this — a scholar from that school should be consulted." Do NOT fill the gap from general knowledge.

4. If passages for a madhab are entirely missing, say so for that madhab rather than staying silent about it (silence could be misread as "no ruling exists").

5. Keep a neutral, respectful tone throughout — you are reporting what classical scholars wrote, not adjudicating between schools or stating which is "correct."

6. End with a short reminder that this is drawn from classical texts for informational/educational purposes and is not a substitute for a qualified local scholar (mufti), especially for personal or complex rulings.
"""


def _format_chunks_for_prompt(chunks_by_madhab: dict[str, list[RetrievedChunk]]) -> str:
    blocks = []
    for madhab, chunks in chunks_by_madhab.items():
        if not chunks:
            blocks.append(f"### {madhab}\n(No relevant passages retrieved.)")
            continue
        block_lines = [f"### {madhab}"]
        for c in chunks:
            block_lines.append(
                f"- Source: {c.source_text} by {c.scholar} (page {c.page_number})\n"
                f"  Passage: {c.text}"
            )
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def synthesize_answer(
    user_query: str,
    route: RouteDecision,
    chunks_by_madhab: dict[str, list[RetrievedChunk]],
    prior_issues: list[str] | None = None,
) -> str:
    context_block = _format_chunks_for_prompt(chunks_by_madhab)

    mode_note = (
        f"The user asked specifically about the {route.madhab} school. "
        f"Answer ONLY for {route.madhab} — do not mention other madhabs' positions."
        if route.scope == "single"
        else "The user did not name a specific madhab. Present the position of "
             "each of the four madhabs separately, clearly labeled."
    )

    revision_note = ""
    if prior_issues:
        issues_str = "\n".join(f"- {i}" for i in prior_issues)
        revision_note = f"""

A previous draft of this answer had these problems — fix them this time:
{issues_str}
"""

    user_prompt = f"""User question: {user_query}

Mode: {mode_note}

Retrieved passages:
{context_block}
{revision_note}
Write the answer following all system rules exactly."""

    return call_llm(
        system=SYNTHESIZER_SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=1500,
    )
