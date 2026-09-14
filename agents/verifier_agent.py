"""
verifier_agent.py — last line of defense before an answer reaches the user.

Checks two things, both critical for this project:
  1. MADHAB INTEGRITY: does the answer attribute every ruling to a madhab
     that actually appears in the retrieved chunks (no invented/mixed
     attributions, no leaking a madhab that wasn't retrieved in "single"
     mode)?
  2. GROUNDING: is each claim traceable to the retrieved passages, or does
     it look like the model filled a gap from unsourced background
     knowledge?

This is a cheap, deterministic pre-check (string/metadata based) PLUS an
LLM-based semantic check. If either fails, the answer is sent back for
revision rather than shown to the user.
"""

from dataclasses import dataclass

from .router_agent import RouteDecision, VALID_MADHABS
from .retriever_agent import RetrievedChunk
from .llm_client import call_llm

VERIFIER_SYSTEM_PROMPT = """You are a fact-checker reviewing a draft fiqh answer against its source passages.

Only fail the answer for SUBSTANTIVE problems:
1. A ruling attributed to a madhab whose passages don't support it at all.
2. A claim with no basis in the provided passages (looks invented from general knowledge, not the sources).
3. Two different madhabs' positions blended into one unlabeled statement (comparative mode only).

Do NOT fail the answer for minor formatting issues: slightly different attribution phrasing is fine as long as the madhab and source are named somewhere in the answer. Do not require the exact phrase "According to the X school, per Y's Z". A well-grounded answer that says "The Hanafi position, per al-Quduri's Mukhtasar, holds that..." is equally acceptable. Be lenient on phrasing, strict on factual grounding and attribution existing at all.

Respond ONLY with JSON: {"passed": true|false, "issues": ["issue 1", "issue 2", ...]}
If there are no substantive issues, return {"passed": true, "issues": []}.
"""


@dataclass
class VerificationResult:
    passed: bool
    issues: list[str]


def _deterministic_madhab_check(
    answer: str, route: RouteDecision, chunks_by_madhab: dict[str, list[RetrievedChunk]]
) -> list[str]:
    """Cheap check that doesn't need an LLM call: in single-madhab mode,
    make sure no OTHER madhab name appears in the answer (would indicate
    a leak/mixup)."""
    issues = []
    if route.scope == "single" and route.madhab:
        other_madhabs = [m for m in VALID_MADHABS if m != route.madhab]
        for m in other_madhabs:
            if m.lower() in answer.lower():
                issues.append(
                    f"Answer mentions '{m}' but query was scoped to {route.madhab} only — possible madhab leak."
                )
    return issues


def _format_chunks_for_check(chunks_by_madhab: dict[str, list[RetrievedChunk]]) -> str:
    lines = []
    for madhab, chunks in chunks_by_madhab.items():
        for c in chunks:
            lines.append(f"[{madhab} / {c.scholar} / {c.source_text}]: {c.text}")
    return "\n".join(lines) if lines else "(no passages retrieved)"


def verify_answer(
    answer: str,
    route: RouteDecision,
    chunks_by_madhab: dict[str, list[RetrievedChunk]],
) -> VerificationResult:
    issues = _deterministic_madhab_check(answer, route, chunks_by_madhab)

    import json

    check_prompt = f"""Draft answer:
{answer}

Source passages it should be grounded in:
{_format_chunks_for_check(chunks_by_madhab)}
"""
    raw = call_llm(system=VERIFIER_SYSTEM_PROMPT, user=check_prompt, max_tokens=400)
    try:
        parsed = json.loads(raw.strip())
        llm_issues = parsed.get("issues", [])
        if not parsed.get("passed", True):
            issues.extend(llm_issues)
    except json.JSONDecodeError:
        issues.append("Verifier LLM output was not valid JSON — treating as unverified.")

    return VerificationResult(passed=(len(issues) == 0), issues=issues)
