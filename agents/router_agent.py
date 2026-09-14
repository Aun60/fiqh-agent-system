"""
router_agent.py — the gatekeeper that decides MADHAB SCOPE for a query.

This is the single most important decision in the whole system. It decides:
  - "single": the user named one madhab (e.g. "in Hanafi fiqh...") →
              retrieval and the answer are scoped to ONLY that madhab.
  - "comparative": the user asked a general fiqh question with no madhab
              named (e.g. "can I combine prayers while traveling?") →
              retrieval pulls from ALL FOUR, and the answer must present
              each madhab's position separately, clearly labeled.
  - "off_topic": not a fiqh question at all → no retrieval, agent declines
              or redirects.

Deliberately conservative: if scope is ambiguous, default to "comparative"
rather than silently guessing a single madhab — showing all four positions
is always safe; picking the wrong one silently is not.
"""

import json
from dataclasses import dataclass

from .llm_client import call_llm

VALID_MADHABS = ["Hanafi", "Shafi'i", "Maliki", "Hanbali"]

ROUTER_SYSTEM_PROMPT = f"""You are a routing classifier for an Islamic fiqh Q&A system covering the four Sunni madhabs: {", ".join(VALID_MADHABS)}.

Given a user question, determine:
1. Is this a fiqh (Islamic jurisprudence / practical ruling) question? If not, scope = "off_topic".
2. Did the user explicitly name ONE specific madhab (e.g. "according to Hanafi", "in the Maliki school", "Shafi'i ruling on...")?
   - If yes: scope = "single", and set "madhab" to exactly one of {VALID_MADHABS} matching what they named.
   - If no madhab was named, or more than one was named, or they asked for "all schools"/"compare": scope = "comparative", "madhab" = null.

Be conservative: only use "single" when a madhab is unambiguously and explicitly named in the question. Never guess a madhab from context or assume a default.

Respond ONLY with JSON, no other text, in this exact shape:
{{"scope": "single" | "comparative" | "off_topic", "madhab": "<one of the four exactly, or null>", "topic_keywords": ["short", "keywords", "for", "retrieval"]}}
"""


@dataclass
class RouteDecision:
    scope: str          # "single" | "comparative" | "off_topic"
    madhab: str | None  # set only when scope == "single"
    topic_keywords: list[str]


def route_query(user_query: str) -> RouteDecision:
    raw = call_llm(
        system=ROUTER_SYSTEM_PROMPT,
        user=user_query,
        max_tokens=300,
    )

    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Fail safe: if the router output is malformed, default to comparative
        # rather than crash or silently pick a madhab.
        return RouteDecision(scope="comparative", madhab=None, topic_keywords=[])

    scope = parsed.get("scope", "comparative")
    madhab = parsed.get("madhab")

    if scope == "single":
        if madhab not in VALID_MADHABS:
            # Guardrail: if the model names something outside the valid set,
            # or scope=single but madhab is missing/invalid, fall back to
            # comparative rather than trust a bad value.
            scope = "comparative"
            madhab = None
    else:
        madhab = None

    return RouteDecision(
        scope=scope,
        madhab=madhab,
        topic_keywords=parsed.get("topic_keywords", []),
    )
