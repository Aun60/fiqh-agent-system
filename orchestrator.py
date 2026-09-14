"""
orchestrator.py — runs the full agent pipeline for one user query:

    router → retriever → synthesizer → verifier → (retry once if failed) → response

If verification fails twice, the system returns a safe fallback message
rather than showing a potentially mixed-up or unsupported answer.
"""

from dataclasses import dataclass, field

from agents.router_agent import route_query
from agents.retriever_agent import retrieve
from agents.synthesizer_agent import synthesize_answer
from agents.verifier_agent import verify_answer

MAX_SYNTHESIS_ATTEMPTS = 3


@dataclass
class PipelineResult:
    answer: str
    scope: str
    madhab: str | None
    verified: bool
    issues: list[str] = field(default_factory=list)


OFF_TOPIC_RESPONSE = (
    "This system is scoped to Islamic fiqh (jurisprudence) questions across "
    "the Hanafi, Shafi'i, Maliki, and Hanbali schools, based on the classical "
    "texts in its knowledge base. Your question doesn't appear to be a fiqh "
    "question — feel free to rephrase if it is."
)

UNVERIFIED_FALLBACK = (
    "I wasn't able to produce an answer I'm confident is accurately "
    "attributed to the correct madhab(s) based on the sources available. "
    "Rather than risk giving you a misattributed ruling, I'd recommend "
    "consulting a qualified scholar directly for this question."
)


def handle_query(user_query: str) -> PipelineResult:
    route = route_query(user_query)

    if route.scope == "off_topic":
        return PipelineResult(
            answer=OFF_TOPIC_RESPONSE, scope="off_topic", madhab=None, verified=True
        )

    chunks_by_madhab = retrieve(user_query, route)

    # If truly nothing was retrieved anywhere, don't let the model improvise.
    if not any(chunks_by_madhab.values()):
        return PipelineResult(
            answer=(
                "No relevant passages were found in the current corpus for this "
                "question. The knowledge base may not yet cover this topic."
            ),
            scope=route.scope,
            madhab=route.madhab,
            verified=True,
        )

    answer = ""
    verified = False
    issues: list[str] = []
    prior_issues: list[str] | None = None

    for attempt in range(1, MAX_SYNTHESIS_ATTEMPTS + 1):
        answer = synthesize_answer(user_query, route, chunks_by_madhab, prior_issues=prior_issues)
        result = verify_answer(answer, route, chunks_by_madhab)
        print(f"\n[DEBUG] Attempt {attempt} — verifier passed={result.passed}")
        if not result.passed:
            print(f"[DEBUG] Verifier issues: {result.issues}")
            print(f"[DEBUG] Draft answer was:\n{answer}\n")
        if result.passed:
            verified = True
            issues = []
            break
        issues = result.issues
        prior_issues = result.issues

    if not verified:
        return PipelineResult(
            answer=UNVERIFIED_FALLBACK,
            scope=route.scope,
            madhab=route.madhab,
            verified=False,
            issues=issues,
        )

    return PipelineResult(
        answer=answer,
        scope=route.scope,
        madhab=route.madhab,
        verified=True,
        issues=[],
    )
