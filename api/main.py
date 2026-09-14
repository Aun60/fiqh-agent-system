"""
api/main.py — FastAPI backend for the fiqh multi-agent RAG system.

Run with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orchestrator import handle_query

app = FastAPI(title="Fiqh Multi-Agent RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    scope: str
    madhab: str | None
    verified: bool
    issues: list[str]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = handle_query(req.query)
    return QueryResponse(
        answer=result.answer,
        scope=result.scope,
        madhab=result.madhab,
        verified=result.verified,
        issues=result.issues,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
