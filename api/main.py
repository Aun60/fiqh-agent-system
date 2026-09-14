"""
api/main.py — FastAPI backend for the fiqh multi-agent RAG system.

Run with:
    uvicorn api.main:app --reload
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from orchestrator import handle_query

load_dotenv()

app = FastAPI(title="Fiqh Multi-Agent RAG API")

# ALLOWED_ORIGINS is a comma-separated list of allowed frontend origins,
# set as an environment variable (in .env locally, or in Render's
# Environment tab for deployment). No trailing slashes, must include the
# scheme, e.g.:
#   ALLOWED_ORIGINS=https://fiqh-agent-system.vercel.app,http://localhost:5500
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5500")
ALLOWED_ORIGINS = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]

print(f"[CORS] Allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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