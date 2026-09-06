"""What goes in and what comes out.

The response is deliberately not the graph's own dict. That carries the whole
payload of every candidate — the full text of twenty-odd rows, several times
over — which is a large response, a slow one, and more of the corpus than a
question needs to return. Citations are the part a caller can act on.
"""

from typing import Any

from pydantic import BaseModel, Field

from agents.constants import MAX_REPAIRS
from backend.constants import MAX_QUESTION_CHARS
from retrieval.constants import BM25_TOP_K, RERANK_TOP_N, VECTOR_TOP_K


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_n: int = Field(RERANK_TOP_N, ge=1, le=20)
    k_vector: int = Field(VECTOR_TOP_K, ge=0, le=50)
    k_bm25: int = Field(BM25_TOP_K, ge=0, le=50)
    max_repairs: int = Field(MAX_REPAIRS, ge=0, le=3)
    rerank: bool = True
    # Whether to include each cited passage's text. Off by default: a caller
    # rendering citations wants the labels, and one checking the answer against
    # the source can ask for the text explicitly.
    include_text: bool = False


class Citation(BaseModel):
    n: int
    label: str
    doc_id: str | None = None
    kind: str | None = None
    rerank_score: float | None = None
    text: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str | None
    # Set when the flow stopped early: "guardrail", "no candidates", or
    # "below scope floor (...)". A refusal is a real answer, not an error, so
    # it arrives as 200 with this filled in rather than as a 4xx.
    refused: str | None = None
    citations: list[Citation] = []
    trace: dict[str, Any] = {}
    # Context relevance, groundedness, answer relevance and latency, computed
    # from work the flow already did. See agents/metrics.py for what they mean.
    metrics: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    corpus_points: int | None = None
    bm25_rows: int | None = None
    reranker: str | None = None
    agent_provider: str | None = None
    guardrail_flows: list[str] = []
    in_flight: int = 0
    detail: str | None = None
