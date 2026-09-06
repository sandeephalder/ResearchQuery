"""The agent graph, behind HTTP and Clerk.

    uv run uvicorn backend.main:app --reload --port 8000

    POST /ask     a question in, a cited answer out          (Clerk required)
    GET  /graph   the flow as mermaid, from the live graph   (Clerk required)
    GET  /me      who Clerk says you are                     (Clerk required)
    GET  /health  readiness, and what is loaded              (open)

`/health` and the OpenAPI schema are the only routes without `require_user`.
That is a short list on purpose — adding a route means deciding which side of
it the new route belongs on.

Two things about this server are unusual, and both come from the stack under it:

**It owns the corpus.** The Qdrant store is embedded and single-process, so
this process holds an exclusive lock on it. Nothing else can run — no notebook
kernel, no `db_populate.py`, no second worker. One uvicorn worker, not four.

**The local models are a single resource.** BGE-M3, the cross-encoder and that
store are shared, stateful and not something to fan out over, so a semaphore
admits `LOCAL_CONCURRENCY` questions to that part of the flow. The model calls
that follow are ordinary async I/O and overlap freely — which is where nearly
all of the wall-clock goes anyway.
"""

import asyncio
import contextlib
import logging
import uuid

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from agents.constants import AGENT_PROVIDER
from tracing import setup as setup_tracing
from agents.graph import mermaid
from agents.identity import Identity
from agents.orchestrator import AgentPipeline
from llm import LLMError
from retrieval.store import RetrievalError

from .auth import Principal, check_configuration, require_user
from .constants import (CORS_ORIGINS, LOCAL_CONCURRENCY, MAX_IN_FLIGHT,
                        REQUEST_TIMEOUT_SECONDS)
from .schemas import AskRequest, AskResponse, Citation, HealthResponse

log = logging.getLogger("backend")


class Runtime:
    """The one AgentPipeline, and the limits around it."""

    def __init__(self):
        self.flow = None
        self.error = None
        self.local = asyncio.Semaphore(LOCAL_CONCURRENCY)
        self.slots = asyncio.Semaphore(MAX_IN_FLIGHT)
        self.in_flight = 0

    @property
    def ready(self):
        return self.flow is not None


runtime = Runtime()


@contextlib.asynccontextmanager
async def lifespan(_app):
    """Build everything once, before the first request.

    Constructing the pipeline loads BGE-M3, opens Qdrant and compiles the NeMo
    rails — tens of seconds, and a lock. Doing it per request would be absurd;
    doing it lazily on the first request would make that request look broken.
    """
    check_configuration()          # a server that cannot authenticate should not start
    # Before the models load, so the load itself is inside the first trace if
    # anything goes wrong there. A no-op unless MLFLOW_TRACING is set.
    if setup_tracing():
        log.info("MLflow tracing enabled")
    log.info("loading models and opening the corpus ...")
    try:
        runtime.flow = AgentPipeline()
        # Touching `.pipeline` is what actually opens Qdrant and loads BGE-M3.
        # Better here, loudly, than inside the first question.
        _ = runtime.flow.pipeline
        log.info("ready: agents on %s, reranker %s",
                 AGENT_PROVIDER, runtime.flow.pipeline.reranker)
    except (RetrievalError, LLMError, OSError) as error:
        # Recorded rather than raised: a server that reports "not ready" and the
        # reason is more useful than one that exits before anything can ask.
        # The half-built pipeline goes with it — `AgentPipeline()` returns
        # before the store is opened, so a flow that exists is not a flow that
        # works, and leaving it in place is what makes /health answer "ok" for
        # a server that cannot retrieve anything.
        if runtime.flow is not None:
            with contextlib.suppress(Exception):
                runtime.flow.close()
            runtime.flow = None
        runtime.error = str(error)
        log.error("startup failed: %s", error)
    try:
        yield
    finally:
        if runtime.flow is not None:
            runtime.flow.close()


app = FastAPI(title="ResearchQuery", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# --------------------------------------------------------------------------- #
# Open
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness and what is loaded. No auth — a load balancer has no token."""
    if not runtime.ready:
        return HealthResponse(status="unavailable", detail=runtime.error)

    flow = runtime.flow
    corpus = rows = None
    with contextlib.suppress(Exception):
        corpus = flow.pipeline.client.get_collection(flow.pipeline.collection).points_count
    with contextlib.suppress(Exception):
        # `.docs`, not `.ids` — the BM25 leg is a LangChain `BM25Retriever` now,
        # and it holds Documents. The old attribute did not raise loudly: this
        # whole block is suppressed, so it simply reported `bm25_rows: null`.
        rows = len(flow.pipeline.index.docs) if flow.pipeline.index else None

    return HealthResponse(
        status="ok", corpus_points=corpus, bm25_rows=rows,
        reranker=flow.pipeline.reranker, agent_provider=flow.provider,
        guardrail_flows=(list(flow.guardrail.rails.config.rails.input.flows)
                         if flow.guardrail else []),
        in_flight=runtime.in_flight)


# --------------------------------------------------------------------------- #
# Clerk required
# --------------------------------------------------------------------------- #

@app.get("/me")
async def me(user: Principal = Depends(require_user)):
    """Who Clerk says you are. The cheapest way to prove a token works."""
    return {"user_id": user.user_id, "session_id": user.session_id,
            "org_id": user.org_id}


@app.get("/graph")
async def graph(_user: Principal = Depends(require_user)):
    """The flow as mermaid, generated from the compiled graph."""
    return {"mermaid": mermaid()}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, user: Principal = Depends(require_user)):
    """One question through the whole chain.

    A refusal — by the guardrail, or by the scope floor after retrieval — comes
    back as 200 with `refused` set. It is what the system decided, not a fault
    in the request, and a client should render it as an answer.
    """
    if not runtime.ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            runtime.error or "still starting")

    if runtime.slots.locked():
        # Refusing beats queueing: an arrival that waits behind the local lock
        # spends its whole timeout not being served, then fails anyway.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "too many questions in flight; retry shortly")

    async with runtime.slots:
        runtime.in_flight += 1
        try:
            result = await asyncio.wait_for(
                _run(request, user), timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            log.warning("timed out for %s: %r", user.user_id, request.question[:80])
            raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT,
                                "the question took too long") from None
        except (RetrievalError, LLMError) as error:
            log.exception("failed for %s", user.user_id)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
        finally:
            runtime.in_flight -= 1

    return result


async def _run(request: AskRequest, user: Principal) -> AskResponse:
    # The semaphore covers only the part that shares the local models. Held
    # across the whole call it would serialise the API waiting too, which is
    # most of the elapsed time and none of the contention.
    # Ids only. The session token that proved them stays in the request and
    # goes no further — an agent that logs its caller should not be able to
    # log a credential.
    identity = Identity(user_id=user.user_id, session_id=user.session_id,
                        request_id=uuid.uuid4().hex)

    async with runtime.local:
        result = await runtime.flow.arun(
            request.question,
            k_vector=request.k_vector, k_bm25=request.k_bm25,
            top_n=request.top_n, rerank=request.rerank,
            max_repairs=request.max_repairs,
            # Reaches every agent, and the NeMo rail action through a
            # ContextVar. `request_id` also keys the checkpointer, so two
            # questions from one session cannot resume into each other.
            identity=identity)

    trace = result["trace"]
    citations = [
        Citation(n=number, label=passage.get("label", ""),
                 doc_id=passage.get("doc_id"), kind=passage.get("kind"),
                 rerank_score=passage.get("rerank_score"),
                 text=passage.get("text") if request.include_text else None)
        for number, passage in enumerate(result.get("passages") or [], start=1)
    ]
    return AskResponse(
        question=result["question"], answer=result["answer"],
        refused=trace.get("refused"), citations=citations,
        # Already computed and already logged; returning it costs nothing and
        # lets a caller show or threshold on it.
        metrics={k: v for k, v in (result.get("metrics") or {}).items()
                 if k not in ("question", "answer")},
        # The draft is dropped: it is the pre-repair answer, and returning both
        # invites a client to show the one the verifier rejected.
        trace={key: trace.get(key) for key in
               ("route", "route_effect", "best_score", "facts", "repairs",
                "unresolved", "guardrail_in", "guardrail_out", "timings")})
