"""The flow, end to end.

    uv run python -m agents.orchestrator "how does detection probability vary with range"
    uv run python -m agents.orchestrator "..." --route-mode filter --no-verify
    uv run python -m agents.orchestrator "..." --guardrail llama-guard --json

The flow itself is a LangGraph graph — see `graph.py`, which holds the nodes,
the three early exits and the repair loop, and which can draw itself. What is
here is the object those nodes call: it owns the agents, the retrieval pipeline
and the two rails, and `arun` seeds the state and invokes the graph.

`arun` returns the keys `retrieval.Pipeline.arun` returns — question,
candidates, passages, reranked, answer, vector_mode, reranker — plus a `trace`.
That is not decoration: `evals/answer_eval.py` builds a `Pipeline` and calls
`retrieve`, `rerank` and `answer` on it, so keeping the shape is what lets the
same harness score this flow against the plain pipeline. Six calls per question
against two is a claim that needs measuring, and that is the measurement.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
import uuid

from llm import LLMError
from tracing import SpanType, annotate, mlflow, setup as setup_tracing
from retrieval.constants import BM25_TOP_K, RERANK_TOP_N, RERANKER, RERANKERS, VECTOR_TOP_K
from retrieval.paths import RetrievalError
from retrieval.pipeline import Pipeline, _report

from .constants import (AGENT_PROVIDER, MAX_REPAIRS, METRICS_ENABLED, ROUTE_MODE,
                        ROUTE_MODES, ROUTE_WIDEN_K, SCOPE_FLOOR)
from .extractor import Extractor
from .generator import Generator
from .graph import build
from .identity import Identity, bind
from .metrics import evaluate, record
from .guardrail import Guardrail, GuardrailError
from .router import Router
from .state import Turn
from .verifier import NullVerifier, Verifier


class AgentPipeline:
    """The five agents around one `retrieval.Pipeline`. Open it once, ask many times.

        with AgentPipeline() as flow:
            result = flow.ask("how does detection probability vary with range")

    Construction loads BGE-M3 (~15 s), opens the embedded Qdrant store, and
    builds the NeMo rails. All three are per-process costs, and the Qdrant store
    admits a single process — so a short-lived instance per question is both
    slow and a lock other processes will trip over.
    """

    def __init__(self, provider=AGENT_PROVIDER, reranker=RERANKER, route_mode=ROUTE_MODE,
                 guardrail=True, verify=True, use_bm25=True, **pipeline_kwargs):
        if route_mode not in ROUTE_MODES:
            raise ValueError(f"route_mode must be one of {ROUTE_MODES}, not {route_mode!r}")
        self.provider = provider
        self.route_mode = route_mode
        # Built on first use, not here. A question the input rail blocks should
        # cost nothing, and building this costs a BGE-M3 load and an exclusive
        # lock on the embedded Qdrant store — which a blocked question would
        # then hold, and tear down, having never asked it anything.
        self._pipeline = None
        self._pipeline_kwargs = dict(provider=provider, reranker=reranker,
                                     use_bm25=use_bm25, **pipeline_kwargs)
        # Each stage builds its own chain, and each chain builds — and caches —
        # its own model. Two stages that name the same provider and budget get
        # the same `BaseChatModel`, and with it one connection pool, which is
        # what the `Clients` registry used to arrange by hand.
        #
        # `provider` is passed down rather than left to each stage's own
        # constant, so `--provider groq` moves the whole flow to groq. Left as
        # the default it is `AGENT_PROVIDER`, and a stage that names a different
        # one in the environment — ROUTER_PROVIDER and friends — still wins,
        # because that is the per-stage split those variables exist to express.
        stage = provider if provider != AGENT_PROVIDER else None
        self.router = Router(provider=stage)
        self.extractor = Extractor(provider=stage)
        self.generator = Generator(provider=stage)
        self.verifier = Verifier(provider=stage) if verify else NullVerifier()
        self.guardrail = Guardrail(verifier=self.verifier) if guardrail else None
        # One graph per instance, compiled once. It is the flow — `arun` only
        # seeds the state and invokes it.
        self.graph = build()

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = Pipeline(**self._pipeline_kwargs)
        return self._pipeline

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None

    # -- retrieval, with the router's answer applied ------------------------- #

    def _retrieve(self, question, route, k_vector, k_bm25):
        """Candidates, widened or filtered according to `route_mode`.

        "widen" is two local queries rather than one: the unrestricted pass,
        then a pass filtered to the routed kind, whose rows are appended if the
        first pass missed them. The reranker is the stage that judges relevance
        and it sees both, so routing changes what reaches the reranker without
        being able to hide anything from it.
        """
        if self.route_mode == "off" or route.filter_kind is None:
            return self.pipeline.retrieve(question, k_vector, k_bm25), None

        if self.route_mode == "filter":
            return self.pipeline.retrieve(question, k_vector, k_bm25,
                                          kind=route.filter_kind), "filter"

        candidates = self.pipeline.retrieve(question, k_vector, k_bm25)
        seen = {candidate["id"] for candidate in candidates}
        widened = self.pipeline.retrieve(question, ROUTE_WIDEN_K, ROUTE_WIDEN_K,
                                         kind=route.filter_kind)
        added = [candidate for candidate in widened if candidate["id"] not in seen]
        return candidates + added, f"widen +{len(added)}"

    @staticmethod
    def _in_scope(passages, reranker):
        """Is anything retrieved good enough to be worth answering from?

        Asked here, after reranking, and not of the guardrail: a rail that runs
        before retrieval has no idea what the corpus holds, so asking it whether
        a question is in scope is asking it to guess. The reranker has read the
        question against the passages, which makes its best score the only
        evidence in the system that actually bears on the matter.
        """
        floor = SCOPE_FLOOR.get(reranker)
        if floor is None or not passages:
            return True, None
        scores = [p["rerank_score"] for p in passages if p["rerank_score"] is not None]
        if not scores:
            return True, None          # the reranker failed; fused order carries no floor
        best = max(scores)
        return best >= floor, best

    # -- the flow ------------------------------------------------------------ #

    @mlflow.trace(name="agent_pipeline", span_type=SpanType.AGENT)
    async def arun(self, question, k_vector=VECTOR_TOP_K, k_bm25=BM25_TOP_K,
                   top_n=RERANK_TOP_N, rerank=True, answer=True, max_repairs=MAX_REPAIRS,
                   thread_id=None, identity=None):
        """Run the graph over one question and package the final state as a Turn.

        There is no `client` parameter any more, and nothing needs one: each
        stage's chain holds a `BaseChatModel` cached by its configuration, so a
        sweep asking a thousand questions shares the same models and the same
        pools across all of them without arranging it. Which provider serves
        which stage is `constants.py`'s to say — the gates on OpenAI nano, the
        reasoning-heavy stages on groq — and an A/B of the flow is a comparison
        of the flow because that mix is a fact about the configuration rather
        than about the call.

        `identity` is who the answer is for. Every agent receives it, and the
        NeMo rail action reads it from the ContextVar this binds — so a stage
        can log against a user or apply a per-user rule without the caller
        threading a new argument through five signatures. It defaults to
        anonymous, which is what the CLI is.

        `thread_id` is the checkpointer's key. It defaults to a fresh one per
        question, and that default matters: the graph compiles with an
        `InMemorySaver`, so a reused thread resumes the checkpoint left by the
        last question — the next run would start holding the previous one's
        candidates and passages. Pass an explicit `thread_id` only when
        resuming a specific run is what you actually want.
        """
        started = time.time()
        identity = identity or Identity.anonymous()
        # `thread_id` keys the checkpointer, so it defaults to something unique
        # per request rather than per user — two questions from one session must
        # not resume into each other.
        thread_id = thread_id or identity.request_id or uuid.uuid4().hex
        # `mlflow.trace.user` and `.session` are MLflow's standard fields, so a
        # trace can be found by who asked for it rather than only by when. Ids
        # only — never the session token that proved them.
        annotate(**{"mlflow.trace.user": identity.user_id,
                    "mlflow.trace.session": identity.session_id,
                    "thread_id": thread_id, "reranker": self.pipeline.reranker
                    if self._pipeline is not None else None})
        # Bound before `ainvoke`, not inside it: LangGraph spawns a task per
        # node and a task inherits the context it was created in, so binding
        # later would leave the earlier nodes anonymous.
        with bind(identity):
            final = await self.graph.ainvoke({
                "question": question,
                "k_vector": k_vector, "k_bm25": k_bm25, "top_n": top_n,
                "rerank": rerank, "answer_wanted": answer, "max_repairs": max_repairs,
                "guardrail_out": [], "repairs": 0, "timings": {},
            }, config={"configurable": {
                # The flow rides here, not in the state: the graph compiles with
                # a checkpointer, and config is the half of a LangGraph run that
                # is not written down.
                "flow": self, "identity": identity, "thread_id": thread_id,
            }})

        turn = Turn(question=question, started=started, **{
            field.name: final[field.name]
            for field in dataclasses.fields(Turn)
            if field.name in final and field.name not in ("question", "started", "answer")
        })
        # `unresolved` is not a node's output: it is what the last verdict still
        # objected to once the repairs ran out, which only the finished run knows.
        turn.identity = identity.to_dict()
        verdicts = final.get("guardrail_out") or []
        if verdicts and not verdicts[-1].get("allowed"):
            turn.unresolved = verdicts[-1].get("problems") or []
        result = turn.finish(final.get("answer"))

        if METRICS_ENABLED:
            # After `finish`, because latency is one of the metrics and it is
            # not known until the run is over. The encoder is passed rather
            # than looked up so a run that never opened the pipeline — a
            # guardrail block — does not open it now just to score itself.
            encoder = self._pipeline.encoder if self._pipeline is not None else None
            result["metrics"] = record(result, evaluate(result, encoder),
                                       identity).copy()
        return result

    def ask(self, question, **kwargs):
        """Blocking `arun`, for notebooks and scripts."""
        return asyncio.run(self.arun(question, **kwargs))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _trace(result):
    """The agent stages, above the retrieval table `_report` already prints."""
    trace = result["trace"]
    route = trace.get("route") or {}
    print(f"\nguardrail  {_verdict(trace.get('guardrail_in'))}")
    if route:
        print(f"route      kind={route.get('kind', '-')} "
              f"confidence={route.get('confidence', 0):.2f} "
              f"lexical={route.get('lexical', 0):.2f} "
              f"({trace.get('route_effect') or 'applied as-is'})"
              f"{'  — ' + route['reason'] if route.get('reason') else ''}")
    if trace.get("best_score") is not None:
        print(f"scope      best rerank score {trace['best_score']:.3f}")
    if trace.get("draft") is not None:
        print(f"extract    {len(trace.get('facts') or [])} facts")

    for index, verdict in enumerate(trace.get("guardrail_out") or []):
        problems = verdict.get("problems") or []
        state = "ok" if verdict.get("allowed") else f"{len(problems)} problem(s)"
        print(f"verify {index + 1}   {state}")
        for problem in problems:
            print(f"           - {problem.get('type')}: {problem.get('detail', '')[:110]}")
    if trace.get("repairs"):
        print(f"repair     {trace['repairs']} rewrite(s)")
    if trace.get("unresolved"):
        print(f"unresolved {len(trace['unresolved'])} problem(s) survived the repair")

    timings = trace.get("timings") or {}
    if timings:
        print("\ntimings    " + "  ".join(f"{k} {v}s" for k, v in timings.items()))

    if trace.get("facts"):
        print("\nextracted facts")
        for fact in trace["facts"]:
            marker = f"[{fact['passage']}]" if fact.get("passage") else "[-]"
            print(f"  {marker} {fact['fact'][:120]}")


def _verdict(verdict):
    if not verdict:
        return "not run"
    return "passed" if verdict.get("allowed") else f"BLOCKED — {verdict.get('message', '')[:80]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardrailed, routed, verified retrieval")
    parser.add_argument("question")
    parser.add_argument("--k-vector", type=int, default=VECTOR_TOP_K)
    parser.add_argument("--k-bm25", type=int, default=BM25_TOP_K)
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N)
    parser.add_argument("--provider", default=AGENT_PROVIDER,
                        help=f"provider for the agent calls (default {AGENT_PROVIDER})")
    parser.add_argument("--reranker", choices=list(RERANKERS), default=RERANKER)
    parser.add_argument("--route-mode", choices=list(ROUTE_MODES), default=ROUTE_MODE,
                        help=f"what to do with the router's answer (default {ROUTE_MODE})")
    parser.add_argument("--max-repairs", type=int, default=MAX_REPAIRS)
    parser.add_argument("--no-guardrail", action="store_true",
                        help="skip both rails — the A/B for what NeMo is buying")
    parser.add_argument("--no-verify", action="store_true",
                        help="run the input rail but not the output one")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-answer", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="hide passage text")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--trace", action="store_true",
                        help="send an MLflow trace of every stage (same as "
                             "MLFLOW_TRACING=1)")
    arguments = parser.parse_args()

    if arguments.trace:
        os.environ["MLFLOW_TRACING"] = "1"
        import importlib

        import tracing as _tracing
        importlib.reload(_tracing)
        if _tracing.setup():
            print(f"tracing to experiment {_tracing.EXPERIMENT!r}", file=sys.stderr)
    else:
        setup_tracing()

    try:
        with AgentPipeline(provider=arguments.provider, reranker=arguments.reranker,
                           route_mode=arguments.route_mode,
                           guardrail=not arguments.no_guardrail,
                           verify=not arguments.no_verify,
                           use_bm25=arguments.k_bm25 > 0) as flow:
            result = flow.ask(arguments.question, k_vector=arguments.k_vector,
                              k_bm25=arguments.k_bm25, top_n=arguments.top_n,
                              rerank=not arguments.no_rerank,
                              answer=not arguments.no_answer,
                              max_repairs=arguments.max_repairs)
    except (RetrievalError, GuardrailError, LLMError) as error:
        sys.exit(f"FAILED: {error}")

    if arguments.json:
        for candidate in result["candidates"]:
            candidate.pop("payload", None)
        for passage in result.get("passages", []):
            passage.pop("payload", None)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif result["trace"].get("refused"):
        _trace(result)
        print(f"\n{'-' * 78}\n{result['answer']}")
    else:
        _report(result, show_text=not arguments.quiet)
        _trace(result)
