"""The flow, as a LangGraph graph. This is what runs.

    uv run python -m agents.graph                  # draw it: mermaid source
    uv run python -m agents.graph --ascii          # draw it: in the terminal

`orchestrator.AgentPipeline` builds one of these and invokes it; there is no
second, hand-rolled path through the stages. Which matters more than it sounds:
a graph you draw and a graph you run should not be two different graphs, and the
picture below is generated from the same object that executed the last query.

    START
      |
      v
    guardrail_in ---(blocked)---------------------> refuse
      |
      v
    router --> retrieve ---(nothing found)--------> refuse
                  |
                  v
                rerank ---(below the scope floor)--> refuse
                  |
                  v
               extract --> generate --> verify ---(problems, repairs left)--> repair
                                          |                                     |
                                          |<------------------------------------+
                                          v
                                         END

Three early exits and one loop. The loop is bounded by `MAX_REPAIRS`, and a
draft that still fails when the repairs run out is returned anyway with the
complaints recorded — the verifier's opinion is evidence, not a veto.

State is a plain dict of the fields in `state.Turn`, which is what the run is
finally packaged as. Two fields accumulate rather than replace: `guardrail_out`
collects one verdict per verify round, and `timings` merges as each node
reports. Everything else is last-write-wins, which is what a linear pipeline
wants.
"""

import argparse
import asyncio
import operator
import sys
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from .constants import MAX_REPAIRS, OUT_OF_SCOPE_MESSAGE, REFUSAL_MESSAGE
from .identity import Identity
from .router import Route


def _merge(left, right):
    """Reducer for `timings`: later nodes add keys, they do not replace the dict."""
    return {**(left or {}), **(right or {})}


def flow_of(config):
    """The live `AgentPipeline` a node needs.

    It travels in `config["configurable"]`, not in the state, because the graph
    compiles with a checkpointer and the checkpointer msgpack-serialises every
    state field. An AgentPipeline holding Qdrant and a loaded BGE-M3 does not
    serialise. Config is not checkpointed, which is exactly what runtime
    dependencies want.

    There is no client to fetch alongside it any more. Each stage's chain builds
    — and caches — its own model, so which provider serves which stage is a fact
    about `constants.py` rather than something the graph has to carry.
    """
    return config["configurable"]["flow"]


def who(config) -> Identity:
    """The caller this run is serving. Anonymous on the CLI."""
    return config["configurable"].get("identity") or Identity.anonymous()


class AgentState(TypedDict, total=False):
    """What travels along the edges, and what the checkpointer writes down.

    Everything here must survive msgpack. `state.Turn` is what it becomes.
    """

    question: str
    k_vector: int
    k_bm25: int
    top_n: int
    rerank: bool
    answer_wanted: bool
    max_repairs: int

    guardrail_in: dict
    guardrail_out: Annotated[list, operator.add]
    # Only declared fields become channels — anything a node returns that is
    # not in this schema is silently dropped, which is the first thing to check
    # when a node sees a KeyError for something the node before it plainly
    # returned.
    route: dict
    route_effect: str
    candidates: list
    passages: list
    reranked: bool
    best_score: float
    vector_mode: str
    reranker: str
    facts: list
    draft: str
    repairs: int
    unresolved: list
    answer: str
    refused: str
    timings: Annotated[dict, _merge]


class _Timed:
    """Times a node and reports it as a state update."""

    def __init__(self, name):
        self.name = name
        self.started = time.time()

    def __call__(self, update):
        update["timings"] = {self.name: round(time.time() - self.started, 2)}
        return update


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

async def guardrail_in(state, config):
    """Stage 1 — the NeMo input rails."""
    timed = _Timed("guardrail_in")
    guardrail = flow_of(config).guardrail
    if guardrail is None:
        return timed({})
    verdict = await guardrail.check_input(state["question"])
    update = {"guardrail_in": verdict.to_dict()}
    if not verdict.allowed:
        update["refused"] = "guardrail"
        update["answer"] = verdict.message or REFUSAL_MESSAGE
    return timed(update)


async def router(state, config):
    """Stage 2 — which kind of row, and how literal the query is.

    Also where the retrieval store is first touched: everything upstream of
    here can run without opening Qdrant or loading BGE-M3, so a blocked
    question pays for neither.
    """
    timed = _Timed("route")
    flow = flow_of(config)
    route = await flow.router.route(
        state["question"], state["k_vector"], state["k_bm25"], identity=who(config),
        enabled=flow.route_mode != "off")
    return timed({
        "route": route.to_dict() if flow.route_mode != "off" else None,
        "vector_mode": flow.pipeline.vector_mode,
        "reranker": flow.pipeline.reranker if state["rerank"] else "none",
    })


async def retrieve(state, config):
    """Both legs, fused — and, in `widen` mode, a second kind-filtered pass."""
    timed = _Timed("retrieve")
    flow = flow_of(config)
    # Rebuilt from the dict rather than carried as an object: `Route.to_dict`
    # emits exactly the fields `Route.__init__` takes, so this round-trips, and
    # the state stays made of things msgpack can write down.
    route = (Route(**state["route"]) if state.get("route")
             else Route(k_vector=state["k_vector"], k_bm25=state["k_bm25"]))
    # `_retrieve` is synchronous throughout — Qdrant, then BM25, then a second
    # pass in widen mode — and measured at 10-18 s. Awaited directly it would
    # hold the event loop for that long, which a CLI never notices and a server
    # cannot afford.
    candidates, effect = await asyncio.to_thread(
        flow._retrieve, state["question"], route, route.k_vector, route.k_bm25)
    update = {"candidates": candidates, "route_effect": effect}
    if not candidates:
        update["refused"] = "no candidates"
        update["answer"] = OUT_OF_SCOPE_MESSAGE
    return timed(update)


async def rerank(state, config):
    """The reranker, then the scope floor its best score is read against."""
    timed = _Timed("rerank")
    flow = flow_of(config)
    if state["rerank"] and flow.pipeline.reranker != "none":
        passages, reranked = await flow.pipeline.rerank(
            state["question"], state["candidates"], state["top_n"])
    else:
        passages, reranked = state["candidates"][:state["top_n"]], False

    in_scope, best = flow._in_scope(passages, flow.pipeline.reranker if reranked else "none")
    update = {"passages": passages, "reranked": reranked, "best_score": best}
    if not in_scope:
        update["refused"] = f"below scope floor ({best:.3f})"
        update["answer"] = OUT_OF_SCOPE_MESSAGE
    elif not state["answer_wanted"]:
        update["refused"] = None
        update["answer"] = None
    return timed(update)


async def extract(state, config):
    """Stage 3 — the numbers, bounds, conditions and table rows."""
    timed = _Timed("extract")
    flow = flow_of(config)
    return timed({"facts": await flow.extractor.extract(
        state["question"], state["passages"], identity=who(config))})


async def generate(state, config):
    """Stage 4 — the draft, citing [1]..[n]."""
    timed = _Timed("generate")
    flow = flow_of(config)
    draft = await flow.generator.draft(
        state["question"], state["passages"], state["facts"], identity=who(config))
    return timed({"draft": draft, "answer": draft})


async def verify(state, config):
    """Stage 5 — the NeMo output rail, backed by `verifier.Verifier`."""
    timed = _Timed("verify")
    flow = flow_of(config)
    if flow.guardrail is None:
        return timed({})
    verdict = await flow.guardrail.check_output(
        state["question"], state["answer"],
        state["passages"], state["facts"])   # identity reaches the rail's action
                                             # through the ContextVar — NeMo
                                             # calls that action, not us.
    return timed({"guardrail_out": [verdict.to_dict()]})


async def repair(state, config):
    """Stage 4 again, with the verifier's complaints as the brief."""
    timed = _Timed("repair")
    flow = flow_of(config)
    problems = state["guardrail_out"][-1].get("problems") or []
    answer = await flow.generator.draft(
        state["question"], state["passages"], state["facts"], problems,
        identity=who(config))
    return timed({"answer": answer, "repairs": state.get("repairs", 0) + 1})


async def refuse(state):
    """The terminus for all three early exits. The message is already set."""
    return {"answer": state.get("answer") or REFUSAL_MESSAGE}


# --------------------------------------------------------------------------- #
# Branches
# --------------------------------------------------------------------------- #

def passed_guardrail(state):
    return "refuse" if state.get("refused") else "router"


def retrieved_anything(state):
    return "refuse" if state.get("refused") else "rerank"


def in_scope(state):
    """Refuse, stop early on `--no-answer`, or go on to the extractor."""
    if state.get("refused"):
        return "refuse"
    return "extract" if state.get("answer_wanted") else END


def verified(state):
    """Repair, or stop.

    Out of repairs with the complaints unresolved, the draft still stands:
    suppressing a flagged answer converts a partly wrong answer into no answer.
    """
    verdicts = state.get("guardrail_out") or []
    if verdicts and not verdicts[-1].get("allowed"):
        if state.get("repairs", 0) < state.get("max_repairs", MAX_REPAIRS):
            return "repair"
    return END


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #

def build():
    """The compiled flow. `orchestrator.AgentPipeline` builds one and invokes it."""
    memory = InMemorySaver()
    graph = StateGraph(AgentState)

    graph.add_node("guardrail_in", guardrail_in)
    graph.add_node("router", router)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("extract", extract)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("repair", repair)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "guardrail_in")
    graph.add_conditional_edges("guardrail_in", passed_guardrail,
                                {"refuse": "refuse", "router": "router"})
    graph.add_edge("router", "retrieve")
    graph.add_conditional_edges("retrieve", retrieved_anything,
                                {"refuse": "refuse", "rerank": "rerank"})
    graph.add_conditional_edges("rerank", in_scope,
                                {"refuse": "refuse", "extract": "extract", END: END})
    graph.add_edge("extract", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", verified, {"repair": "repair", END: END})
    graph.add_edge("repair", "verify")
    graph.add_edge("refuse", END)

    return graph.compile(checkpointer=memory)


def mermaid():
    """Mermaid source, rendered locally — no call out to mermaid.ink."""
    return build().get_graph().draw_mermaid()


def ascii_art():
    """The same graph in the terminal. Needs `grandalf`."""
    try:
        return build().get_graph().draw_ascii()
    except ImportError:
        return "ascii rendering needs grandalf: uv add --dev grandalf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the agent flow")
    parser.add_argument("--ascii", action="store_true", help="draw it in the terminal")
    parser.add_argument("--out", help="write to a file instead of stdout")
    arguments = parser.parse_args()

    drawing = ascii_art() if arguments.ascii else mermaid()
    if arguments.out:
        with open(arguments.out, "w") as handle:
            handle.write(drawing if arguments.ascii else f"```mermaid\n{drawing}```\n")
        print(f"wrote {arguments.out}", file=sys.stderr)
    else:
        print(drawing)
