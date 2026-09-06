"""Every node, every branch, no network.

    uv run pytest tests/test_agent_wiring.py

These are the checks that catch what actually breaks when the flow is edited: a
node reading a state field nobody wrote, a reducer replacing where it should
append, a branch pointing at the wrong node, a `Route` that no longer survives
the round trip through the checkpointer. None of it needs a model, and none of
it should spend one — so the chains are stubbed and retrieval is real.

What these cannot tell you is whether the answers are any good. That is
`test_agent_quality.py`, and it needs `--live`.
"""

import pytest

from agents.identity import Identity, current
from tests.stubs import Picky, Stub

WHO = Identity(user_id="user_test", session_id="sess_test", request_id="req_test")


@pytest.fixture
def stubbed(flow):
    """The flow with every chain stubbed and the guardrail off."""
    flow.guardrail, saved = None, flow.guardrail
    stub = Stub().install(flow)
    yield flow, stub
    flow.guardrail = saved


@pytest.fixture
def result(stubbed):
    flow, stub = stubbed
    import asyncio
    return asyncio.get_event_loop().run_until_complete(flow.arun("how does detection "
                                                                 "probability vary with range",
                                                                 identity=WHO)), stub


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #

class TestFullChain:
    """One run through every node, with the models stubbed."""

    def test_router_output_reaches_the_trace(self, result):
        run, _ = result
        assert (run["trace"]["route"] or {}).get("kind") == "figure"

    def test_routing_widens_rather_than_filters(self, result):
        """The default mode adds a kind-filtered pass to the unfiltered one.

        Filtering on a guess is how the gold row becomes unreachable rather than
        merely low, which is why `widen` is the default — so a run that reports
        `filter` has had its default changed, not just its output.
        """
        run, _ = result
        assert (run["trace"]["route_effect"] or "").startswith("widen")

    def test_extracted_facts_survive_the_state(self, result):
        run, _ = result
        assert len(run["trace"]["facts"]) == 1

    def test_an_answer_is_produced(self, result):
        run, _ = result
        assert run["answer"]

    def test_the_reranker_keeps_top_n(self, result):
        run, _ = result
        assert len(run["passages"]) == 5

    def test_every_stage_ran(self, result):
        run, stub = result
        assert sorted(set(stub.calls)) == ["extract", "generate", "route"]

    def test_identity_is_recorded(self, result):
        run, _ = result
        assert (run["trace"].get("identity") or {}).get("user_id") == "user_test"

    def test_identity_does_not_leak_past_the_run(self, result):
        """The ContextVar is bound for the run and reset after it.

        Eight questions in flight each see their own caller, and that is only
        true if the binding does not outlive its run.
        """
        assert str(current()) == "anonymous"


# --------------------------------------------------------------------------- #
# The repair loop
# --------------------------------------------------------------------------- #

class TestRepairLoop:
    """The one cycle in the graph, and the only edge that runs a node twice."""

    @pytest.fixture
    def repaired(self, flow):
        import asyncio

        picky = Picky()
        saved_verifier, saved_rail = flow.verifier, flow.guardrail
        flow.verifier = picky
        flow.guardrail.verifier = picky
        Stub().install(flow)
        run = asyncio.get_event_loop().run_until_complete(
            flow.arun("how does detection probability vary with range", identity=WHO))
        yield run, picky
        flow.verifier, flow.guardrail = saved_verifier, saved_rail
        flow.guardrail.verifier = saved_verifier

    def test_verify_runs_once_per_draft(self, repaired):
        """`guardrail_out` accumulates rather than replaces.

        It is an `Annotated[list, operator.add]` channel for this reason, and a
        reducer that replaced would leave one verdict here and no evidence that
        the repair had been checked.
        """
        run, _ = repaired
        assert len(run["trace"]["guardrail_out"]) == 2

    def test_one_repair(self, repaired):
        run, _ = repaired
        assert run["trace"]["repairs"] == 1

    def test_nothing_left_unresolved(self, repaired):
        run, _ = repaired
        assert run["trace"]["unresolved"] == []

    def test_identity_reaches_the_rails_own_action(self, repaired):
        """NeMo calls the rail action, so nothing can pass it an argument.

        It arrives through the ContextVar instead, and this is the only test
        that proves that path works.
        """
        _, picky = repaired
        assert picky.identity and picky.identity.user_id == "user_test"


# --------------------------------------------------------------------------- #
# The three early exits
# --------------------------------------------------------------------------- #

class TestEarlyExits:

    def test_no_answer_stops_after_rerank(self, stubbed):
        import asyncio
        flow, _ = stubbed
        run = asyncio.get_event_loop().run_until_complete(
            flow.arun("how does detection probability vary with range", answer=False))
        assert run["answer"] is None

    def test_scope_floor_refuses(self, stubbed):
        """Nothing clears a threshold above 1.0, so the floor must fire.

        The floor is where "out of scope" is decided — after retrieval, on the
        reranker's best score, because a rail running before retrieval cannot
        know what the corpus holds.
        """
        import asyncio

        from agents import constants
        import agents.orchestrator as orchestrator

        flow, _ = stubbed
        saved = constants.SCOPE_FLOOR.copy()
        try:
            constants.SCOPE_FLOOR["cross-encoder"] = 2.0
            orchestrator.SCOPE_FLOOR["cross-encoder"] = 2.0
            run = asyncio.get_event_loop().run_until_complete(
                flow.arun("how does detection probability vary with range"))
            assert (run["trace"]["refused"] or "").startswith("below scope")
        finally:
            constants.SCOPE_FLOOR.update(saved)
            orchestrator.SCOPE_FLOOR.update(saved)


# --------------------------------------------------------------------------- #
# The graph itself
# --------------------------------------------------------------------------- #

class TestGraph:
    """The drawing and the run come from the same compiled object."""

    def test_every_node_is_in_the_drawing(self):
        from agents.graph import mermaid

        drawing = mermaid()
        for node in ("guardrail_in", "router", "retrieve", "rerank", "extract",
                     "generate", "verify", "repair", "refuse"):
            assert node in drawing, f"{node} missing from the drawn graph"

    def test_state_declares_every_field_a_node_returns(self):
        """Only declared fields become channels.

        Anything a node returns that is not in `AgentState` is dropped silently,
        and the symptom is a KeyError in the node after it for something the
        node before plainly returned.
        """
        import dataclasses

        from agents.graph import AgentState
        from agents.state import Turn

        declared = set(AgentState.__annotations__)
        carried = {f.name for f in dataclasses.fields(Turn)}
        # `started`, `identity` and `answer` are set by `arun`, not by a node.
        missing = carried - declared - {"started", "identity", "answer", "question"}
        assert not missing, f"Turn fields no channel carries: {sorted(missing)}"
