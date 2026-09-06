"""Canned chain outputs, for the tests that must not touch a network.

Each agent stage is `prompt | model | parser`, and it is the whole chain that is
replaced here rather than the transport under it. A stage's contract is "a dict
in, a parsed value out", so a `RunnableLambda` honouring that contract is a
complete stand-in — and unlike a stubbed socket, it cannot let a test pass on a
reply the real chain would have failed to parse.

Retrieval and reranking are left real. Both are local, and a stubbed retriever
would leave the wiring tests testing almost nothing.
"""

from langchain_core.runnables import RunnableLambda

ROUTE = {"kind": "figure", "confidence": 0.8, "lexical": 0.2, "reason": "a plot shows it"}
FACTS = [{"passage": 1, "fact": "p_D falls as range rises"}]
DRAFT = "Detection probability falls with range [1]."
VERDICT = {"ok": True, "problems": []}


class Stub:
    """Canned chain outputs, and a record of which stages were asked."""

    def __init__(self):
        self.calls = []

    def _chain(self, stage, reply):
        def run(_inputs):
            self.calls.append(stage)
            return reply
        return RunnableLambda(run)

    def install(self, flow):
        """Replace every stage's chain on `flow`. Returns self, for chaining."""
        flow.router.chain = self._chain("route", ROUTE)
        flow.extractor.chain = self._chain("extract", FACTS)
        flow.generator.chain = self._chain("generate", DRAFT)
        if hasattr(flow.verifier, "chain"):
            flow.verifier.chain = self._chain("verify", VERDICT)
        return self


class Picky:
    """A verifier that objects once, then accepts — so the repair edge is taken."""

    def __init__(self):
        self.seen = 0
        self.identity = None

    async def check(self, question, draft, passages, facts, identity=None):
        from agents.verifier import Report

        self.seen += 1
        self.identity = identity
        if self.seen == 1:
            return Report(ok=False, problems=[
                {"type": "flipped_logic", "detail": "planted, to force one repair"}])
        return Report(ok=True)
