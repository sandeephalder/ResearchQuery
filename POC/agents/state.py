"""One question's passage through the flow, recorded as it happens.

Every stage writes what it decided here rather than returning it up a chain, so
a finished run can be read backwards: an answer that is wrong can be traced to
the router that sent retrieval to the wrong modality, the extractor that missed
the number, or the verifier that let it through. `retrieval/pipeline.py` does
the same thing for its two legs by carrying both ranks on every candidate — a
run should show its own working.

`to_dict` produces the shape `retrieval.Pipeline.arun` returns, with the trace
alongside it, so the eval harness cannot tell the two apart.
"""

import dataclasses
import time


@dataclasses.dataclass
class Turn:
    """The state of one question, from the guardrail to the answer."""

    question: str
    started: float = dataclasses.field(default_factory=time.time)

    # stage 1 and 5 — the rails
    guardrail_in: dict = None
    guardrail_out: list = dataclasses.field(default_factory=list)

    # stage 2 — routing, and what it changed about retrieval
    route: dict = None
    route_effect: str = None

    # retrieval
    candidates: list = dataclasses.field(default_factory=list)
    passages: list = dataclasses.field(default_factory=list)
    reranked: bool = False
    best_score: float = None
    vector_mode: str = None
    reranker: str = None

    # stages 3 and 4
    facts: list = dataclasses.field(default_factory=list)
    draft: str = None
    repairs: int = 0
    unresolved: list = dataclasses.field(default_factory=list)

    answer: str = None
    refused: str = None
    # Who the run was for, so a saved trace can be read back per user. Ids
    # only — never the session token that proved them.
    identity: dict = None
    timings: dict = dataclasses.field(default_factory=dict)

    def refuse(self, message, reason):
        """Stop here. The message is the answer — a refusal is a real result."""
        self.refused = reason
        return self.finish(message)

    def finish(self, answer):
        self.answer = answer
        self.timings["total"] = round(time.time() - self.started, 2)
        return self.to_dict()

    def to_dict(self):
        return {
            # The keys `retrieval.Pipeline.arun` returns, so the eval harness
            # and `_report` work against either without knowing which they have.
            "question": self.question,
            "candidates": self.candidates,
            "passages": self.passages,
            "reranked": self.reranked,
            "answer": self.answer,
            "vector_mode": self.vector_mode,
            "reranker": self.reranker,
            "trace": {
                "guardrail_in": self.guardrail_in,
                "guardrail_out": self.guardrail_out,
                "route": self.route,
                "route_effect": self.route_effect,
                "best_score": self.best_score,
                "facts": self.facts,
                "draft": self.draft,
                "repairs": self.repairs,
                "unresolved": self.unresolved,
                "refused": self.refused,
                "identity": self.identity,
                "timings": self.timings,
            },
        }
