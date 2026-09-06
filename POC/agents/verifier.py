"""Stage 5's substance: does the draft survive a reading of the passages?

This runs as a NeMo output rail — `guardrail.Guardrail` registers `check` as the
action behind the `check answer grounding` flow. NeMo owns the rail; what the
rail asks is here, because the built-in action returns a yes/no and a repair
loop needs to know what was wrong.

Four faults, and the first is the one worth the call. The evaluation found
abstentions quadrupling on image queries, with the gold document already in
context — a model looking at an extracted figure description and deciding it is
not really evidence. That is a false abstention, and it is invisible to every
other stage: retrieval succeeded, the reranker succeeded, and the answer is a
polite refusal that scores zero.

The others are flipped logic (an inequality reversed, a null result reported as
a positive one), unsupported claims, and miscitation.

One repair. A verifier that can send a draft back indefinitely is a loop whose
exit condition is a model's opinion, and the second repair has nothing new to
work from — same passages, same facts, one more complaint.
"""

import logging

from tracing import SpanType, mlflow

from .chains import verifier_chain
from .constants import ANSWER_MAX_CHARS
from .identity import Identity

log = logging.getLogger("agents.verifier")

PROBLEM_TYPES = ("false_abstention", "flipped_logic", "unsupported_claim", "miscitation")


class Report:
    """The verifier's finding. `ok` is the decision; `problems` is the repair brief."""

    __slots__ = ("ok", "problems", "ran")

    def __init__(self, ok=True, problems=(), ran=True):
        self.ok = ok
        self.problems = list(problems)
        self.ran = ran

    def to_dict(self):
        return {"ok": self.ok, "ran": self.ran, "problems": self.problems}

    def __repr__(self):
        return f"Report(ok={self.ok}, ran={self.ran}, problems={len(self.problems)})"


class Verifier:
    """One call. Raises `LLMError`, so the rail can decide what a failure means."""

    def __init__(self, provider=None, model=None, max_chars=ANSWER_MAX_CHARS):
        self.chain = verifier_chain(**({"provider": provider} if provider else {}),
                                    model=model, max_chars=max_chars)

    @mlflow.trace(name="verify", span_type=SpanType.AGENT)
    async def check(self, question, draft, passages, facts, identity=None):
        if not (draft or "").strip():
            return Report(ok=True, ran=False)

        reply = await self.chain.ainvoke({"question": question, "draft": draft,
                                          "passages": passages, "facts": facts})
        problems = []
        for problem in reply.get("problems") or []:
            if isinstance(problem, str):
                problems.append({"type": "unsupported_claim", "detail": problem})
                continue
            if not isinstance(problem, dict):
                continue
            detail = str(problem.get("detail", "")).strip()
            if not detail:
                continue
            kind = str(problem.get("type", "")).strip().lower()
            problems.append({"type": kind if kind in PROBLEM_TYPES else "unsupported_claim",
                             "detail": detail})

        # The two fields can disagree — a model that lists problems and then
        # says ok, or the reverse. The problems are the evidence and the flag is
        # a summary of them, so the problems win.
        ok = bool(reply.get("ok", True)) and not problems
        # Worth a warning rather than a debug line: a draft that failed
        # verification is the signal the whole stage exists to produce, and it
        # is the one a per-user investigation would start from.
        (log.info if ok else log.warning)(
            "verified for %s: %s", identity or Identity.anonymous(),
            "ok" if ok else ", ".join(p["type"] for p in problems))
        return Report(ok=ok, problems=problems)


class NullVerifier:
    """What runs with --no-verify: the stage is skipped, and says so."""

    async def check(self, *_args, **_kwargs):
        return Report(ok=True, ran=False)
