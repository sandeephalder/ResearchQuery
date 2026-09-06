"""Stage 2: which kind of row is likely to hold the answer, and how literal is the query.

The corpus indexes three kinds of row — prose, figure descriptions and table
transcriptions — and `retrieval.Pipeline.retrieve` already takes a `kind`
filter. So the router's whole job is to fill in arguments a function already
has, which is why it is one small call and no new retrieval code.

What is done with the answer is `ROUTE_MODE`, and the default is not the obvious
one. Filtering on the router's guess is what the flow diagram draws, and it is
the wrong default here: the measured weakness of this corpus is modality —
image queries score 2.41 against text's 2.73, with four times the abstentions —
and a hard filter on a wrong guess is how the gold row becomes unreachable
rather than merely low. "widen" runs the extra filtered pass and adds it to the
candidates instead, so a wrong guess costs the reranker a few more rows to look
at and nothing else. Both legs are local and sub-millisecond; this is paid in
reranker input, not latency.
"""

import logging
import sys

from llm import LLMError

from tracing import SpanType, mlflow

from .chains import router_chain
from .constants import BM25_TOP_K, ROUTE_KINDS, VECTOR_TOP_K
from .identity import Identity

log = logging.getLogger("agents.router")


class Route:
    """Where the router thinks the answer lives, and how sure it is."""

    __slots__ = ("kind", "confidence", "lexical", "reason", "k_vector", "k_bm25")

    def __init__(self, kind="any", confidence=0.0, lexical=0.0, reason="",
                 k_vector=VECTOR_TOP_K, k_bm25=BM25_TOP_K):
        self.kind = kind
        self.confidence = confidence
        self.lexical = lexical
        self.reason = reason
        self.k_vector = k_vector
        self.k_bm25 = k_bm25

    @property
    def filter_kind(self):
        """The kind to actually pass to `retrieve`, or None for no filter."""
        return None if self.kind == "any" else self.kind

    def to_dict(self):
        return {"kind": self.kind, "confidence": self.confidence, "lexical": self.lexical,
                "reason": self.reason, "k_vector": self.k_vector, "k_bm25": self.k_bm25}

    def __repr__(self):
        return (f"Route(kind={self.kind!r}, confidence={self.confidence:.2f}, "
                f"lexical={self.lexical:.2f}, k_vector={self.k_vector}, k_bm25={self.k_bm25})")


class Router:
    """One call, or none. Falls back to the unrouted defaults on any failure."""

    def __init__(self, provider=None, model=None):
        self.chain = router_chain(**({"provider": provider} if provider else {}), model=model)

    @mlflow.trace(name="route", span_type=SpanType.AGENT)
    async def route(self, question, k_vector=VECTOR_TOP_K, k_bm25=BM25_TOP_K,
                    identity=None, enabled=True):
        """The routed kind, or the unrouted defaults.

        `enabled` is `--route-mode off`, and it skips the call rather than just
        its effect — which is the point of measuring the mode at all.
        """
        default = Route(k_vector=k_vector, k_bm25=k_bm25, reason="router not run")
        if not enabled:
            return default

        try:
            reply = await self.chain.ainvoke({"question": question})
        except LLMError as error:
            # An unrouted query is the pipeline as it already works. That is a
            # working default, so the failure costs the routing and nothing else
            # — but it is announced, so a silently unrouted sweep is not read as
            # evidence that routing does not help.
            print(f"warning: routing failed ({error}); retrieving unrouted", file=sys.stderr)
            default.reason = f"routing failed: {error}"
            return default

        log.info("routed for %s: %s", identity or Identity.anonymous(),
                 reply.get("kind", "any"))
        kind = str(reply.get("kind", "any")).strip().lower()
        if kind not in ROUTE_KINDS:
            kind = "any"
        return Route(kind=kind,
                     confidence=_number(reply.get("confidence"), 0.0),
                     lexical=_number(reply.get("lexical"), 0.0),
                     reason=str(reply.get("reason", ""))[:200],
                     k_vector=k_vector, k_bm25=k_bm25)


def _number(value, fallback):
    """A float in 0-1, whatever the model put in the field."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return fallback
