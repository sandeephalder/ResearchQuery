"""Stage 4: the draft, from the extracted facts and the passages they came from.

The flow this implements said the generator should work from the extractor's
facts *only*. It works from both, and the reason is what the evaluation already
measured: 92% of this pipeline's failures happen with the gold document already
in context. The failure being addressed is not "the model saw too much" — it is
"the model had the evidence and fumbled it". Putting a lossy summariser between
the passages and the writer treats a problem the system does not have, and costs
one it would then acquire: a generator that never saw [3] cannot cite [3]
faithfully.

So the facts lead — they are what a stage dedicated to reading for values found,
and they are where the numbers come from — and the passages remain, so the
citation is checkable and an extraction that missed something is recoverable.

The repair path is the same call with the verifier's complaints appended. Not a
fresh start: the fault the verifier catches is usually local — an inequality the
wrong way round, a refusal made with the evidence in hand — and rewriting from
scratch loses the parts that were right.
"""

import logging

from tracing import SpanType, mlflow

from .chains import generator_chain
from .constants import ANSWER_MAX_CHARS
from .identity import Identity

log = logging.getLogger("agents.generator")


class Generator:
    """The one stage where answer quality is actually bought."""

    def __init__(self, provider=None, model=None, max_chars=ANSWER_MAX_CHARS):
        self.chain = generator_chain(**({"provider": provider} if provider else {}),
                                     model=model, max_chars=max_chars)

    @mlflow.trace(name="generate", span_type=SpanType.AGENT)
    async def draft(self, question, passages, facts, problems=(), identity=None):
        """An answer citing [1]..[n]. `problems` turns this into the repair call."""
        answer = await self.chain.ainvoke({"question": question, "passages": passages,
                                           "facts": facts, "problems": problems})
        log.info("%s for %s: %d chars from %d facts",
                 "repaired" if problems else "drafted",
                 identity or Identity.anonymous(), len(answer), len(facts))
        return answer
