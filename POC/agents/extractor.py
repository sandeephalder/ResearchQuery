"""Stage 3: the hard evidence in the passages, isolated before anything is written.

This stage already existed, buried. `INFER_SYSTEM_PROMPT` opens by asking the
answer model to "quote or extract the exact numbers, conditional statements, or
statistical thresholds" inside a <thinking> block — extraction, done by the
model that is about to write the answer, in the same breath and on the same
budget. Promoting it to its own call buys two things: the extraction can be
read, and the generator's prompt can start at the answer instead of repeating
the work. `GENERATOR_SYSTEM_PROMPT` has no thinking block for exactly this
reason.

An empty extraction is a real result, not a failure. It says the passages carry
nothing that bears on the question, which is what the generator needs in order
to abstain honestly, and what the verifier needs in order to tell an honest
abstention from a false one.
"""

import logging
import sys

from llm import LLMError
from llm.chains import numbered

from tracing import SpanType, mlflow

from .chains import as_evidence, extractor_chain
from .constants import EXTRACTOR_MAX_CHARS
from .identity import Identity

log = logging.getLogger("agents.extractor")


class Extractor:
    """One call over all the passages, not one per passage.

    The same reason the reranker is listwise: a value in [2] that is qualified
    by a condition stated in [4] is only found by a reader holding both.
    """

    def __init__(self, provider=None, model=None, max_chars=EXTRACTOR_MAX_CHARS):
        self.chain = extractor_chain(**({"provider": provider} if provider else {}),
                                     model=model, max_chars=max_chars)

    @mlflow.trace(name="extract", span_type=SpanType.AGENT)
    async def extract(self, question, passages, identity=None):
        """[{"passage": n, "fact": "..."}], or [] if there is nothing to take."""
        if not passages:
            return []

        try:
            entries = await self.chain.ainvoke({"question": question, "passages": passages})
        except LLMError as error:
            # The generator still has the passages, so a failed extraction is a
            # weaker answer rather than no answer. Saying so matters: an empty
            # list means "nothing to extract" everywhere else in the flow, and
            # the verifier reads it as evidence that an abstention was honest.
            print(f"warning: extraction failed ({error}); the generator will work "
                  f"from the passages alone", file=sys.stderr)
            return []

        facts = []
        for entry in entries:
            fact = str(entry.get("fact", "")).strip()
            if not fact:
                continue
            try:
                passage = int(entry.get("passage"))
            except (TypeError, ValueError):
                passage = None
            # An attribution to a passage that was never sent is a fabricated
            # citation the generator would inherit. Keep the fact, drop the
            # attribution, and let the generator find it in the passages.
            if passage is not None and not 1 <= passage <= len(passages):
                passage = None
            facts.append({"passage": passage, "fact": fact})
        log.info("extracted %d facts from %d passages for %s", len(facts),
                 len(passages), identity or Identity.anonymous())
        return facts


__all__ = ["Extractor", "as_evidence", "numbered"]
