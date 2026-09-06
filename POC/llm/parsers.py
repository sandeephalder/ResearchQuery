"""Output parsers, for models that were asked for JSON and sent something near it.

Every structured stage in this repo — the reranker, the router, the extractor,
the verifier — asks for JSON and gets JSON inside a markdown fence, or JSON
after a sentence of preamble, or JSON that stopped mid-object because the reply
outgrew its budget. The last is the common failure and the least deserving of
one: the entries that did arrive are perfectly good, and a stage that raises on
them throws away work that was paid for.

LangChain already solves two thirds of this. `parse_json_markdown` finds the
JSON in a fenced or prefixed reply, and `parse_partial_json` closes a truncated
one. What is added here is the last fallback — reading the complete `{...}`
objects out of the text and ignoring the brackets entirely — and the decision to
return what survived rather than raise.

These are `BaseOutputParser`s, so they compose into a chain with `|` and carry
their own format instructions into the prompt.
"""

import json
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.utils.json import parse_json_markdown, parse_partial_json

# Non-greedy and brace-free inside, so this matches the individual elements of a
# truncated array rather than the truncated array itself.
_ELEMENT_RE = re.compile(r"\{[^{}]*}")
# A closed thinking block, or an unclosed one that ran out of budget — both are
# scratchpad, and neither is the answer.
_THINKING_RE = re.compile(r"<think(?:ing)?>.*?(?:</think(?:ing)?>|$)", re.DOTALL | re.IGNORECASE)


def _salvage(text):
    """Whatever complete JSON objects are in the text, in order."""
    salvaged = []
    for element in _ELEMENT_RE.findall(text or ""):
        try:
            parsed = json.loads(element)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            salvaged.append(parsed)
    return salvaged


def _loads(text):
    """The outermost JSON value in `text`, closing it if the reply was cut short."""
    for attempt in (parse_json_markdown, parse_partial_json):
        try:
            parsed = attempt(text)
        except Exception:                       # noqa: BLE001 — both raise broadly
            continue
        if parsed is not None:
            return parsed
    return None


class JsonListParser(BaseOutputParser):
    """A list of objects. Returns what survived; never raises on a short reply.

    Used by every stage that asks for an array — the reranker's scores and the
    extractor's facts. An empty list is a real answer in both: it means the
    model found nothing to say, which the caller needs to be able to tell apart
    from a stage that failed.
    """

    def parse(self, text):
        parsed = _loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
        return _salvage(text)

    def get_format_instructions(self):
        return "Reply with a JSON array and nothing else — no prose, no markdown fence."

    @property
    def _type(self):
        return "json_list"


class JsonObjectParser(BaseOutputParser):
    """A single object. A model that returned an array of one is common enough
    to handle rather than reject, and a reply with nothing usable in it gets
    `default` — which is how the verifier's fallback stays a decision the
    caller made rather than an exception it has to interpret."""

    default: dict = {}

    def parse(self, text):
        parsed = _loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
        salvaged = _salvage(text)
        return salvaged[0] if salvaged else dict(self.default)

    def get_format_instructions(self):
        return "Reply with a single JSON object and nothing else."

    @property
    def _type(self):
        return "json_object"


class AnswerParser(BaseOutputParser):
    """The answer, with the reasoning scratchpad removed.

    `INFER_SYSTEM_PROMPT` asks the model to work through the passages in a
    <thinking> block before answering. That block is scratchpad: it is not the
    answer, it should not reach a reader, and left in place it is what an
    evaluation judge grades. So it is stripped here, in the one component that
    knows the prompt asked for it.
    """

    def parse(self, text):
        stripped = _THINKING_RE.sub("", text or "").strip()
        # If the model spent the whole budget thinking and never answered,
        # handing back what it did say beats handing back an empty string.
        return stripped or (text or "").strip()

    @property
    def _type(self):
        return "answer"


class RankingParser(JsonListParser):
    """The reranker's array as `[(index, score)]`, best first. Indices are 0-based.

    Ids the model invented or repeated are dropped. Ids it never returned keep
    their retrieval order at the end, scored None: a passage the reranker forgot
    — or never got to, because the reply was cut short — is not evidence that
    the passage is bad.

    `count` is set per call, because the parser cannot otherwise know how many
    passages were sent and therefore which ids are real.
    """

    count: int = 0

    def parse(self, text):
        entries = super().parse(text)

        ranked, seen = [], set()
        for entry in entries:
            try:
                number = int(entry["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 1 <= number <= self.count or number in seen:
                continue
            seen.add(number)
            try:
                score = float(entry.get("score"))
            except (TypeError, ValueError):
                score = None
            ranked.append((number - 1, score))

        if not ranked:
            raise OutputParserException(
                f"Reranker returned no usable ids: {(text or '')[:200]!r}")
        ranked.extend((index, None) for index in range(self.count) if index + 1 not in seen)
        return ranked

    @property
    def _type(self):
        return "ranking"
