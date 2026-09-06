"""The four agent stages, each as an LCEL chain.

    router      which kind of row holds the answer, and how literal the query is
    extractor   the numbers, bounds, conditions and table rows in the passages
    generator   the draft, citing [1]..[n] — and the repair, which is the same
                call with the verifier's complaints appended
    verifier    what is wrong with the draft, if anything

Every one is `prompt | model | text | parser`, and the four differ only in which
prompt, which budget and which parser. Collecting them here rather than leaving
a hand-built message list in each stage is what the rewrite bought: the stages
below are now about *what the call means*, and this file is about how it is made.

## Which provider runs which stage

Each stage names its own provider and model in `constants.py`, because they do
not all want the same one — the gates are cheapest on OpenAI's nano models and
the reasoning-heavy stages on groq. `chat_model` caches by configuration, so two
stages that do land on the same provider and budget share one model object, and
with it one connection pool. That is what `Clients` used to arrange by hand.

## Reasoning models break token budgets, three times over

Every free model on OpenRouter reasons before it answers, and reasoning is
billed and budgeted as output. That broke three stages, each silently:

    rerank      1200 -> 3000    gpt-oss-120b emits 4,000-4,800 characters first
    extract     1500 -> 3000    2,428 completion tokens for a 1,118-char reply
    router       200 -> 1500    still mid-preamble at 800
    verify       800 -> 2000    precautionary, not measured

The failures were silent because each stage's parse failure looks like a real
answer: an extractor that returns no array reports "no evidence in the
passages", and a verifier that returns nothing parseable falls back to
`{"ok": true}` and approves. A stage that fails by approving is worse than one
that fails loudly, which is why the verifier's budget was raised on suspicion
rather than on evidence.
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from llm.chains import numbered, prompt_for
from llm.models import chat_model, guarded, text_of
from llm.parsers import JsonListParser, JsonObjectParser

from .constants import (ANSWER_MAX_CHARS, EXTRACTOR_MAX_CHARS, EXTRACTOR_MAX_TOKENS,
                        EXTRACTOR_MODEL, EXTRACTOR_PROVIDER, EXTRACTOR_SYSTEM_PROMPT,
                        GENERATOR_MAX_TOKENS, GENERATOR_MODEL, GENERATOR_PROVIDER,
                        GENERATOR_SYSTEM_PROMPT, REPAIR_INSTRUCTION, ROUTER_MAX_TOKENS,
                        ROUTER_MODEL, ROUTER_PROVIDER, ROUTER_SYSTEM_PROMPT,
                        VERIFIER_MAX_TOKENS, VERIFIER_MODEL, VERIFIER_PROVIDER,
                        VERIFIER_SYSTEM_PROMPT)

ROUTER_PROMPT = prompt_for(ROUTER_SYSTEM_PROMPT)
EXTRACTOR_PROMPT = prompt_for(EXTRACTOR_SYSTEM_PROMPT)
GENERATOR_PROMPT = prompt_for(GENERATOR_SYSTEM_PROMPT)
VERIFIER_PROMPT = prompt_for(VERIFIER_SYSTEM_PROMPT)


def as_evidence(facts):
    """Extracted facts as `[n] fact` lines, for the generator and the verifier."""
    if not facts:
        return "(nothing extracted — the passages carry no evidence on this question)"
    lines = []
    for fact in facts:
        passage = fact.get("passage")
        marker = f"[{passage}] " if passage else ""
        lines.append(f"{marker}{fact.get('fact', '')}")
    return "\n".join(lines)


def _stage(prompt, parser, provider, model_override, max_tokens, to_message, name):
    """One stage: build the human message, ask the model, read the reply.

    `to_message` is the only part that differs between the four, and it takes
    the chain's input dict and returns the text of the user turn. Everything
    else — the system prompt, the deterministic temperature, the budget, the
    salvaging parser — is the same shape every time.
    """
    llm = chat_model(provider, model_override=model_override,
                     temperature=0.0, max_tokens=max_tokens)
    chain = (RunnableLambda(lambda inputs: {"messages": [HumanMessage(to_message(inputs))]})
             | prompt | llm | RunnableLambda(text_of))
    if parser is not None:
        chain = chain | parser
    # Guarded, so a provider's own exception reaches the stage above as the
    # `LLMError` it is written to degrade on. Without this a Gemini quota error
    # and a groq rate limit are two unrelated types, and neither is caught.
    return guarded(chain, name)


# --------------------------------------------------------------------------- #
# Stage 2 — the router
# --------------------------------------------------------------------------- #

def router_chain(provider=ROUTER_PROVIDER, model=None):
    """`{"question": ...}` in, `{"kind", "confidence", "lexical", "reason"}` out.

    The default on an unparseable reply is `{}`, which the caller reads as "no
    route" — the pipeline as it already works, which is a working default.
    """
    return _stage(ROUTER_PROMPT, JsonObjectParser(default={}),
                  provider, model or ROUTER_MODEL, ROUTER_MAX_TOKENS,
                  lambda inputs: f"Question: {inputs['question']}", "route")


# --------------------------------------------------------------------------- #
# Stage 3 — the extractor
# --------------------------------------------------------------------------- #

def extractor_chain(provider=EXTRACTOR_PROVIDER, model=None, max_chars=EXTRACTOR_MAX_CHARS):
    """`{"question", "passages"}` in, `[{"passage": n, "fact": "..."}]` out.

    One call over all the passages, not one per passage — the same reason the
    reranker is listwise: a value in [2] that is qualified by a condition stated
    in [4] is only found by a reader holding both.
    """
    def message(inputs):
        passages = "\n\n".join(numbered(inputs["passages"], max_chars))
        return f"Question: {inputs['question']}\n\nPassages:\n\n{passages}"

    return _stage(EXTRACTOR_PROMPT, JsonListParser(),
                  provider, model or EXTRACTOR_MODEL, EXTRACTOR_MAX_TOKENS,
                  message, "extract")


# --------------------------------------------------------------------------- #
# Stage 4 — the generator, and the repair
# --------------------------------------------------------------------------- #

def describe_problem(problem):
    """One complaint as a line, whether it arrived structured or as a string."""
    if isinstance(problem, str):
        return problem
    kind = problem.get("type", "problem")
    return f"{kind}: {problem.get('detail', '')}".strip().rstrip(":")


def generator_chain(provider=GENERATOR_PROVIDER, model=None, max_chars=ANSWER_MAX_CHARS):
    """`{"question", "passages", "facts", "problems"}` in, a cited draft out.

    `problems` turns this into the repair call. Not a fresh start: the fault the
    verifier catches is usually local — an inequality the wrong way round, a
    refusal made with the evidence in hand — and rewriting from scratch loses
    the parts that were right.
    """
    def message(inputs):
        parts = [f"Question: {inputs['question']}",
                 f"Extracted facts:\n{as_evidence(inputs.get('facts') or [])}",
                 "Context passages:\n\n"
                 + "\n\n".join(numbered(inputs["passages"], max_chars))]
        problems = inputs.get("problems") or ()
        if problems:
            parts.append(REPAIR_INSTRUCTION.format(
                problems="\n".join(f"- {describe_problem(p)}" for p in problems)))
        return "\n\n".join(parts)

    # No parser: the generator's output is the answer, and there is no
    # scratchpad to strip — `GENERATOR_SYSTEM_PROMPT` deliberately has no
    # thinking block, because the extractor is that block promoted to its own
    # call and leaving one here would pay for the work twice.
    return _stage(GENERATOR_PROMPT, None,
                  provider, model or GENERATOR_MODEL, GENERATOR_MAX_TOKENS,
                  message, "generate")


# --------------------------------------------------------------------------- #
# Stage 5 — the verifier
# --------------------------------------------------------------------------- #

def verifier_chain(provider=VERIFIER_PROVIDER, model=None, max_chars=ANSWER_MAX_CHARS):
    """`{"question", "draft", "passages", "facts"}` in, `{"ok", "problems"}` out.

    The default on an unparseable reply is `{"ok": True}` — a verifier that
    could not be read must not block a draft on an opinion it does not have.
    """
    def message(inputs):
        passages = "\n\n".join(numbered(inputs["passages"], max_chars))
        return (f"Question: {inputs['question']}\n\n"
                f"Extracted facts:\n{as_evidence(inputs.get('facts') or [])}\n\n"
                f"Passages:\n\n{passages}\n\n"
                f"Draft answer:\n{inputs['draft']}")

    return _stage(VERIFIER_PROMPT, JsonObjectParser(default={"ok": True}),
                  provider, model or VERIFIER_MODEL, VERIFIER_MAX_TOKENS,
                  message, "verify")


__all__ = ["router_chain", "extractor_chain", "generator_chain", "verifier_chain",
           "as_evidence", "describe_problem", "numbered"]
