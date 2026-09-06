"""The three calls the corpus needs, each as an LCEL chain.

    describe_figure()   a figure image plus its page context, in; a factual
                        description for the retrieval index, out.
    rerank_passages()   a question plus ~20 retrieved candidates, in; the same
                        candidates ordered by relevance, out.
    answer_question()   a question plus the surviving passages, in; a grounded
                        answer, out.

Each is `prompt | model | parser`, which is the whole reason for the rewrite:
the retry policy, the OpenAI dialect and the shape of the response belong to the
integration, and what is left in this file is the prompt, the budget and how the
reply is read. All three are deterministic by default (temperature 0), because a
description that changes between runs quietly invalidates an index, and a
ranking or an answer that changes between runs cannot be evaluated.

Every chain is a `Runnable`, so the ordinary LangChain verbs work on them —
`ainvoke` for one, `abatch` for a corpus, `with_retry` for a flaky provider —
and none of that had to be written here.

    from llm.chains import answer_question
    answer = await answer_question("How does detection probability vary with range?",
                                   passages)

## Passages are never templated

The passage text carries markdown tables and LaTeX, which is to say it carries
braces. A `ChatPromptTemplate` treats those as variables and fails on the first
table it sees, so the prompts here place a `MessagesPlaceholder` and the human
message is built as a value rather than formatted into a template. The system
prompt, which is ours and has no braces, stays a template.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from .constants import (DEFAULT_PROVIDER, INFER_MAX_TOKENS, INFER_SYSTEM_PROMPT,
                        INFER_TEMPERATURE, PARSE_MAX_TOKENS, PARSE_SYSTEM_PROMPT,
                        PARSE_TEMPERATURE, RERANK_MAX_CHARS, RERANK_MAX_TOKENS, RERANK_MODEL,
                        RERANK_SYSTEM_PROMPT, RERANK_TEMPERATURE)
from .images import encode_image
from .models import LLMError, chat_model, guarded, resolve_provider, text_of, vision_model
from .parsers import AnswerParser, RankingParser


def numbered(passages, max_chars=None):
    """Passages as `[n] label\\ntext` blocks, numbered from 1.

    The same numbering runs through every stage that sees a passage: the
    reranker sorts by it, the extractor attributes to it, the generator cites
    it, the verifier checks against it. A passage that is [3] to the reranker
    must be [3] to the reader, which is why there is one of these and everything
    imports it.

    A passage is a dict carrying `text` and an optional `label` or `source`, or
    a plain string.
    """
    blocks = []
    for number, passage in enumerate(passages, start=1):
        if isinstance(passage, dict):
            label = passage.get("label") or passage.get("source") or passage.get("id") or ""
            text = passage.get("text", "")
        else:
            label, text = "", str(passage)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        blocks.append(f"[{number}]{' ' + str(label) if label else ''}\n{text}")
    return blocks


def prompt_for(system):
    """A system prompt, then whatever messages the caller built.

    The placeholder is what keeps retrieved text out of the template engine —
    see the note at the top of this module. The system prompt is passed as a
    `SystemMessage` rather than a `("system", ...)` template for the same
    reason, and it is not a hypothetical one: `RERANK_SYSTEM_PROMPT` ends by
    showing the model the array it should return, braces and all, which a
    template reads as a variable named `"id"`.
    """
    return ChatPromptTemplate.from_messages(
        [SystemMessage(content=system), MessagesPlaceholder("messages")])


# --------------------------------------------------------------------------- #
# Parsing a figure
# --------------------------------------------------------------------------- #

PARSE_PROMPT = prompt_for(PARSE_SYSTEM_PROMPT)


def parse_messages(image_uri, caption=None, title=None, heading=None, context=None):
    """A figure, plus the page context that tells the model what it is looking at."""
    lines = []
    if title:
        lines.append(f"Paper: {title}")
    if heading:
        lines.append(f"Section: {heading}")
    if caption:
        lines.append(f"Caption: {caption}")
    if context:
        lines.append(f"Surrounding text: {context}")
    lines.append("Describe the figure.")

    return [HumanMessage(content=[
        {"type": "text", "text": "\n".join(lines)},
        {"type": "image_url", "image_url": {"url": image_uri}},
    ])]


def _unreadable(description):
    """"" for a figure the model could not read — the prompt's own escape hatch."""
    return "" if description.strip().upper().startswith("UNREADABLE") else description


def describe_figure_chain(provider=None, model=None, temperature=PARSE_TEMPERATURE,
                          max_tokens=None):
    """image_uri and page context in, an indexable description out."""
    provider = provider or DEFAULT_PROVIDER
    budget = max_tokens or resolve_provider(provider).get("parse_max_tokens") or PARSE_MAX_TOKENS
    llm = vision_model(provider, model=model, temperature=temperature, max_tokens=budget)
    chain = (RunnableLambda(lambda inputs: {"messages": parse_messages(**inputs)})
            | PARSE_PROMPT
            | llm
            | RunnableLambda(text_of)
            | RunnableLambda(_unreadable))
    return guarded(chain, "describe_figure")


async def describe_figure(image, *, caption=None, title=None, heading=None, context=None,
                          provider=None, model=None, **kwargs):
    """Describe a figure for the retrieval index. Returns "" if unreadable."""
    chain = describe_figure_chain(provider, model, **kwargs)
    return await chain.ainvoke({"image_uri": encode_image(image), "caption": caption,
                                "title": title, "heading": heading, "context": context})


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #

RERANK_PROMPT = prompt_for(RERANK_SYSTEM_PROMPT)


def rerank_messages(question, passages, max_chars=RERANK_MAX_CHARS):
    """The numbered passages, truncated — the model is scoring, not reading."""
    context = "\n\n".join(numbered(passages, max_chars))
    return [HumanMessage(content=f"Question: {question}\n\nPassages:\n\n{context}\n\n"
                                 f"Score all {len(passages)} passages.")]


def rerank_chain(provider=None, model=None, temperature=RERANK_TEMPERATURE,
                 max_tokens=None, max_chars=None):
    """question and passages in, `[(index, score)]` best first out.

    One call for the whole list, not one per passage: the model comparing
    candidates against each other is most of what makes a listwise reranker
    better than the retrieval score.
    """
    provider = provider or DEFAULT_PROVIDER
    config = resolve_provider(provider)
    # A provider whose rate limit binds before quality does says so in the
    # registry, the way `parse_max_tokens` does for the vision call.
    max_chars = max_chars or config.get("rerank_max_chars") or RERANK_MAX_CHARS
    budget = max_tokens or config.get("rerank_max_tokens") or RERANK_MAX_TOKENS
    llm = chat_model(provider, model=model, model_override=RERANK_MODEL,
                     temperature=temperature, max_tokens=budget)

    def to_messages(inputs):
        return {"messages": rerank_messages(inputs["question"], inputs["passages"], max_chars),
                # The parser cannot otherwise know how many passages were sent,
                # and therefore which ids the model invented.
                "count": len(inputs["passages"])}

    def parse(inputs):
        return RankingParser(count=inputs["count"]).parse(inputs["text"])

    chain = (RunnableLambda(to_messages)
             | {"text": RERANK_PROMPT | llm | RunnableLambda(text_of),
                "count": lambda inputs: inputs["count"]}
             | RunnableLambda(parse))
    return guarded(chain, "rerank")


async def rerank_passages(question, passages, *, top_n=None, provider=None, model=None,
                          **kwargs):
    """Order candidates by relevance. Returns [(index, score)], best first.

    Indices are positions in `passages`, so the caller keeps its own payloads
    and only takes the order back.
    """
    if not passages:
        return []
    ranked = await rerank_chain(provider, model, **kwargs).ainvoke(
        {"question": question, "passages": passages})
    return ranked[:top_n] if top_n else ranked


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #

INFER_PROMPT = prompt_for(INFER_SYSTEM_PROMPT)


def infer_messages(question, passages):
    """Numbered passages so the model can cite them as [1], [2], ..."""
    blocks = numbered(passages)
    context = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return [HumanMessage(content=f"Context passages:\n\n{context}\n\nQuestion: {question}")]


def answer_chain(provider=None, model=None, temperature=INFER_TEMPERATURE,
                 max_tokens=INFER_MAX_TOKENS):
    """question and passages in, a grounded answer citing [1]..[n] out."""
    llm = chat_model(provider or DEFAULT_PROVIDER, model=model,
                     temperature=temperature, max_tokens=max_tokens)
    chain = (RunnableLambda(lambda inputs: {"messages": infer_messages(inputs["question"],
                                                                      inputs["passages"])})
             | INFER_PROMPT
             | llm
             | RunnableLambda(text_of)
             | AnswerParser())
    return guarded(chain, "answer")


async def answer_question(question, passages=(), *, provider=None, model=None, **kwargs):
    """Answer a question from retrieved passages, scratchpad stripped."""
    return await answer_chain(provider, model, **kwargs).ainvoke(
        {"question": question, "passages": list(passages)})


__all__ = ["numbered", "prompt_for", "describe_figure", "describe_figure_chain",
           "parse_messages", "rerank_passages", "rerank_chain", "rerank_messages",
           "answer_question", "answer_chain", "infer_messages", "LLMError"]
