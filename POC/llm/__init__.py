"""LLM access for the RAG pipeline, on LangChain.

    describe_figure   figure image + page context -> description for the index
    rerank_passages   question + retrieved candidates -> the candidates, in order
    answer_question   question + retrieved passages -> grounded answer

Each is an LCEL chain in `chains.py`; `models.py` builds the `BaseChatModel`
behind it. There is no client object to open or close — a LangChain chat model
is long-lived and holds its own pooled HTTP client, so the async context manager
the old gateway needed is gone, and with it the `async with` around every caller.

Calls go straight to a provider by default (`LLM_PROVIDER`, default "gemini").
Set it to "gateway" to route through the local LiteLLM proxy instead, or to
"ollama" for a local model — same chains, no container required.

    from llm import answer_question, chat_model

    answer = await answer_question(question, passages)      # the chain
    model = chat_model("groq", temperature=0)               # the raw model

The three short names `parse`, `rerank` and `infer` are kept as blocking
wrappers, because notebooks and one-off scripts call them.
"""

import asyncio

from .chains import (answer_chain, answer_question, describe_figure, describe_figure_chain,
                     infer_messages, numbered, parse_messages, rerank_chain, rerank_messages,
                     rerank_passages)
from .constants import DEFAULT_PROVIDER, PROVIDERS
from .images import encode_image
from .models import LLMError, available, chat_model, model_name, resolve_provider, vision_model

# The async names these had before the rewrite. Kept because `enrich.py`, the
# notebooks and the eval harness call them, and renaming a working call site is
# not what this change is for.
aparse = describe_figure
arerank = rerank_passages
ainfer = answer_question


def parse(image, **kwargs):
    """Blocking `describe_figure`, for notebooks and scripts."""
    return asyncio.run(describe_figure(image, **kwargs))


def rerank(question, passages, **kwargs):
    """Blocking `rerank_passages`, for notebooks and scripts."""
    return asyncio.run(rerank_passages(question, passages, **kwargs))


def infer(question, passages=(), **kwargs):
    """Blocking `answer_question`, for notebooks and scripts."""
    return asyncio.run(answer_question(question, passages, **kwargs))


__all__ = [
    # chains
    "describe_figure", "rerank_passages", "answer_question",
    "describe_figure_chain", "rerank_chain", "answer_chain",
    # models
    "chat_model", "vision_model", "model_name", "resolve_provider", "available",
    # message builders, shared with the agents
    "numbered", "parse_messages", "rerank_messages", "infer_messages", "encode_image",
    # blocking wrappers and the older async names
    "parse", "rerank", "infer", "aparse", "arerank", "ainfer",
    "LLMError", "PROVIDERS", "DEFAULT_PROVIDER",
]
