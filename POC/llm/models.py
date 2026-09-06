"""One function that returns a LangChain chat model, and a cache in front of it.

    from llm.models import chat_model

    model = chat_model("groq")                       # the provider's chat model
    vision = chat_model("openai", kind="vision")     # its vision model
    hot = chat_model("groq", temperature=0.7, max_tokens=256)

This is what replaced `LLMClient`. The hand-rolled client existed to do four
things — pick a base URL, attach a key, retry the retryable statuses and dig the
content out of the response — and a LangChain integration does all four. What is
left is the part that was never about HTTP: which provider serves which stage,
and with what budget.

`init_chat_model` builds every provider here, including the OpenAI-compatible
gateways: an entry with a `base_url` is a ChatOpenAI (or ChatOllama) pointed
somewhere else, which is the supported way to reach OpenRouter, LiteLLM,
DeepSeek and GLM.

Models are cached by their full configuration. A `BaseChatModel` is a
long-lived, thread-safe object holding an HTTP client with its own pool, so two
stages on the same provider and budget should share one — which is what the
`Clients` class did with connection pools, done here without the async context
manager, because a chat model has nothing to close.
"""

import functools
import os

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda

from .constants import (DEFAULT_PROVIDER, INFER_MODEL, MAX_RETRIES, PARSE_MODEL, PROVIDERS,
                        REQUEST_TIMEOUT_SECONDS)


class LLMError(RuntimeError):
    """A model could not be built, or a call returned something unusable.

    Kept as the package's own exception rather than letting provider SDK errors
    escape: every caller in this repo catches it to decide whether a failed
    stage costs quality or the whole answer, and that decision should not have
    to enumerate five vendors' exception types.
    """


def resolve_provider(name):
    try:
        return PROVIDERS[name]
    except KeyError:
        raise LLMError(f"Unknown provider {name!r}. Known: {', '.join(PROVIDERS)}") from None


def api_key(config):
    names = config["key_env"]
    if not names:
        return "local"                      # Ollama and friends ignore the header
    for name in names:
        key = os.getenv(name)
        if key:
            return key
    raise LLMError(f"No API key found. Set one of: {', '.join(names)}")


def model_name(provider, kind="chat", requested=None, override=None):
    """Which model id this provider should be asked for.

    The precedence is caller, then the stage's environment override, then the
    provider's registry default — so `EXTRACTOR_MODEL=...` changes one stage and
    `LLM_INFER_MODEL=...` changes every provider's chat default, without either
    having to know about the other.
    """
    config = resolve_provider(provider)
    name = (requested or override
            or (PARSE_MODEL if kind == "vision" else INFER_MODEL)
            or config.get(f"{kind}_model"))
    if not name:
        raise LLMError(f"Provider {provider!r} has no {kind} model; pass model=...")
    return name


# Rate limiting is per provider, not per model: the cap that binds is the
# account's, and two stages on the same key share it whether or not they share a
# model. One limiter per provider, made once, is what makes that true.
@functools.lru_cache(maxsize=None)
def _rate_limiter(provider):
    rate = resolve_provider(provider).get("requests_per_second")
    if not rate:
        return None
    return InMemoryRateLimiter(requests_per_second=rate,
                               check_every_n_seconds=0.1,
                               max_bucket_size=max(1.0, rate))


def _kwargs(init, config, *, key, timeout, temperature, max_tokens, limiter):
    """Canonical settings, translated into the names one integration accepts.

    The four integrations disagree about three of these — ChatOllama has
    `num_predict` where the others have `max_tokens` and takes its timeout
    through `client_kwargs`, and ChatGoogleGenerativeAI has no base URL to set —
    so the translation is explicit rather than a dict that is passed through and
    hoped for. Passing an unknown keyword to a pydantic model is an error at
    construction, which is the right time to find out, but not a message anyone
    should have to read twice.
    """
    common = {"temperature": temperature, "rate_limiter": limiter,
              # Nothing in this repo streams: every call is awaited whole and
              # then parsed. Saying so lets the integrations that would
              # otherwise stream by default (and lose `usage_metadata` doing it)
              # take the simpler path.
              "disable_streaming": True}
    base_url = config.get("base_url")

    if init == "ollama":
        return {**common, "base_url": base_url, "num_predict": max_tokens,
                "client_kwargs": {"timeout": timeout}}
    if init == "google_genai":
        # No base_url: this is the first-party API, not an OpenAI-dialect proxy.
        return {**common, "api_key": key, "max_tokens": max_tokens,
                "timeout": timeout, "max_retries": MAX_RETRIES}

    # openai and groq take the same names, and groq only needs base_url when it
    # is being pointed at something that is not groq.
    kwargs = {**common, "api_key": key, "max_tokens": max_tokens,
              "timeout": timeout, "max_retries": MAX_RETRIES}
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


@functools.lru_cache(maxsize=None)
def chat_model(provider=None, kind="chat", model=None, temperature=0.0,
               max_tokens=None, timeout=None, model_override=None):
    """A configured `BaseChatModel`. Cached, so callers need not hold one.

    `kind` is "chat" or "vision", and only selects the registry default when
    `model` is not given. `model_override` is the stage-level environment
    variable — ROUTER_MODEL, EXTRACTOR_MODEL and friends — kept separate from
    `model` so a caller passing an explicit model still wins over the
    environment.
    """
    provider = provider or DEFAULT_PROVIDER
    config = resolve_provider(provider)
    name = model_name(provider, kind, model, model_override)
    # A self-hosted model answers in minutes where an API answers in seconds, so
    # the wait is a property of the provider, not of the caller.
    timeout = timeout or config.get("timeout") or REQUEST_TIMEOUT_SECONDS

    try:
        return init_chat_model(
            name, model_provider=config["init"],
            **_kwargs(config["init"], config, key=api_key(config), timeout=timeout,
                      temperature=temperature, max_tokens=max_tokens,
                      limiter=_rate_limiter(provider)))
    except LLMError:
        raise
    except Exception as error:                  # noqa: BLE001 — integrations raise broadly
        raise LLMError(f"cannot build {provider}/{name}: {error}") from error


def vision_model(provider=None, **kwargs):
    """The provider's vision model, or a clear failure if it has none."""
    provider = provider or DEFAULT_PROVIDER
    if resolve_provider(provider).get("vision_model") is None and "model" not in kwargs:
        raise LLMError(f"Provider {provider!r} has no vision model; "
                       f"use --provider openai, or pass model=...")
    return chat_model(provider, kind="vision", **kwargs)


def text_of(message):
    """The assistant's text out of an `AIMessage`, or a useful failure.

    A reasoning model that spends its whole budget thinking returns an empty
    `content` with the reasoning in `additional_kwargs`. That is a token budget
    problem, not a model failure, and the difference is worth a sentence: the
    fix is to raise `max_tokens`, and nothing about the prompt needs touching.
    """
    content = message.content
    if isinstance(content, list):
        # Some integrations return content blocks rather than a string.
        content = "".join(block.get("text", "") for block in content
                          if isinstance(block, dict) and block.get("type") == "text")
    if content and content.strip():
        return content.strip()

    extra = getattr(message, "additional_kwargs", None) or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    if reasoning:
        raise LLMError(
            f"Model returned {len(str(reasoning))} characters of reasoning and no answer — "
            "raise max_tokens for this stage")
    raise LLMError("Provider returned an empty completion")


def available(provider=None):
    """Model ids the provider currently serves — the first thing to check on 400s.

    Not every integration exposes a listing, and the ones that do reach it
    differently, so this asks the underlying SDK client and says plainly when
    there is nothing to ask.
    """
    provider = provider or DEFAULT_PROVIDER
    model = chat_model(provider)
    for attribute in ("client", "async_client", "_client"):
        client = getattr(model, attribute, None)
        listing = getattr(getattr(client, "_client", client), "models", None)
        if listing is None or not hasattr(listing, "list"):
            continue
        try:
            return [entry.id for entry in listing.list()]
        except Exception as error:              # noqa: BLE001 — SDKs raise broadly
            raise LLMError(f"cannot list models for {provider}: {error}") from error
    raise LLMError(f"{provider} exposes no model listing through its LangChain integration")


# --------------------------------------------------------------------------- #
# Every failure a caller is expected to handle, as one exception
# --------------------------------------------------------------------------- #

# Exceptions that mean this repo has a bug, not that the provider does. They are
# let through untranslated, because "the model failed" is the wrong sentence for
# a misspelled key in a prompt-builder, and a stage that degrades gracefully on
# a NameError hides it for as long as the degradation looks plausible.
OUR_BUGS = (TypeError, AttributeError, KeyError, IndexError, NameError, ValueError)


def _translate(error):
    """A provider or parser failure as `LLMError`; anything else unchanged.

    Every caller in this repo catches `LLMError` to decide whether a failed
    stage costs quality or the whole answer — the router falls back to unrouted
    retrieval, the extractor to the passages alone, the reranker to fused order.
    That contract predates LangChain and has to survive it, and it would not on
    its own: each integration raises its own vendor's exceptions, so a groq rate
    limit and a Gemini quota error arrive as two unrelated types and neither is
    the one the caller is watching for.
    """
    if isinstance(error, LLMError):
        return error
    if isinstance(error, OutputParserException):
        return LLMError(f"could not read the model's reply: {str(error)[:300]}")
    if isinstance(error, OUR_BUGS):
        return error
    return LLMError(f"{type(error).__name__}: {str(error)[:300]}")


def guarded(chain, name=None):
    """`chain`, with provider and parser failures arriving as `LLMError`.

    Wrapping rather than translating in place, because the failure can come from
    any step — the rate limiter, the HTTP call, the parser — and a `Runnable`
    that owns the whole chain is the only place that sees all of them.
    """
    def run(inputs, config=None):
        try:
            return chain.invoke(inputs, config)
        except Exception as error:              # noqa: BLE001 — translated, not swallowed
            raise _translate(error) from error

    async def arun(inputs, config=None):
        try:
            return await chain.ainvoke(inputs, config)
        except Exception as error:              # noqa: BLE001 — translated, not swallowed
            raise _translate(error) from error

    return RunnableLambda(run, afunc=arun, name=name or getattr(chain, "name", None))
