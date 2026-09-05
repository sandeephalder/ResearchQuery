"""Two ways to call a model, both through the LiteLLM gateway.

    parse()  — document parsing: a figure image plus its page context, in;
               a factual description for the retrieval index, out.
    infer()  — query time: a question plus retrieved passages, in;
               a grounded answer, out.

They differ in more than their prompt: `parse` is a vision call made ~10k times
over a corpus, so it downscales images and defaults to a cheap alias; `infer` is
a text call made once per user question. Both are deterministic by default
(temperature 0), because a description that changes between runs quietly
invalidates an index and an answer that changes between runs cannot be evaluated.

Async is the primary form — enrichment fans out over thousands of figures. The
sync wrappers exist for notebooks and one-off calls.

    from llm import parse, infer

    description = parse("data/processed/pdf/images/2401.01872v2/img-0.png",
                        caption="Fig 1: Scatter plot of complete cases ...")
    answer = infer("How does detection probability vary with range?", passages)

Reuse one gateway when making many calls, so connections are pooled:

    async with LLMGateway() as gateway:
        results = await asyncio.gather(*(gateway.parse(p) for p in paths))
"""

import asyncio
import base64
import os

import httpx
import pymupdf

from .constants import (CONNECT_TIMEOUT_SECONDS, DEFAULT_PROVIDER, IMAGE_MIME_TYPE,
                        INFER_MAX_TOKENS, INFER_MODEL, INFER_SYSTEM_PROMPT, INFER_TEMPERATURE,
                        JPEG_QUALITY, MAX_IMAGE_EDGE, MAX_RETRIES, MAX_RETRY_BACKOFF_SECONDS,
                        PARSE_MAX_TOKENS, PARSE_MODEL, PARSE_SYSTEM_PROMPT, PARSE_TEMPERATURE,
                        PROVIDERS, REQUEST_TIMEOUT_SECONDS, RETRYABLE_STATUS_CODES)


class LLMError(RuntimeError):
    """A gateway call failed, or returned something unusable."""


def _resolve_provider(name):
    try:
        return PROVIDERS[name]
    except KeyError:
        raise LLMError(f"Unknown provider {name!r}. Known: {', '.join(PROVIDERS)}") from None


def _api_key(config):
    names = config["key_env"]
    if not names:
        return "local"                      # Ollama and friends ignore the header
    for name in names:
        key = os.getenv(name)
        if key:
            return key
    raise LLMError(f"No API key found. Set one of: {', '.join(names)}")


def encode_image(image, max_edge=MAX_IMAGE_EDGE, quality=JPEG_QUALITY):
    """Downscale an image to a JPEG data URI.

    Figures are rendered at 200 dpi, which is far more than a vision model reads
    and is billed by the tile. Halving until the long edge fits `max_edge` keeps
    axis labels legible while cutting the image token cost several-fold.
    """
    pixmap = pymupdf.Pixmap(image)        # accepts a path or the encoded bytes

    while max(pixmap.width, pixmap.height) > max_edge:
        pixmap.shrink(1)                      # halves both edges

    if pixmap.colorspace is None or pixmap.colorspace.n != 3:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)   # CMYK/greyscale -> RGB
    if pixmap.alpha:
        # A fifth of the rendered figures carry an alpha channel, and converting
        # colorspace does not drop it. JPEG has no alpha, so flatten explicitly.
        pixmap = pymupdf.Pixmap(pixmap, 0)

    payload = base64.b64encode(pixmap.tobytes("jpeg", jpg_quality=quality)).decode()
    return f"data:{IMAGE_MIME_TYPE};base64,{payload}"


def _parse_messages(image_uri, caption=None, title=None, heading=None, context=None):
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

    return [
        {"role": "system", "content": PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "\n".join(lines)},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ]},
    ]


def _infer_messages(question, passages):
    """Numbered passages so the model can cite them as [1], [2], ..."""
    blocks = []
    for number, passage in enumerate(passages, start=1):
        if isinstance(passage, dict):
            label = passage.get("source") or passage.get("id") or ""
            text = passage.get("text", "")
            blocks.append(f"[{number}]{' ' + label if label else ''}\n{text}")
        else:
            blocks.append(f"[{number}]\n{passage}")

    context = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return [
        {"role": "system", "content": INFER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{context}\n\nQuestion: {question}"},
    ]


class LLMClient:
    """A pooled connection to one provider — direct, or via the LiteLLM proxy.

    Every provider in the registry speaks the OpenAI dialect, so `provider` only
    selects a base URL, a path prefix, a key and the default models.
    """

    def __init__(self, provider=DEFAULT_PROVIDER, base_url=None, api_key=None,
                 timeout=REQUEST_TIMEOUT_SECONDS):
        self.provider = provider
        self.config = _resolve_provider(provider)
        self.base_url = (base_url or self.config["base_url"]).rstrip("/")
        self.chat_path = self.config["chat_path"]
        self.models_path = self.config["models_path"]
        self.api_key = api_key or _api_key(self.config)
        self.timeout = httpx.Timeout(timeout, connect=CONNECT_TIMEOUT_SECONDS)
        self._client = None

    def _model_for(self, requested, kind):
        model = requested or (PARSE_MODEL if kind == "vision" else INFER_MODEL) \
            or self.config[f"{kind}_model"]
        if model is None:
            raise LLMError(f"Provider {self.provider!r} has no {kind} model; pass model=...")
        return model

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"})
        return self

    async def __aexit__(self, *exc_info):
        await self._client.aclose()
        self._client = None

    async def models(self):
        """Aliases the gateway currently serves — the first thing to check on 400s."""
        try:
            response = await self._client.get(self.models_path)
            response.raise_for_status()
        except (httpx.TransportError, httpx.TimeoutException) as error:
            hint = ("Start it with: docker_images/light_llm/run.sh" if self.provider == "gateway"
                    else "ollama serve" if self.provider == "ollama" else "check your network")
            raise LLMError(f"Cannot reach {self.provider} at {self.base_url} "
                           f"({type(error).__name__}). {hint}") from error
        except httpx.HTTPStatusError as error:
            raise LLMError(f"{self.provider} rejected {self.models_path}: HTTP "
                           f"{error.response.status_code} — {error.response.text[:200]}") from error
        return [model["id"] for model in response.json().get("data", [])]

    async def chat(self, messages, model, **params):
        """One chat completion, retrying the statuses worth retrying."""
        if self._client is None:
            raise LLMError("LLMClient must be used as an async context manager")

        payload = {"model": model, "messages": messages, **params}
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.post(self.chat_path, json=payload)
                if response.status_code == 200:
                    return self._content(response.json())
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    await self._backoff(attempt, response)
                    continue
                raise LLMError(f"{model}: HTTP {response.status_code} — {response.text[:300]}")
            except (httpx.TransportError, httpx.TimeoutException) as error:
                last_error = error
                if attempt < MAX_RETRIES:
                    await self._backoff(attempt)
                    continue
                raise LLMError(f"{model}: {type(error).__name__} — {error}") from error
        raise LLMError(f"{model}: exhausted {MAX_RETRIES} attempts — {last_error}")

    @staticmethod
    def _content(body):
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"Unexpected gateway response: {str(body)[:300]}") from error
        if not content or not content.strip():
            # Reasoning models answer only after the thinking block. Spend the
            # whole budget thinking and `content` arrives empty, which is a token
            # limit problem, not a model failure — say so.
            reasoning = (body.get("choices") or [{}])[0].get("message", {}).get("reasoning")
            if reasoning:
                raise LLMError(
                    f"Model returned {len(reasoning)} characters of reasoning and no answer — "
                    "raise max_tokens (or parse_max_tokens for this provider)")
            raise LLMError("Provider returned an empty completion")
        return content.strip()

    @staticmethod
    async def _backoff(attempt, response=None):
        delay = 2 ** (attempt - 1)
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        await asyncio.sleep(min(delay, MAX_RETRY_BACKOFF_SECONDS))

    # -- the two entry points ------------------------------------------------ #

    async def parse(self, image, *, caption=None, title=None, heading=None, context=None,
                    model=None, temperature=PARSE_TEMPERATURE, max_tokens=None):
        """Describe a figure for the retrieval index. Returns "" if unreadable."""
        messages = _parse_messages(encode_image(image), caption, title, heading, context)
        budget = max_tokens or self.config.get("parse_max_tokens") or PARSE_MAX_TOKENS
        description = await self.chat(messages, self._model_for(model, "vision"),
                                      temperature=temperature, max_tokens=budget)
        return "" if description.strip().upper().startswith("UNREADABLE") else description

    async def infer(self, question, passages=(), *, model=None,
                    temperature=INFER_TEMPERATURE, max_tokens=INFER_MAX_TOKENS):
        """Answer a question from retrieved passages."""
        return await self.chat(_infer_messages(question, passages),
                               self._model_for(model, "chat"),
                               temperature=temperature, max_tokens=max_tokens)


# The class was proxy-only when it was written; the name survives as an alias.
LLMGateway = LLMClient


# --------------------------------------------------------------------------- #
# Convenience wrappers — one gateway per call. For bulk work open an LLMGateway
# once and reuse it instead.
# --------------------------------------------------------------------------- #

async def aparse(image, *, provider=DEFAULT_PROVIDER, **kwargs):
    async with LLMClient(provider) as client:
        return await client.parse(image, **kwargs)


async def ainfer(question, passages=(), *, provider=DEFAULT_PROVIDER, **kwargs):
    async with LLMClient(provider) as client:
        return await client.infer(question, passages, **kwargs)


def parse(image, **kwargs):
    """Blocking `aparse`, for notebooks and scripts."""
    return asyncio.run(aparse(image, **kwargs))


def infer(question, passages=(), **kwargs):
    """Blocking `ainfer`, for notebooks and scripts."""
    return asyncio.run(ainfer(question, passages, **kwargs))


if __name__ == "__main__":
    # Round-trip check against a running gateway:  uv run python -m llm.client [image]
    import sys

    async def _check():
        async with LLMGateway() as gateway:
            print(f"gateway   : {gateway.base_url}")
            print(f"aliases   : {', '.join(await gateway.models())}")

            answer = await gateway.infer(
                "What did the experiment measure?",
                [{"source": "demo", "text": "We measured detection probability against range."}])
            print(f"infer     : {answer[:160]}")

            if len(sys.argv) > 1:
                description = await gateway.parse(sys.argv[1], caption="Figure 1")
                print(f"parse     : {description[:300]}")
            else:
                print("parse     : skipped (pass an image path to test it)")

    try:
        asyncio.run(_check())
    except LLMError as error:
        sys.exit(f"FAILED: {error}")
