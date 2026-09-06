"""The OpenAI Batch API, through the official SDK.

Batch trades latency for half price: requests go up as a JSONL file, the
provider works through them within a completion window, and the results come
back as another JSONL. For figure description — thousands of independent calls
with nobody waiting — that is the right shape.

This is the one call in the repo that is not a chat completion and therefore not
a LangChain chain: LangChain models a conversation, and a 24-hour queue of ten
thousand of them is a different object with a different lifecycle. What it is
now is the `openai` SDK — which `langchain-openai` already depends on, so this
costs no new dependency — instead of four hand-rolled httpx calls.

The file format is still built here, because a batch line is a request that
LangChain would otherwise send, and it has to be written down rather than sent.
`request_line` takes LangChain messages and converts them, so the prompt in a
batch and the prompt in a live call cannot drift apart.
"""

import json

from langchain_core.messages import convert_to_openai_messages

from .models import LLMError, api_key, resolve_provider

CHAT_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"

# Terminal states — anything else means the batch is still moving.
DONE_STATES = frozenset({"completed", "failed", "expired", "cancelled"})


class BatchAPI:
    """The batch endpoints of one provider. Same key and base URL as its chat model.

        async with BatchAPI("openai") as api:
            file_id = await api.upload("requests.jsonl")
            batch = await api.create(file_id)
    """

    def __init__(self, provider="openai", timeout=600.0):
        from openai import AsyncOpenAI

        config = resolve_provider(provider)
        self.provider = provider
        # `base_url` is set for the OpenAI-compatible gateways and absent for
        # OpenAI itself, where the SDK's own default is correct.
        self._client = AsyncOpenAI(api_key=api_key(config), timeout=timeout,
                                   **({"base_url": config["base_url"]}
                                      if config.get("base_url") else {}))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self._client.close()

    async def upload(self, path):
        """Upload a JSONL of requests; returns the file id."""
        with open(path, "rb") as handle:
            uploaded = await self._call(self._client.files.create(file=handle, purpose="batch"),
                                        "upload")
        return uploaded.id

    async def create(self, input_file_id, window=COMPLETION_WINDOW):
        batch = await self._call(self._client.batches.create(
            input_file_id=input_file_id, endpoint=CHAT_ENDPOINT, completion_window=window),
            "create batch")
        return batch.model_dump()

    async def retrieve(self, batch_id):
        batch = await self._call(self._client.batches.retrieve(batch_id), "retrieve batch")
        return batch.model_dump()

    async def cancel(self, batch_id):
        batch = await self._call(self._client.batches.cancel(batch_id), "cancel")
        return batch.model_dump()

    async def download(self, file_id):
        """Fetch a result file. Returns the decoded JSONL text."""
        content = await self._call(self._client.files.content(file_id), f"download {file_id}")
        return content.text

    @staticmethod
    async def _call(awaitable, what):
        """One SDK call, with its errors turned into this package's own.

        Every caller in `enrich.py` catches `LLMError` to decide whether a phase
        can continue, and that decision should not have to enumerate the SDK's
        exception hierarchy.
        """
        try:
            return await awaitable
        except Exception as error:              # noqa: BLE001 — the SDK raises broadly
            raise LLMError(f"{what}: {type(error).__name__} — {str(error)[:400]}") from error


def request_line(custom_id, model, messages, temperature, max_tokens):
    """One line of the batch input file.

    `messages` are LangChain messages — the same ones `llm.chains` would send —
    converted here to the wire format the file needs.
    """
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": CHAT_ENDPOINT,
        "body": {"model": model, "messages": convert_to_openai_messages(messages),
                 "temperature": temperature, "max_tokens": max_tokens},
    }, ensure_ascii=False)


def parse_result_line(line):
    """(custom_id, content, error) from one line of the batch output file."""
    record = json.loads(line)
    custom_id = record.get("custom_id", "")
    if record.get("error"):
        return custom_id, None, str(record["error"])[:300]
    response = record.get("response") or {}
    if response.get("status_code") != 200:
        return custom_id, None, f"HTTP {response.get('status_code')}: {str(response.get('body'))[:200]}"
    try:
        content = response["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return custom_id, None, f"unexpected body: {str(response.get('body'))[:200]}"
    return custom_id, (content or "").strip(), None
