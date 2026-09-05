"""Minimal OpenAI Batch API client.

Batch trades latency for half price: requests go up as a JSONL file, the
provider works through them within a completion window, and the results come
back as another JSONL. For figure description — thousands of independent calls
with nobody waiting — that is the right shape.

The protocol is four calls (upload, create, retrieve, download), so this wraps
them rather than pulling in an SDK. It works against api.openai.com directly or
through the LiteLLM proxy, which passes /v1/batches through.
"""

import json

import httpx

from .client import LLMClient, LLMError

FILES_PATH = "/v1/files"
BATCHES_PATH = "/v1/batches"
CHAT_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"

# Terminal states — anything else means the batch is still moving.
DONE_STATES = frozenset({"completed", "failed", "expired", "cancelled"})


class BatchAPI(LLMClient):
    """LLMClient plus the batch endpoints. Same auth, same base URL."""

    async def upload(self, path):
        """Upload a JSONL of requests; returns the file id."""
        with open(path, "rb") as handle:
            response = await self._client.post(
                FILES_PATH,
                files={"file": (path.rsplit("/", 1)[-1], handle, "application/jsonl")},
                data={"purpose": "batch"},
                timeout=httpx.Timeout(600.0, connect=30.0))
        return self._json(response, "upload")["id"]

    async def create(self, input_file_id, window=COMPLETION_WINDOW):
        response = await self._client.post(BATCHES_PATH, json={
            "input_file_id": input_file_id,
            "endpoint": CHAT_ENDPOINT,
            "completion_window": window})
        return self._json(response, "create batch")

    async def retrieve(self, batch_id):
        return self._json(await self._client.get(f"{BATCHES_PATH}/{batch_id}"), "retrieve batch")

    async def cancel(self, batch_id):
        return self._json(await self._client.post(f"{BATCHES_PATH}/{batch_id}/cancel"), "cancel")

    async def download(self, file_id):
        """Fetch a result file. Returns the decoded JSONL text."""
        response = await self._client.get(f"{FILES_PATH}/{file_id}/content",
                                          timeout=httpx.Timeout(600.0, connect=30.0))
        if response.status_code != 200:
            raise LLMError(f"download {file_id}: HTTP {response.status_code} — {response.text[:300]}")
        return response.text

    @staticmethod
    def _json(response, what):
        if response.status_code not in (200, 201):
            raise LLMError(f"{what}: HTTP {response.status_code} — {response.text[:400]}")
        return response.json()


def request_line(custom_id, model, messages, temperature, max_tokens):
    """One line of the batch input file."""
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": CHAT_ENDPOINT,
        "body": {"model": model, "messages": messages,
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
