"""Configuration for LLM calls routed through the LiteLLM gateway.

Every call goes to the proxy defined in `docker_images/light_llm/`, never to a
provider directly. Provider choice, keys, fallbacks and caching live in that
gateway's `config.yml`; this package only names a model alias.
"""

import os

from dotenv import load_dotenv

# Keys live in POC/.env. load_dotenv() walks up from the working directory, so
# this resolves whether a script runs from POC/ or from Ingestion/.
load_dotenv()

# --- Providers -------------------------------------------------------------- #
# Every entry speaks the OpenAI chat-completions dialect, so one client covers
# all of them; only the base URL, the path prefix and the key differ.
#
# "gateway" routes through the local LiteLLM proxy, which buys caching, fallbacks
# and spend tracking at the cost of running three containers. The direct entries
# skip it — same code path, no Docker, no memory held by a VM.
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

PROVIDERS = {
    "gateway": {
        "base_url": LITELLM_BASE_URL,
        "chat_path": "/v1/chat/completions", "models_path": "/v1/models",
        "key_env": ("LITELLM_API_KEY", "LIGHT_LLM_API_KEY", "LITELLM_MASTER_KEY"),
        # Aliases resolved by config.yml, not provider model ids.
        "vision_model": "vision", "chat_model": "chat",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "chat_path": "/chat/completions", "models_path": "/models",
        "key_env": ("GEMINI_API_KEY",),
        "vision_model": "gemini-2.5-flash", "chat_model": "gemini-2.5-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "chat_path": "/v1/chat/completions", "models_path": "/v1/models",
        "key_env": ("OPENAI_API_KEY",),
        "vision_model": "gpt-4.1-mini", "chat_model": "gpt-4.1-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai",
        "chat_path": "/v1/chat/completions", "models_path": "/v1/models",
        "key_env": ("GROQ_API_KEY",),
        # Verified against /v1/models on 2026-09-05: the Llama 4 vision models are
        # gone from Groq's lineup; qwen3.8-27b is the one that accepts images.
        "vision_model": "qwen/qwen3.8-27b",
        "chat_model": "openai/gpt-oss-120b",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "chat_path": "/v1/chat/completions", "models_path": "/v1/models",
        "key_env": ("DEEPSEEK_API_KEY",),
        "vision_model": None,                 # no vision endpoint as of writing
        "chat_model": "deepseek-chat",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "chat_path": "/chat/completions", "models_path": "/models",
        "key_env": ("GLM_API_KEY",),
        # glm-5/glm-5.3-flash reject image content ("取值范围 ['text']"): text only.
        "vision_model": None, "chat_model": "glm-5.3-flash",
    },
    "ollama": {
        "base_url": OLLAMA_BASE_URL,
        "chat_path": "/v1/chat/completions", "models_path": "/v1/models",
        "key_env": (),                        # local server, no auth
        "vision_model": "qwen3-vl:4b", "chat_model": "qwen3-vl:4b",
        # Qwen3-VL always emits a reasoning block — `think: false` and
        # reasoning_effort are both ignored by this build — and the answer only
        # starts after it. At the default 300 the budget is spent thinking and
        # `content` comes back empty, so it needs room for both.
        "parse_max_tokens": 1600,
    },
}

# Direct by default: the gateway is opt-in, so no container needs to be running.
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Set these to override a provider's default model without touching the registry.
PARSE_MODEL = os.getenv("LLM_PARSE_MODEL")
INFER_MODEL = os.getenv("LLM_INFER_MODEL")

# --- Request behaviour ------------------------------------------------------ #
REQUEST_TIMEOUT_SECONDS = 120.0
CONNECT_TIMEOUT_SECONDS = 10.0
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_RETRY_BACKOFF_SECONDS = 60

PARSE_TEMPERATURE = 0.0        # descriptions feed an index; reruns should agree
PARSE_MAX_TOKENS = 300
INFER_TEMPERATURE = 0.0
INFER_MAX_TOKENS = 1024

# --- Image handling --------------------------------------------------------- #
MAX_IMAGE_EDGE = 1024          # px; figures are downscaled before upload
JPEG_QUALITY = 80
IMAGE_MIME_TYPE = "image/jpeg"

# --- Prompts ---------------------------------------------------------------- #
PARSE_SYSTEM_PROMPT = """\
You describe figures from research papers so they can be found by text search.

Write 2-4 factual sentences covering, where visible:
- what kind of figure it is (line plot, bar chart, diagram, schematic, photograph, screenshot)
- the quantities on each axis, including units, and the series or conditions compared
- the trend, comparison or relationship the figure demonstrates, with approximate values
- any labelled components, if it is a diagram rather than a plot

Rules:
- Describe only what is visible. Never infer results the figure does not show.
- Do not repeat the caption back; add what the caption does not say.
- No preamble. Do not begin with "This figure" or "The image shows".
- If the figure is unreadable or blank, reply exactly: UNREADABLE
"""

INFER_SYSTEM_PROMPT = """\
You answer questions using only the provided context passages from research papers.

Rules:
- Ground every claim in the passages. Cite the passages you used as [1], [2], ...
- A passage may be a figure or table description rather than prose; use it the same way.
- If the passages do not support an answer, say so plainly and stop. Do not fill the gap
  from prior knowledge.
- Be direct and specific. Prefer the paper's own numbers and terminology.
"""
