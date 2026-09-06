"""Which LangChain chat model runs each call, and what to ask it.

The registry below no longer describes an HTTP endpoint. It names a LangChain
integration package and the model ids to hand it, and `llm/models.py` turns an
entry into a `BaseChatModel`. That is the whole of the change: retries, backoff,
the OpenAI dialect and the response shape are the integration's problem now, so
what is left here is configuration and prompts.

Providers fall into two groups, and the split is not cosmetic:

    native      a first-party integration — ChatOpenAI, ChatGroq,
                ChatGoogleGenerativeAI, ChatOllama. `init_chat_model` builds
                these from the `init` string.
    compatible  an OpenAI-dialect endpoint with no integration of its own —
                OpenRouter, the LiteLLM proxy, DeepSeek, GLM. `ChatOpenAI` with
                a `base_url` is the supported way to reach those, and it is what
                the entries with `base_url` get.

Everything else — the budgets, the temperatures, the prompts — is carried over
unchanged, because it was measured against these models and none of it is made
truer or falser by the client that sends it.
"""

import os

from dotenv import load_dotenv

# Keys live in POC/.env. load_dotenv() walks up from the working directory, so
# this resolves whether a script runs from POC/ or from Ingestion/.
load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Providers -------------------------------------------------------------- #
# `init` is what `langchain.chat_models.init_chat_model` is given as its
# `model_provider`. `base_url` marks the OpenAI-compatible entries, which are
# built as ChatOpenAI pointed elsewhere rather than through init_chat_model —
# the same code path the integration itself documents for third-party gateways.
PROVIDERS = {
    "gateway": {
        "init": "openai",
        "base_url": f"{LITELLM_BASE_URL.rstrip('/')}/v1",
        "key_env": ("LITELLM_API_KEY", "LIGHT_LLM_API_KEY", "LITELLM_MASTER_KEY"),
        # Aliases resolved by the proxy's config.yml, not provider model ids.
        "vision_model": "vision", "chat_model": "chat",
    },
    "gemini": {
        # The first-party integration, not the OpenAI-compatibility shim the
        # httpx client had to use. It speaks Gemini's own API, which is where
        # the vision call is best supported.
        "init": "google_genai",
        "key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        # Was gemini-2.5-flash, which a new key can no longer call:
        #
        #     404 NOT_FOUND: This model models/gemini-2.5-flash is no longer
        #     available to new users. Please update your code to use
        #     models/gemini-3.6-flash
        #
        # An existing key keeps working, so this only bites on a fresh one —
        # which is the worst way for it to bite, because it reads as a broken
        # setup rather than a retired model. `gemini-3.6-flash` is Google's own
        # named replacement; 3.7 and 3.8 are also live, as is the floating
        # `gemini-flash-latest`, which is deliberately not the default here: a
        # model id that changes underneath a corpus is how an index quietly
        # stops matching the descriptions that built it.
        #
        # One behavioural difference worth knowing, and it is not small for this
        # repo: the 3.x flash models use fixed sampling and **ignore
        # temperature**. Everything here asks for 0.0 so that a description
        # feeding the index, a ranking and an answer are reproducible, and on
        # these models that request is dropped. Reruns will differ. Where that
        # matters — re-parsing figures, or an A/B whose two arms must be
        # comparable — use a provider that honours it.
        "vision_model": "gemini-3.6-flash", "chat_model": "gemini-3.6-flash",
    },
    "openai": {
        "init": "openai",
        "key_env": ("OPENAI_API_KEY",),
        "vision_model": "gpt-4.1-mini", "chat_model": "gpt-4.1-mini",
    },
    # An aggregator, not a lab: one key and one OpenAI-dialect endpoint in front
    # of most other providers' models. Useful here as the overflow valve — when
    # groq's free tier hits its daily cap, the same `openai/gpt-oss-120b` is
    # reachable through this without changing anything but the provider name.
    "openrouter": {
        "init": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": ("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"),
        # The chat default is a `:free` model, because an OpenRouter account with
        # no credits can call only those — everything else 402s.
        "vision_model": "openai/gpt-4.1-mini",
        "chat_model": "nvidia/nemotron-3-super-120b-a12b:free",
    },
    "groq": {
        "init": "groq",
        "key_env": ("GROQ_API_KEY",),
        # Verified against /v1/models on 2026-09-05: the Llama 4 vision models are
        # gone from Groq's lineup; qwen3.8-27b is the one that accepts images.
        "vision_model": "qwen/qwen3.8-27b",
        "chat_model": "openai/gpt-oss-120b",
        # The free tier allows 8,000 tokens per minute, counting the reasoning
        # gpt-oss-120b emits and the prompt together. 20 passages at the default
        # 1,200 chars asks for ~9,000 and comes back HTTP 413 — a plan limit, not
        # a retryable error, so the fix is to send less rather than to retry.
        #
        # 900 chars keeps a 20-candidate rerank near 4,700 tokens, and matters
        # because the median figure row is 858 chars — at 500 the reranker saw
        # the caption and half the description, and the description states the
        # trend last, which is the half that answers the question.
        "rerank_max_chars": 900,
        "rerank_max_tokens": 2000,
        # Requests per second allowed onto the wire, enforced by LangChain's
        # InMemoryRateLimiter rather than by catching 429s afterwards. Six calls
        # per question put two concurrent questions over the free tier's cap.
        "requests_per_second": 0.5,
    },
    "deepseek": {
        "init": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "key_env": ("DEEPSEEK_API_KEY",),
        "vision_model": None,                 # no vision endpoint as of writing
        "chat_model": "deepseek-chat",
    },
    "glm": {
        "init": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": ("GLM_API_KEY",),
        # glm-5/glm-5.3-flash reject image content ("取值范围 ['text']"): text only.
        "vision_model": None, "chat_model": "glm-5.3-flash",
    },
    "ollama": {
        # ChatOllama talks to the native /api endpoints, not /v1 — which is the
        # one behavioural gain from dropping the hand-rolled client here, since
        # the OpenAI shim is where ollama's own options (`keep_alive`, the
        # thinking toggles, native tool calls) get lost.
        "init": "ollama",
        "base_url": OLLAMA_BASE_URL,
        "key_env": (),                        # local server, no auth
        "vision_model": "qwen3-vl:4b", "chat_model": "llama3.1:8b",
        # A CPU host is bound by prompt *prefill*, not generation: it ingests
        # perhaps 30-60 tokens/s, so the reranker's 20 passages are most of the
        # wait before a single token comes back. Halving the per-passage budget
        # halves that wait, and 400 chars still carries a figure's caption and
        # the opening of its description.
        "rerank_max_chars": 400,
        # Llama emits no reasoning block, so the array starts immediately and
        # 800 tokens is ample — every token generated is ~0.15s on CPU.
        "rerank_max_tokens": 800,
        # A 20-candidate rerank on an 8B model without a GPU runs into minutes,
        # and the 120s default would abort it while the server was still working.
        "timeout": 900,
        # Qwen3-VL always emits a reasoning block and the answer only starts
        # after it. At the default 300 the budget is spent thinking and the
        # content comes back empty, so it needs room for both.
        "parse_max_tokens": 1600,
    },
}

# Direct by default: the gateway is opt-in, so no container needs to be running.
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Set these to override a provider's default model without touching the registry.
PARSE_MODEL = os.getenv("LLM_PARSE_MODEL")
INFER_MODEL = os.getenv("LLM_INFER_MODEL")
# Reranking is a cheaper judgement than answering — a small fast model is usually the
# right trade. Falls back to the provider's chat model when unset.
RERANK_MODEL = os.getenv("LLM_RERANK_MODEL")

# --- Request behaviour ------------------------------------------------------ #
# Retries are the integration's now: every LangChain chat model takes
# `max_retries` and backs off exponentially on the statuses worth retrying,
# which is the whole of what the hand-rolled `_backoff` did.
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 3

# How a chain gets JSON out of a model that was asked for it.
#
#   "parse"  prompt for JSON and parse the reply, salvaging a truncated one.
#            Works on every provider, including the free reasoning models that
#            wrap their JSON in prose and have no usable tool-calling.
#   "tools"  `with_structured_output`, which binds the schema as a tool and
#            lets the provider guarantee the shape. Better where it is
#            supported; silently unavailable on several models used here.
STRUCTURED_OUTPUT = os.getenv("LLM_STRUCTURED_OUTPUT", "parse")
STRUCTURED_OUTPUT_MODES = ("parse", "tools")

PARSE_TEMPERATURE = 0.0        # descriptions feed an index; reruns should agree
# Raised from 300 when PARSE_SYSTEM_PROMPT became an extraction prompt. It now asks
# for exhaustive transcription — every axis label, every numeric value, and whole
# tables in markdown — where it used to ask for "2-4 factual sentences". Under the
# old prompt descriptions averaged 741 characters (~185 tokens); under this one they
# routinely exceed 300 and would be truncated mid-table, which is silent: the row is
# indexed, just missing its tail.
PARSE_MAX_TOKENS = 1500
INFER_TEMPERATURE = 0.0
INFER_MAX_TOKENS = 1024
RERANK_TEMPERATURE = 0.0       # a ranking that moves between runs cannot be evaluated
# 20 candidates of {"id": n, "score": n} is ~250 tokens, so this is almost entirely
# headroom for reasoning models: measured against openai/gpt-oss-120b, which spends
# 4,000-4,800 characters thinking before it emits the first bracket. At 1200 the
# budget went on reasoning and the array arrived empty or cut off in 6 of 25 calls.
RERANK_MAX_TOKENS = 3000
# Per passage, at rerank time only. The judgement is "does this answer the question",
# which the opening of a chunk settles; sending all 2048 tokens of 20 candidates costs
# ~8x more for a verdict that rarely changes. A provider may lower this via
# `rerank_max_chars` when its rate limit binds before quality does.
RERANK_MAX_CHARS = 1200

# --- Image handling --------------------------------------------------------- #
MAX_IMAGE_EDGE = 1024          # px; figures are downscaled before upload
JPEG_QUALITY = 80
IMAGE_MIME_TYPE = "image/jpeg"

# --- Prompts ---------------------------------------------------------------- #
PARSE_SYSTEM_PROMPT = """\
You extract precise, structured data from figures and tables in research papers so they can be accurately queried for specific values.

Extract all visible information exhaustively:
- Identify the figure type (line plot, bar chart, diagram, schematic, photograph, screenshot).
- List exact axis labels, including units, and explicitly state all legend items or conditions compared.
- Transcribe specific numerical values, key data points (peaks, troughs, intersections), error margins, and statistical significance markers (e.g., p-values, asterisks).
- Transcribe verbatim any text or equations visible within the figure itself.
- If it is a table, transcribe the entire table using standard Markdown formatting without skipping any rows or columns.
- For structural diagrams, list all labelled components and their exact flow/relationships.

Rules:
- Prioritize raw data and exact text transcription over high-level trend summaries. 
- Describe only what is visible. Never infer results the figure does not show.
- Do not repeat the caption back; extract the internal contents of the image/table.
- No preamble. Do not begin with "This figure" or "The image shows".
- If the figure is unreadable or blank, reply exactly: UNREADABLE
"""

INFER_SYSTEM_PROMPT = """\
You answer questions using only the provided context passages from research papers.

Rules:
- Before providing your final answer, you must write a brief <thinking> block. Inside this block, explicitly quote or extract the exact numbers, conditional statements, or statistical thresholds from the passages that address the question.
- Ground every claim in the passages. Cite the passages you used as [1], [2], ... in your final answer.
- A passage may be a figure or table description rather than prose; treat its extracted data as fact.
- If the passages do not support an answer, say so plainly in your final answer and stop. Do not fill the gap from prior knowledge.
- Be direct and specific. Prefer the paper's own numbers and terminology.
"""

RERANK_SYSTEM_PROMPT = """\
You rank retrieved passages by how well they answer a question.

You are given a question and numbered passages from research papers. Score each passage
0-10 for how much it helps answer that specific question:

- 8-10  contains the answer, or a substantial part of it
- 5-7   directly about the question but stops short of answering it
- 2-4   same topic or paper, does not bear on the question
- 0-1   unrelated, or matched only on a shared word

Rules:
- Judge relevance to the question only. Length, fluency, and how established the paper
  looks are not evidence.
- A figure or table description is judged the same way as prose. A plot that shows the
  relationship asked about answers the question.
- Passages come from different papers and may contradict each other. Do not resolve the
  disagreement — score each on its own.
- A passage that merely repeats the question's wording without new information is a 2.

Reply with a JSON array and nothing else — no prose, no markdown fence. One object per
passage you were given, best first, every id exactly once:

[{"id": 3, "score": 9}, {"id": 1, "score": 6}, {"id": 2, "score": 1}]
"""
