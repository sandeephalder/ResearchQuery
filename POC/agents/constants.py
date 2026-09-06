"""Knobs and prompts for the agent flow.

Retrieval constants live in `retrieval/constants.py` and indexing constants in
`Ingestion/constants.py`; both are imported from there, never restated. What is
here is only what the agents themselves need — the models each stage calls, the
budgets they are allowed, and the prompts that define what they are.

Every stage's model is a separate setting. That is not over-configuration: the
guardrail is a 86M classifier, the extractor and verifier are ordinary chat
calls, and the generator is the one place where answer quality is bought. Tying
them to one model would mean paying generator prices for a yes/no gate.
"""

import os

from llm.constants import DEFAULT_PROVIDER
from retrieval.constants import (ANSWER_MAX_CHARS, BM25_TOP_K,   # noqa: F401 (re-exported)
                                 POC_DIR,
                                 RERANK_TOP_N, VECTOR_TOP_K)

# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
# The flow's own provider, for router / extractor / generator / verifier.
AGENT_PROVIDER = os.getenv("AGENT_PROVIDER", "openrouter")

# Per stage, because the stages are not alike and neither is their pricing.
#
# The *gates* — the guardrail rail and the router — are short classification
# calls: a few hundred tokens in, a word or a small object out. OpenAI's nano
# models are the cheapest tokens either provider sells ($0.10/$0.40 per M for
# gpt-4.1-nano, against $0.15/$0.60 for gpt-oss-120b on groq), and they emit no
# reasoning block, so nothing is billed for thinking that nobody reads.
#
# The *reasoning-heavy* stages — extract, generate, verify — are where quality
# is bought, and where groq's gpt-oss-120b is 2.67x cheaper per token than
# gpt-4.1-mini. Its reasoning tokens are billed as output (measured at ~1,200 a
# call in `llm/constants.py`), which eats some of that margin, but not all of it.
#
# Each falls back to AGENT_PROVIDER, so setting that alone moves the whole flow.
#
# Three of these were hardcoded to "openrouter" while this comment claimed
# otherwise, which meant `AGENT_PROVIDER=groq` moved the router and left extract,
# generate and verify on a provider whose daily free allowance was spent — a 429
# from a provider nobody had selected, reported by a server whose /health said
# it was running on the other one.
ROUTER_PROVIDER = os.getenv("ROUTER_PROVIDER", AGENT_PROVIDER)
EXTRACTOR_PROVIDER = os.getenv("EXTRACTOR_PROVIDER", AGENT_PROVIDER)
GENERATOR_PROVIDER = os.getenv("GENERATOR_PROVIDER", AGENT_PROVIDER)
VERIFIER_PROVIDER = os.getenv("VERIFIER_PROVIDER", AGENT_PROVIDER)

# The guardrail is NeMo's, and configures its own model — see the rails section
# below. Everything here is the flow's own provider.

# --------------------------------------------------------------------------- #
# Per-request metrics
# --------------------------------------------------------------------------- #
# The RAG triad plus latency, computed from work the flow already did — the
# reranker's scores, the verifier's findings, and one local embedding pass. No
# judge, so no extra call per question. See `metrics.py` for what each number
# does and does not mean.
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "1").lower() not in ("0", "false", "no")
METRICS_LOG_PATH = os.getenv(
    "METRICS_LOG_PATH", os.path.join(POC_DIR, "logs", "requests.jsonl"))
# The question and answer text alongside the scores. On, because a score you
# cannot trace back to the question it came from is not much use for tuning —
# turn it off where the questions are sensitive.
METRICS_INCLUDE_TEXT = os.getenv("METRICS_INCLUDE_TEXT", "1").lower() not in (
    "0", "false", "no")

# --------------------------------------------------------------------------- #
# Stage 1 and 5 — NeMo Guardrails
# --------------------------------------------------------------------------- #
# The rails themselves — which flows run, in which order — live in
# `agents/rails/config.yml`, checked in and reviewable. What is here is only
# what changes between machines: which model runs them, and where Llama Guard
# is hosted.
RAILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rails")

# Two gates, in this order:
#
#   1. llama guard check input   llama-guard3 on a machine you host. Primary,
#      because it is free, local, and the content-safety classifier Meta trained
#      for exactly this.
#   2. self check input          a chat model, prompted in rails/prompts.yml.
#      Second, because Llama Guard classifies *content harm* and does not catch
#      prompt injection — "ignore all previous instructions" is not one of its
#      hazard categories, and this rail is what stops it.
#
# Neither is OpenAI. `engine: openai` in the rails config is the OpenAI
# *dialect*, not the vendor: groq speaks it, and so does ollama at /v1.
GUARDRAIL_ENGINE = os.getenv("GUARDRAIL_ENGINE", "openai")
# Which `llm.PROVIDERS` entry builds the rails' `main` model. NeMo takes a
# LangChain model directly — `LLMRails(config, llm=...)` — so the rail runs on
# the same registry, the same key resolution and the same retry policy as every
# other stage, instead of on a second client configured in YAML. The default is
# the local host, because `GUARDRAIL_MODEL` is a local model.
GUARDRAIL_PROVIDER = os.getenv("GUARDRAIL_PROVIDER", "ollama")
# The `main` model exists because NeMo wants one configured, not because a flow
# calls it: `self check input` is off by default. That rail parses the reply for
# a leading Yes or No, and every model available to run it locally or free
# reasons out loud first — which NeMo resolves by blocking. Measured: it refused
# "how does detection probability vary with range" on three different models.
#
# What that costs is injection detection, which is not one of Llama Guard's
# hazard categories. GUARDRAIL_SELF_CHECK=1 turns it back on, and it needs a
# model that answers immediately: gpt-oss-120b on groq and gpt-4.1-mini on
# OpenAI are both verified 4/4 on the probes.
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", "llama3.1:8b")
GUARDRAIL_BASE_URL = os.getenv("GUARDRAIL_BASE_URL", "")
GUARDRAIL_API_KEY = os.getenv("GUARDRAIL_API_KEY", "ollama")
GUARDRAIL_SELF_CHECK = os.getenv("GUARDRAIL_SELF_CHECK", "").lower() in ("1", "true", "yes")
SELF_CHECK_FLOW = "self check input"
# Defined below, once LLAMA_GUARD_BASE_URL exists: the guardrail's own model
# shares the Llama Guard host unless told otherwise. Local means local.

# The primary gate, and on by default. Set GUARDRAIL_LLAMA_GUARD=0 to run the
# second rail alone — which is also what happens automatically when the host is
# unreachable, so a machine that is switched off degrades to the free API rather
# than to no gate at all.
GUARDRAIL_LLAMA_GUARD = os.getenv("GUARDRAIL_LLAMA_GUARD", "1").lower() not in (
    "0", "false", "no", "")
LLAMA_GUARD_MODEL = os.getenv("LLAMA_GUARD_MODEL", "llama-guard3:1b")
# ollama's OpenAI-compatible endpoint, which is the /v1 under the server root.
_OLLAMA = (os.getenv("LLAMA_GUARD_BASE_URL")
           or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
LLAMA_GUARD_BASE_URL = _OLLAMA if _OLLAMA.endswith("/v1") else _OLLAMA + "/v1"
LLAMA_GUARD_FLOW = "llama guard check input"
GUARDRAIL_PARAMETERS = {"base_url": GUARDRAIL_BASE_URL or LLAMA_GUARD_BASE_URL,
                        "api_key": GUARDRAIL_API_KEY}
# ollama ignores the header, but the OpenAI-dialect client insists on one.
LLAMA_GUARD_API_KEY = os.getenv("LLAMA_GUARD_API_KEY", "ollama")

# A rail that errors is a gate that did not run. Failing open is the default
# because the corpus is public arXiv and a network blip should not read as a
# refusal — but it is announced on stderr, and one variable makes it strict.
GUARDRAIL_FAIL_CLOSED = os.getenv("GUARDRAIL_FAIL_CLOSED", "").lower() in ("1", "true", "yes")

REFUSAL_MESSAGE = (
    "I can't help with that one. This assistant answers questions about the "
    "indexed arXiv papers — try asking about a method, a result or a figure in "
    "the literature.")

OUT_OF_SCOPE_MESSAGE = (
    "Nothing in the indexed papers bears on that question. This corpus is a "
    "slice of arXiv, so it can only answer what those papers discuss.")

# --------------------------------------------------------------------------- #
# Stage 2 — router
# --------------------------------------------------------------------------- #
# What the router does with its answer:
#   "widen"   retrieve twice — once unrestricted, once filtered to the named
#             kind — and send the union to the reranker. A wrong guess costs a
#             few extra candidates; it cannot hide the gold row.
#   "filter"  a hard `kind=` filter, as the flow diagram draws it. Cheaper by
#             one local query, and a wrong guess makes the answer unreachable.
#   "off"     ignore the route; retrieve as `retrieval.Pipeline` does.
# "widen" is the default because the measured failure mode of this corpus is
# modality (image queries score 2.41 against text's 2.73, with four times the
# abstentions), and a hard filter on a guess makes exactly that failure worse.
ROUTE_MODE = os.getenv("ROUTE_MODE", "widen")
ROUTE_MODES = ("widen", "filter", "off")
ROUTE_KINDS = ("text", "figure", "table", "any")

# The router's reply is one small JSON object, so the gate wants the cheapest
# model that can produce it rather than the best one available.
ROUTER_MODEL = os.getenv("ROUTER_MODEL")   # None -> the provider's chat model
# 1,500 for a 200-token answer, because the budget is not the answer's size —
# it is the answer's size plus however long the model thinks first. At 200 a
# reasoning model returns its thinking and no JSON, `json_object` salvages
# nothing, and the run continues unrouted with a warning that reads like a
# provider fault. Measured on nemotron-3.5-lightning: the reply was still
# mid-preamble at 800.
ROUTER_MAX_TOKENS = 1500
# How deep the extra kind-filtered pass goes in "widen" mode. Both legs are
# local and sub-millisecond, so this is paid in reranker input, not in latency.
ROUTE_WIDEN_K = int(os.getenv("ROUTE_WIDEN_K", "6"))

ROUTER_SYSTEM_PROMPT = """\
You route a question to the right kind of content in a corpus of research papers.

The corpus holds three kinds of row, indexed separately:
- text    prose from the paper body: methods, definitions, argument, discussion
- figure  an extracted description of a plot, chart, diagram or photograph, including
          transcribed axis labels, legend items and data points
- table   an extracted transcription of a table, row by row

Decide which kind is most likely to carry the answer, and how much the question turns on
literal tokens (a symbol, a unit, a model name, a dataset name, an exact identifier) rather
than on meaning.

Reply with a JSON object and nothing else — no prose, no markdown fence:

{"kind": "figure", "confidence": 0.8, "lexical": 0.3, "reason": "asks how a quantity varies, which a plot shows"}

- "kind" is one of "text", "figure", "table", "any". Use "any" when the question does not
  favour one, which is the honest answer for most questions.
- "confidence" is 0.0-1.0, how sure you are of the kind.
- "lexical" is 0.0-1.0, how much the question hinges on exact tokens. A question naming
  CIFAR-100, SNR or mmol/L is high. A conceptual question is low.
- "reason" is one short clause.
"""

# --------------------------------------------------------------------------- #
# Scope — decided after retrieval, not before
# --------------------------------------------------------------------------- #
# A guardrail that runs before retrieval cannot know what the corpus holds, so
# it cannot judge whether a question is in scope; asking it to guess is how a
# legitimate arXiv question gets refused. The reranker already answers the
# question directly — it has read the question against the passages — so scope
# is its top score against a floor.
#
# The two rerankers score on unrelated scales, so the floor is per-reranker:
#   llm             the 0-10 rubric in RERANK_SYSTEM_PROMPT, where 2-4 is
#                   "same topic, does not bear on the question". A best
#                   candidate at 4 has nothing to answer with.
#   cross-encoder   bge-reranker-v2-m3 through a sigmoid, so 0-1. Measured on
#                   three probes: an on-topic question's best candidate scored
#                   0.976, "who won the 2018 world cup" 0.425, "a good recipe
#                   for carbonara" 0.0015.
# Three probes is calibration, not tuning. `evals/answer_eval.py` is what would
# settle either number.
SCOPE_FLOOR = {"llm": 4.0, "cross-encoder": 0.5, "none": None}

# --------------------------------------------------------------------------- #
# Stage 3 — extractor
# --------------------------------------------------------------------------- #
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL")
# 3,000 for the same reason RERANK_MAX_TOKENS is: a reasoning model spends the
# start of its budget thinking, and an array that never opens parses to nothing
# — which this stage reports as "no evidence in the passages", the one answer
# that is indistinguishable from a real result. Measured on
# nemotron-3-super-120b over five passages: 1,873 completion tokens for a
# 1,029-character reply, so 1,500 truncated it every time and did so silently.
EXTRACTOR_MAX_TOKENS = 3000
# The extractor reads for values, not for prose, but a truncated table row is a
# wrong number rather than a missing one — so it gets the same budget the
# answer model does.
EXTRACTOR_MAX_CHARS = 4000

EXTRACTOR_SYSTEM_PROMPT = """\
You isolate the hard evidence in retrieved passages, so that a later step can answer a question
using only what the passages actually say.

You are given a question and numbered passages from research papers. Some passages are prose;
others are extracted descriptions of a figure or a transcription of a table, and their contents
are facts of the same standing as prose.

Pull out every item in the passages that bears on the question:
- numeric values, with their units and their symbol as the paper writes it
- statistical bounds: confidence intervals, error margins, p-values, significance thresholds
- the direction and condition of a relationship ("decreases as rho approaches rho_max",
  "only when the SNR exceeds 10 dB")
- table rows, kept as rows
- definitions and named methods the question turns on

Rules:
- Quote the paper's own numbers and symbols. Do not round, convert, or restate them.
- Record the condition attached to a value. A number without its condition is a wrong number.
- Attribute every item to the passage it came from, by that passage's number.
- Extract only what is present. If a passage does not bear on the question, skip it.
- If no passage carries anything relevant, reply with an empty array. That is a real answer,
  and a later step is expecting it.

Reply with a JSON array and nothing else — no prose, no markdown fence:

[{"passage": 2, "fact": "p_D,max = 0.9 for rho <= rho_max, 0 otherwise"},
 {"passage": 3, "fact": "rho_max = 600 m in scenario A, 1000 m in scenario B"}]
"""

# --------------------------------------------------------------------------- #
# Stage 4 — generator
# --------------------------------------------------------------------------- #
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL")
GENERATOR_MAX_TOKENS = 1024

# Deliberately not INFER_SYSTEM_PROMPT. That prompt opens with a <thinking>
# block asking the model to "quote or extract the exact numbers, conditional
# statements, or statistical thresholds" — which is stage 3, done again at
# generator prices. The extractor having already run is the whole reason this
# prompt can start at the answer.
GENERATOR_SYSTEM_PROMPT = """\
You answer questions about research papers, from evidence that has already been extracted for you.

You are given a question, a list of extracted facts each tagged with the passage it came from,
and the numbered passages themselves.

Rules:
- Build the answer from the extracted facts. They are what a previous step found in the passages,
  and they are where the numbers come from.
- The passages are there so you can cite accurately and catch what the extraction missed. If a
  passage plainly answers the question and no fact captured it, use the passage and say so.
- Cite the passages you used as [1], [2], ... Every claim carrying a number, a bound or a
  direction needs a citation.
- Keep the paper's own numbers, units, symbols and terminology. Never round or convert.
- Carry the condition with the claim. "p_D falls with range" is wrong if the paper said it falls
  only beyond rho_min.
- If the facts and passages do not support an answer, say so plainly and stop. Do not fill the
  gap from prior knowledge. But do not refuse merely because the answer is partial — give what
  the passages support and name what is missing.
- Answer directly. No preamble, no restatement of the question.
"""

# The repair prompt. The verifier's complaint is appended to a second call
# rather than started fresh, because the draft is usually mostly right — the
# failure the verifier catches is a flipped inequality or a refusal that had
# the evidence in front of it, not a wholesale miss.
REPAIR_INSTRUCTION = """\
A checker read your draft against the passages and found the problems listed below. Rewrite the
answer so that none of them remain. Change only what the complaints require; keep everything
the checker did not object to, including the citations.

Problems found:
{problems}
"""

# --------------------------------------------------------------------------- #
# Stage 5 — verifier
# --------------------------------------------------------------------------- #
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL")
# Raised from 800 as insurance, not from a measurement: an unparseable reply
# leaves `json_object` returning its {"ok": True} default, so this stage fails
# by approving. On a reasoning model that is one truncation away.
VERIFIER_MAX_TOKENS = 2000
# One. A verifier that can send a draft back indefinitely is a loop with a
# model's opinion as its exit condition, and the second repair has nothing new
# to work from — the same passages, the same facts, one more complaint.
MAX_REPAIRS = int(os.getenv("MAX_REPAIRS", "1"))

VERIFIER_SYSTEM_PROMPT = """\
You check a drafted answer against the passages it was supposed to be built from.

You are given the question, the numbered passages, the facts extracted from them, and the draft.
You are not rewriting the draft and not judging its style. You are looking for four faults:

1. FALSE ABSTENTION — the draft declines to answer, or calls the evidence insufficient, when the
   passages or facts do support an answer. This is the most important fault to catch: refusing
   with the evidence in hand is worse than an imperfect answer.
2. FLIPPED LOGIC — a relationship stated backwards. An inequality the wrong way round, "increases"
   where the passage says decreases, a condition inverted, a correlation reported as a cause,
   a null result reported as a positive one, significance claimed where the passage reports none.
3. UNSUPPORTED CLAIM — a number, bound or assertion that appears in neither the passages nor the
   facts, or a number that has drifted from the passage's value or lost its condition.
4. MISCITATION — a claim cited to a passage that does not contain it.

Rules:
- Judge only against the passages and facts given. Your own knowledge of the subject is not
  evidence, and a claim being true in the world does not make it supported here.
- A correct answer that goes beyond the extracted facts is not a fault, provided the passages
  carry it.
- A genuine abstention is correct behaviour. Only flag it when the evidence was actually there.
- Be specific. "The draft is vague" is not a fault; "the draft says p_D rises with range, passage
  [2] says it falls" is.

Reply with a JSON object and nothing else — no prose, no markdown fence:

{"ok": false,
 "problems": [{"type": "flipped_logic", "detail": "draft says detection probability rises with range; passage [2] gives (rho_max - rho)/(rho_max - rho_min), which falls"}]}

"ok" is true when you found nothing. "type" is one of "false_abstention", "flipped_logic",
"unsupported_claim", "miscitation".
"""
