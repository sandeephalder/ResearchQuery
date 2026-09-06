# LLM

Every model call in the repo, and the registry that decides who serves it.

```python
from llm import chat_model, answer_question, rerank_passages, describe_figure

model  = chat_model("groq")                              # a BaseChatModel, cached
answer = await answer_question(question, passages)       # the chain
order  = await rerank_passages(question, candidates)     # [(index, score)]
text   = await describe_figure("img-0.png", caption="Figure 1")
```

There is **no client to open or close**. A LangChain chat model is long-lived and
holds its own pooled HTTP client, and `chat_model` caches by full configuration —
so two stages on the same provider and budget share one pool without anything
arranging it. The `async with LLMGateway()` that used to wrap every caller is
gone, and so is the `Clients` registry that pooled connections by hand.

## What this package is now

The hand-rolled client existed to do four things — pick a base URL, attach a key,
retry the retryable statuses, and dig the content out of the response. A LangChain
integration does all four. What is left is the part that was never about HTTP.

| file | |
| --- | --- |
| [`constants.py`](constants.py) | the provider registry, the budgets, the prompts |
| [`models.py`](models.py) | `chat_model()` — one function, eight providers |
| [`chains.py`](chains.py) | the three calls the corpus needs, as LCEL chains |
| [`parsers.py`](parsers.py) | reading a reply that is *nearly* the JSON you asked for |
| [`images.py`](images.py) | a figure on disk, as something a vision model accepts |
| [`batch.py`](batch.py) | the OpenAI Batch API — the one call that is not a chain |
| [`enrich.py`](enrich.py) | describe every parsed figure offline, at half price |

## The three chains

Each is `prompt | model | parser`, and they differ in more than their prompt.

| | | |
| --- | --- | --- |
| `describe_figure` | a figure image plus its page context, in; a factual description for the index, out | a vision call made ~10k times over a corpus, so it downscales images and defaults to a cheap alias |
| `rerank_passages` | a question plus ~20 candidates, in; the same candidates in relevance order, out | one listwise call, not twenty pairwise ones |
| `answer_question` | a question plus the surviving passages, in; a grounded answer, out | strips the `<thinking>` block the prompt asked for |

All three are deterministic by default (temperature 0), because a description
that changes between runs quietly invalidates an index, and a ranking or an
answer that changes between runs cannot be evaluated.

Each is a `Runnable`, so the ordinary LangChain verbs work on them — `ainvoke`
for one, `abatch` for a corpus, `with_retry` for a flaky provider — and none of
that had to be written here.

### Passages are never templated

The passage text carries markdown tables and LaTeX, which is to say it carries
braces. A `ChatPromptTemplate` treats those as variables and fails on the first
table it sees, so the prompts place a `MessagesPlaceholder` and the human message
is built as a *value* rather than formatted into a template.

The system prompt is passed as a `SystemMessage` for the same reason, and that is
not hypothetical: `RERANK_SYSTEM_PROMPT` ends by showing the model the array it
should return —

```json
[{"id": 3, "score": 9}]
```

— which a template reads as a variable named `"id"`, and which is exactly how the
first version of this failed.

## The provider registry

Eight entries, all verified to build. `LLM_PROVIDER` picks the default.

| | integration | note |
| --- | --- | --- |
| `groq` | `ChatGroq` | fast and free, at 8,000 TPM and 200,000 tokens/day |
| `openai` | `ChatOpenAI` | no per-minute cap; what the enrichment batches ran on |
| `gemini` | `ChatGoogleGenerativeAI` | the first-party API, not the OpenAI shim |
| `openrouter` | `ChatOpenAI` + `base_url` | the overflow valve; 50 free requests/day |
| `ollama` | `ChatOllama` | native `/api`, not the `/v1` shim |
| `gateway` | `ChatOpenAI` + `base_url` | a local LiteLLM proxy |
| `deepseek`, `glm` | `ChatOpenAI` + `base_url` | text only, no vision model |

Providers fall into two groups and the split is not cosmetic: a first-party
integration, or an OpenAI-dialect endpoint reached by `ChatOpenAI` with a
`base_url`. `init_chat_model` builds both.

### The four integrations disagree about three settings

`_kwargs()` translates canonical settings into the names one integration accepts,
explicitly rather than passing a dict through and hoping:

- `ChatOllama` has `num_predict` where the others have `max_tokens`, and takes
  its timeout through `client_kwargs`.
- `ChatGoogleGenerativeAI` has no base URL to set — it is the first-party API,
  not a proxy.
- Passing an unknown keyword to a pydantic model is an error at construction,
  which is the right time to find out but not a message anyone should read twice.

### Rate limiting is a declaration, not a rescue

A provider can carry `requests_per_second`, and `chat_model` gives it a LangChain
`InMemoryRateLimiter`. That paces requests onto the wire instead of catching 429s
after the fact — the difference between a slow sweep and a failed one. It is per
*provider*, not per model: the cap that binds is the account's, and two stages on
one key share it whether or not they share a model.

## Every failure a caller can handle is one exception

`guarded()` wraps each chain so provider and parser failures arrive as `LLMError`.

That contract predates LangChain and has to survive it, and it would not on its
own: each integration raises its own vendor's exceptions, so a groq rate limit
and a Gemini quota error arrive as two unrelated types and neither is the one the
caller is watching for. Every caller in this repo catches `LLMError` to decide
whether a failed stage costs quality or the whole answer — the router falls back
to unrouted retrieval, the extractor to the passages alone, the reranker to fused
order.

Verified live: with OpenRouter's daily cap exhausted, a run degraded exactly as
designed — *"routing failed (…429…); retrieving unrouted"*, then *"extraction
failed; the generator will work from the passages alone"* — rather than crashing.

**`OUR_BUGS` are let through untranslated.** `TypeError`, `AttributeError`,
`KeyError`, `IndexError`, `NameError` and `ValueError` mean this repo has a bug,
not that the provider does. "The model failed" is the wrong sentence for a
misspelled key in a prompt-builder, and a stage that degrades gracefully on a
`NameError` hides it for as long as the degradation looks plausible.

## Reading a reply that is nearly JSON

A model asked for JSON returns JSON inside a markdown fence, or after a sentence
of preamble, or JSON that stopped mid-object because the reply outgrew its
budget. The last is the common failure and the least deserving of one: the
entries that did arrive are perfectly good, and a stage that raises on them
throws away work that was paid for.

LangChain solves two thirds of it — `parse_json_markdown` finds the JSON,
`parse_partial_json` closes a truncated one. [`parsers.py`](parsers.py) adds the
last fallback (read the complete `{...}` objects out of the text, ignore the
brackets) and the decision to return what survived rather than raise.

| parser | |
| --- | --- |
| `JsonListParser` | a list of objects; `[]` is a real answer, not a failure |
| `JsonObjectParser` | one object, with a caller-supplied default |
| `RankingParser` | the reranker's array as `[(index, score)]`, invented ids dropped, forgotten ids kept at the end scored `None` |
| `AnswerParser` | the answer with the `<thinking>` scratchpad removed |

A passage the reranker forgot — or never got to, because the reply was cut short
— is not evidence that the passage is bad. That is why forgotten ids keep their
retrieval order rather than being dropped.

## Reasoning models break token budgets

Every free reasoning model bills and budgets its reasoning as output. That broke
three stages, each silently, because each stage's parse failure looks like a real
answer: an extractor that returns no array reports "no evidence in the passages",
and a verifier that returns nothing parseable falls back to `{"ok": true}` and
approves.

| stage | was | is | measured |
| --- | ---: | ---: | --- |
| rerank | 1200 | 3000 | `gpt-oss-120b` emits 4,000-4,800 characters first |
| extract | 1500 | 3000 | 2,428 completion tokens for a 1,118-character reply |
| router | 200 | 1500 | still mid-preamble at 800 |
| verify | 800 | 2000 | precautionary, not measured |

A stage that fails by approving is worse than one that fails loudly, which is why
the verifier's budget was raised on suspicion rather than on evidence.

`text_of()` says so when it happens: a reasoning model that spends its whole
budget thinking returns empty content with the reasoning in `additional_kwargs`,
and that is a token-budget problem, not a model failure. The fix is `max_tokens`,
and nothing about the prompt needs touching.

## Batch is the one call that is not a chain

[`batch.py`](batch.py) uses the `openai` SDK directly — which `langchain-openai`
already depends on, so it costs no new dependency.

LangChain models a conversation; a 24-hour queue of ten thousand of them is a
different object with a different lifecycle. What *is* shared is the prompt:
`request_line` takes LangChain messages and converts them with
`convert_to_openai_messages`, so the prompt in a batch and the prompt in a live
call cannot drift apart.

Batch is half price and the work is offline, so the only cost of the 24-hour
window is that the run outlives the process — which is why every phase of
[`enrich.py`](enrich.py) keeps its state on disk and can be re-run.

```bash
uv run python -m llm.enrich submit     # build, upload, create batches
uv run python -m llm.enrich status     # where are they
uv run python -m llm.enrich collect    # merge finished results back
uv run python -m llm.enrich merge      # write descriptions into the doc JSONs
```

## Provider notes worth knowing before you debug

**`gemini-2.5-flash` is retired for new API keys.** An existing key keeps working,
so this only bites on a fresh one — which is the worst way for it to bite, because
it reads as a broken setup rather than a retired model:

```
404 NOT_FOUND: This model models/gemini-2.5-flash is no longer available to
new users. Please update your code to use models/gemini-3.6-flash
```

**The Gemini 3.x flash models ignore `temperature`.** Everything here asks for 0.0
so that a description feeding the index, a ranking and an answer are reproducible,
and on those models the request is silently dropped. Where that matters —
re-parsing figures, or an A/B whose two arms must be comparable — use a provider
that honours it.

**`gemini-flash-latest` is deliberately not the default.** A model id that changes
underneath a corpus is how an index quietly stops matching the descriptions that
built it.

**groq rejects `n>1`.** Anything asking for multiple completions on one request
gets `400 'n' : number must be at most 1`, which is why RAGAS's `answer_relevancy`
runs at `strictness=1` here.

**groq's `gpt-oss-120b` rejects LangChain's tool-calling structured output** —
`Tool call validation failed: attempted to call tool 'json' which was not in
request.tools`. LangChain binds the schema as a tool, the model answers with a
`json` tool the request never declared, and the provider rejects its own reply.
This is why `STRUCTURED_OUTPUT` defaults to `"parse"`: ask for JSON in the prompt
and read it with a tolerant parser, which works on every provider here.
