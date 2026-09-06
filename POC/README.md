# ResearchQuery

Ask a question about a corpus of arXiv papers and get an answer that cites the
passages it came from — including the figures and tables, which are indexed as
text a retriever can actually reach.

Built on **LangChain** and **LangGraph** end to end: the retrievers, the fusion,
the rerankers, every model call and the flow itself are LangChain components,
not wrappers around them.

```bash
cd POC
uv sync
uv run python -m retrieval.bm25 --build          # once, ~6 s
uv run python -m agents.orchestrator "how does detection probability vary with range"
```

## What is in the corpus

| | rows | |
| --- | ---: | --- |
| `text` | 67,530 | a chunk of a section's markdown |
| `figure` | 9,458 | the caption plus a VLM's transcription of the plot |
| `table` | 3,375 | the caption plus the table as markdown |
| | **80,365** | points in Qdrant, one vector space, told apart by `kind` |

Indexing figures and tables *as text* is the point of the whole design. A
question like "how does detection probability vary with range" is answered by a
plot, and a plot is invisible to a text retriever unless something has read it
first. `llm/enrich.py` does that reading offline through the OpenAI Batch API,
at half price, and `Ingestion/db_populate.py` indexes the result alongside the
prose so one query reaches all three kinds.

## The five packages

| | | |
| --- | --- | --- |
| [`Ingestion/`](Ingestion/) | 1,775 | fetch, parse PDFs, chunk, embed, load Qdrant |
| [`llm/`](llm/) | 1,757 | the provider registry and the three LCEL chains every stage is built from |
| [`retrieval/`](retrieval/) | 1,290 | the query side: two retrievers, fused, reranked |
| [`agents/`](agents/) | 2,313 | the LangGraph flow — guardrail, route, extract, generate, verify |
| [`backend/`](backend/) | 558 | the flow behind FastAPI and Clerk |
| [`evals/`](evals/) | 697 | RAGAS scoring over the benchmark |
| [`tests/`](tests/) | 680 | wiring without a network, quality with DeepEval |

## How a question is answered

```
question
   │
   ▼
guardrail_in ──(blocked)────────────────────────────────► refusal
   │            NeMo rails, llama-guard3 on your own host
   ▼
router ───────► which kind of row holds the answer? one call
   │
   ▼
retrieve ──┬── Qdrant   10 candidates   dense BGE-M3, meaning
           └── BM25     10 candidates   literal tokens
                    │
              EnsembleRetriever — reciprocal rank fusion  ──► ~19 unique
                    │
   ▼
rerank ───────► a cross-encoder, or one listwise LLM call ──► top 5
   │            below the scope floor ──────────────────────► "not in this corpus"
   ▼
extract ──────► the numbers, bounds, conditions and table rows
   │
   ▼
generate ─────► a draft citing [1]..[5]
   │
   ▼
verify ───────► NeMo output rail. Problems? one repair, then accept
   │
   ▼
answer
```

The picture comes out of the same compiled graph that served the last query —
`uv run python -m agents.graph` draws it, locally, with no call to mermaid.ink.

## Which LangChain component does what

Nothing below is hand-rolled. That is the whole claim of the rewrite: the parts
that were bespoke HTTP, bespoke fusion and a bespoke inverted index are now
library components, and what is left in this repo is the part that is actually
about *this* corpus.

| stage | component |
| --- | --- |
| every model call | `init_chat_model` → `ChatGroq` / `ChatOpenAI` / `ChatGoogleGenerativeAI` / `ChatOllama` |
| embedding | a custom `Embeddings` over BGE-M3 — `retrieval/embeddings.py` |
| vector search | `QdrantVectorStore`, on the existing collection |
| lexical search | `BM25Retriever` with this corpus's own tokenizer |
| fusion | `EnsembleRetriever` — its weighted RRF is the fusion this used to compute by hand |
| reranking | a `BaseDocumentCompressor`: local cross-encoder, or one listwise call |
| every prompt | `ChatPromptTemplate` + `MessagesPlaceholder` |
| reading replies | `BaseOutputParser`s that salvage a truncated reply — `llm/parsers.py` |
| the flow | `StateGraph` with an `InMemorySaver` |
| the rails | `LLMRails(config, llm=...)` — NeMo, on a LangChain model |
| evaluation | RAGAS metrics, on the same models |
| tests | DeepEval, judged by a `DeepEvalBaseLLM` over the same registry |

### Three places a library was extended rather than replaced

Each is a small subclass with a reason, and each is documented where it lives.

**`CorpusVectorStore`** (`retrieval/store.py`) overrides `_document_from_point`.
`QdrantVectorStore` expects a payload with the text under one key and a *nested*
metadata dict under another; `db_populate.py` writes a flat payload. A four-line
override against re-embedding 80,365 rows to suit a default.

**`TracedEnsembleRetriever`** (`retrieval/pipeline.py`) records each leg's rank
and the fused score onto every surviving document. LangChain's returns the fused
order and discards the arithmetic, and a run should show its own working.

**`ScoringCrossEncoderReranker`** (`retrieval/rerank.py`) keeps the score.
LangChain's `CrossEncoderReranker` returns documents in their new order and drops
the numbers — and the scope floor downstream is a threshold on the best of those
numbers, so a reranker that forgets what it scored takes the floor with it.

## Where models come from

One registry, `llm/PROVIDERS`, and one function:

```python
from llm import chat_model, answer_question

model = chat_model("groq")                       # a BaseChatModel, cached
answer = await answer_question(question, passages)
```

Eight providers, all verified to build: `groq`, `openai`, `gemini`, `openrouter`,
`ollama`, `gateway` (a local LiteLLM proxy), `deepseek`, `glm`. `LLM_PROVIDER`
picks the default; each agent stage can name its own, because they do not all
want the same one — the gates are cheapest on small hosted models and the
reasoning-heavy stages on a large one.

There is **no client to open or close**. A LangChain chat model is long-lived and
holds its own pooled HTTP client, and `chat_model` caches by configuration — so
two stages on the same provider and budget share one pool without anything
arranging it.

### Everything that can run on your own hardware does

| stage | where | model |
| --- | --- | --- |
| embedding | local | `BAAI/bge-m3` |
| BM25 | local | no model |
| reranking | local, on MPS/CUDA | `BAAI/bge-reranker-v2-m3` |
| input guardrail | your ollama host | `llama-guard3:1b` |
| route / extract / generate / verify | hosted | one large general model |

## Measured

A live run, groq, cross-encoder reranking, cold start excluded:

```
route      kind=figure confidence=0.78 lexical=0.20 (widen +4)
           — asks how a quantity varies with another, typically shown in a plot
scope      best rerank score 0.976
extract    8 facts
timings    route 12.1s  retrieve 7.3s  rerank 15.1s  extract 3.0s  generate 2.5s
           total 39.9s
```

Retrieval on the same question: 19 candidates — 10 from the vector leg, 10 from
BM25, **1 found by both**. That overlap is the design working. The two legs are
deliberately unlike each other, and legs that fail the same way buy nothing.

| | |
| --- | --- |
| BM25 index build | 6.0 s for 80,363 rows; 50 ms per query |
| BGE-M3 load | ~15 s, once per process |
| cross-encoder, 20 passages | 4.06 s on MPS, 6.79 s on CPU |
| wiring test suite | 16 tests, 137 s, no network |

## Evaluating it

Two suites, and the split is what they measure.

```bash
uv run pytest                                    # 16 wiring tests, no network
uv run pytest --live                             # + DeepEval answer quality
uv run python -m evals.answer_eval               # RAGAS over a fixed 100 questions
```

The RAGAS sweep runs **the same 100 questions every time** — drawn once into
`evals/sample_100.json`, stratified so all four query sources are readable. A
fresh draw per run would make every comparison a comparison of two different
question sets. `--daily` swaps in a rotating hundred seeded by the date, so
thirty days cover roughly 3,000 of the 3,045.

**A hundred a day is a hard ceiling**, enforced in `evals/budget.py` rather than
left to whoever types `--limit`. In one working session this flow exhausted
groq's daily token allowance, OpenRouter's free-request count and DeepSeek's
balance; a sweep that runs until the keys die produces no result and costs the
rest of the day. `EVAL_DAILY_LIMIT` moves or removes it.

Both suites log into the same MLflow experiment — the run holds the scores, the
`@mlflow.trace` spans hold what produced them.

[`evals/`](evals/) scores the pipeline with **RAGAS** — faithfulness,
answer relevancy, context precision, context recall, factual correctness — which
replaced a hand-written 0-3 judge. Five numbers instead of one is not decoration:
`context_recall` is the retriever's score and `faithfulness` is the generator's,
so a failing question says *which stage* failed. The evaluation that motivated
this pipeline found that 92% of failures happen with the gold document already in
context, and one score cannot see that.

[`tests/`](tests/) guards the flow with **DeepEval**. One setting there is worth
knowing before trusting any faithfulness number anywhere:
`penalize_ambiguous_claims=True`. Without it DeepEval scores a pure fabrication
**1.00**, because it counts only direct contradictions and treats a claim the
passages never mention as merely unverifiable. Measured on this corpus's shape:

| answer | default | penalized |
| --- | --- | --- |
| grounded, correct | 1.00 | 1.00 |
| invented — rainfall, from a passage about detection probability | **1.00** | 0.00 |
| flipped — "rises with range" | **1.00** | 0.00 |

## Tracing

Six model calls across five stages, two retrievers, a fusion and a reranker.
When an answer is wrong the question is always *which stage* — so every stage
carries an `@mlflow.trace`, and the flow can hand you the tree.

```bash
uv run python -m agents.orchestrator "..." --trace
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

```
agent_pipeline              AGENT       total latency, question, final answer
  guardrail_in              GUARDRAIL   did a rail block it, and which
  route                     AGENT       kind, confidence, effect on retrieval
  retrieve                  RETRIEVER   how many candidates, from which leg
    qdrant_vector           RETRIEVER   the semantic leg's 10
    bm25                    RETRIEVER   the lexical leg's 10
    fuse                    RETRIEVER   what survived, and its rank in each leg
  rerank                    RERANKER    the 5 kept, and their scores
  extract                   AGENT       the facts, or [] if there were none
  generate                  AGENT       the draft
  verify                    GUARDRAIL   the verdict, and any repair after it
```

Two layers compose here. `mlflow.langchain.autolog()` traces every LCEL chain,
chat model and LangGraph node for free — since the rewrite that is most of this
repo. The decorators add what autolog cannot see: that a call *was* the router,
that it chose `figure`, and that the choice widened retrieval by four candidates.

**Off unless asked.** `MLFLOW_TRACING` gates it, and when unset `tracing.py`
calls `mlflow.tracing.disable()` — so the decorators stay in the source and cost
nothing on a CLI run or a test. The default backend is a local
`sqlite:///mlflow.db`, not MLflow's own `./mlruns`, which as of 3.16 raises
rather than writes. Point `MLFLOW_TRACKING_URI` at a server for anything shared.

Traces carry the caller: `mlflow.trace.user` and `mlflow.trace.session` are set
from the `Identity` a run serves, so a trace is findable by who asked for it.

```python
mlflow.search_traces(filter_string="metadata.`mlflow.trace.user` = 'user_123'")
```

## Known rough edges

**The BM25 index holds 80,363 rows against Qdrant's 80,365.** Both are built from
`db_populate.rows_for_document` with the same chunk parameters, so the two rows
are a rebuild-order artefact rather than a disagreement about content. A lexical
hit that Qdrant cannot resolve is dropped with a warning rather than hidden.

**The embedded Qdrant store admits one process.** Not one writer and many
readers — one. A notebook kernel holding it open is why an indexing run or the
server will refuse to start. Running Qdrant in Docker is the fix, and
`QDRANT_URL` is how you point at it.

**Free tiers run out fast.** Six model calls per question plus the grader's is
enough to exhaust groq's daily token allowance and OpenRouter's free-model
count in a session. `--provider` and `--judge-provider` exist for that.

**`gemini-2.5-flash` is retired for new API keys**, and its replacements ignore
`temperature`. Everything here asks for 0.0 so a description feeding the index,
a ranking and an answer are reproducible; on `gemini-3.x-flash` that request is
dropped and reruns will differ. `llm/constants.py` says so at the entry.
