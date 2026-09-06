# Retrieval

The query side of the pipeline. Indexing produced a Qdrant collection; this turns
a question into a cited answer.

```
question
   ├── QdrantVectorStore ──► 10 candidates   (dense BGE-M3 vectors, meaning)
   └── BM25Retriever     ──► 10 candidates   (literal tokens)
                      │
                 EnsembleRetriever — reciprocal rank fusion  ──►  ~19 unique
                      │
                 a BaseDocumentCompressor  ──►  top 5
                      │
                 llm.chains.answer_chain  ──►  answer citing [1]..[5]
```

Every box is a LangChain component. The fusion that was ~40 lines of hand-written
reciprocal-rank arithmetic is `EnsembleRetriever`; the inverted index that was a
term-major sparse matrix with its own IDF, length normalisation and on-disk format
is `BM25Retriever`; the Qdrant queries are `QdrantVectorStore`.

## Running it

```bash
uv run python -m retrieval.bm25 --build                       # once, ~6 s
uv run python -m retrieval.pipeline "how does detection probability vary with range"
```

Both run from anywhere — unlike `Ingestion/`, paths here resolve against the
package, not the working directory.

Useful variations:

```bash
--kind figure          # search only figures (or text, table)
--no-rerank            # keep the fused order, to see what reranking bought
--no-answer            # retrieval only — touches no API at all
--vector-mode hybrid   # dense + BGE-M3 sparse, fused inside Qdrant
--k-vector 20 --k-bm25 20 --top-n 8
--json                 # machine-readable, for the eval harness
--reranker llm         # the listwise call instead of the local cross-encoder
--provider groq        # rerank and answer through another provider
```

The report marks which leg found each candidate, so a run shows its own working.
A real run, cross-encoder reranking:

```
19 candidates (10 vector, 10 bm25, 1 both) | vector mode: dense | reranker: cross-encoder

     vec  bm25   fused    rr  source
* 1   #1    #1  0.0164 0.98  figure · 2401.14085v2 · C. The multi-object filter · p.4
* 2   #2     -  0.0081 0.86  figure · 2409.14049v2 · Z −H ˆΨ · p.7
  3    -    #2  0.0081       text · 2401.14085v2 · C. The multi-object filter · p.4
  4   #3     -  0.0079       text · 2412.03633v3 · Detection Head. · p.5
```

**One candidate in nineteen was found by both legs.** That is the design working,
not a warning sign — see below.

## Documents in, dicts out

Inside, a passage is a `Document`, because that is what every LangChain component
here takes and returns. At the boundary — what `retrieve`, `rerank` and `arun`
hand back — it becomes a dict carrying both legs' ranks and the fused score.

That is not nostalgia. `evals/answer_eval.py` and `agents/` read those keys, and
`as_candidate` / `as_documents` in [`pipeline.py`](pipeline.py) are the two sides
of the boundary. Keeping the shape is what lets the eval harness score the agent
flow against the plain pipeline without being able to tell them apart.

## Why two retrievers

BGE-M3 emits a dense **and** a sparse vector, and the collection holds both — so
`--vector-mode hybrid` is already available without a second model. BM25 is a
third leg, not a replacement, because BGE-M3's sparse output is *learned* lexical
matching: it expands and reweights terms with the same model that produced the
dense vector, so it tends to fail where the dense vector fails. BM25 matches the
literal token and fails elsewhere — on the queries that turn on an exact symbol,
model name or unit (`SNR`, `CIFAR-100`, `mmol/L`).

Two legs are only worth their cost when they fail differently, which is what the
**1-of-19 overlap** above is evidence of, and which is also why `--vector-mode`
defaults to `dense`: pairing BGE-M3's sparse leg *with* BM25 would spend two of
the three legs on lexical overlap.

Fusion is by rank, not score. A cosine similarity of 0.7 and a BM25 score of 14
are not comparable; reciprocal rank fusion only asks how high each leg placed a
row, and rewards the rows both legs placed highly. `EnsembleRetriever` does this
with `weight / (rank + c)` and `c = 60`, the constant from the paper.

`id_key="_id"` on the ensemble is load-bearing: without it LangChain deduplicates
on page content, and two chunks that share an opening collapse into one.

## Reranking

Retrieval scores *similarity*. The reranker is the first stage to read the
question and the passage together and judge whether the passage **answers** it —
a different question, and the reason a passage merely about the topic stops
outranking the one that settles it.

Both rerankers are `BaseDocumentCompressor`s, so either drops into the pipeline
without it knowing which it got.

| reranker | 20 passages | cost |
| --- | --- | --- |
| `bge-reranker-v2-m3` on MPS (**default**) | 4.06 s | free |
| the same on CPU | 6.79 s | free |
| `ms-marco-MiniLM-L-6-v2`, 88 MB | 0.16 s | free |
| `gpt-4.1-mini`, listwise | ~2 s | $3.61 of a $5.41 full sweep |
| `glm-5.3-flash`, listwise | 42.8 s | paid, 1,661 reasoning tokens |
| `llama3.1:8b`, 4-core CPU | ~321 s (projected) | free |

Reranking is ~73% of the token bill of a 3,045-question sweep, so this is the
single biggest cost decision in the pipeline — which is why the local
cross-encoder is the default. `bge-reranker-v2-m3` was trained alongside `bge-m3`,
the model that produced the vectors being reranked, and reads 8192 tokens, so no
row in this corpus is truncated before it is judged. `ms-marco-MiniLM-L-6-v2` is
26x smaller and 25x faster but caps at 512 tokens, which cuts the p90 table row
in half.

What a cross-encoder cannot do is weigh candidates against each other: it scores
each (question, passage) pair independently, where the listwise call sees all
twenty at once. Whether that is worth 270x the latency is unmeasured —
`--reranker llm` against `--reranker cross-encoder` in `evals/answer_eval.py` is
the comparison that settles it.

### The score has to survive the stage that produced it

LangChain's `CrossEncoderReranker` returns the documents in their new order and
discards the numbers. The scope floor in `agents/` is a threshold on the *best*
of those numbers, so a reranker that forgets what it scored takes the floor with
it. `ScoringCrossEncoderReranker` writes it back onto `metadata["rerank_score"]`,
onto a copy — the candidate list is what the run reports, and a reranked view
should not silently rewrite the list it came from.

### What actually goes wrong on the listwise call

Measured over 25 live queries against `openai/gpt-oss-120b`, 7 rerank calls
failed. All three causes are worth knowing, because two look like model failures
and are not:

| | |
| --- | --- |
| **Truncated array** | The reply outgrew `max_tokens` and stopped mid-object. The scores that *did* arrive are fine, so the parser closes the array with LangChain's `parse_partial_json`, and falls back to reading complete `{...}` objects out of the text. Ids that never arrived are handled by the missing-id rule: they keep their fused order at the end, scored `None`. |
| **Budget spent reasoning** | `gpt-oss-120b` emits 4,000-4,800 characters of reasoning before the first bracket, so at `RERANK_MAX_TOKENS = 1200` the array never started. Now 3,000. |
| **Provider TPM cap** | `HTTP 413` — groq's free tier allows 8,000 tokens/minute counting reasoning, and 20 passages at 1,200 chars asks for ~9,000. Not retryable as-is, so groq carries `rerank_max_chars: 900` in the provider registry, the way ollama carries `parse_max_tokens`. |

All three are handled in [`llm/parsers.py`](../llm/parsers.py). A reranker that
fails outright leaves the fused order standing and says so on stderr — a broken
stage should cost quality, not look like a working one.

## The Qdrant store, and the one adaptation it needed

`CorpusVectorStore` is `QdrantVectorStore` with `_document_from_point` overridden,
and the reason is worth knowing before adding a field to the index.

`QdrantVectorStore` expects a point's payload to hold the text under one key and
*a nested dict of metadata* under another:

```python
{"page_content": "...", "metadata": {"kind": "figure", ...}}
```

`db_populate.py` writes a flat payload instead:

```python
{"text": "...", "kind": "figure", "doc_id": "...", "heading": "...", ...}
```

`content_payload_key="text"` handles the first half. The second half has no
setting, because there is no nested dict to name — so the override lifts the whole
flat payload into `Document.metadata`. Four lines, against re-embedding 80,365
rows to suit a default, and it keeps every retriever, compressor and chain
downstream unmodified. Doing it by "everything that is not the text" rather than
by naming fields also means a payload key added at index time arrives without
this file hearing about it.

`validate_collection_config=False` because the collection predates this code and
its distance and vector names are already known-good.

## The BM25 index

Not in Qdrant, because a named sparse vector cannot be added to an existing
collection — putting it there means a recreate, and re-embedding 80k rows to gain
a term-frequency count.

It stays aligned with Qdrant because both are built from
`db_populate.rows_for_document` with the same chunk parameters, and both key rows
by `db_populate.point_id`. So a BM25 hit is an id Qdrant can resolve, and **the
index stores the text only to return it** — Qdrant remains the only home for
payloads, and `LexicalRetriever` resolves a lexical hit's payload there in one
round trip.

```bash
uv run python -m retrieval.bm25 --build
uv run python -m retrieval.bm25 --stats
uv run python -m retrieval.bm25 --search "detection probability" --kind figure -k 5
```

| | |
| --- | --- |
| rows | 80,363 — text 67,530, figure 9,458, table 3,375 |
| mean tokens per row | 139 |
| on disk | 211 MB (`data/processed/pdf/bm25/bm25.pkl`) |
| build | 6.0 s — read 0.9 s, tokenize 3.7 s, fit 1.2 s |
| search | ~50 ms over 80,363 rows |

Scoring is `rank_bm25`'s Okapi with Lucene's defaults (`k1=1.2`, `b=0.75`,
untuned). A query is 50 ms rather than the sub-millisecond of the hand-written
posting-list index it replaced, because `rank_bm25` scores every document rather
than only those holding a query term. That is the trade the rewrite made
deliberately: 50 ms is far below the 7-15 s the vector leg takes, and the 275
lines of index it removed were 275 lines that could be wrong.

### Two things that are still this corpus's, not a library's

**The tokenizer**, passed in as `preprocess_func` — which is the hook
`BM25Retriever` provides for exactly this. Compound identifiers are indexed whole
*and* in parts, so `CIFAR-100` is found as `cifar-100`, `cifar` or `100`. Parts of
a purely numeric token are dropped. No stemming: the vocabulary is dominated by
symbols, units and model names where `GAs` and `GA` are different things, and
BGE-M3's sparse leg already covers the morphological variants a stemmer would
catch.

`TOKENIZER_VERSION` is stamped into the saved index and checked on load. An index
built by a different tokenizer scores correctly and matches nothing, which is the
failure that does not announce itself.

**The kind filter.** `BM25Retriever` has no notion of metadata filters, and the
router needs one. `CorpusBM25Retriever.scored` scores everything and *then* keeps
the rows of the requested kind, which is the honest way round — filtering before
scoring would change the IDF statistics the scores mean. Rows scoring zero are
dropped rather than padding the list to `k`: a rank is a claim that the leg found
something.

### Why the CLI re-imports itself

```python
if __name__ == "__main__":
    from retrieval.bm25 import main as _main
    _main()
```

Running the file as `python -m retrieval.bm25` makes its module `__main__`, so a
`CorpusBM25Retriever` built there pickles as `__main__.CorpusBM25Retriever` and
will not unpickle anywhere else. Calling `main` from the imported module gives the
class its real name. This is not hypothetical — it is how the first build of this
index failed.

## Providers

`LLM_PROVIDER` in `POC/.env` selects the default; `--provider` overrides it per
run. All eight entries in `llm.PROVIDERS` are verified to build.

```bash
uv run python -m retrieval.pipeline "..." --provider glm
```

As measured in this repo's last session: **groq** exhausts its 200,000
tokens-per-day allowance in a working session at six calls per question;
**openrouter** allows 50 free-model requests per day; **deepseek** reports
insufficient balance and **gemini** depleted prepayment credits. **glm** is the
one currently answering. None of that is a property of the code, and all of it is
why `--provider` exists.

A cross-encoder run with `--no-answer` touches no API at all, which makes it the
right way to check retrieval when every key is capped.

## Measured results

The full 3,045-question evaluation is written up in
[`Ingestion/README.md`](../Ingestion/README.md#end-to-end-evaluation). The short
version: mean answer score 2.61/3, 93% scoring 2 or better, and the gold document
reaches the reranker 99.6% of the time with a median fused rank of 1.

Retrieval is not the bottleneck — 92% of failures happen with the gold document
already in context. The live gap is modality: text queries score 2.73, image and
table queries 2.41-2.46, with abstentions quadrupling on image queries.

Those numbers predate the rewrite and the RAGAS scoring that replaced the 0-3
judge. Re-running the sweep is what would restate them in the new metrics, and
until it runs they should be read as the baseline to beat rather than as the
current result.

## Open questions

- **Nothing here is tuned.** `k1`, `b`, the 10/10 split, the RRF constant and
  `--vector-mode` are all defaults. The benchmark's `qrels.json` can settle every
  one of them; none has been measured.
- **The reranker's value is unmeasured.** `--no-rerank` is the A/B, and
  `--reranker llm` against `cross-encoder` is the cost decision underneath it.
- **`EnsembleRetriever` supports weights.** Both legs are currently equal. A
  corpus this lexical may well want the BM25 leg weighted up, and that is one
  argument and one sweep away from being known.
