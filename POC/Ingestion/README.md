# Ingestion

Acquires the source data for the multimodal PDF RAG: the **Open RAG Benchmark**
corpus ([`vectara/open_ragbench`](https://huggingface.co/datasets/vectara/open_ragbench))
and the **1000 arXiv PDFs** those documents were parsed from.

Ingestion runs in two stages, in order — the PDF stage depends on `pdf_urls.json`,
which only exists after the dataset snapshot lands.

| Stage | Source | Lands in | Size |
| --- | --- | --- | --- |
| 1. Dataset snapshot | HuggingFace Hub | `data/dataset/` | 715 MB |
| 2. PDF download | `arxiv.org` | `data/raw_dataset/pdf/raw_pdf/` | 3.1 GB |

## Running it

Use the wrapper at the repo root — it handles the working directory, picks a
Python runner, and tees output to `logs/`:

```bash
./run_ingestion.sh            # start (skips PDFs already on disk)
./run_ingestion.sh retry      # re-attempt only the ids in failed_downloads.json
./run_ingestion.sh force      # re-download everything, ignoring local files
./run_ingestion.sh status     # report progress, download nothing
```

Or invoke the module directly. `constants.py` resolves paths relative to the
current directory, so this **must** run from `Ingestion/`:

```bash
cd Ingestion && uv run ingestion.py [--retry-failed] [--force] [--skip-dataset]
```

A cold run takes a while: ~1000 HTTP fetches at 10 concurrent, plus the 715 MB
snapshot. It is safe to interrupt — see [Resume](#resume-and-failure-handling).

## What lands on disk

```
data/
├── dataset/                       # stage 1 — HuggingFace snapshot (parsed corpus)
│   ├── README.md                  #   upstream dataset card
│   └── pdf/arxiv/
│       ├── corpus/                #   1000 × {PAPER_ID}.json — parsed multimodal content
│       ├── pdf_urls.json          #   {paper_id: arxiv_url} — drives stage 2
│       ├── queries.json           #   3045 queries
│       ├── qrels.json             #   gold {doc_id, section_id} per query
│       └── answers.json           #   reference answers
└── raw_dataset/pdf/
    ├── raw_pdf/                   # stage 2 — 1000 × {PAPER_ID}.pdf
    └── failed_downloads.json      #   written only when downloads fail
```

`data/` is gitignored in full. Everything here is reproducible from the two
commands above.

### The parsed corpus

Each `corpus/{PAPER_ID}.json` carries the multimodal content already extracted
from the PDF (upstream used Mistral OCR), so downstream chunking does **not**
need to re-parse the PDFs:

```json
{
  "title": "...", "id": "...", "authors": [...], "categories": [...],
  "abstract": "...", "updated": "...", "published": "...",
  "sections": [
    {
      "text":   "markdown with inline ![table_0](table_0) / ![img-2.jpeg](img-2.jpeg) placeholders",
      "tables": {"table_0": "| markdown | table |\n| :--: | :--: |\n..."},
      "images": {"img-2.jpeg": "data:image/jpeg;base64,/9j/4AAQ..."}
    }
  ]
}
```

The placeholders sit at the exact position the element appeared in the text —
that positional binding is what makes table- and image-grounded retrieval
possible, so preserve it when chunking.

Measured over a 40-document sample: ~17 sections/doc, ~8 images/doc, ~4
tables/doc, which extrapolates to roughly **16.8k sections, 7.9k images and 4k
tables** across the corpus. Section length is median 3.0k chars, p90 10.9k, max
113k — too long to embed whole, so sections need sub-chunking.

Images are inline base64, which is most of the corpus's 709 MB. Extract them to
files once and store paths; do not carry base64 through the pipeline.

### The evaluation split

`queries.json` / `qrels.json` / `answers.json` give a labeled eval set with
**section-level** gold labels, not just document-level:

| Query source | Count |
| --- | --- |
| `text` | 1914 |
| `text-image` | 763 |
| `text-table-image` | 220 |
| `text-table` | 148 |

**1131 of 3045 queries (37%) are unanswerable from text alone.** Scoring the
text-only subset against the multimodal subset separately is the cheapest way to
tell whether the multimodal path is actually earning its cost.

## Resume and failure handling

Both stages are safe to interrupt and re-run.

**Stage 1** relies on the HuggingFace cache, so an interrupted snapshot resumes
where it stopped.

**Stage 2** resumes on its own terms:

- The pending set is computed *before* any request is issued, so the progress bar
  reflects real remaining work and finished files are never re-fetched.
- A file counts as complete only if it is non-empty **and** starts with `%PDF`.
  A truncated write or an HTML error page saved under a `.pdf` name is re-fetched,
  not silently skipped.
- Downloads are written to `<name>.pdf.part` and atomically renamed into place, so
  a killed run cannot leave a half-written file that the next resume mistakes for
  complete.
- `408/425/429/500/502/503/504` retry with exponential backoff honouring
  `Retry-After` (capped at 60s). Other 4xx fail immediately as permanent.

Anything still failing after 3 attempts is recorded in
`data/raw_dataset/pdf/failed_downloads.json`:

```json
{
  "2409.17266v2": {
    "url": "https://arxiv.org/pdf/2409.17266v2",
    "reason": "http_status",
    "attempts": 3,
    "detail": "status code 503"
  }
}
```

`reason` is `http_status`, `not_a_pdf`, or the exception type name. Entries clear
on a later success, and the file is deleted once nothing is outstanding — so a
stale log never triggers phantom retries.

`./run_ingestion.sh retry` re-attempts only these ids, and passes `--skip-dataset`
so it goes straight to the PDFs rather than re-verifying the 715 MB snapshot.

## Files

| Path | Purpose |
| --- | --- |
| `ingestion.py` | `DATAIngestor` — `download_dataset()` (HF snapshot) and `download_pdf()` (async fetch with resume) |
| `constants.py` | Paths, dataset name, retry/resume settings |
| `../run_ingestion.sh` | Wrapper: start / retry / force / status, with logging |

### Notable constants

| Constant | Default | Meaning |
| --- | --- | --- |
| `HUGGINGFACE_DATASET_NAME` | `vectara/open_ragbench` | Source dataset |
| `RETRYABLE_STATUS_CODES` | `{408,425,429,500,502,503,504}` | Retried; everything else 4xx is permanent |
| `MAX_RETRY_BACKOFF_SECONDS` | `60` | Ceiling on `Retry-After` and exponential backoff |
| `PDF_MAGIC_BYTES` | `b"%PDF"` | Completeness check used by resume |

Concurrency defaults to 10 (`download_pdf(max_concurrent=...)`). arXiv rate-limits,
so raising it tends to produce more `429`s rather than a faster run.

## Known rough edges

- `constants.py` uses paths relative to the cwd, so `ingestion.py` only works when
  run from `Ingestion/`. Anchoring them to `Path(__file__).parent` would remove the
  constraint.
- `ingestion.py` imports `from constants import ...` (script style), which breaks
  if `Ingestion` is imported as a package from elsewhere.
- The upstream dataset card notes Mistral OCR "struggles with unstructured PDFs" —
  parse quality in `corpus/` is uneven, which is why the raw PDFs are kept.

## End-to-end evaluation

Run on 2026-09-06 over the full evaluation split: **3,045 questions, 5.8 hours,
$3.60**.

> **This section records what was measured then, on the harness of the time.**
> `evals/answer_eval.py` has since been rebuilt on [RAGAS](../evals/), which
> reports five metrics on a 0-1 scale in place of the single 0-3 judge score
> below. The commands still run, but the numbers they now print are **not
> comparable to the ones in this section** — the columns are different measures,
> not a rescaling of the same one. Read what follows as the baseline this
> pipeline started from; re-running the sweep is what would restate it.

```bash
uv run python -m evals.answer_eval --limit 3045 --reranker cross-encoder \
    --concurrency 8 --out runs/full_xenc.jsonl
uv run python -m evals.answer_eval --report runs/full_xenc.jsonl
```

The run is resumable — every record is appended and flushed as it is scored, so
re-running the same command continues from where the file stops.

### What was measured, and what could not be

Each question is scored on its own. The pipeline's answer is compared against
the reference in `answers.json` by a judge model (gpt-4.1-mini) scoring 0-3 on
how much of the reference's substance survived. The judge is instructed to
ignore length and wording, and explicitly **not** to penalise correct detail the
brief reference omits — otherwise it marks down the answers that used the
retrieved passages best.

**Section-level retrieval scoring is not reported, because it cannot be.** The
gold `section_id` in `qrels.json` indexes *vectara's* parse of the PDF, not the
one `data_process.py` produces. Measured over the 396 documents carrying gold
labels, the gold id is out of range of our parse entirely in 23% of them, and
the counts diverge in both directions (`2412.02582v2`: gold max 14, ours 143;
`2406.17972v3`: gold max 48, ours 5). An early BM25-only run scored 97.2%
doc-level recall@10 but 18.5% section-level — the gap is the numbering, not the
retriever. Everything below is therefore **doc-level**.

### The configuration it was measured against

| | |
| --- | --- |
| collection | 80,363 points — 67,530 text, 9,458 figure, 3,375 table |
| retrieval | 10 candidates from Qdrant (dense BGE-M3) + 10 from BM25, fused by RRF |
| reranker | `BAAI/bge-reranker-v2-m3` on MPS, top 5 kept |
| answer | gpt-4.1-mini, `INFER_SYSTEM_PROMPT`, cites passages as `[1]..[5]` |
| judge | gpt-4.1-mini |

### Headline

| | |
| --- | --- |
| mean score (0-3) | **2.61** |
| scored 2 or better | **93%** |
| scored 3 (fully correct) | 72% |
| abstained | 2% |
| gold document among the candidates | **99.6%** |
| gold document survived reranking | 99.2% |
| judge failures (unscored) | 11 of 3,045 |

```
3  ████████████████████████████████████  2,195   72.3%
2  ██████████                              624   20.6%
1  █                                        98    3.2%
0  █                                       117    3.9%
```

### Retrieval is not the bottleneck

The median fused rank of the gold document is **1**, and it reaches the reranker
99.6% of the time. Averaged over the corpus a question draws 15.8 unique
candidates from the 10+10 fetch, so the two legs agree on roughly a quarter of
what they return.

That makes the failure taxonomy the useful table. Of the 215 questions scoring
below 2:

| cause | n | share of failures | share of all |
| --- | --- | --- | --- |
| answered wrong with the gold document in context | 149 | 69.3% | 4.9% |
| abstained with the gold document in context | 49 | 22.8% | 1.6% |
| gold document never retrieved | 10 | 4.7% | 0.3% |
| gold document dropped by the reranker | 7 | 3.3% | 0.2% |

**92% of failures happen after retrieval has already succeeded.** Tuning `k1`,
`b`, the 10/10 split or the RRF constant would move the bottom two rows, which
together account for 0.5% of the benchmark. The generation and grounding step is
where the remaining quality is.

### Modality is the real gap

| source | n | score | ≥2 | =3 | abstained | gold@20 |
| --- | --- | --- | --- | --- | --- | --- |
| text | 1,906 | **2.73** | 96% | 79% | 1% | 100% |
| text-image | 760 | **2.41** | 88% | 60% | 4% | 99% |
| text-table | 148 | **2.41** | 86% | 62% | 3% | 100% |
| text-table-image | 220 | **2.46** | 89% | 63% | 4% | 100% |

Every multimodal slice trails text by ~0.3, and the gap is not retrieval —
`gold@20` is 99-100% everywhere. It is concentrated in the *fully correct* rate,
which falls from 79% to 60-63%, and in abstentions, which quadruple from 1% to
4%. The figures and tables are being **found and not used**: the VLM description
is enough to match the question but not always enough to answer it.

That is the sharpest actionable finding here, and it is one section-level
scoring could never have produced. The 9,458 figure descriptions average 741
characters and are written to be *searchable* — `PARSE_SYSTEM_PROMPT` asks for
2-4 factual sentences ordered as figure type, then axes, then the trend. For
retrieval that ordering is right. For answering a question about a specific
value it may not be, and the 4% abstention rate on image queries is the model
saying so.

### By question type

| type | n | score | ≥2 |
| --- | --- | --- | --- |
| extractive | 1,245 | 2.72 | 92% |
| abstractive | 1,789 | 2.54 | 94% |

Extractive questions score higher on average but fail harder: they clear the
bar less often while scoring 3 more often, which is what a question with one
right answer looks like.

### Cost and throughput

| | |
| --- | --- |
| wall clock | 20,962s (5.8h), ~6.9s per question |
| API cost | $3.60 (gpt-4.1-mini, answer + judge only) |
| reranking | free — local cross-encoder, ~4s of the 6.9s |

Reranking is the whole latency budget and none of the bill. With an LLM
reranker instead it would be ~73% of the token cost — $10.81 rather than $3.60
at standard pricing, or $5.41 batched.

### Open questions this run does not answer

- **Is the LLM reranker better?** Only the cross-encoder arm was run. The A/B is
  `--reranker llm` against the same seed, and it is the one comparison that
  would justify the extra $7.
- **Would richer figure descriptions close the modality gap?** The evidence
  points at description content, not retrieval, but re-enriching with an
  answer-oriented prompt and re-running the `text-image` slice is what would
  prove it.
- **Nothing in the retrieval layer is tuned.** `k1`, `b`, the 10/10 split, the
  RRF constant and `--vector-mode` are all defaults. Given 92% of failures are
  post-retrieval, tuning them has a ceiling of about half a percent.
- **11 questions went unjudged** on judge errors and are excluded from every
  number above.


## Licence

The dataset is CC-BY-NC-4.0. See `data/dataset/README.md` for the upstream card.
