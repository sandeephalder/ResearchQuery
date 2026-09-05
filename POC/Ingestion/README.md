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

## Licence

The dataset is CC-BY-NC-4.0. See `data/dataset/README.md` for the upstream card.
