# Evals

Does the pipeline answer what the benchmark asked? Scored with **RAGAS**, on the
same LangChain models the pipeline itself runs on.

```bash
uv run python -m evals.answer_eval --judge-provider glm            # the fixed 100
uv run python -m evals.answer_eval --no-rerank --out runs/rerank_off.jsonl
uv run python -m evals.answer_eval --report runs/rerank_off.jsonl
```

## The sample is fixed, and that is the point

The default is **100 questions, always the same 100**, drawn once into
[`sample_100.json`](sample_100.json) and committed.

A fresh random draw per run would make every comparison a comparison of two
different question sets, and the differences worth measuring here are smaller
than that noise. The full split is 3,045 questions and this flow makes six model
calls per question plus five RAGAS metrics — so a hundred is what you can afford
to run on a change, and it is only worth running if it is the same hundred.

```bash
uv run python -m evals.sample --show     # what is in it
uv run python -m evals.sample            # redraw it (seed 0, deterministic)
```

| source | sampled | pool | |
| --- | ---: | ---: | --- |
| `text` | 48 | 1,914 | |
| `text-image` | 25 | 763 | the modality this corpus is weakest at |
| `text-table-image` | 14 | 220 | |
| `text-table` | 13 | 148 | |

All 100 carry a reference answer, so none of the three reference-needing metrics
is ever skipped.

**Stratified with a floor of 10 per source.** Proportional allocation would give
7 and 5 to the two table cells, and a per-source mean over five questions moves
twenty points on one bad answer — it reads as noise. Modality is precisely what
this corpus is weak at (image queries scored 2.41 against text's 2.73 on the old
scale, with four times the abstentions), so a sample that cannot resolve it is
the wrong sample.

What the floor costs: the sample is no longer corpus-representative, so the
**"all" row is a mean over this sample**, not an estimate of quality over the
corpus. The per-source rows are the ones to read, and they are the reason for the
floor. `uv run python -m evals.sample --no-floor` draws a pure proportional one.

`--limit 20` takes the first 20 of the sample rather than resampling, so a short
run is a strict subset of the full one. `--no-sample` draws at random — a spot
check, not something to compare two runs with.

## What replaced the judge

This used to ask one model to score our answer against the reference 0-3. That is
now five RAGAS metrics, and the change buys three things a single score could not.

| metric | what it asks | needs a reference? |
| --- | --- | --- |
| `faithfulness` | is every claim in the answer supported by the passages it was given? | no |
| `answer_relevancy` | does the answer address the question that was asked? | no |
| `context_precision` | of the passages retrieved, were the useful ones ranked first? | yes |
| `context_recall` | did retrieval find what the reference answer needed? | yes |
| `factual_correctness` | does the answer agree with the reference, claim by claim? | yes |

**A wrong answer says which stage was wrong.** `context_recall` is the retriever's
score and `context_precision` is the reranker's; `faithfulness` is the
generator's. A question that fails with high recall and low faithfulness had the
evidence and fumbled it, which is a different bug from one that never retrieved
it — and the evaluation that motivated this pipeline found the first kind was 92%
of failures. `_diagnose()` in [`answer_eval.py`](answer_eval.py) reads that off
the metrics and prints it beside each failure:

```
  [0.31] how does detection probability vary with range
         gold retrieved but dropped by rerank — a reranker failure
```

**Three of the five need no reference answer.** `faithfulness` and
`answer_relevancy` score any question, including a user's, so the same metrics
run in an offline sweep and against production traffic.

**Nobody has to trust a rubric written here.** The prompts are RAGAS's, published
and used elsewhere, which a hand-written judge in this file was not.

## Everything runs on the project's own models

RAGAS takes a LangChain model and a LangChain `Embeddings` directly, so the
grader is built by `llm.models.chat_model` from the same registry as every stage
it is grading, and `answer_relevancy` embeds through the same BGE-M3 that built
the index.

```bash
JUDGE_PROVIDER=glm JUDGE_MODEL=glm-5.3 uv run python -m evals.answer_eval --limit 40
```

**The grader must not be the model being graded.** Same provider is fine; the same
*model* judging its own answer is not an independent measurement, and it fails in
the flattering direction. `--judge-provider` defaults away from the pipeline's own.

`answer_relevancy` is close to free: it reverse-generates questions from the
answer and compares them to the real one by cosine, and the comparison is local,
on weights already resident.

## Three things that are not defaults, and why

**`RELEVANCY_STRICTNESS = 1`.** RAGAS defaults to 3 — it reverse-generates three
questions and averages — and asks for them as `n=3` on a single completion, which
groq rejects outright:

```
400 'n' : number must be at most 1
```

What the lower value costs is variance: one reverse-generated question is a
noisier estimate than the mean of three, so a single question's
`answer_relevancy` is indicative and a sweep's mean is the measurement. Raise it
via `RAGAS_RELEVANCY_STRICTNESS` on a provider that allows `n>1`.

**`UNSCORED` is `None`, not `0.0`.** A metric that could not be computed and a
metric that scored zero are different facts, and a run that cannot tell them apart
reports a broken grader as a failing pipeline. `_print_metric_errors` tallies what
failed and why:

```
metrics that could not be computed:
  context_precision    3x RateLimitError
  faithfulness         1x RateLimitError
```

**Abstention is detected by phrase, not by a model.** Every RAGAS metric reads a
refusal as a low score, and the distinction is the finding this evaluation exists
to track: the system prompt tells the model to refuse when the passages do not
support an answer, so a refusal is *correct* on a failed retrieval and a *failure*
on a successful one. Paired with `gold_kept`, an abstention with the gold passage
in context is a false one. It is a heuristic, deliberately — cheap, deterministic,
and it spends no call.

## A hundred a day, and no more

The budget is enforced in [`budget.py`](budget.py), not left to whoever types
`--limit`. In one working session this flow exhausted groq's 200,000
tokens-per-day allowance, OpenRouter's 50 free-model requests and DeepSeek's
balance — measured, not predicted. A sweep that runs until the keys die produces
no result and costs the rest of the day.

```
daily budget: 40/100 spent on 2026-09-06, 60 left
```

It counts **questions scored**, not calls or tokens, because that is the number
a caller controls. Records already in the output file do not count again, so
resuming an interrupted sweep is not charged twice for work already paid for.
When the day is spent the run stops with the three ways out:

```
STOPPED: today's evaluation budget is spent: 100/100 questions scored on 2026-09-06.
    raise it for one run with EVAL_DAILY_LIMIT=200
    turn it off entirely with EVAL_DAILY_LIMIT=0
    or wait — it resets at 00:00 UTC
```

The ledger (`runs/eval_budget.json`, UTC-keyed) is also the record of how much
evaluation has been run, which is worth having when a metric moves and the
question is what changed.

## Two samples, for two different questions

```bash
uv run python -m evals.answer_eval              # the fixed 100 — comparable
uv run python -m evals.answer_eval --daily      # today's 100 — cumulative
```

| | `sample_100.json` (default) | `--daily` |
| --- | --- | --- |
| which questions | the same 100 every run | a different 100 each UTC day |
| seeded by | 0, committed to the repo | the date |
| use it to | compare two configurations | accumulate coverage |

The daily sample rotates: measured, today's 100 and tomorrow's 100 overlap by 2,
so thirty days cover roughly 3,000 of the 3,045. It is deterministic from the
date, so every machine agrees on today's hundred without sharing state and
yesterday's run is reproducible with `--daily 2026-09-05`.

**A daily sample is not comparable across days.** A change in the mean could be
the change you made or could be the different questions. Use the fixed sample to
decide whether something helped, and the daily one to find failures the fixed
hundred never reach.

## MLflow

```bash
MLFLOW_TRACING=1 uv run python -m evals.answer_eval --judge-provider glm
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

A sweep produces two things MLflow holds separately, and both are wanted:

**A run**, carrying the configuration that produced the result and the aggregate
that came out — the five metric means, the abstention rate, gold retrieval and
retention, seconds per question, and the same five **per query source**. That
last part is the reason to bother: this corpus's weakness is modality, and an
aggregate that hides a 0.3 drop on image queries is the number that lets it keep
happening. The per-question records go in as a table, because the question
actually asked afterwards is "which ones failed, and why".

**A trace per question**, from the `@mlflow.trace` decorators on every stage —
so a row in that table can be opened to see which retriever found what, what the
reranker scored it, and what the generator did with it. `mlflow.langchain.autolog()`
adds the judge's own calls, so a metric that scored oddly can be read back to the
prompt that produced it.

Both suites — this one and [DeepEval](../tests/) — log into the same experiment,
so a change can be read against both from one place.

## Per-sample, not `ragas.evaluate`

`evaluate()` scores a whole dataset at once, which would mean holding every answer
until the sweep finished. A full sweep is thousands of API calls over hours, so
every record is appended as it is scored and a re-run picks up where the file
stops:

```bash
uv run python -m evals.answer_eval --limit 3045 --out runs/full.jsonl   # ^C anytime
uv run python -m evals.answer_eval --limit 3045 --out runs/full.jsonl   # resumes
```

Per-sample `single_turn_ascore` is what keeps that. The five metrics for one
question run concurrently — they are independent calls — and a metric that fails
returns `UNSCORED` rather than taking the run with it.

## Re-running only what failed

The point of a sweep is the questions it gets wrong, and re-running those under
one changed setting is the cheapest experiment available.

```bash
uv run python -m evals.answer_eval --failures-from runs/full.jsonl --no-rerank
uv run python -m evals.answer_eval --failures-from runs/full.jsonl \
    --failing-metric context_recall
```

`--failing-metric` picks which of the five decides, and that is worth having
rather than fixed: the questions with the worst `context_recall` are the
retriever's to answer for, and the ones with the worst `faithfulness` are the
generator's. Which set is worth re-running depends on what you changed.

## Why there is an import shim

[`ragas_compat.py`](ragas_compat.py) registers a stand-in for
`langchain_community.chat_models.vertexai` before RAGAS is imported.

RAGAS 0.4.3 opens `ragas/llms/base.py` with an unguarded import of that module,
and `langchain-community` 0.4 — the release that pairs with `langchain-core` 1.x —
no longer ships it; Vertex moved to `langchain-google-vertexai` several releases
ago. So importing RAGAS into a LangChain v1 environment fails at import time, on a
symbol RAGAS only uses for an `isinstance` check against an LLM this repo never
passes it.

The alternative is `langchain-community` 0.3, which requires `langchain-core`
<0.4, which is the entire stack this project was rewritten onto. The shim is the
smaller thing to give up, and it is safe to try deleting — if RAGAS still imports,
it was no longer needed.

## Not measured yet

The full 3,045-question sweep has not been re-run since the rewrite. The baseline
to beat, from the old 0-3 judge, is mean 2.61/3 with 93% scoring 2 or better —
text 2.73, image and table 2.41-2.46. Those numbers are not comparable to a RAGAS
score and should be read as *what the pipeline was*, not as what it is.

The A/Bs worth running first, in order of how much they would change:

- `--no-verify` — the verifier is two of the six calls per question and targets
  the one failure mode the evaluation actually identified.
- `--route-mode off` against `widen` against `filter` — the routing question, and
  the cheapest of the three to settle.
- `--reranker cross-encoder` against `llm` — ~73% of a sweep's token bill.
