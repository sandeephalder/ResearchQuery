"""Per-question evaluation with RAGAS: does the pipeline answer what was asked?

    uv run python -m evals.answer_eval --limit 40
    uv run python -m evals.answer_eval --limit 40 --no-rerank      # the A/B
    uv run python -m evals.answer_eval --source text-image         # one modality
    uv run python -m evals.answer_eval --report runs/rerank_on.jsonl

Retrieval recall is not measured from the benchmark's labels, and deliberately
so. `qrels.json` labels a gold `(doc_id, section_id)`, but the `section_id`
indexes vectara's parse of the PDF rather than `data_process.py`'s: it is out of
range of ours in 23% of documents, and the counts diverge in both directions. So
section-level scoring measures the numbering, not the retriever. Doc-level
scoring is valid but saturated — every candidate in a typical run comes from the
gold document, and a metric already at 100% cannot rank two configurations. The
gold document is still recorded per question, not as the metric but so a bad
answer can be read as "never retrieved it" or "had it and fumbled it".

## What replaced the judge

This used to ask one model to score our answer against the reference 0-3. That
is now five RAGAS metrics — see `evals/metrics.py` for what each one asks — and
the change buys three things a single score could not:

**A wrong answer says which stage was wrong.** `context_recall` is the
retriever's score and `context_precision` is the reranker's; `faithfulness` is
the generator's. A question that fails with high recall and low faithfulness had
the evidence and fumbled it, which is a different bug from one that never
retrieved it — and the evaluation that motivated this pipeline found the first
kind was 92% of failures.

**Three of the five need no reference answer.** faithfulness and
answer_relevancy score any question, including a user's, so the same metrics run
in an offline sweep and against production traffic.

**Nobody has to trust a rubric written here.** The prompts are RAGAS's,
published and used elsewhere, which a hand-written judge in this file was not.

Abstentions are still counted apart from wrong answers, because every RAGAS
metric reads a refusal as a low score and the distinction is the finding this
evaluation exists to track: the system prompt tells the model to refuse when the
passages do not support an answer, so a refusal is correct on a failed retrieval
and a failure on a successful one. `evals.metrics.abstained` is the detector.
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LLMError
from retrieval.constants import INGESTION_DIR, RERANKER, RERANKERS
from retrieval.paths import RetrievalError
from retrieval.pipeline import Pipeline

from tracing import evaluation_run, log_metrics, log_table, setup as setup_tracing

from .budget import Budget, BudgetExhausted
from .metrics import METRIC_NAMES, NEEDS_REFERENCE, Scorer, abstained
from .sample import SAMPLE_PATH, daily_seed, draw as draw_sample, load as load_sample

BENCHMARK_DIR = os.path.join(INGESTION_DIR, "data", "dataset", "pdf", "arxiv")

# Both default to a provider without a tokens-per-minute cap, overriding the
# pipeline's own default. groq is the right choice for testing one query by hand,
# but a sweep makes three calls per question against an 8,000 TPM budget shared
# across the account — it throttles to a crawl, and 429s land as rerank failures
# that would read as a quality result rather than a rate limit.
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "openai")
# The grader must not be the model being graded — the same model judging its own
# answer is not an independent measurement — so this defaults away from the
# pipeline's provider as well as away from a per-minute cap.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openai")
JUDGE_MODEL = os.getenv("JUDGE_MODEL")
# RAGAS asks for structured output and several of its metrics decompose an
# answer claim by claim, so the verdict is longer than a 0-3 score was.
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "2048"))
# Retrieval and reranking are local and single-process; only the API calls
# parallelise, and 8 in flight keeps a sweep latency-bound without tripping
# a provider's per-minute limits.
CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "8"))

def load_benchmark():
    try:
        return tuple(json.load(open(os.path.join(BENCHMARK_DIR, name)))
                     for name in ("queries.json", "qrels.json", "answers.json"))
    except FileNotFoundError as error:
        raise RetrievalError(
            f"Benchmark files missing from {BENCHMARK_DIR}. Fetch them with:\n"
            f"    cd Ingestion && uv run fetch_data.py --part dataset") from error


def _already_done(out_path):
    """qids already in the output file, so an interrupted sweep resumes.

    A full sweep is thousands of API calls over hours. Writing only at the end
    means a laptop sleeping, a session restarting or a provider outage costs the
    whole run — so every record is appended as it is judged, and a re-run picks
    up where the file stops.
    """
    done = {}
    if not out_path or not os.path.exists(out_path):
        return done
    with open(out_path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue                  # a half-written last line from a kill
            if record.get("qid"):
                done[record["qid"]] = record
    return done


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


# The metric a bare `--failures-from` sorts on. factual_correctness is the
# closest thing to "did it answer the question", which is what the old 0-3 score
# was; the threshold is on a 0-1 scale now, and 0.5 is the value below which an
# answer disagrees with the reference on more claims than it agrees.
HEADLINE_METRIC = "factual_correctness"
FAILING_BELOW = float(os.getenv("EVAL_FAILING_BELOW", "0.5"))


def failing_qids(path, threshold=FAILING_BELOW, metric=HEADLINE_METRIC):
    """qids that scored below `threshold` in an earlier run, worst first.

    The point of a sweep is the questions it gets wrong, and re-running those
    under one changed setting is the cheapest experiment available — far cheaper
    than another full sweep, and it isolates the change to the cases where it
    could possibly matter.

    `metric` picks which of the five decides. That is worth having rather than
    fixed: the questions with the worst `context_recall` are the retriever's to
    answer for, and the ones with the worst `faithfulness` are the generator's,
    so which set is worth re-running depends on what changed.
    """
    if not os.path.exists(path):
        raise RetrievalError(f"No previous run at {path}")
    failures = []
    for line in open(path):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        score = record.get(metric)
        if score is not None and score < threshold:
            failures.append((score, record["qid"]))
    if not failures:
        raise RetrievalError(f"No question in {path} scored {metric} below {threshold}")
    return [qid for _, qid in sorted(failures)]


async def run(limit, source, provider, judge_provider, rerank, top_n, out_path, seed,
              reranker=RERANKER, concurrency=CONCURRENCY, failures_from=None,
              failing_metric=HEADLINE_METRIC, sample_path=SAMPLE_PATH, daily=None):
    """Score `limit` questions and write a record per question.

    Which questions is decided in one of three ways, in this order:

        --failures-from   the questions an earlier run got wrong, worst first
        the fixed sample  `evals/sample_100.json` — the default, and the only
                          one of the three under which two runs are comparable
        a random draw     `--no-sample`, for a spot check
    """
    queries, qrels, answers = load_benchmark()
    drawn_from = "fixed sample"

    if failures_from:
        pool = failing_qids(failures_from, metric=failing_metric)
        drawn_from = f"failures in {os.path.basename(failures_from)}"
        print(f"{len(pool)} questions scored {failing_metric} below "
              f"{FAILING_BELOW} in {failures_from}")
    elif daily:
        # A different hundred each day, deterministic from the date, so thirty
        # days cover 3,000 of the 3,045. Not comparable across days — a change
        # in the mean could be your change or could be the different questions.
        pool = draw_sample(queries, date=daily)["qids"]
        drawn_from = f"daily sample for {daily}"
    elif sample_path:
        # The same hundred questions every run. A fresh draw per run would make
        # every comparison a comparison of two different question sets, and the
        # differences worth measuring here are smaller than that noise.
        pool = [qid for qid in load_sample(sample_path)["qids"] if qid in queries]
        if not pool:
            raise RetrievalError(
                f"None of the qids in {sample_path} are in this benchmark — "
                f"redraw it with: uv run python -m evals.sample")
    else:
        pool = sorted(queries)
        drawn_from = f"random draw, seed {seed}"

    if source:
        pool = [qid for qid in pool if queries[qid].get("source") == source]
    if not pool:
        raise RetrievalError(f"No queries with source {source!r}. Present: "
                             f"{sorted({q.get('source') for q in queries.values()})}")

    if failures_from or sample_path or daily:
        # Both arrive in a meaningful order — failures worst-first, the sample
        # sorted — so a smaller --limit takes a prefix rather than resampling,
        # which keeps a --limit 20 run a subset of the --limit 100 one.
        sample = pool[:limit]
    else:
        random.seed(seed)
        sample = pool if limit >= len(pool) else random.sample(pool, limit)

    done = _already_done(out_path)
    pending = [qid for qid in sample if qid not in done]

    # Charged on what is actually scored, so resuming an interrupted sweep is
    # not billed twice for work already in the file.
    budget = Budget()
    allowed = budget.clamp(len(pending))
    if allowed < len(pending):
        print(f"{budget.describe()} — scoring {allowed} of {len(pending)} "
              f"remaining; run again tomorrow, or raise EVAL_DAILY_LIMIT")
        pending = pending[:allowed]
    else:
        print(budget.describe())

    print(f"{len(sample)} questions ({drawn_from}) | "
          f"rerank {reranker if rerank else 'off'} | "
          f"pipeline {provider} | ragas grader {judge_provider} | "
          f"concurrency {concurrency}")
    if done:
        print(f"resuming: {len(done)} already in {out_path}, {len(pending)} to do")
    print()

    records, started = list(done.values()), time.time()
    handle = open(out_path, "a") if out_path else None
    scored_now = 0
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        with Pipeline(provider=provider, reranker=reranker) as pipeline:
            # One chain, reused for every question in the sweep: the model
            # behind it is cached by configuration, so the whole run shares one
            # pool without a context manager to hold it open.
            # One scorer for the whole sweep: building a metric compiles its
            # prompts, and the embeddings it needs for answer_relevancy are the
            # pipeline's own, already resident.
            scorer = Scorer(provider=judge_provider, model=JUDGE_MODEL,
                            embeddings=pipeline.embeddings,
                            max_tokens=JUDGE_MAX_TOKENS)

            async def finish(qid, candidates, passages):
                """The metered half: LLM rerank (if any), answer, RAGAS."""
                question = queries[qid]["query"]
                ok = passages is not None
                if passages is None:
                    passages, ok = await pipeline.rerank(
                        question, candidates, top_n)
                result_answer = await pipeline.answer(question, passages)
                # The passages as the grader sees them: exactly the text the
                # generator was given, so faithfulness is measured against what
                # the answer actually had rather than against the whole corpus.
                contexts = [p["text"] for p in passages]
                scores = await scorer.score(question, result_answer, contexts,
                                            reference=answers.get(qid))
                retrieved = [c["doc_id"] for c in candidates]
                gold = qrels[qid]["doc_id"]
                return {
                    "qid": qid, "source": queries[qid].get("source"),
                    "type": queries[qid].get("type"),
                    "question": question, "gold_doc": gold,
                    "gold_retrieved": gold in retrieved,
                    "gold_rank": retrieved.index(gold) + 1 if gold in retrieved else None,
                    "gold_kept": gold in [p["doc_id"] for p in passages],
                    "n_candidates": len(candidates),
                    "reranked": ok, "reranker": reranker if rerank else "none",
                    "answer": result_answer, "reference": answers.get(qid),
                    "abstained": abstained(result_answer), **scores,
                }

            position = 0
            for chunk in _chunks(pending, concurrency):
                # Sequential: one embedded-Qdrant client, one BGE-M3, one
                # cross-encoder. None of them is safe to drive in parallel,
                # and none of them is the stage worth parallelising.
                prepared = []
                for qid in chunk:
                    question = queries[qid]["query"]
                    candidates = pipeline.retrieve(question)
                    passages = None
                    if not candidates:
                        passages = []
                    elif not rerank:
                        passages = candidates[:top_n]
                    elif reranker == "cross-encoder":
                        passages, _ = await pipeline.rerank(
                            question, candidates, top_n)
                    prepared.append((qid, candidates, passages))

                # Concurrent: the API calls, which are latency-bound.
                results = await asyncio.gather(
                    *(finish(qid, cands, passages)
                      for qid, cands, passages in prepared),
                    return_exceptions=True)

                for (qid, _, _), record in zip(prepared, results):
                    position += 1
                    if isinstance(record, Exception):
                        print(f"  {position:>4}/{len(pending)} [!] {qid} "
                              f"{type(record).__name__}: {str(record)[:70]}")
                        continue
                    records.append(record)
                    scored_now += 1
                    if handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                    headline = record.get("factual_correctness")
                    mark = "-" if headline is None else f"{headline:.2f}"
                    rate = (time.time() - started) / position
                    eta = (len(pending) - position) * rate / 60
                    print(f"  {position:>4}/{len(pending)} [{mark:>4}] "
                          f"{record['source']:16} eta {eta:5.0f}m  "
                          f"{record['question'][:52]}")
    finally:
        if handle:
            handle.close()
        budget.consume(scored_now)

    if out_path:
        print(f"\nwrote {out_path}")
    elapsed = time.time() - started
    summarise(records, elapsed)
    _log_to_mlflow(records, elapsed, {
        "questions": len(sample), "scored": scored_now, "drawn_from": drawn_from,
        "pipeline_provider": provider, "judge_provider": judge_provider,
        "reranker": reranker if rerank else "none", "top_n": top_n,
        "daily": daily, "sample": None if daily else sample_path,
        "daily_budget_remaining": budget.remaining,
    })
    return records


def _log_to_mlflow(records, elapsed, params):
    """One MLflow run per sweep: the configuration, the aggregate, the rows.

    The five metric means go in as metrics — which is what makes two sweeps
    comparable in the UI — and the per-question records go in as a table,
    because the question actually asked afterwards is "which ones failed, and
    why", and that is a table rather than a number.

    Per-source means are logged too. This corpus's weakness is modality, and an
    aggregate that hides a 0.3 drop on image queries is the number that lets it
    keep happening.
    """
    if not records:
        return
    with evaluation_run(f"ragas-{params.get('drawn_from', 'sweep')}", params) as run:
        if run is None:
            return
        scored = [r for r in records if any(r.get(m) is not None for m in METRIC_NAMES)]
        aggregate = {m: _mean(scored, m) for m in METRIC_NAMES}
        aggregate["abstention_rate"] = (
            sum(1 for r in scored if r.get("abstained")) / len(scored) if scored else None)
        aggregate["gold_retrieved"] = (
            sum(1 for r in scored if r.get("gold_retrieved")) / len(scored) if scored else None)
        aggregate["gold_kept"] = (
            sum(1 for r in scored if r.get("gold_kept")) / len(scored) if scored else None)
        aggregate["seconds_per_question"] = elapsed / len(records) if records else None
        log_metrics(aggregate)

        for source in sorted({r.get("source") for r in scored if r.get("source")}):
            rows = [r for r in scored if r.get("source") == source]
            log_metrics({f"{source}/{m}": _mean(rows, m) for m in METRIC_NAMES})

        # Drop the long text so the table stays readable in the UI; the answers
        # are in the JSONL, which is where anyone reading one would look.
        log_table([{k: v for k, v in r.items()
                    if k not in ("answer", "reference", "metric_errors")}
                   for r in records], "per_question.json")


def _mean(rows, key):
    """The mean of a metric over the rows that have one, or None."""
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def _cell(value):
    return "   -" if value is None else f"{value:.2f}"


def summarise(records, elapsed=None):
    """The five metrics, overall and per query source.

    Split by source because that is where this corpus's weakness lives: image
    queries scored 2.41 against text's 2.73 on the old scale, with four times
    the abstentions, and an aggregate number hides exactly that.
    """
    scored = [r for r in records if any(r.get(m) is not None for m in METRIC_NAMES)]
    if not scored:
        print("\nno question was scored successfully")
        _print_metric_errors(records)
        return

    def block(name, rows):
        if not rows:
            return
        abstentions = sum(1 for r in rows if r.get("abstained")) / len(rows)
        found = sum(1 for r in rows if r["gold_retrieved"]) / len(rows)
        kept = sum(1 for r in rows if r["gold_kept"]) / len(rows)
        print(f"{name:16} {len(rows):>4} "
              f"{_cell(_mean(rows, 'faithfulness')):>6} "
              f"{_cell(_mean(rows, 'answer_relevancy')):>6} "
              f"{_cell(_mean(rows, 'context_precision')):>6} "
              f"{_cell(_mean(rows, 'context_recall')):>6} "
              f"{_cell(_mean(rows, 'factual_correctness')):>6} "
              f"{abstentions:>10.0%} {found:>8.0%} {kept:>7.0%}")

    print(f"\n{'':16} {'n':>4} {'faith':>6} {'a-rel':>6} {'c-prec':>6} "
          f"{'c-rec':>6} {'fact':>6} {'abstained':>10} {'gold@20':>8} {'kept':>7}")
    block("all", scored)
    print()
    for name in sorted({r["source"] for r in scored if r["source"]}):
        block(name, [r for r in scored if r["source"] == name])

    # A failure is only worth reading with the stage that caused it attached,
    # which is the whole reason there are five metrics instead of one.
    # `is not None`, not `or 0`: an unscored question is not a failing one, and
    # listing it as "below 0.5" is the same category error as scoring it zero.
    failures = sorted((r for r in scored if r.get(HEADLINE_METRIC) is not None
                       and r[HEADLINE_METRIC] < FAILING_BELOW),
                      key=lambda r: r[HEADLINE_METRIC])
    if failures:
        print(f"\n{len(failures)} question(s) with {HEADLINE_METRIC} below {FAILING_BELOW}:")
        for record in failures[:10]:
            print(f"\n  [{_cell(record.get(HEADLINE_METRIC))}] {record['question'][:70]}")
            print(f"       {_diagnose(record)}")

    _print_metric_errors(records)
    unscored = len(records) - len(scored)
    if unscored:
        print(f"\n{unscored} question(s) no metric could score")
    if elapsed:
        print(f"\n{elapsed:.0f}s for {len(records)} questions "
              f"({elapsed/len(records):.1f}s each)")


def _diagnose(record):
    """Which stage a failing question failed at, read off the metrics.

    The order matters: retrieval cannot be excused by anything downstream, and a
    faithful answer to a question the passages did not cover is a retrieval
    failure however well it reads.
    """
    if not record["gold_retrieved"]:
        return "gold doc never retrieved — a retrieval failure"
    if not record["gold_kept"]:
        return "gold retrieved but dropped by rerank — a reranker failure"
    recall = record.get("context_recall")
    if recall is not None and recall < 0.5:
        return f"gold doc in context but the passages missed it (c-rec {recall:.2f})"
    if record.get("abstained"):
        return "ABSTAINED with the evidence in context — a false abstention"
    faith = record.get("faithfulness")
    if faith is not None and faith < 0.7:
        return f"answer not grounded in its own passages (faith {faith:.2f})"
    if record.get(HEADLINE_METRIC) is None:
        # Reached when the headline metric could not be computed. Saying the
        # answer disagreed with the reference would report a broken grader as a
        # failing pipeline, which is the one thing this harness must not do.
        return f"{HEADLINE_METRIC} was not scored — see the metric errors below"
    return "evidence retrieved and used; the answer disagrees with the reference"


def _print_metric_errors(records):
    """What the grader could not compute, and why. Counted, not listed twice.

    A metric that failed is not a pipeline result, and a sweep that quietly
    reports fewer numbers than it promised is how a broken grader gets read as a
    failing pipeline.
    """
    tally = {}
    for record in records:
        for metric, reason in (record.get("metric_errors") or {}).items():
            tally.setdefault(metric, {}).setdefault(reason.split(":")[0], 0)
            tally[metric][reason.split(":")[0]] += 1
    if not tally:
        return
    print("\nmetrics that could not be computed:")
    for metric in sorted(tally):
        detail = ", ".join(f"{count}x {kind}" for kind, count in sorted(tally[metric].items()))
        print(f"  {metric:20} {detail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score the pipeline's answers per question with RAGAS")
    parser.add_argument("--limit", type=int, default=100,
                        help="questions to evaluate (default 100 — the fixed sample)")
    parser.add_argument("--sample", default=SAMPLE_PATH,
                        help="the fixed sample to draw from (default evals/sample_100.json)")
    parser.add_argument("--no-sample", action="store_true",
                        help="ignore the fixed sample and draw at random — a spot "
                             "check, not something to compare two runs with")
    parser.add_argument("--source", help="only this query source, e.g. text-image")
    parser.add_argument("--provider", default=None,
                        help=f"provider for rerank and answer (default {EVAL_PROVIDER})")
    parser.add_argument("--judge-provider", default=JUDGE_PROVIDER,
                        help=f"provider for the RAGAS metrics (default {JUDGE_PROVIDER}); "
                             f"keep it off the pipeline's own model, or the grader "
                             f"is marking its own work")
    parser.add_argument("--reranker", choices=list(RERANKERS), default=RERANKER,
                        help=f"llm (default) or cross-encoder \u2014 the comparison "
                             f"that decides ~73%% of the sweep's token cost")
    parser.add_argument("--no-rerank", action="store_true", help="the A/B against reranking")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="same seed = same questions")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"API calls in flight (default {CONCURRENCY})")
    parser.add_argument("--out", help="write per-question records as JSONL")
    parser.add_argument("--failures-from", metavar="JSONL",
                        help=f"re-run only the questions scoring below {FAILING_BELOW} "
                             f"in an earlier run, worst first \u2014 the cheap A/B for "
                             f"one changed setting")
    parser.add_argument("--failing-metric", default=HEADLINE_METRIC, choices=list(METRIC_NAMES),
                        help="which metric --failures-from sorts on: context_recall "
                             "selects the retriever's failures, faithfulness the "
                             "generator's")
    parser.add_argument("--daily", nargs="?", const="today", metavar="YYYY-MM-DD",
                        help="use the rotating sample for a date (default today, "
                             "UTC) instead of the fixed one \u2014 a different 100 "
                             "each day, so 30 days cover 3,000 of the 3,045")
    parser.add_argument("--report", help="re-print the summary from a saved JSONL and exit")
    arguments = parser.parse_args()

    daily = None
    if arguments.daily:
        import datetime
        daily = (datetime.datetime.now(datetime.timezone.utc).date().isoformat()
                 if arguments.daily == "today" else arguments.daily)
    setup_tracing()

    try:
        if arguments.report:
            summarise([json.loads(line) for line in open(arguments.report)])
        else:
            asyncio.run(run(arguments.limit, arguments.source,
                            arguments.provider or EVAL_PROVIDER,
                            arguments.judge_provider, not arguments.no_rerank,
                            arguments.top_n, arguments.out, arguments.seed,
                            arguments.reranker, arguments.concurrency,
                            arguments.failures_from, arguments.failing_metric,
                            None if (arguments.no_sample or arguments.daily)
                            else arguments.sample,
                            daily))
    except BudgetExhausted as error:
        sys.exit(f"STOPPED: {error}")
    except (RetrievalError, LLMError) as error:
        sys.exit(f"FAILED: {error}")
