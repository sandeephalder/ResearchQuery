"""Per-request quality signals, computed without a single extra model call.

The RAG triad — context relevance, groundedness, answer relevance — is usually
measured with an LLM judge, which is three more calls per question. Everything
here comes from work the flow has already done:

    context relevance   the reranker's own score. It is the one stage that read
                        the question against each passage and said how well they
                        match, which is the definition of the metric.
    groundedness        the verifier's findings, plus whether the answer's
                        citations resolve to passages that were actually sent.
    answer relevance    cosine between the question and the answer, through the
                        BGE-M3 encoder that is already loaded and local.
    latency             the per-stage timings the graph records anyway.

**These are not RAGAS.** Read them as cheap, continuous signals that move when
quality moves, not as calibrated scores comparable to a published benchmark.
Two are worth being explicit about:

`answer_relevance` is a direct question-answer embedding similarity. RAGAS
instead has a model generate questions from the answer and compares *those* to
the original, which measures something subtly different and costs a call. A
direct cosine will read low for a correct answer phrased very differently from
the question, and high for a fluent answer that restates it and says nothing.

`groundedness` is a heuristic over the verifier's problem list and citation
validity, not an entailment check. It catches a fabricated citation and a claim
the verifier objected to; it will not catch a well-cited sentence that subtly
misstates its source — that is what the verifier itself is for.

Set `METRICS_JUDGE=1` if you would rather pay for a judged score; nothing here
implements one yet, and `evals/answer_eval.py` is the harness that would.
"""

import dataclasses
import json
import logging
import math
import os
import re
import time

from .constants import METRICS_INCLUDE_TEXT, METRICS_LOG_PATH

log = logging.getLogger("agents.metrics")

# `[3]`, and the bracketed form some models emit instead — 【3†L1-L4】.
_CITATION_RE = re.compile(r"\[(\d{1,2})\]|【(\d{1,2})†")

# Which verifier findings bear on groundedness. A false abstention is a
# failure of recall, not of grounding: the answer that was not given cannot be
# unsupported by the passages.
_GROUNDING_FAULTS = frozenset({"flipped_logic", "unsupported_claim", "miscitation"})


@dataclasses.dataclass
class Metrics:
    """One request's signals. Any of the three may be None when unmeasurable."""

    context_relevance: float | None = None
    groundedness: float | None = None
    answer_relevance: float | None = None
    latency_seconds: float = 0.0
    stage_latency: dict = dataclasses.field(default_factory=dict)

    # The workings, so a number that looks wrong can be explained rather than
    # just distrusted.
    citations_found: int = 0
    citations_valid: int = 0
    grounding_faults: list = dataclasses.field(default_factory=list)
    abstained: bool = False
    refused: str | None = None
    passages: int = 0

    def to_dict(self):
        return dataclasses.asdict(self)


def _citations(answer):
    """The passage numbers an answer cites, in order, with repeats."""
    return [int(a or b) for a, b in _CITATION_RE.findall(answer or "")]


def _context_relevance(passages, reranker):
    """Mean reranker score over the passages that were actually sent.

    The scales differ, so they are normalised to 0-1 here: the cross-encoder is
    already a sigmoid, and the LLM reranker's rubric is 0-10. Without a
    reranker there is no score — fused RRF rank carries no notion of "how
    relevant", only "how agreed-upon" — so the metric is None rather than a
    number that cannot be compared with the others.
    """
    scores = [p.get("rerank_score") for p in passages
              if p.get("rerank_score") is not None]
    if not scores:
        return None
    if reranker == "llm":
        scores = [s / 10.0 for s in scores]
    return round(sum(scores) / len(scores), 4)


def _groundedness(answer, passages, verdicts):
    """Citation validity, voided by any grounding fault the verifier found.

    A citation that points past the end of the passage list is a fabricated
    source, which is the failure this catches cheaply and reliably. The
    verifier's own objections are the other half: if it said a claim was
    unsupported, no amount of well-formed citation makes the answer grounded.
    """
    cited = _citations(answer)
    valid = [n for n in cited if 1 <= n <= len(passages)]
    faults = []
    for verdict in verdicts or []:
        for problem in verdict.get("problems") or []:
            if problem.get("type") in _GROUNDING_FAULTS:
                faults.append(problem["type"])

    if not cited:
        # An answer with no citations is either an abstention — handled by the
        # caller — or an ungrounded one. Neither deserves a score above zero.
        score = 0.0
    else:
        score = len(valid) / len(cited)
    # Only the *surviving* faults count: one the repair fixed is a stage doing
    # its job, not a defect in the answer that shipped.
    if faults and verdicts and not (verdicts[-1].get("allowed", True)):
        score = 0.0
    return round(score, 4), valid, cited, faults


def _answer_relevance(embeddings, question, answer):
    """Cosine between the question and the answer, through the local embeddings.

    Free and offline: BGE-M3 is already resident, and one more forward pass is
    ~60 ms. See the module docstring for what this does and does not mean.

    `embeddings` is the `retrieval.embeddings.BGEM3Embeddings` the pipeline
    holds — a LangChain `Embeddings`, so this asks it the way anything else
    would.
    """
    if embeddings is None or not (answer or "").strip():
        return None
    try:
        question_vector = embeddings.embed_query(question)
        answer_vector = embeddings.embed_query(answer)
    except Exception:                       # noqa: BLE001 — a metric must not break a run
        log.warning("could not embed the answer; answer_relevance unavailable")
        return None
    dot = sum(a * b for a, b in zip(question_vector, answer_vector))
    norm = (math.sqrt(sum(a * a for a in question_vector))
            * math.sqrt(sum(b * b for b in answer_vector)))
    return round(dot / norm, 4) if norm else None


def evaluate(result, encoder=None):
    """Metrics for one finished run. `result` is what `arun` returns."""
    trace = result.get("trace") or {}
    passages = result.get("passages") or []
    answer = result.get("answer") or ""
    refused = trace.get("refused")

    groundedness, valid, cited, faults = _groundedness(
        answer, passages, trace.get("guardrail_out"))

    # A refusal is a decision, not a bad answer, and a run that was never asked
    # to answer (`--no-answer`, or retrieval-only) has nothing to score at all.
    # Both would otherwise land as groundedness 0.0 and drag down every average
    # with runs that behaved exactly as asked.
    abstained = bool(refused)
    unscorable = abstained or not answer.strip()
    return Metrics(
        context_relevance=_context_relevance(passages, result.get("reranker")),
        groundedness=None if unscorable else groundedness,
        answer_relevance=None if unscorable else _answer_relevance(
            encoder, result["question"], answer),
        latency_seconds=(trace.get("timings") or {}).get("total", 0.0),
        stage_latency={k: v for k, v in (trace.get("timings") or {}).items() if k != "total"},
        citations_found=len(cited), citations_valid=len(valid),
        grounding_faults=faults, abstained=abstained, refused=refused,
        passages=len(passages))


def record(result, metrics, identity):
    """One JSON line per request, keyed by user and session.

    Written to a file rather than only to the logger because these are meant to
    be read back in aggregate — "what has context relevance done this week, for
    this user" is a question about a table, not a log stream.

    The session *token* never appears here. Ids do.
    """
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user_id": identity.user_id, "session_id": identity.session_id,
        "request_id": identity.request_id,
        **metrics.to_dict(),
    }
    if METRICS_INCLUDE_TEXT:
        entry["question"] = result.get("question")
        entry["answer"] = result.get("answer")

    log.info("user=%s session=%s ctx=%s grounded=%s answer_rel=%s %.1fs%s",
             identity.user_id or "anonymous", identity.session_id or "-",
             metrics.context_relevance, metrics.groundedness,
             metrics.answer_relevance, metrics.latency_seconds,
             f" refused={metrics.refused}" if metrics.refused else "")

    if not METRICS_LOG_PATH:
        return entry
    try:
        os.makedirs(os.path.dirname(METRICS_LOG_PATH), exist_ok=True)
        with open(METRICS_LOG_PATH, "a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as error:
        # A metrics sink that cannot write must not take the answer with it.
        log.warning("could not write %s: %s", METRICS_LOG_PATH, error)
    return entry


# --------------------------------------------------------------------------- #
# Reading it back
# --------------------------------------------------------------------------- #

def summarise(path=METRICS_LOG_PATH, group="user_id"):
    """The log as a table, grouped by user or by session.

        uv run python -m agents.metrics
        uv run python -m agents.metrics session_id

    Averages skip the None entries rather than counting them as zero — a
    refusal has no groundedness, and averaging it as 0 would make a system that
    correctly declines look like one that answers badly.
    """
    rows = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.setdefault(entry.get(group) or "anonymous", []).append(entry)

    def mean(entries, field):
        values = [e[field] for e in entries if e.get(field) is not None]
        return round(sum(values) / len(values), 3) if values else None

    print(f"{group:26} {'n':>4} {'ctx':>6} {'grnd':>6} {'ans':>6} {'p50 s':>7} "
          f"{'abstain':>8}")
    for key, entries in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        latencies = sorted(e["latency_seconds"] for e in entries)
        p50 = latencies[len(latencies) // 2] if latencies else 0
        abstained = sum(1 for e in entries if e.get("abstained"))
        print(f"{str(key)[:26]:26} {len(entries):>4} "
              f"{_fmt(mean(entries, 'context_relevance')):>6} "
              f"{_fmt(mean(entries, 'groundedness')):>6} "
              f"{_fmt(mean(entries, 'answer_relevance')):>6} "
              f"{p50:>7.1f} {abstained / len(entries):>7.0%}")


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    import sys

    field = sys.argv[1] if len(sys.argv) > 1 else "user_id"
    try:
        summarise(group=field)
    except FileNotFoundError:
        raise SystemExit(f"no metrics yet at {METRICS_LOG_PATH}")
