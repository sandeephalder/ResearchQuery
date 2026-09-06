"""The RAGAS metric set, built on this repo's own models.

    scorer = Scorer(provider="openai")
    scores = await scorer.score(question, answer, contexts, reference)

Five metrics, and each answers a question the pipeline can act on. Three of them
need no reference answer, which matters more than it sounds: they can be run
over questions the benchmark has no gold answer for, including a user's.

    faithfulness         is every claim in the answer supported by the passages
                         it was given? This is the hallucination metric, and the
                         one the verifier stage exists to prevent failing.
    answer_relevancy     does the answer address the question that was asked,
                         rather than a nearby one? Uses embeddings, not a
                         judgement — so it is the cheapest of the five here,
                         because the embeddings are local.
    context_precision    of the passages retrieved, how many were useful, and
                         were they ranked first? This is the reranker's score.
    context_recall       did retrieval find what the reference answer needed?
                         This is the retriever's score, and the two together are
                         what separates "retrieval missed it" from "retrieval
                         had it and the answer fumbled it".
    factual_correctness  does the answer agree with the reference, claim by
                         claim? This replaced a hand-written 0-3 judge.

The last two need `reference`, so they are skipped when there is not one.

## Everything runs on the project's own models

RAGAS takes a LangChain model and a LangChain `Embeddings` directly, so the
grader is built by `llm.models.chat_model` from the same registry as every stage
it is grading, and `answer_relevancy` embeds through the same BGE-M3 that built
the index. Two consequences worth stating:

- **The grader must not be the model being graded.** Same provider is fine;
  the same *model* judging its own answer is not an independent measurement.
  `JUDGE_PROVIDER` defaults away from the pipeline's provider for this reason.
- **`answer_relevancy` is free.** It reverse-generates questions from the answer
  and compares them to the real one by cosine — the comparison is local, on
  weights already resident.

## Why per-sample rather than `ragas.evaluate`

`evaluate()` scores a whole dataset at once, which would mean holding every
answer until the sweep finished. A sweep is thousands of API calls over hours,
and the existing harness appends each question's record as it is judged so an
interrupted run resumes. Per-sample `single_turn_ascore` keeps that.
"""

import asyncio
import os

from . import ragas_compat                                           # noqa: F401

from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (FactualCorrectness, Faithfulness,
                           LLMContextPrecisionWithReference, LLMContextRecall,
                           ResponseRelevancy)

from llm.models import LLMError, chat_model

# What a metric returns when it could not be computed. Distinct from 0.0, which
# is a real score meaning "nothing in the answer was supported" — a run that
# cannot tell those apart reports a broken grader as a failing pipeline.
UNSCORED = None

# How many questions `answer_relevancy` reverse-generates from the answer before
# comparing them to the real one. RAGAS defaults to 3, and asks for them as
# `n=3` on a single completion — which groq rejects outright:
#
#     'n' : number must be at most 1
#
# So the default here is 1. What that costs is variance: one reverse-generated
# question is a noisier estimate than the mean of three, so a single question's
# answer_relevancy should be read as indicative and a sweep's mean as the
# measurement. Raise it on a provider that allows n>1 — OpenAI does.
RELEVANCY_STRICTNESS = int(os.getenv("RAGAS_RELEVANCY_STRICTNESS", "1"))

# Needs `reference`; skipped on a question the benchmark has no gold answer for.
NEEDS_REFERENCE = ("context_precision", "context_recall", "factual_correctness")

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "factual_correctness")


class Scorer:
    """The five metrics, built once and reused for a whole sweep.

    Building a metric compiles its prompts; doing that per question would be
    thousands of needless rebuilds. The models behind them are cached by
    `chat_model`, so a sweep shares one pool across every question in flight.
    """

    def __init__(self, provider=None, model=None, embeddings=None, max_tokens=2048,
                 timeout=180.0, strictness=RELEVANCY_STRICTNESS):
        # A grader that reasons before answering still has to fit its verdict in
        # the budget, and RAGAS asks for structured output — the same failure
        # the pipeline's own stages hit at 1,500.
        self.llm = LangchainLLMWrapper(
            chat_model(provider, model=model, temperature=0.0,
                       max_tokens=max_tokens, timeout=timeout))
        # Local and already loaded when a sweep is running, so answer_relevancy
        # costs one forward pass rather than an API call.
        self.embeddings = (LangchainEmbeddingsWrapper(embeddings)
                           if embeddings is not None else None)

        self.faithfulness = Faithfulness(llm=self.llm)
        self.context_precision = LLMContextPrecisionWithReference(llm=self.llm)
        self.context_recall = LLMContextRecall(llm=self.llm)
        # `FactualCorrectness`, not `AnswerCorrectness`. The latter blends an
        # LLM F1 over statements with an embedding similarity, and needs an
        # `AnswerSimilarity` sub-metric it does not build from `embeddings=`
        # alone — every score failed with `AssertionError: AnswerSimilarity
        # must be set`. `FactualCorrectness` is claim-by-claim and LLM-only,
        # which is what this metric is documented above as measuring.
        self.factual_correctness = FactualCorrectness(llm=self.llm)
        self.answer_relevancy = (ResponseRelevancy(llm=self.llm, embeddings=self.embeddings,
                                                   strictness=strictness)
                                 if self.embeddings is not None else None)

    def sample(self, question, answer, contexts, reference=None):
        return SingleTurnSample(user_input=question, response=answer or "",
                                retrieved_contexts=list(contexts),
                                reference=reference)

    async def _one(self, metric, sample, name):
        """One metric, or `UNSCORED` and the reason.

        A metric that fails must not take the run with it: a grader's rate limit
        is not a fact about the pipeline, and a sweep that dies on question 300
        of 3,000 has measured nothing.
        """
        if metric is None:
            return name, UNSCORED, "metric not configured"
        try:
            value = await metric.single_turn_ascore(sample)
        except Exception as error:                  # noqa: BLE001 — see the docstring
            return name, UNSCORED, f"{type(error).__name__}: {str(error)[:120]}"
        try:
            return name, round(float(value), 4), None
        except (TypeError, ValueError):
            return name, UNSCORED, f"non-numeric score {value!r}"

    async def score(self, question, answer, contexts, reference=None):
        """{metric: score or None} plus `metric_errors` for whatever failed.

        The five run concurrently: they are independent API calls, and a sweep
        already limits how many questions are in flight, so this is where the
        remaining latency is.
        """
        sample = self.sample(question, answer, contexts, reference)
        wanted = [("faithfulness", self.faithfulness),
                  ("answer_relevancy", self.answer_relevancy)]
        if reference:
            wanted += [("context_precision", self.context_precision),
                       ("context_recall", self.context_recall),
                       ("factual_correctness", self.factual_correctness)]

        results = await asyncio.gather(
            *(self._one(metric, sample, name) for name, metric in wanted))

        scores = {name: UNSCORED for name in METRIC_NAMES}
        errors = {}
        for name, value, error in results:
            scores[name] = value
            if error:
                errors[name] = error
        if errors:
            scores["metric_errors"] = errors
        return scores


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #

# The system prompt tells the model to refuse when the passages do not support an
# answer, so a refusal is correct behaviour on a failed retrieval and a failure
# on a successful one. RAGAS has no metric for that distinction — every one of
# its scores reads a refusal as a low score — and the distinction is the single
# finding this pipeline's evaluation turned up: abstentions quadrupling on image
# queries with the gold document already in context.
#
# So it is detected here, by phrase, and it is a heuristic rather than a
# judgement: cheap, deterministic, and it does not spend a call. Paired with
# `gold_kept`, an abstention with the gold passage in context is a false one.
ABSTENTION_MARKERS = (
    "do not support", "does not support", "not supported by",
    "no information", "does not contain", "do not contain",
    "cannot answer", "can't answer", "unable to answer",
    "not mentioned", "insufficient information", "no relevant",
    "the passages do not", "the provided passages",
)


def abstained(answer):
    """Did the answer decline, rather than answer?

    Only the opening is read. An answer that resolves the question and then
    notes a limitation is not an abstention, and the refusals this prompt
    produces say so in their first sentence.
    """
    text = (answer or "").strip().lower()
    if not text:
        return True
    return any(marker in text[:400] for marker in ABSTENTION_MARKERS)
