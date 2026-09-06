"""Is the answer any good? DeepEval, against real models.

    uv run pytest tests/test_agent_quality.py --live

Skipped without `--live`, because every test here costs API calls — the flow's
six per question, plus the judge's. `test_agent_wiring.py` is the suite that
runs on every change; this is the one that runs before believing a change helped.

## What each metric is for

    faithfulness          is every claim in the answer supported by the passages
                          the flow retrieved? The hallucination check, and what
                          the verifier stage exists to keep passing.
    answer relevancy      does the answer address the question asked?
    contextual relevancy  were the passages the reranker kept actually about the
                          question? This one grades retrieval, not the answer,
                          and separates "retrieval missed it" from "had it and
                          fumbled it".
    citations (GEval)     does the answer cite [1]..[n], and only passages it
                          was given? A grounded answer with an invented citation
                          is still a fabrication, and no stock metric looks for
                          it.
    false abstention      does the answer refuse *while holding the evidence*?
        (GEval)           This is the finding the whole flow was built around —
                          abstentions quadrupled on image queries with the gold
                          document already in context — and it is invisible to
                          every other metric, all of which read a refusal as
                          merely a low score.

## Thresholds are deliberately not tight

These are regression guards, not a leaderboard. A threshold set at the current
measured value turns every run of a stochastic system into a coin flip, and a
suite that fails at random gets ignored. They are set where a real regression
would trip them and ordinary variance would not; `evals/answer_eval.py` is what
measures quality, over thousands of questions rather than three.
"""

import pytest
from deepeval.metrics import (AnswerRelevancyMetric, ContextualRelevancyMetric,
                              FaithfulnessMetric, GEval)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

pytestmark = pytest.mark.live

# Questions the corpus can answer, one per row kind — because modality is this
# corpus's measured weakness, and a suite that only asks text questions would
# not see the failure that matters.
QUESTIONS = [
    pytest.param("how does detection probability vary with range", id="figure"),
    pytest.param("what methods are used for adversarial robustness evaluation", id="text"),
]


@pytest.fixture(scope="module")
def answered(flow):
    """One real run per question, reused by every metric that grades it.

    Module-scoped and computed once: the flow costs six model calls per
    question, and running it again per metric would multiply that by five for
    no new information.
    """
    import asyncio

    runs = {}

    def run(question):
        if question not in runs:
            runs[question] = asyncio.get_event_loop().run_until_complete(
                flow.arun(question))
        return runs[question]

    return run


def case(run):
    """A finished run as a DeepEval test case.

    `retrieval_context` is the passages the generator actually saw, not the
    whole candidate list — faithfulness has to be measured against the evidence
    the answer was given, or it grades the corpus rather than the answer.
    """
    return LLMTestCase(
        input=run["question"],
        actual_output=run["answer"] or "",
        retrieval_context=[p["text"] for p in run.get("passages", [])],
    )


# Built in fixtures rather than at module level, and that is not a style
# preference: constructing a DeepEval metric resolves its judge, and with no
# `model=` that resolution is OpenAI's — which raises at *import* time if
# OPENAI_API_KEY is unset, before any skip marker can apply. A module-level
# metric therefore breaks collection of the whole suite on a machine that was
# never going to run these tests. Deferring construction keeps `--live` the only
# thing that decides whether a judge is needed.

CITATION_CRITERIA = (
    "Determine whether the actual output cites its sources as bracketed numbers "
    "like [1] or [2], and whether every number it cites is within the range of "
    "the retrieval context provided. An answer that makes factual claims with no "
    "citation at all should score low. An answer that cites a number higher than "
    "the number of retrieved passages is fabricating a source and should score 0. "
    "An answer that correctly declines to answer needs no citations and should "
    "not be penalised."
)

FALSE_ABSTENTION_CRITERIA = (
    "Determine whether the actual output declines to answer despite the retrieval "
    "context containing information that addresses the input question. "
    "Score 1.0 if the output answers the question, or if it declines AND the "
    "retrieval context genuinely does not address the question. "
    "Score 0.0 if the output declines to answer, says the passages do not support "
    "an answer, or hedges into a non-answer, while the retrieval context clearly "
    "contains material that addresses the question."
)

ALL_PARAMS = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
              LLMTestCaseParams.RETRIEVAL_CONTEXT]


@pytest.fixture(scope="module")
def citation_metric(evaluator):
    return GEval(name="Citations", criteria=CITATION_CRITERIA,
                 evaluation_params=ALL_PARAMS, threshold=0.6, model=evaluator)


@pytest.fixture(scope="module")
def false_abstention_metric(evaluator):
    return GEval(name="Does not falsely abstain", criteria=FALSE_ABSTENTION_CRITERIA,
                 evaluation_params=ALL_PARAMS, threshold=0.6, model=evaluator)


@pytest.fixture
def question_id(request):
    """The parametrised id ("figure", "text"), so a metric is logged per question.

    A mean over two very different questions would hide exactly the split this
    corpus is weak at.
    """
    return getattr(request.node, "callspec", None) and request.node.callspec.id or "all"


@pytest.mark.parametrize("question", QUESTIONS)
class TestAnswerQuality:

    def test_answer_is_faithful_to_its_passages(self, answered, question,
                                               question_id, evaluator, graded):
        """Nothing in the answer that the passages do not support.

        The threshold is the highest of the five, because this is the failure
        the pipeline is least allowed to have: a fluent, well-cited, confident
        fabrication is worse than a refusal.

        `penalize_ambiguous_claims=True` is not optional here, and the default
        is a trap. DeepEval labels each claim `yes`, `no` or `idk`, and by
        default only `no` — a direct contradiction of the passages — lowers the
        score. A claim the passages simply never mention is `idk`, and is
        ignored. Measured against this corpus's own shape:

            answer                                          default  penalized
            grounded, correct                                  1.00       1.00
            invented — rainfall, from a passage about p_D      1.00       0.00
            flipped — "rises with range"                       1.00       0.00

        A fabrication scoring 1.00 is the exact failure mode the verifier stage
        exists to catch, and a test that scores it 1.00 would certify the bug.
        """
        graded(case(answered(question)),
               FaithfulnessMetric(threshold=0.7, model=evaluator,
                                  penalize_ambiguous_claims=True),
               f"faithfulness/{question_id}")

    def test_answer_addresses_the_question(self, answered, question, question_id,
                                           evaluator, graded):
        graded(case(answered(question)),
               AnswerRelevancyMetric(threshold=0.6, model=evaluator),
               f"answer_relevancy/{question_id}")

    def test_retrieved_passages_are_about_the_question(self, answered, question,
                                                       question_id, evaluator, graded):
        """Grades retrieval, not the answer.

        Low here with a good answer means the reranker kept noise the generator
        managed to ignore; low here with a bad answer means retrieval is the bug.

        **The threshold is well below the measured value on purpose.** Measured
        at 0.25 on both questions — roughly one passage in four judged relevant
        out of a top-5. Setting the bar at 0.25 would make every run a coin
        flip, which is the failure mode that gets a suite ignored; setting it at
        0.15 means a *drop* trips it and ordinary variance does not.

        0.25 is not a good number, and it is the most interesting result this
        suite produced. Whether it means the reranker keeps three weak passages
        out of five, or that the judge is strict about what "relevant" means for
        a figure description, is unsettled — `evals/answer_eval.py`'s
        `context_precision` over a hundred questions is what would settle it.
        """
        graded(case(answered(question)),
               ContextualRelevancyMetric(threshold=0.15, model=evaluator),
               f"contextual_relevancy/{question_id}")

    def test_answer_cites_only_passages_it_was_given(self, answered, question,
                                                     question_id, citation_metric, graded):
        graded(case(answered(question)), citation_metric, f"citations/{question_id}")

    def test_answer_does_not_abstain_with_evidence_in_hand(self, answered, question,
                                                           question_id,
                                                           false_abstention_metric, graded):
        graded(case(answered(question)), false_abstention_metric,
               f"false_abstention/{question_id}")


class TestRefusal:
    """The flow must still refuse what it should refuse."""

    def test_out_of_corpus_question_is_refused(self, flow):
        """Scope is decided after retrieval, on the reranker's best score.

        A question this corpus has nothing on should fall below the floor and be
        refused — and this needs no judge, because the refusal is a decision the
        flow records rather than a text to grade.
        """
        import asyncio

        run = asyncio.get_event_loop().run_until_complete(
            flow.arun("what is a good recipe for carbonara"))
        trace = run["trace"]
        assert trace.get("refused"), (
            f"an out-of-corpus question was answered rather than refused: "
            f"{(run['answer'] or '')[:200]}")
