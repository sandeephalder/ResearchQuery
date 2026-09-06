"""Fixtures for both test files, and the rule about which one needs a network.

Two kinds of test live here, and the split is the point:

    test_agent_wiring.py    every node, every branch, no network. Stubbed
                            chains, real retrieval. Runs on every change.
    test_agent_quality.py   DeepEval metrics against real model output. Costs
                            API calls, so it is opt-in.

`--live` (or `LIVE_TESTS=1`) enables the second. Without it the quality tests
are skipped rather than failed: a test that cannot run is not a test that
failed, and a suite that goes red because a key is missing gets ignored.

## One pipeline for the whole session

Constructing an `AgentPipeline` loads BGE-M3 (~15 s), loads the cross-encoder,
opens the embedded Qdrant store and compiles the NeMo rails. The store admits a
*single process*, so a fixture per test would not merely be slow — the second
one would fail on a lock the first still held. Session scope is not an
optimisation here, it is the only thing that works.
"""

import asyncio
import os

import pytest

QUESTION = "how does detection probability vary with range"

# DeepEval gathers a metric's sub-calls under a timeout and marks the metric
# timed out when they overrun. Its default is tuned for a fast hosted judge;
# the judges available here are not that. `faithfulness` alone extracts truths
# from five passages of up to 4,000 characters, then claims from the answer,
# then a verdict per claim — measured at over 200 s on glm-5.3-flash, which the
# default cancels midway and reports as a failing pipeline.
#
# Raised rather than removed: a judge that has genuinely hung should still end
# the run, and 15 minutes is far past any healthy call.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "900")
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "60")
# Nothing here should reach Confident AI's hosted dashboard: these tests run on
# a corpus of papers and a local store, and a test suite should not post
# anywhere by default.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="run the tests that call real models (costs API calls)")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: needs real models and API keys")


def pytest_collection_modifyitems(config, items):
    import os
    if config.getoption("--live") or os.getenv("LIVE_TESTS", "").lower() in ("1", "true", "yes"):
        return
    skip = pytest.mark.skip(reason="needs --live (or LIVE_TESTS=1): calls real models")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def event_loop():
    """One loop for the session, because the pipeline fixture is session-scoped.

    pytest-asyncio's default is a loop per test, and an async session fixture
    built on one loop cannot be awaited from another.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def flow():
    """The AgentPipeline, built once. See the module docstring for why."""
    from agents.orchestrator import AgentPipeline

    pipeline = AgentPipeline()
    # Touching `.pipeline` is what opens Qdrant and loads BGE-M3. Doing it here
    # keeps the cost in fixture setup rather than in whichever test ran first.
    _ = pipeline.pipeline
    yield pipeline
    pipeline.close()


@pytest.fixture(scope="session")
def evaluator():
    """The DeepEval judge — a LangChain model, not OpenAI's default."""
    from tests.evaluator import LangChainEvaluator

    return LangChainEvaluator()


@pytest.fixture(scope="session")
def mlflow_run():
    """One MLflow run for the whole live suite, or None when tracing is off.

    Session-scoped so all eleven tests land in one run rather than eleven, and
    so the run's parameters describe the configuration once. The traces the
    stages emit are separate objects — a run holds the scores, the traces hold
    what produced them, and MLflow keeps both under the same experiment.
    """
    import os

    from tracing import evaluation_run

    from tests.evaluator import JUDGE_MODEL, JUDGE_PROVIDER

    with evaluation_run("deepeval-agent-quality", {
        "suite": "deepeval",
        "judge_provider": JUDGE_PROVIDER,
        "judge_model": JUDGE_MODEL or "provider default",
        "agent_provider": os.getenv("AGENT_PROVIDER", "default"),
    }) as run:
        yield run


@pytest.fixture
def graded(mlflow_run):
    """Score a test case, log it to MLflow, then assert — in that order.

    `assert_test` raises on a failing metric, so anything logged after it never
    runs for exactly the cases worth recording. Measuring first means a failing
    score reaches MLflow and *then* fails the test, which is the way round that
    makes the run useful: the chart shows the regression rather than a gap.
    """
    from deepeval import assert_test

    from tracing import log_metrics

    def run(case, metric, label):
        try:
            metric.measure(case)
            log_metrics({label: metric.score})
        except Exception:                   # noqa: BLE001 — never let telemetry
            pass                            # decide whether a test passes
        assert_test(case, [metric])

    return run


@pytest.fixture
def question():
    return QUESTION
