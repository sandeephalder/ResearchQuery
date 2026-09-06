"""MLflow tracing for the whole flow, off unless you ask for it.

    uv run python -m agents.orchestrator "..." --trace
    MLFLOW_TRACING=1 uv run python -m retrieval.pipeline "..."
    MLFLOW_TRACING=1 MLFLOW_TRACKING_URI=http://localhost:5000 uv run uvicorn backend.main:app

Then read them:

    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

A question through this system is six model calls across five stages, two
retrievers, a fusion and a reranker. When an answer is wrong the question is
always *which stage* — and a stack of nested spans with each stage's inputs and
outputs answers it in a way that reading four log lines does not.

## Two layers, and they compose

**`mlflow.langchain.autolog()`** traces every LCEL chain, chat model and
LangGraph node for free — the prompts actually sent, the replies, the token
counts. Since the rewrite that is *most* of this repo, so most of the tracing is
had for nothing.

**`@mlflow.trace` decorators** on the stages, in `retrieval/` and `agents/`.
Autolog sees a chat model being called; it cannot see that the call was the
router, that the router chose `figure`, or that the choice widened retrieval by
four candidates. The decorators put that structure back.

## Off by default, and genuinely off

Tracing is opt-in because a decorator that always fires would write an `mlruns/`
tree beside the corpus on every CLI run and every test. When `MLFLOW_TRACING` is
unset this module calls `mlflow.tracing.disable()`, so the decorators stay in the
source and cost nothing at runtime.

That is also why every decorated function imports `mlflow` at module level: the
decorators have to exist whether or not tracing is on, and only their *effect* is
switched.

## Reading a trace

    span                        what to look at when the answer is wrong
    ────────────────────────────────────────────────────────────────────
    agent_pipeline              total latency, the question, the final answer
      guardrail_in              did a rail block it, and which
      route                     kind, confidence, and what it did to retrieval
      retrieve                  how many candidates, from which leg
        qdrant_vector           the semantic leg's 10
        bm25                    the lexical leg's 10
        fuse                    what survived, and its rank in each leg
      rerank                    the 5 kept, and their scores — the scope floor
      extract                   the facts, or [] if the passages carried none
      generate                  the draft
      verify                    the verdict, and any repair that followed
"""

import contextlib
import os

# The hint is addressed to a coding assistant, not to someone running a query,
# and it prints on every import. Set before mlflow is imported to suppress it.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow                                                        # noqa: E402
from mlflow.entities import SpanType                                 # noqa: E402

ENABLED = os.getenv("MLFLOW_TRACING", "").lower() in ("1", "true", "yes")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "research-query")

# Where traces go when nothing says otherwise. **Not** MLflow's own default,
# which is the `./mlruns` file store — and as of MLflow 3.16 that store raises
# rather than writes:
#
#     MlflowException: The filesystem tracking backend (e.g., './mlruns') is in
#     maintenance mode and will not receive further updates. Please migrate to a
#     database backend (e.g., 'sqlite:///mlflow.db')
#
# So `MLFLOW_TRACING=1` on its own would fail at `set_experiment`, before a
# single span was recorded, on a question about storage that nobody asking for a
# trace has in mind. sqlite is the backend MLflow's own message recommends, it
# needs no server, and the file sits beside the corpus rather than in whatever
# directory the command happened to run from.
#
# Point `MLFLOW_TRACKING_URI` at a real server for anything shared.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRACKING_URI = f"sqlite:///{os.path.join(_HERE, 'mlflow.db')}"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI

_configured = False


def setup(experiment=None, tracking_uri=None, autolog=True):
    """Point MLflow somewhere and switch tracing on. Idempotent.

    Called by the CLI entry points and by the FastAPI lifespan. Calling it more
    than once is harmless, which matters because a process can reach it through
    several of those at once.

    Returns whether tracing ended up enabled, so a caller can say so rather than
    leaving someone to wonder why an experiment is empty.
    """
    global _configured
    if not ENABLED:
        # Not merely "do not configure" — actively off, so a decorator does not
        # write a local mlruns/ tree as a side effect of being in the source.
        mlflow.tracing.disable()
        return False
    if _configured:
        return True

    mlflow.set_tracking_uri(tracking_uri or TRACKING_URI)
    mlflow.set_experiment(experiment or EXPERIMENT)
    mlflow.tracing.enable()
    if autolog:
        # Every LCEL chain, chat model and LangGraph node, without a decorator.
        mlflow.langchain.autolog()
    _configured = True
    return True


def annotate(**metadata):
    """Attach metadata to the trace in progress, if there is one.

    Used to hang the caller's identity on a run — `mlflow.trace.user` and
    `mlflow.trace.session` are MLflow's standard fields, so a trace can be found
    by who asked for it:

        mlflow.search_traces(
            filter_string="metadata.`mlflow.trace.user` = 'user_123'")

    A no-op when tracing is off, so callers need not guard the call.
    """
    if not ENABLED:
        return
    try:
        mlflow.update_current_trace(metadata={k: str(v) for k, v in metadata.items()
                                              if v is not None})
    except Exception:                       # noqa: BLE001 — telemetry must never
        pass                                # break the thing it is observing


# Disable at import when off, so a module that imports a decorator but never
# calls `setup` — a notebook, a test — still writes nothing.
if not ENABLED:
    mlflow.tracing.disable()


# --------------------------------------------------------------------------- #
# Evaluation runs
# --------------------------------------------------------------------------- #
#
# An eval sweep produces two kinds of thing MLflow wants separately: **traces**,
# one per question, showing which stage did what; and a **run**, holding the
# configuration that produced them and the aggregate that came out. Both suites
# — RAGAS in `evals/` and DeepEval in `tests/` — log into the same experiment, so
# a change can be read against both from one place.
#
# Everything here degrades to a no-op when tracing is off, so a caller never has
# to guard the call. That matters more than it sounds: an eval harness that
# raises because MLflow is not configured has turned observability into a
# dependency of the thing it observes.


@contextlib.contextmanager
def evaluation_run(name, params=None, tags=None):
    """An MLflow run for one sweep, or nothing at all when tracing is off.

    Yields the active run (or None), so a caller can hang extra data on it
    without asking again whether tracing is enabled.
    """
    if not setup():
        yield None
        return
    with mlflow.start_run(run_name=name) as run:
        log_params(params or {})
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})
        yield run


def log_params(params):
    """Configuration, as MLflow params. Values are stringified; None is dropped."""
    if not ENABLED or not params:
        return
    try:
        mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})
    except Exception:                       # noqa: BLE001 — telemetry must never
        pass                                # break the thing it is observing


def log_metrics(metrics, step=None):
    """Numbers, as MLflow metrics. Non-numeric and None values are dropped.

    Dropping rather than coercing is deliberate: a metric that could not be
    computed is `None` in this repo, and logging it as 0.0 would put a broken
    grader and a failing pipeline on the same line of the same chart.
    """
    if not ENABLED or not metrics:
        return
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            clean[key] = float(value)
        elif isinstance(value, (int, float)) and value == value:   # not NaN
            clean[key] = float(value)
    if not clean:
        return
    try:
        mlflow.log_metrics(clean, step=step)
    except Exception:                       # noqa: BLE001
        pass


def log_table(rows, artifact_file):
    """Per-question records, as an MLflow table artifact.

    The aggregate goes in as metrics and the rows go in here, because the
    question a sweep actually gets asked afterwards is "which ones failed, and
    why" — and that is a table, not a number.
    """
    if not ENABLED or not rows:
        return
    try:
        mlflow.log_table(data=list(rows), artifact_file=artifact_file)
    except Exception:                       # noqa: BLE001
        pass


__all__ = ["mlflow", "SpanType", "setup", "annotate", "ENABLED", "EXPERIMENT",
           "evaluation_run", "log_params", "log_metrics", "log_table"]
