# Tests

Two suites, and the split is which one needs a network.

| | what it checks | cost | when |
| --- | --- | --- | --- |
| [`test_agent_wiring.py`](test_agent_wiring.py) | every node, every branch, the repair loop | none — chains stubbed, retrieval real | every change |
| [`test_agent_quality.py`](test_agent_quality.py) | is the answer faithful, relevant, cited, non-abstaining | six model calls per question plus the judge's | before believing a change helped |

```bash
uv run pytest                      # the wiring suite; quality is skipped
uv run pytest --live               # both
uv run pytest tests/test_agent_quality.py --live -q
```

`--live` (or `LIVE_TESTS=1`) is the only switch. Without it the quality tests
are **skipped, not failed**: a test that cannot run is not a test that failed,
and a suite that goes red because a key is missing gets ignored.

## Choosing the judge

DeepEval judges with a model, and by default that model is OpenAI's, configured
by its own environment variables. Here it is a LangChain chat model from
`llm.PROVIDERS` — the same registry, keys and retry policy as the stages being
judged — through `tests/evaluator.py`.

```bash
EVAL_JUDGE_PROVIDER=glm EVAL_JUDGE_MODEL=glm-5.3 uv run pytest --live
```

**The judge must not be the model under test.** Same provider is fine; the same
*model* grading its own answer is not an independent measurement, and it fails
in the flattering direction.

## Two settings that are not defaults, and why

**`penalize_ambiguous_claims=True` on faithfulness.** DeepEval labels each claim
`yes`, `no` or `idk`, and by default only `no` — a direct contradiction — lowers
the score. A claim the passages never mention is `idk`, and is ignored. Measured
on this corpus's own shape:

| answer | default | penalized |
| --- | --- | --- |
| grounded, correct | 1.00 | 1.00 |
| invented — rainfall, from a passage about detection probability | **1.00** | 0.00 |
| flipped — "rises with range" | **1.00** | 0.00 |

A fabrication scoring 1.00 is the exact failure the verifier stage exists to
catch. A test that scores it 1.00 certifies the bug.

**Raised DeepEval timeouts**, set in [`conftest.py`](conftest.py). `faithfulness`
extracts truths from five passages of up to 4,000 characters, then claims, then
a verdict per claim — measured at over 200 s on `glm-5.3-flash`, which the stock
gather timeout cancels midway and reports as a failing pipeline rather than as a
slow judge.

## MLflow

```bash
MLFLOW_TRACING=1 EVAL_JUDGE_PROVIDER=glm uv run pytest --live
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The whole live suite is one MLflow run, and every metric is logged under
`<metric>/<question id>` — per question, not averaged, because a mean over two
deliberately different questions would hide exactly the modality split the suite
exists to watch.

The `graded` fixture **measures, logs, then asserts**, in that order.
`assert_test` raises on a failing metric, so anything logged after it never runs
for precisely the cases worth recording. Measuring first means a failing score
reaches MLflow and *then* fails the test — the chart shows the regression rather
than a gap where it should be.

The stages emit their own traces alongside, so a low score can be opened to see
which retriever found what and what the generator did with it. Logging failures
never fail a test: telemetry does not get to decide whether the code is correct.

## Thresholds are loose on purpose

These are regression guards, not a leaderboard. A threshold set at the current
measured value turns every run of a stochastic system into a coin flip, and a
suite that fails at random gets ignored. They sit where a real regression trips
them and ordinary variance does not. `evals/answer_eval.py` is what measures
quality, over a hundred questions rather than two.

Measured on the first full live run (23m34s, pipeline on groq, judge glm-5.3):

| metric | measured | threshold | |
| --- | --- | --- | --- |
| faithfulness | 1.00 | 0.70 | with `penalize_ambiguous_claims` |
| answer relevancy | pass | 0.60 | |
| false abstention | pass | 0.60 | |
| citations | 0.00 on one question | 0.60 | a real defect — see below |
| contextual relevancy | **0.25** | 0.15 | the most interesting number here |

**Contextual relevancy is 0.25** — roughly one passage in four judged relevant
out of a top-5. The threshold sits at 0.15 so a *drop* trips it, not so the test
goes green. Whether 0.25 means the reranker keeps three weak passages, or that
the judge is strict about what "relevant" means for a figure description, is
unsettled; `context_precision` over a hundred questions is what would settle it.

**The citation failure is a pipeline defect, not a calibration problem.** The
generator copied `[55]` — a bibliography reference sitting verbatim in a
retrieved passage — into its answer, where it is indistinguishable from a
citation marker pointing at passage 55 of 5. The judge was right to score it 0:

> it includes the bracketed citation [55] for 'Standard APS', which exceeds the
> total of 5 retrieved passages … a bibliography reference carried over verbatim
> from the source text

Bracketed numerals in the source text collide with the citation scheme the whole
answer contract is built on. The fix belongs in the generator prompt or in a
post-check, and this test is what will confirm it.

## Where the stub goes

Each agent stage is `prompt | model | parser`, and the wiring suite replaces the
whole chain rather than the transport under it. A stage's contract is "a dict in,
a parsed value out", so a `RunnableLambda` honouring that contract is a complete
stand-in — and unlike a stubbed socket it cannot let a test pass on a reply the
real chain would have failed to parse.

Retrieval and reranking stay real. Both are local, and a stubbed retriever would
leave the wiring tests testing almost nothing.

## One pipeline for the whole session

Constructing an `AgentPipeline` loads BGE-M3 (~15 s) and the cross-encoder, opens
the embedded Qdrant store and compiles the NeMo rails. That store admits a
**single process**, so a fixture per test would not merely be slow — the second
would fail on a lock the first still held. Session scope is not an optimisation
here, it is the only thing that works. It is also why the suite is not run in
parallel.
