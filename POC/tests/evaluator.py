"""DeepEval, running on this repo's own models.

DeepEval judges with a model, and by default that model is OpenAI's, configured
by its own environment variables. Here it is a LangChain chat model built by
`llm.models.chat_model` — the same registry, the same key resolution and the
same retry policy as every stage being judged. `DeepEvalBaseLLM` is the
documented extension point for exactly this.

    evaluator = LangChainEvaluator("groq")
    metric = FaithfulnessMetric(model=evaluator, threshold=0.7)

## The judge must not be the model under test

Same provider is fine. The same *model* grading its own answer is not an
independent measurement, and it fails in the flattering direction — so
`EVAL_JUDGE_PROVIDER` defaults away from the pipeline's own provider, and the
fixtures say so where they pick it.

## Structured output is off by default, and that is a measured decision

DeepEval asks for a pydantic schema back (`a_generate(prompt, schema=...)`), and
`with_structured_output` is the obvious way to give it one. It is off here,
because on the models this repo actually uses it does not work:

    groq / openai/gpt-oss-120b
    400 Tool call validation failed: attempted to call tool 'json'
        which was not in request.tools

LangChain binds the schema as a tool; the model answers with a `json` tool the
request never declared, and the provider rejects its own reply. The free
reasoning models on OpenRouter fail differently and for the same underlying
reason — they have no usable tool calling and wrap their JSON in prose.

So the default returns the raw string and lets DeepEval parse it with
`trimAndLoadJson`, which is the documented path for a custom model and is built
for exactly this. `EVAL_STRUCTURED_OUTPUT=1` opts back in on a provider where it
works, and even then a failure at call time falls back to text rather than
failing the test — a judge that cannot be asked nicely is still a usable judge.
"""

import os

from deepeval.models import DeepEvalBaseLLM

from llm.models import chat_model, model_name, text_of

# Away from AGENT_PROVIDER's default so the grader is not the model being
# graded. Overridden per run when a provider's daily cap has been reached.
JUDGE_PROVIDER = os.getenv("EVAL_JUDGE_PROVIDER", "groq")
# The provider's own default is usually its cheapest model, and a cheap model is
# a poor judge — it is the one stage where paying more is straightforwardly
# worth it, because a judge that cannot tell a fabrication from a fact certifies
# the bug rather than catching it.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL") or None
JUDGE_MAX_TOKENS = int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "2048"))
# See the module docstring: off because it is broken on the models used here,
# not because it is worse in principle.
STRUCTURED_OUTPUT = os.getenv("EVAL_STRUCTURED_OUTPUT", "").lower() in ("1", "true", "yes")


class LangChainEvaluator(DeepEvalBaseLLM):
    """A LangChain chat model, as a DeepEval judge."""

    def __init__(self, provider=None, model=None, temperature=0.0,
                 max_tokens=JUDGE_MAX_TOKENS, timeout=180.0):
        self.provider = provider or JUDGE_PROVIDER
        model = model or JUDGE_MODEL
        self._name = model_name(self.provider, "chat", model)
        self._model = chat_model(self.provider, model=model, temperature=temperature,
                                 max_tokens=max_tokens, timeout=timeout)
        super().__init__(self._name)

    def load_model(self, *_args, **_kwargs):
        return self._model

    def get_model_name(self, *_args, **_kwargs):
        return f"{self.provider}/{self._name}"

    # -- generation ---------------------------------------------------------- #

    def _structured(self, schema):
        """The model bound to `schema`, or None if that is not on offer."""
        if schema is None or not STRUCTURED_OUTPUT:
            return None
        try:
            return self._model.with_structured_output(schema)
        except (NotImplementedError, AttributeError, ValueError):
            return None

    def generate(self, prompt, schema=None, **_kwargs):
        bound = self._structured(schema)
        if bound is not None:
            try:
                return bound.invoke(prompt)
            except Exception:               # noqa: BLE001 — fall back, see the docstring
                pass
        return text_of(self._model.invoke(prompt))

    async def a_generate(self, prompt, schema=None, **_kwargs):
        bound = self._structured(schema)
        if bound is not None:
            try:
                return await bound.ainvoke(prompt)
            except Exception:               # noqa: BLE001 — fall back, see the docstring
                pass
        return text_of(await self._model.ainvoke(prompt))
