"""Stages 1 and 5, both run as NeMo Guardrails rails.

NeMo owns the rail mechanism at both ends of the flow; nothing in between. It is
asked for `GenerationOptions(rails=["input"])` and `rails=["output"]`, which run
the rails and return the message unchanged when they pass — so retrieval,
reranking and answering stay with `retrieval.Pipeline` and the agents, and no
second, ungrounded answer is ever generated.

    input rail    llama guard check input, on a locally hosted llama-guard3.
                  Optionally `self check input`, NeMo's own flow, prompted in
                  rails/prompts.yml.
    output rail   check answer grounding, whose action is `verify_answer`,
                  registered here and backed by `verifier.Verifier`. Not NeMo's
                  built-in `self check output`, whose action can only answer yes
                  or no — a repair loop needs to know what was wrong.

Reading a rail's verdict is by identity, not by parsing: a rail that passes
hands back the message it was given, and a rail that blocks substitutes a
refusal. That is NeMo's contract, and it means the wrapper never has to guess
whether a refusal came from the rail or from the text being checked.

## The rails run on a LangChain model

`LLMRails(config, llm=...)` takes a chat model, and the one it is given comes
from `llm.models.chat_model` like every other stage's. So the `main` model in
`rails/config.yml` is documentation rather than configuration: the provider
registry decides who serves it, the same keys and the same retry policy apply,
and there is no second client to configure or to get wrong.

`llama_guard` is still NeMo's to build, because `llm=` names only the main
model. Its engine, base URL and key are overridden onto the config at load.

## Nothing is staged on the instance

Everything the output rail's check needs travels in the call: the evidence goes
in as a `context` message, and the problems come back through
`GenerationOptions(output_vars=[...])`, set by the Colang flow. That is what
lets one `Guardrail` serve many questions at once — which the eval harness does,
eight at a time. The one thing that could not travel that way used to be the
open LLM client, which had to ride a `ContextVar`; a chain builds its own model
now, so that is gone and the identity is the only ambient thing left.
"""

import sys

from llm import LLMError
from llm.models import chat_model
from tracing import SpanType, mlflow

from .identity import current as current_identity
from .constants import (GUARDRAIL_API_KEY, GUARDRAIL_ENGINE, GUARDRAIL_FAIL_CLOSED,
                        GUARDRAIL_LLAMA_GUARD, GUARDRAIL_MODEL, GUARDRAIL_PROVIDER,
                        GUARDRAIL_SELF_CHECK, LLAMA_GUARD_API_KEY, LLAMA_GUARD_BASE_URL,
                        LLAMA_GUARD_FLOW, LLAMA_GUARD_MODEL, RAILS_DIR, REFUSAL_MESSAGE,
                        SELF_CHECK_FLOW)


class GuardrailError(RuntimeError):
    """The rails could not be built, or a rail failed with fail-closed set."""


class Verdict:
    """What a rail decided, and what it found.

    `allowed` is the decision. `problems` is only ever populated by the output
    rail — the input rail is a gate and has nothing to report beyond blocking.
    """

    __slots__ = ("allowed", "problems", "message", "rail")

    def __init__(self, allowed, *, problems=(), message=None, rail=""):
        self.allowed = allowed
        self.problems = list(problems)
        self.message = message
        self.rail = rail

    def __bool__(self):
        return self.allowed

    def to_dict(self):
        return {"allowed": self.allowed, "rail": self.rail,
                "problems": self.problems, "message": self.message}

    def __repr__(self):
        return (f"Verdict({'allowed' if self.allowed else 'blocked'}, "
                f"rail={self.rail!r}, problems={len(self.problems)})")


class Guardrail:
    """The rails, loaded once. Building them costs a config parse and a model init.

        guardrail = Guardrail(verifier=verifier)
        if not await guardrail.check_input(question):
            return REFUSAL_MESSAGE

    `verifier` is what the output rail's action actually runs — see
    `verifier.Verifier`. It is injected rather than imported so that the rail
    and the check it performs can be tested apart.
    """

    def __init__(self, config_path=RAILS_DIR, verifier=None,
                 llama_guard=GUARDRAIL_LLAMA_GUARD, fail_closed=GUARDRAIL_FAIL_CLOSED,
                 provider=GUARDRAIL_PROVIDER, model=GUARDRAIL_MODEL):
        self.verifier = verifier
        self.fail_closed = fail_closed
        self.config_path = config_path
        self.llama_guard = llama_guard
        self.provider = provider
        self.model = model
        self.rails = self._build(config_path, llama_guard)
        # The second gate on its own, built the first time the first one cannot
        # be reached. A machine that is switched off should cost the local rail,
        # not the whole gate — and building this is a config parse, so it is not
        # worth paying for until it is needed.
        self._without_llama_guard = None if llama_guard else self.rails
        # Latched once the host has failed. Retrying it per question costs a
        # connection timeout per question — measured at ~4.5s against a machine
        # that had gone off the network — to re-learn what the first failure
        # already established. It stays latched for the life of the process; a
        # host that comes back is picked up by restarting, which is the same
        # deal every other model in this repo offers.
        self._llama_guard_down = False

    @property
    def fallback(self):
        if self._without_llama_guard is None:
            self._without_llama_guard = self._build(self.config_path, llama_guard=False)
        return self._without_llama_guard

    # -- construction -------------------------------------------------------- #

    def _main_model(self):
        """The chat model the rails run on, from the same registry as every stage."""
        try:
            return chat_model(self.provider, model=self.model, temperature=0.0)
        except LLMError as error:
            raise GuardrailError(f"cannot build the guardrail model: {error}") from error

    def _build(self, config_path, llama_guard):
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError as error:
            raise GuardrailError(
                "nemoguardrails is not installed. Add it with `uv add nemoguardrails`, "
                "or run with --guardrail off.") from error

        try:
            config = RailsConfig.from_path(config_path)
        except Exception as error:                      # noqa: BLE001 — NeMo raises broadly
            raise GuardrailError(f"cannot load rails from {config_path}: {error}") from error

        self._apply_overrides(config, llama_guard)
        rails = LLMRails(config, llm=self._main_model())
        # `check answer grounding` in flows.co executes this by name.
        rails.register_action(self._check_output_action, name="verify_answer")
        return rails

    def _apply_overrides(self, config, llama_guard):
        """Environment wins over the checked-in YAML, for the models only.

        Which flows run is a design decision and stays in the file. Which model
        runs them is deployment, and changes between a laptop, a free-tier key
        and a sweep — so it is an environment variable.

        The `main` entry is only kept consistent with what was actually passed
        as `llm=`; NeMo does not build it. `llama_guard` it does build, so that
        one is configured here in full.
        """
        for model in config.models:
            if model.type == "main":
                model.engine = GUARDRAIL_ENGINE
                model.model = self.model
                model.parameters["api_key"] = GUARDRAIL_API_KEY
            elif model.type == "llama_guard":
                model.model = LLAMA_GUARD_MODEL
                model.parameters["base_url"] = LLAMA_GUARD_BASE_URL
                model.parameters["api_key"] = LLAMA_GUARD_API_KEY

        flows = config.rails.input.flows
        if not GUARDRAIL_SELF_CHECK and SELF_CHECK_FLOW in flows:
            flows.remove(SELF_CHECK_FLOW)
        if llama_guard:
            if LLAMA_GUARD_FLOW not in flows:
                # First, not appended: it is the primary gate, and it is local.
                flows.insert(0, LLAMA_GUARD_FLOW)
        else:
            # A configured-but-unused llama_guard model is a connection to a
            # machine that may be off. Drop it rather than let NeMo dial it.
            config.models = [m for m in config.models if m.type != "llama_guard"]

    # -- stage 1 -------------------------------------------------------------- #

    @mlflow.trace(name="guardrail_in", span_type=SpanType.GUARDRAIL)
    async def check_input(self, question):
        """Does the question pass the input rails?

        Llama Guard first, then the self-check. If the machine hosting the first
        is unreachable the whole rail run aborts — NeMo has no notion of a
        partly-run rail — so the call is made again against the second gate
        alone. Degrading from two gates to one is worth announcing; degrading to
        none silently is not.
        """
        messages = [{"role": "user", "content": question}]
        if not self._llama_guard_down:
            try:
                return await self._run(messages, question, rails=["input"], rail="input",
                                       raise_on_error=self.llama_guard)
            except GuardrailError as error:
                self._llama_guard_down = True
                # What is left may be nothing. Llama Guard is the only input
                # rail by default, so a host that is off leaves the flow with no
                # gate at all — which is a different sentence from "one gate
                # instead of two", and has to read like one.
                remaining = list(self.fallback.config.rails.input.flows)
                left = (f"falling back to {', '.join(remaining)}" if remaining
                        else "THERE IS NO INPUT GATE LEFT — every question will "
                             "pass unchecked until the host is back")
                print(f"warning: the local Llama Guard rail could not be reached "
                      f"({error}); {left}, for the rest of this process",
                      file=sys.stderr)
        return await self._run(messages, question, rails=["input"], rail="input",
                               rails_object=self.fallback)

    # -- stage 5 -------------------------------------------------------------- #

    @mlflow.trace(name="guardrail_out", span_type=SpanType.GUARDRAIL)
    async def check_output(self, question, draft, passages=(), facts=()):
        """Does the draft survive the output rail, and what did it object to?

        Everything the check needs travels in the call: the evidence goes in as
        a `context` message and the problems come back as an output variable.
        Nothing is staged on the instance, so one `Guardrail` serves many
        questions at once.
        """
        return await self._run(
            [{"role": "context", "content": {"passages": list(passages),
                                             "facts": list(facts)}},
             {"role": "user", "content": question},
             {"role": "assistant", "content": draft}],
            draft, rails=["output"], rail="output", output_vars=["verify_problems"])

    async def _check_output_action(self, context=None, bot_message=None, **_):
        """The `verify_answer` action. `is_blocked` is NeMo's convention.

        Everything arrives in `context`: the draft (as `bot_message`, which NeMo
        puts there and passes as a parameter only sometimes — the same fallback
        its own `self_check_output` action performs), and the question, passages
        and facts the caller put there. Nothing is read off `self` but the
        verifier, so concurrent questions cannot see each other's evidence.
        """
        context = context or {}
        draft = bot_message if bot_message is not None else context.get("bot_message")
        if self.verifier is None or not (draft or "").strip():
            return {"is_blocked": False, "problems": []}
        try:
            result = await self.verifier.check(
                context.get("user_message", ""), draft,
                context.get("passages") or [], context.get("facts") or [],
                # NeMo calls this action, so there is no parameter to thread —
                # which is exactly what the ContextVar is for.
                identity=current_identity())
        except LLMError as error:
            # The same rule the reranker follows: a stage that could not run
            # should cost quality, not masquerade as a stage that ran and
            # approved. Announce it, and do not block on an absent opinion.
            print(f"warning: output rail could not run ({error}); draft not checked",
                  file=sys.stderr)
            return {"is_blocked": False, "problems": []}
        return {"is_blocked": not result.ok, "problems": result.problems}

    # -- the rail call -------------------------------------------------------- #

    async def _run(self, messages, original, *, rails, rail, output_vars=(),
                   rails_object=None, raise_on_error=False):
        from nemoguardrails.rails.llm.options import GenerationOptions

        options = GenerationOptions(rails=rails,
                                    **({"output_vars": list(output_vars)} if output_vars else {}))
        try:
            result = await (rails_object or self.rails).generate_async(
                messages=messages, options=options)
        except Exception as error:                      # noqa: BLE001 — NeMo raises broadly
            if raise_on_error or self.fail_closed:
                raise GuardrailError(f"{rail} rail failed: {error}") from error
            print(f"warning: {rail} rail failed ({error}); allowing through",
                  file=sys.stderr)
            return Verdict(True, rail=rail)

        content = _content(result)
        problems = (getattr(result, "output_data", None) or {}).get("verify_problems") or []
        # A rail that passes returns what it was given; one that blocks
        # substitutes its refusal. Comparing against the original is how the
        # decision is read, and it cannot be confused by the text's own wording.
        if content is not None and content.strip() == original.strip():
            return Verdict(True, problems=problems, rail=rail)
        return Verdict(False, problems=problems,
                       message=content or REFUSAL_MESSAGE, rail=rail)


def _content(result):
    """The assistant text out of whatever shape NeMo returned.

    `generate_async` returns a plain dict without GenerationOptions and a
    GenerationResponse with them, whose `response` is a list of messages in
    current versions and was a bare string in older ones. All three turn up
    depending on how the call was made.
    """
    response = getattr(result, "response", result)
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("content")
    if isinstance(response, list) and response:
        last = response[-1]
        return last.get("content") if isinstance(last, dict) else str(last)
    return None
