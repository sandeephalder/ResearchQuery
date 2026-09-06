"""Who is asking, carried through the flow.

Every agent gets an `Identity`, so a stage can log against a user, apply a
per-user rule, or attribute a cost without the caller having to thread a new
argument through five signatures each time it wants one.

It travels two ways, and both are needed:

    explicitly   each agent method takes `identity=`, which is what makes it
                 visible in a signature and settable in a test
    ambiently    a ContextVar, for the code that cannot be passed an argument —
                 the NeMo rail action, which NeMo itself calls, and anything
                 deeper like the LLM client

The ContextVar is per-task, so concurrent questions cannot read each other's
caller. That is the same reason `guardrail.py` carries its client this way.

`Identity.anonymous()` is what the CLI uses. Nothing requires a real user, so a
stage that logs identity does not have to special-case its absence.
"""

import contextlib
import contextvars
import dataclasses


@dataclasses.dataclass(frozen=True)
class Identity:
    """The caller, as far as the flow is concerned."""

    user_id: str | None = None
    session_id: str | None = None
    # Distinguishes two questions from the same session. The graph's
    # checkpointer keys on this, so it is also what keeps one run from resuming
    # into another.
    request_id: str | None = None

    @classmethod
    def anonymous(cls):
        """No authenticated caller — the CLI, a notebook, the smoke test."""
        return cls()

    @property
    def is_anonymous(self):
        return self.user_id is None

    def to_dict(self):
        return {"user_id": self.user_id, "session_id": self.session_id,
                "request_id": self.request_id}

    def __str__(self):
        """What a log line should show. Never the whole session token."""
        if self.is_anonymous:
            return "anonymous"
        session = f"/{self.session_id[-8:]}" if self.session_id else ""
        return f"{self.user_id}{session}"


_CURRENT = contextvars.ContextVar("identity", default=Identity.anonymous())


def current() -> Identity:
    """The identity of whoever this task is serving.

    For code that cannot be handed one: a NeMo action, a logging filter, a
    provider hook. Anything that *can* take a parameter should take one.
    """
    return _CURRENT.get()


@contextlib.contextmanager
def bind(identity):
    """Make `identity` current for the duration of one run."""
    token = _CURRENT.set(identity or Identity.anonymous())
    try:
        yield identity
    finally:
        _CURRENT.reset(token)
