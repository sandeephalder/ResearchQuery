"""One import shim, so RAGAS loads under LangChain v1.

`ragas.llms.base` opens with an unguarded

    from langchain_community.chat_models.vertexai import ChatVertexAI

and `langchain-community` 0.4 — the release that pairs with `langchain-core` 1.x
— no longer ships that module; Vertex moved to `langchain-google-vertexai`
several releases ago. So importing RAGAS 0.4.3 into a LangChain v1 environment
fails at import time, on a symbol RAGAS only uses for an `isinstance` check
against an LLM this repo never passes it.

Registering a stand-in before RAGAS is imported is the whole fix. It is a
deliberate, one-line-of-surface hack rather than a hidden one:

- **What it costs.** Nothing here can use Vertex AI through RAGAS. Nothing here
  wants to: every model comes from `llm.PROVIDERS`, and Vertex is not in it.
- **When it can be deleted.** As soon as RAGAS guards that import, or moves to
  `langchain-google-vertexai`. Deleting it is safe to try — if RAGAS still
  imports, the shim was no longer needed.
- **Why not pin around it.** The alternative is `langchain-community` 0.3, which
  requires `langchain-core` <0.4, which is the entire stack this project was
  rewritten onto. The shim is the smaller thing to give up.

Import this module before anything from `ragas`.
"""

import sys
import types


def install():
    """Register the stand-in, unless the real module is there. Idempotent."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return False
    try:
        __import__(name)
    except ImportError:
        pass
    else:
        return False                        # a real one exists; leave it alone

    module = types.ModuleType(name)
    module.__doc__ = __doc__

    class ChatVertexAI:                     # noqa: D401 — a placeholder, not a client
        """Stands in for the moved integration. Never instantiated."""

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "Vertex AI is not available here. Install langchain-google-vertexai "
                "and add a provider to llm/constants.py if it is wanted.")

    module.ChatVertexAI = ChatVertexAI
    sys.modules[name] = module
    return True


installed = install()
