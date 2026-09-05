"""LLM access for the RAG pipeline.

    parse  — figure image + page context -> description for the index (parsing time)
    infer  — question + retrieved passages -> grounded answer (query time)

Calls go straight to a provider by default (`LLM_PROVIDER`, default "gemini").
Set it to "gateway" to route through the local LiteLLM proxy instead, or to
"ollama" for a local model — same code, no container required.
"""

from .client import (LLMClient, LLMError, LLMGateway, ainfer, aparse, encode_image, infer,
                     parse)
from .constants import DEFAULT_PROVIDER, PROVIDERS

__all__ = ["parse", "infer", "aparse", "ainfer", "LLMClient", "LLMGateway", "LLMError",
           "encode_image", "PROVIDERS", "DEFAULT_PROVIDER"]
