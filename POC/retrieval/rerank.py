"""Two ways to reorder candidates, both as LangChain document compressors.

Retrieval scores *similarity*; reranking is the first stage that reads the
question and the passage together and judges whether the passage **answers** it.
That judgement is worth making twice over, because the two ways of making it
have opposite costs:

    cross-encoder   BAAI/bge-reranker-v2-m3, local. ~4s for 20 passages on
                    Apple silicon, no API, no rate limit, no per-query cost.
                    Trained alongside bge-m3 — the same model family that
                    produced the vectors being reranked — and reads 8192 tokens,
                    so no row in this corpus is truncated before it is judged.

    llm             One listwise call. Slower and metered, but it sees all the
                    candidates at once and can weigh them against each other,
                    which a cross-encoder scoring independent pairs cannot.

Measured on this corpus, 20 candidates: bge-reranker-v2-m3 takes 4.06s on MPS
(6.79s on CPU), ms-marco-MiniLM-L-6-v2 0.16s, glm-5.3-flash 42.8s, and
llama3.1:8b on a 4-core CPU an estimated 321s. For a 3,045-question benchmark
the LLM reranker is ~73% of the token bill, so which one is used is the single
biggest cost decision in the pipeline — and whether it is worth the money is
exactly what `evals/answer_eval.py` exists to settle.

Both are `BaseDocumentCompressor`s, which is what lets either be dropped into a
`ContextualCompressionRetriever` without the pipeline knowing which it got. The
one thing added to the stock `CrossEncoderReranker` is that the score is kept:
LangChain's returns the documents in their new order and discards the numbers,
and the scope floor downstream is a threshold on the best score, so a reranker
that forgets what it scored takes the floor with it.
"""

import sys

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

from tracing import SpanType, mlflow

from .constants import (CROSS_ENCODER_MAX_LENGTH, CROSS_ENCODER_MODEL, RERANK_TOP_N,
                        RERANKERS)


def device():
    """The fastest backend actually present.

    Not FlagEmbedding's `FlagReranker`, which this used first: it looks for CUDA
    and otherwise pins the model to CPU, ignoring `devices="mps"`. On Apple
    silicon that leaves the GPU idle — measured 6.79s on CPU against 4.06s on
    MPS for the same 20 passages, to identical scores.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cross_encoder(model_name=CROSS_ENCODER_MODEL, max_length=CROSS_ENCODER_MAX_LENGTH):
    """The local cross-encoder, on the fastest device present.

    `model_kwargs` reaches `sentence_transformers.CrossEncoder` untouched, which
    is how the device, the length limit and the activation survive the wrapper.
    Sigmoid makes a score a 0-1 relevance probability rather than an unbounded
    logit, so it is comparable across queries — which the scope floor requires,
    since 0.5 has to mean the same thing on every question.
    """
    import torch
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    on = device()
    print(f"loading {model_name} on {on} ...", file=sys.stderr, flush=True)
    return HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs={"max_length": max_length, "device": on,
                      "activation_fn": torch.nn.Sigmoid()})


class ScoringCrossEncoderReranker(BaseDocumentCompressor):
    """`CrossEncoderReranker`, with the score written back onto the document.

    LangChain's own returns the reordered documents and drops the numbers. The
    scope floor is a threshold on the best of those numbers, so they have to
    survive the stage that produced them.
    """

    model: object
    top_n: int = RERANK_TOP_N

    model_config = {"arbitrary_types_allowed": True}

    @mlflow.trace(name="cross_encoder_rerank", span_type=SpanType.RERANKER)
    def compress_documents(self, documents, query, callbacks: Callbacks = None):
        if not documents:
            return []
        scores = self.model.score([(query, document.page_content) for document in documents])
        ranked = sorted(zip(documents, scores), key=lambda pair: -float(pair[1]))
        return [_scored(document, float(score)) for document, score in ranked[:self.top_n]]


class LLMListwiseReranker(BaseDocumentCompressor):
    """One listwise call through `llm.chains.rerank_chain`.

    Listwise, not pairwise: the model comparing candidates against each other is
    most of what makes this better than the retrieval score, and it costs one
    request instead of twenty.
    """

    provider: str | None = None
    model: str | None = None
    top_n: int = RERANK_TOP_N

    model_config = {"arbitrary_types_allowed": True}

    def _chain(self):
        from llm.chains import rerank_chain
        return rerank_chain(self.provider, self.model)

    @staticmethod
    def _passages(documents):
        return [{"label": document.metadata.get("label", ""), "text": document.page_content}
                for document in documents]

    def _reorder(self, documents, ranked):
        return [_scored(documents[index], score) for index, score in ranked[:self.top_n]]

    @mlflow.trace(name="llm_rerank", span_type=SpanType.RERANKER)
    def compress_documents(self, documents, query, callbacks: Callbacks = None):
        if not documents:
            return []
        ranked = self._chain().invoke({"question": query,
                                       "passages": self._passages(documents)})
        return self._reorder(documents, ranked)

    @mlflow.trace(name="llm_rerank", span_type=SpanType.RERANKER)
    async def acompress_documents(self, documents, query, callbacks: Callbacks = None):
        if not documents:
            return []
        ranked = await self._chain().ainvoke({"question": query,
                                              "passages": self._passages(documents)})
        return self._reorder(documents, ranked)


def _scored(document, score):
    """A copy of the document carrying its rerank score in metadata.

    A copy, not a mutation: the same `Document` objects are the candidate list
    the run reports, and a reranked view should not silently rewrite the list it
    was derived from.
    """
    return Document(page_content=document.page_content,
                    metadata={**document.metadata, "rerank_score": score})


def build(name, top_n=RERANK_TOP_N, provider=None, model=None):
    """The compressor for a reranker name, or None for "none".

    Loading is deferred to here rather than done at pipeline construction, so a
    cross-encoder run does not pay for weights the question never reaches and an
    LLM run does not load them at all.
    """
    if name not in RERANKERS:
        raise ValueError(f"reranker must be one of {RERANKERS}, not {name!r}")
    if name == "none":
        return None
    if name == "cross-encoder":
        return ScoringCrossEncoderReranker(model=cross_encoder(), top_n=top_n)
    return LLMListwiseReranker(provider=provider, model=model, top_n=top_n)


__all__ = ["build", "cross_encoder", "device", "ScoringCrossEncoderReranker",
           "LLMListwiseReranker"]
