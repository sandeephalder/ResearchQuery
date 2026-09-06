"""Query time: two retrievers fused, a reranker, and a grounded answer.

    uv run python -m retrieval.pipeline "how does detection probability vary with range"
    uv run python -m retrieval.pipeline "..." --kind figure --top-n 8 --no-answer
    uv run python -m retrieval.pipeline "..." --no-rerank --json

Four stages, each of which can be turned off to see what it was buying:

    retrieve   10 candidates from Qdrant, 10 from BM25
    fuse       merged by reciprocal rank, so agreement between the legs counts
    rerank     a compressor scores all ~20 against the question
    answer     the survivors go to `llm.chains.answer_chain`, which cites them

All four are LangChain components now:

    vector leg   `CorpusVectorStore.as_retriever`
    lexical leg  `CorpusBM25Retriever`
    fusion       `EnsembleRetriever`, whose weighted RRF is the fusion this
                 used to compute by hand
    reranking    a `BaseDocumentCompressor`, either the local cross-encoder or
                 one listwise LLM call
    answering    `llm.chains.answer_chain`

The two legs are deliberately unlike each other. Qdrant answers with meaning —
it finds the passage that says the same thing in different words, which is most
of what a benchmark question needs. BM25 answers with the literal token, which
is what rescues the query that hinges on a symbol or a model name the embedder
smoothed away. Fusing legs that fail the same way buys nothing, which is why the
vector leg defaults to dense only: BGE-M3's sparse vector is itself lexical, and
pairing it with BM25 would spend both legs on the same kind of match.

Fusion is by rank, not score, because the two legs' scores are not comparable —
a cosine similarity of 0.7 and a BM25 score of 14 say nothing about each other.
Reciprocal rank fusion only asks how high each leg placed a row, and rewards the
rows both legs placed highly.

## Documents in, dicts out

Inside, a passage is a `Document`, because that is what every LangChain
component here takes and returns. At the boundary — what `retrieve`, `rerank`
and `arun` hand back — it becomes the same dict this pipeline has always
returned, carrying both legs' ranks and the fused score. That is not nostalgia:
`evals/answer_eval.py` and `agents/` read those keys, and a run should show its
own working. `as_candidate` and `as_documents` are the two sides of that
boundary.
"""

import argparse
import asyncio
import json
import os
import sys

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from llm import LLMError
from llm.chains import answer_chain
from llm.constants import DEFAULT_PROVIDER
from tracing import SpanType, mlflow

from . import rerank as reranking
from .bm25 import load as load_bm25
from .constants import (ANSWER_MAX_CHARS, BM25_TOP_K, RERANK_TOP_N, RERANKER, RERANKERS,
                        RRF_K, VECTOR_MODE, VECTOR_MODES, VECTOR_TOP_K)
from .embeddings import BGEM3Embeddings
from .paths import RetrievalError, indexing
from .store import kind_filter, label, open_store


# --------------------------------------------------------------------------- #
# The two legs
# --------------------------------------------------------------------------- #

class ScoredVectorRetriever(BaseRetriever):
    """The Qdrant leg, keeping the similarity it retrieved on.

    `as_retriever` discards the score, and the run report shows it. One
    `similarity_search_with_score` and a stamp on the metadata is the whole of
    the difference.
    """

    store: object
    k: int = VECTOR_TOP_K
    kind: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @mlflow.trace(name="qdrant_vector", span_type=SpanType.RETRIEVER)
    def _get_relevant_documents(self, query, *, run_manager=None, **kwargs):
        if self.k <= 0:
            return []
        hits = self.store.similarity_search_with_score(
            query, k=self.k, filter=kind_filter(self.kind))
        return [Document(page_content=document.page_content,
                         metadata={**document.metadata, "vector_score": float(score)})
                for document, score in hits]


class LexicalRetriever(BaseRetriever):
    """The BM25 leg, resolved against Qdrant.

    The BM25 index stores ids and term statistics and nothing else, which keeps
    it small and removes any chance of the two stores disagreeing about what a
    row says. So a lexical hit is an id, and its payload is fetched here — in
    one round trip, and only for the rows the vector leg did not already carry.
    """

    index: object
    client: object
    collection: str
    k: int = BM25_TOP_K
    kind: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @mlflow.trace(name="bm25", span_type=SpanType.RETRIEVER)
    def _get_relevant_documents(self, query, *, run_manager=None, **kwargs):
        if self.k <= 0 or self.index is None:
            return []
        hits = self.index.scored(query, self.k, self.kind)
        if not hits:
            return []
        records = self.client.retrieve(collection_name=self.collection,
                                       ids=[point_id for point_id, _ in hits],
                                       with_payload=True, with_vectors=False)
        payloads = {str(record.id): dict(record.payload or {}) for record in records}

        documents = []
        for point_id, score in hits:
            payload = payloads.get(point_id)
            if payload is None:
                # The BM25 index was built from a corpus this collection does
                # not hold. Dropping the row is right; hiding it is not.
                print(f"warning: {point_id} is in the BM25 index but not in Qdrant — "
                      f"rebuild one of them", file=sys.stderr)
                continue
            text = payload.pop("text", "")
            documents.append(Document(page_content=text,
                                      metadata={**payload, "_id": point_id,
                                                "bm25_score": float(score)}))
        return documents


class TracedEnsembleRetriever(EnsembleRetriever):
    """`EnsembleRetriever` that says where each row came from.

    The fusion is LangChain's, unchanged — weighted reciprocal rank over the two
    legs' orders. What is added is the record of it: each surviving document
    carries the rank it held in each leg and the fused score it was sorted by,
    because a row found by both legs appearing once, carrying both its ranks, is
    what makes a run readable afterwards.
    """

    @mlflow.trace(name="fuse", span_type=SpanType.RETRIEVER)
    def weighted_reciprocal_rank(self, doc_lists):
        ranks, scores = [], []
        for documents in doc_lists:
            ranks.append({document.metadata.get("_id"): position
                          for position, document in enumerate(documents, start=1)})
            scores.append({document.metadata.get("_id"): document.metadata
                           for document in documents})

        fused = super().weighted_reciprocal_rank(doc_lists)
        # Recomputed rather than read out of the parent, which does not return
        # it: the same weighted sum, over the ranks just recorded.
        totals = {}
        for leg, weight in zip(ranks, self.weights):
            for point_id, rank in leg.items():
                totals[point_id] = totals.get(point_id, 0.0) + weight / (rank + self.c)

        annotated = []
        for document in fused:
            point_id = document.metadata.get("_id")
            metadata = dict(document.metadata)
            metadata["vector_rank"] = ranks[0].get(point_id) if ranks else None
            metadata["bm25_rank"] = ranks[1].get(point_id) if len(ranks) > 1 else None
            metadata["fused_score"] = totals.get(point_id, 0.0)
            # A row only the other leg found still needs the score that leg gave
            # it, and `unique_by_key` kept whichever copy it saw first.
            for leg in scores:
                other = leg.get(point_id) or {}
                for key in ("vector_score", "bm25_score"):
                    if key in other and key not in metadata:
                        metadata[key] = other[key]
            annotated.append(Document(page_content=document.page_content, metadata=metadata))
        return annotated


# --------------------------------------------------------------------------- #
# The boundary between Documents and the dicts everything downstream reads
# --------------------------------------------------------------------------- #

def as_candidate(document):
    """A retrieved `Document` as the candidate dict this pipeline returns."""
    metadata = dict(document.metadata)
    return {
        "id": metadata.get("_id"),
        "metadata": metadata,
        "label": label(metadata),
        "kind": metadata.get("kind"),
        "doc_id": metadata.get("doc_id"),
        "text": document.page_content,
        "vector_rank": metadata.get("vector_rank"),
        "vector_score": metadata.get("vector_score"),
        "bm25_rank": metadata.get("bm25_rank"),
        "bm25_score": metadata.get("bm25_score"),
        "fused_score": metadata.get("fused_score"),
        "rerank_score": metadata.get("rerank_score"),
    }


def as_documents(candidates):
    """Candidate dicts back as `Document`s, for a compressor to read."""
    return [Document(page_content=candidate["text"],
                     metadata={**(candidate.get("metadata") or {}),
                               "_id": candidate["id"], "label": candidate["label"]})
            for candidate in candidates]


class Pipeline:
    """One question in, a cited answer out. Holds the model and the stores open.

    Open it once and ask many questions: BGE-M3 takes ~15 s to load, and the
    embedded Qdrant store admits a single process, so a short-lived client per
    query is both slow and a lock other processes will trip over.

        with Pipeline() as pipeline:
            result = pipeline.ask("how does detection probability vary with range")
    """

    def __init__(self, provider=DEFAULT_PROVIDER, collection=indexing.QDRANT_COLLECTION,
                 vector_mode=VECTOR_MODE, use_bm25=True, reranker=RERANKER):
        if vector_mode not in VECTOR_MODES:
            raise RetrievalError(f"vector_mode must be one of {VECTOR_MODES}, not {vector_mode!r}")
        if reranker not in RERANKERS:
            raise RetrievalError(f"reranker must be one of {RERANKERS}, not {reranker!r}")
        self.provider = provider
        self.collection = collection
        self.vector_mode = vector_mode
        self.reranker = reranker
        self.embeddings = BGEM3Embeddings()
        self.store, self.where = open_store(embeddings=self.embeddings, collection=collection,
                                            vector_mode=vector_mode)
        self.client = self.store.client
        self.index = load_bm25() if use_bm25 else None
        # Built on first use: a cross-encoder run should not pay for weights the
        # question never reaches, and an LLM run should not load them at all.
        self._compressor = None

    # `encoder` was the BGE-M3 wrapper this held before the embeddings became a
    # LangChain `Embeddings`. The metrics still want it under that name.
    @property
    def encoder(self):
        return self.embeddings

    @property
    def compressor(self):
        if self._compressor is None and self.reranker != "none":
            self._compressor = reranking.build(self.reranker, provider=self.provider)
        return self._compressor

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    # -- stage 1: the two legs, fused --------------------------------------- #

    def retriever(self, k_vector=VECTOR_TOP_K, k_bm25=BM25_TOP_K, kind=None):
        """The fused retriever for one set of budgets. Cheap to build per query.

        Both legs are rebuilt because `k` and `kind` belong to the retriever
        rather than to the call — which is LangChain's shape, and the reason the
        router's kind filter is a second retriever rather than an argument. The
        expensive parts, the loaded model and the open store, are held on the
        pipeline and shared.
        """
        legs = [ScoredVectorRetriever(store=self.store, k=k_vector, kind=kind)]
        if self.index is not None and k_bm25 > 0:
            legs.append(LexicalRetriever(index=self.index, client=self.client,
                                         collection=self.collection, k=k_bm25, kind=kind))
        # `id_key` is what makes a row found by both legs one row: without it
        # LangChain dedupes on page content, and two chunks that share an
        # opening would collapse into one.
        return TracedEnsembleRetriever(retrievers=legs, id_key="_id", c=RRF_K)

    @mlflow.trace(name="retrieve", span_type=SpanType.RETRIEVER)
    def retrieve(self, question, k_vector=VECTOR_TOP_K, k_bm25=BM25_TOP_K, kind=None):
        """Candidates from both legs, merged and ordered by fused rank."""
        documents = self.retriever(k_vector, k_bm25, kind).invoke(question)
        return [as_candidate(document) for document in documents]

    # -- stages 2 and 3: rerank, then answer -------------------------------- #

    @mlflow.trace(name="rerank", span_type=SpanType.RERANKER)
    async def rerank(self, question, candidates, top_n=RERANK_TOP_N):
        """The candidates reordered. Falls back to fused order on failure.

        A reranker that fails should cost quality, not the answer — but silently
        returning the retrieval order would make a broken stage look like a
        working one, so the degradation is announced.
        """
        if not candidates:
            return candidates, False
        compressor = self.compressor
        if compressor is None:
            return candidates[:top_n], False

        documents = as_documents(candidates)
        try:
            if isinstance(compressor, reranking.ScoringCrossEncoderReranker):
                # Seconds of model work. Awaited directly it would hold the
                # event loop for that long, which a CLI never notices and a
                # server cannot afford.
                ranked = await asyncio.to_thread(
                    compressor.compress_documents, documents, question)
            else:
                ranked = await compressor.acompress_documents(documents, question)
        except LLMError as error:
            print(f"warning: reranking failed ({error}); keeping fused order",
                  file=sys.stderr)
            return candidates[:top_n], False

        # The compressor returns the top_n it kept, in its order. What comes
        # back is the candidate that went in, with the score written on: the
        # fused rank and both legs' scores stay attached to the row they
        # describe, and the reranker only decides the order.
        by_id = {candidate["id"]: candidate for candidate in candidates}
        reranked = []
        for document in ranked[:top_n]:
            candidate = by_id.get(document.metadata.get("_id"))
            if candidate is None:
                continue
            reranked.append({**candidate,
                             "rerank_score": document.metadata.get("rerank_score")})
        return reranked, True

    @mlflow.trace(name="answer", span_type=SpanType.LLM)
    async def answer(self, question, candidates):
        """A grounded answer citing the passages by their position in the list."""
        passages = [{"label": candidate["label"],
                     "text": candidate["text"][:ANSWER_MAX_CHARS]}
                    for candidate in candidates]
        return await answer_chain(self.provider).ainvoke(
            {"question": question, "passages": passages})

    # -- the whole thing ----------------------------------------------------- #

    @mlflow.trace(name="retrieval_pipeline", span_type=SpanType.CHAIN)
    async def arun(self, question, k_vector=VECTOR_TOP_K, k_bm25=BM25_TOP_K,
                   top_n=RERANK_TOP_N, kind=None, rerank=True, answer=True):
        candidates = await asyncio.to_thread(self.retrieve, question, k_vector, k_bm25, kind)
        rerank = rerank and self.reranker != "none"
        result = {"question": question, "candidates": candidates,
                  "reranked": False, "answer": None,
                  "vector_mode": self.vector_mode,
                  "reranker": self.reranker if rerank else "none"}

        if not candidates:
            return result
        if not (rerank or answer):
            result["passages"] = candidates[:top_n]
            return result

        if rerank:
            passages, ok = await self.rerank(question, candidates, top_n)
            result["reranked"] = ok
        else:
            passages = candidates[:top_n]
        result["passages"] = passages
        if answer:
            result["answer"] = await self.answer(question, passages)
        return result

    def ask(self, question, **kwargs):
        """Blocking `arun`, for notebooks and scripts."""
        return asyncio.run(self.arun(question, **kwargs))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _report(result, show_text=True):
    candidates = result["candidates"]
    passages = result.get("passages", [])
    kept = {candidate["id"] for candidate in passages}

    both = sum(1 for c in candidates if c["vector_rank"] and c["bm25_rank"])
    print(f"\n{len(candidates)} candidates "
          f"({sum(1 for c in candidates if c['vector_rank'])} vector, "
          f"{sum(1 for c in candidates if c['bm25_rank'])} bm25, {both} both) "
          f"| vector mode: {result['vector_mode']} "
          f"| reranker: {result.get('reranker', 'llm')}")

    print(f"\n{'':2} {'vec':>5} {'bm25':>5} {'fused':>7} {'rr':>5}  source")
    for position, candidate in enumerate(candidates, start=1):
        vector = f"#{candidate['vector_rank']}" if candidate["vector_rank"] else "-"
        lexical = f"#{candidate['bm25_rank']}" if candidate["bm25_rank"] else "-"
        rerank = candidate["rerank_score"]
        # The reranked list carries the scores; the candidate list is pre-rerank.
        for passage in passages:
            if passage["id"] == candidate["id"] and passage["rerank_score"] is not None:
                rerank = passage["rerank_score"]
        marker = "*" if candidate["id"] in kept else " "
        print(f"{marker}{position:>2}{vector:>5} {lexical:>5} {candidate['fused_score']:7.4f} "
              f"{('' if rerank is None else f'{rerank:.2f}'):>4}  {candidate['label'][:78]}")

    if result["reranked"]:
        print(f"\n* kept by the {result.get('reranker')} reranker, in the order it "
              f"returned (cited below as [1]..[{len(passages)}])")
    else:
        print("\n* kept by fused rank (no reranking)")

    if show_text:
        for number, passage in enumerate(passages, start=1):
            print(f"\n[{number}] {passage['label']}")
            print(f"    {passage['text'][:280].replace(chr(10), ' ')}")

    if result["answer"]:
        print(f"\n{'-' * 78}\n{result['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid retrieval, reranking, answer")
    parser.add_argument("question")
    parser.add_argument("--k-vector", type=int, default=VECTOR_TOP_K,
                        help=f"candidates from Qdrant (default {VECTOR_TOP_K}; 0 disables)")
    parser.add_argument("--k-bm25", type=int, default=BM25_TOP_K,
                        help=f"candidates from BM25 (default {BM25_TOP_K}; 0 disables)")
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N,
                        help=f"passages kept after reranking (default {RERANK_TOP_N})")
    parser.add_argument("--kind", choices=["text", "figure", "table"],
                        help="restrict retrieval to one row kind")
    parser.add_argument("--vector-mode", choices=list(VECTOR_MODES), default=VECTOR_MODE,
                        help="dense only, or dense+BGE-M3-sparse fused in Qdrant")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        help=f"LLM provider for rerank and answer (default {DEFAULT_PROVIDER})")
    parser.add_argument("--reranker", choices=list(RERANKERS), default=RERANKER,
                        help=f"how to reorder candidates (default {RERANKER}); "
                             f"cross-encoder is local, free and ~270x faster")
    parser.add_argument("--no-rerank", action="store_true", help="keep the fused order")
    parser.add_argument("--no-answer", action="store_true", help="retrieve only")
    parser.add_argument("--quiet", action="store_true", help="hide passage text")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--trace", action="store_true",
                        help="send an MLflow trace of every stage (same as "
                             "MLFLOW_TRACING=1)")
    arguments = parser.parse_args()

    if arguments.trace:
        os.environ["MLFLOW_TRACING"] = "1"
        import importlib

        import tracing as _tracing
        importlib.reload(_tracing)
        if _tracing.setup():
            print(f"tracing to experiment {_tracing.EXPERIMENT!r}", file=sys.stderr)

    try:
        with Pipeline(provider=arguments.provider, vector_mode=arguments.vector_mode,
                      use_bm25=arguments.k_bm25 > 0, reranker=arguments.reranker) as pipeline:
            result = pipeline.ask(
                arguments.question, k_vector=arguments.k_vector, k_bm25=arguments.k_bm25,
                top_n=arguments.top_n, kind=arguments.kind,
                rerank=not arguments.no_rerank, answer=not arguments.no_answer)
    except (RetrievalError, LLMError) as error:
        sys.exit(f"FAILED: {error}")

    if arguments.json:
        for candidate in result["candidates"]:
            candidate.pop("metadata", None)
        for passage in result.get("passages", []):
            passage.pop("metadata", None)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif not result["candidates"]:
        print("no candidates — neither leg matched")
    else:
        _report(result, show_text=not arguments.quiet)
