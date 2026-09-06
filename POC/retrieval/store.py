"""The corpus as a LangChain `VectorStore`, over the collection indexing built.

`QdrantVectorStore` does everything the hand-written Qdrant queries did — dense
search, server-side RRF over dense and sparse, payload filters — and it comes
with `as_retriever`, which is what lets the rest of the query side be ordinary
LangChain components rather than bespoke code.

One thing had to be adapted, and it is worth knowing about before adding a field
to the index. `QdrantVectorStore` expects a point's payload to hold the text
under one key and *a nested dict of metadata* under another:

    {"page_content": "...", "metadata": {"kind": "figure", ...}}

`db_populate.py` writes a flat payload instead:

    {"text": "...", "kind": "figure", "doc_id": "...", "heading": "...", ...}

`content_payload_key="text"` handles the first half. The second half has no
setting, because there is no nested dict to name — so `_document_from_point` is
overridden to lift the flat payload into `Document.metadata`. That is a four-line
override of a documented classmethod, against re-embedding 80,365 rows to suit a
default, and it keeps every retriever, compressor and chain downstream unmodified.
"""

import os

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from .constants import VECTOR_MODE, VECTOR_MODES
from .embeddings import BGEM3Embeddings, BGEM3SparseEmbeddings
from .paths import RetrievalError, indexing, qdrant_path


class CorpusVectorStore(QdrantVectorStore):
    """`QdrantVectorStore` that reads this corpus's flat payloads.

    Everything else — `similarity_search_with_score`, `as_retriever`, the
    filters — is inherited unchanged.
    """

    @classmethod
    def _document_from_point(cls, scored_point, collection_name, content_payload_key,
                             metadata_payload_key):
        payload = dict(scored_point.payload or {})
        text = payload.pop(content_payload_key, "")
        # Everything that is not the text is metadata. Doing it this way rather
        # than naming the fields means a payload key added at index time arrives
        # here without this file having to hear about it.
        payload["_id"] = str(scored_point.id)
        payload["_collection_name"] = collection_name
        return Document(page_content=text, metadata=payload)


def connect():
    """(client, where). Honours QDRANT_URL; otherwise opens the embedded store."""
    url = os.getenv(indexing.QDRANT_URL_ENV)
    if url:
        return QdrantClient(url=url), url

    path = qdrant_path()
    if not os.path.isdir(path):
        raise RetrievalError(
            f"No Qdrant store at {path}. Build it first:\n"
            f"    cd Ingestion && uv run db_populate.py")
    try:
        return QdrantClient(path=path), path
    except RuntimeError as error:
        if "already accessed" not in str(error):
            raise
        raise RetrievalError(
            "The embedded Qdrant store is locked by another process — an indexing run "
            "or a notebook kernel still has it open. Embedded mode allows a single "
            "process, not one writer and many readers. Stop that process, or run a "
            f"server and set {indexing.QDRANT_URL_ENV}.") from error


def open_store(client=None, embeddings=None, collection=None, vector_mode=VECTOR_MODE):
    """(store, where). The vector store on the existing collection.

    `vector_mode` decides which of the two vectors on every point is searched:

        dense    the semantic leg only, which is what the pipeline pairs with
                 BM25 — BGE-M3's sparse vector is itself a learned lexical
                 match, so using both would spend both legs on lexical overlap.
        hybrid   dense and sparse fused server-side by Qdrant's own RRF, which
                 is what `db_populate.py --search` does. One BGE-M3 forward pass
                 emits both, so it costs nothing extra to encode.

    `validate_collection_config` is off because the collection predates this
    code and its distance and vector names are already known-good; the check
    would only re-derive what `db_populate.py` guaranteed when it built them.
    """
    if vector_mode not in VECTOR_MODES:
        raise RetrievalError(f"vector_mode must be one of {VECTOR_MODES}, not {vector_mode!r}")

    where = None
    if client is None:
        client, where = connect()
    embeddings = embeddings or BGEM3Embeddings()
    hybrid = vector_mode == "hybrid"

    store = CorpusVectorStore(
        client=client,
        collection_name=collection or indexing.QDRANT_COLLECTION,
        embedding=embeddings,
        sparse_embedding=BGEM3SparseEmbeddings(embeddings) if hybrid else None,
        retrieval_mode=RetrievalMode.HYBRID if hybrid else RetrievalMode.DENSE,
        vector_name=indexing.DENSE_VECTOR,
        sparse_vector_name=indexing.SPARSE_VECTOR,
        content_payload_key="text",
        # Named for completeness; nothing reads it, because the override above
        # takes the whole payload instead.
        metadata_payload_key="metadata",
        validate_collection_config=False,
    )
    return store, where


def kind_filter(kind):
    """A Qdrant filter restricting retrieval to one row kind, or None for all.

    The three kinds — text, figure and table — share one collection and one
    vector space so that a single query reaches all of them. This is what the
    router narrows with, and what makes the benchmark's image queries separable
    from its text ones.
    """
    if not kind:
        return None
    return models.Filter(must=[models.FieldCondition(
        key="kind", match=models.MatchValue(value=kind))])


def label(metadata):
    """One line of provenance — what this passage is and which paper it came from."""
    parts = [metadata.get("kind", "text"), metadata.get("doc_id", "")]
    heading = (metadata.get("heading") or "").strip()
    if heading:
        parts.append(heading[:70])
    pages = metadata.get("pages") or []
    if isinstance(pages, str):
        # Payload values arrive as strings from the embedded store; a page list
        # written as "[67, 68]" should still read as a page range.
        pages = [p for p in pages.strip("[]").split(",") if p.strip()]
    if pages:
        first, last = str(pages[0]).strip(), str(pages[-1]).strip()
        parts.append(f"p.{first}" if len(pages) == 1 else f"pp.{first}-{last}")
    return " · ".join(str(part) for part in parts if part)


def as_document(candidate):
    """A pipeline candidate back as a `Document`, for a compressor to read."""
    return Document(page_content=candidate.get("text", ""),
                    metadata={**(candidate.get("metadata") or {}), "_id": candidate.get("id")})


def iter_rows(paths=None):
    """(point_id, kind, indexed text) for every row, in document order.

    Exactly the rows `db_populate` sends to Qdrant, produced by the same code
    with the same chunk parameters — that identity is what lets the BM25 index
    return ids Qdrant can resolve.
    """
    import glob
    import json

    from .paths import CHUNK_CHARS, OVERLAP_CHARS, db_populate, docs_dir

    for path in sorted(paths or glob.glob(os.path.join(docs_dir(), "*.json"))):
        with open(path) as handle:
            document = json.load(handle)
        for row in db_populate.rows_for_document(document, CHUNK_CHARS, OVERLAP_CHARS):
            yield db_populate.point_id(row), row["kind"], db_populate.embed_input(row)


__all__ = ["CorpusVectorStore", "connect", "open_store", "kind_filter", "label",
           "as_document", "iter_rows", "RetrievalError", "indexing"]
