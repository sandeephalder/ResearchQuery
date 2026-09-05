"""Chunk the parsed corpus, embed it with BGE-M3, and load it into Qdrant.

    cd Ingestion && uv run db_populate.py                 # index everything
    cd Ingestion && uv run db_populate.py --limit 20      # a slice, for a first look
    cd Ingestion && uv run db_populate.py --search "how does detection probability vary with range"

Three kinds of row share one collection and one vector space:

    text     a chunk of a section's markdown, placeholders left in place
    figure   the figure's caption plus its VLM description
    table    the table's caption plus its markdown

They are indexed together so a single query reaches all of them, and told apart
by the `kind` payload field — which is what lets the benchmark's text-only
queries be scored separately from the image and table ones.

BGE-M3 emits a dense and a sparse vector in one forward pass, so hybrid search
needs no second model. Both go on the same point as named vectors, and Qdrant
fuses them server-side with RRF.

Re-running is safe. Point ids are derived from (doc_id, section_id, kind, ord),
so a second run overwrites rather than duplicates, and documents already indexed
are skipped unless --force is given.
"""

import argparse
import json
import glob
import os
import re
import sys
import uuid

from tqdm import tqdm

from constants import (CHARS_PER_TOKEN, CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, DENSE_VECTOR,
                       EMBED_BATCH_SIZE, EMBED_DIM, EMBED_MODEL, INDEXED_DOCS_JSON,
                       LOCAL_PROCESSED_DIR, MAX_ROW_TOKENS, PROCESSED_DOCS_DIR_NAME,
                       QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL_ENV, SPARSE_VECTOR,
                       UPSERT_BATCH_SIZE)

PLACEHOLDER_RE = re.compile(r"!\[([^\]]+)\]\(([^)]+)\)")
# Fixed namespace so point ids are stable across runs: re-indexing a document
# overwrites its points instead of creating a second copy.
NAMESPACE = uuid.UUID("6f4d4b7a-0000-4000-8000-726573656172")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def _paragraphs(text):
    """(start, end) spans of paragraph blocks, so offsets survive chunking."""
    spans, position = [], 0
    for block in text.split("\n\n"):
        if block.strip():
            spans.append((position, position + len(block)))
        position += len(block) + 2
    return spans


def chunk_text(text, target_chars, overlap_chars):
    """Split on paragraph boundaries into (start, end) char spans.

    Paragraph-aligned so a placeholder — always its own paragraph — is never cut
    in half, which would sever a chunk from the figure it refers to.
    """
    spans = _paragraphs(text)
    if not spans:
        return []

    chunks, current = [], []
    for span in spans:
        if current and span[1] - current[0][0] > target_chars:
            chunks.append((current[0][0], current[-1][1]))
            # Re-open with the tail of the previous chunk for continuity.
            back, size = [], 0
            for previous in reversed(current):
                back.insert(0, previous)
                size += previous[1] - previous[0]
                if size >= overlap_chars:
                    break
            current = back
        current.append(span)
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def rows_for_document(document, target_chars, overlap_chars):
    """Every indexable row for one parsed document."""
    rows = []
    doc_id, title = document["id"], document.get("title", "")

    for section in document["sections"]:
        heading = section.get("heading") or ""
        text = section["text"]
        assets = {**section["images"], **section["tables"]}

        for ordinal, (start, end) in enumerate(chunk_text(text, target_chars, overlap_chars)):
            body = text[start:end].strip()
            if not body:
                continue
            # Assets whose placeholder falls inside this chunk travel with it.
            inside = [asset_id for asset_id, record in assets.items()
                      if start <= record["char_offset"] < end]
            rows.append({
                "kind": "text", "doc_id": doc_id, "section_id": section["section_id"],
                "ord": ordinal, "heading": heading, "title": title,
                "pages": section.get("pages", []), "text": body,
                "asset_ids": inside, "char_start": start, "char_end": end,
            })

        for asset_id, record in section["images"].items():
            description = record.get("description") or ""
            if not description:
                continue          # UNREADABLE or not yet enriched — nothing to embed
            rows.append({
                "kind": "figure", "doc_id": doc_id, "section_id": section["section_id"],
                "ord": asset_id, "heading": heading, "title": title,
                "pages": [record["page"]], "page": record["page"],
                "text": _join(record.get("caption"), description),
                "asset_id": asset_id, "path": record.get("path"),
                "bbox": record.get("bbox"), "caption": record.get("caption"),
                "char_offset": record.get("char_offset"),
            })

        for asset_id, record in section["tables"].items():
            rows.append({
                "kind": "table", "doc_id": doc_id, "section_id": section["section_id"],
                "ord": asset_id, "heading": heading, "title": title,
                "pages": [record["page"]], "page": record["page"],
                "text": _join(record.get("caption"), record.get("markdown")),
                "asset_id": asset_id, "rows": record.get("rows"),
                "cols": record.get("cols"), "caption": record.get("caption"),
                "char_offset": record.get("char_offset"),
            })
    return rows


def _join(caption, body):
    """Caption first: it carries the authors' own words, which queries echo."""
    return "\n\n".join(part for part in (caption, body) if part)


def embed_input(row):
    """What actually gets embedded — heading included, so a chunk keeps its context."""
    text = row["text"]
    if row["heading"]:
        text = f"{row['heading']}\n\n{text}"
    return text[: MAX_ROW_TOKENS * CHARS_PER_TOKEN]


def point_id(row):
    return str(uuid.uuid5(NAMESPACE, f"{row['doc_id']}:{row['section_id']}:"
                                     f"{row['kind']}:{row['ord']}"))


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #

def connect():
    from qdrant_client import QdrantClient
    url = os.getenv(QDRANT_URL_ENV)
    if url:
        return QdrantClient(url=url), url
    os.makedirs(os.path.dirname(QDRANT_PATH) or ".", exist_ok=True)
    return QdrantClient(path=QDRANT_PATH), QDRANT_PATH


def ensure_collection(client, collection, recreate=False):
    from qdrant_client import models

    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={DENSE_VECTOR: models.VectorParams(
                size=EMBED_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )
        # Filters on these run for every evaluation query, so index them.
        for field, schema in (("doc_id", models.PayloadSchemaType.KEYWORD),
                              ("kind", models.PayloadSchemaType.KEYWORD),
                              ("section_id", models.PayloadSchemaType.INTEGER)):
            client.create_payload_index(collection, field_name=field, field_schema=schema)
    return collection


def to_sparse(weights):
    """BGE-M3 lexical weights -> Qdrant sparse vector."""
    from qdrant_client import models
    return models.SparseVector(indices=[int(k) for k in weights],
                               values=[float(v) for v in weights.values()])


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

class Populator:
    def __init__(self, collection=QDRANT_COLLECTION, batch_size=EMBED_BATCH_SIZE):
        self.collection = collection
        self.batch_size = batch_size
        self.docs_dir = os.path.join(LOCAL_PROCESSED_DIR, PROCESSED_DOCS_DIR_NAME)
        self.state_path = os.path.join(LOCAL_PROCESSED_DIR, INDEXED_DOCS_JSON)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            print(f"loading {EMBED_MODEL} ...", flush=True)
            self._model = BGEM3FlagModel(EMBED_MODEL, use_fp16=True)
        return self._model

    def _indexed(self):
        try:
            with open(self.state_path) as handle:
                return set(json.load(handle))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _record(self, indexed):
        with open(self.state_path + ".part", "w") as handle:
            json.dump(sorted(indexed), handle)
        os.replace(self.state_path + ".part", self.state_path)

    def encode(self, texts):
        out = self.model.encode(texts, batch_size=self.batch_size,
                                return_dense=True, return_sparse=True,
                                return_colbert_vecs=False)
        return out["dense_vecs"], out["lexical_weights"]

    def run(self, limit=None, force=False, recreate=False,
            chunk_tokens=CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS):
        from qdrant_client import models

        client, where = connect()
        ensure_collection(client, self.collection, recreate)
        print(f"qdrant: {where} | collection: {self.collection}")

        indexed = set() if (force or recreate) else self._indexed()
        paths = sorted(glob.glob(os.path.join(self.docs_dir, "*.json")))
        pending = [p for p in paths
                   if force or recreate or os.path.basename(p)[:-5] not in indexed]
        done = len(paths) - len(pending)
        if limit:
            pending = pending[:limit]
        print(f"{len(paths)} documents | {done} already indexed | {len(pending)} to do")
        if not pending:
            return

        target = chunk_tokens * CHARS_PER_TOKEN
        overlap = overlap_tokens * CHARS_PER_TOKEN
        counts, buffer = {"text": 0, "figure": 0, "table": 0}, []

        for path in tqdm(pending, desc="Indexing"):
            document = json.load(open(path))
            rows = rows_for_document(document, target, overlap)
            for row in rows:
                counts[row["kind"]] += 1
            buffer.extend(rows)

            if len(buffer) >= UPSERT_BATCH_SIZE:
                self._flush(client, buffer, models)
                buffer = []
            indexed.add(document["id"])
            self._record(indexed)

        if buffer:
            self._flush(client, buffer, models)

        total = client.count(self.collection).count
        print(f"\nrows added: {counts['text']:,} text | {counts['figure']:,} figure | "
              f"{counts['table']:,} table")
        print(f"collection now holds {total:,} points")

    def _flush(self, client, rows, models):
        dense, sparse = self.encode([embed_input(row) for row in rows])
        points = [
            models.PointStruct(
                id=point_id(row),
                vector={DENSE_VECTOR: dense[i].tolist(),
                        SPARSE_VECTOR: to_sparse(sparse[i])},
                payload=row,
            )
            for i, row in enumerate(rows)
        ]
        client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query, k=5, kind=None):
        """Hybrid dense + sparse, fused server-side with RRF."""
        from qdrant_client import models

        client, _ = connect()
        dense, sparse = self.encode([query])
        query_filter = (models.Filter(must=[models.FieldCondition(
            key="kind", match=models.MatchValue(value=kind))]) if kind else None)

        result = client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=dense[0].tolist(), using=DENSE_VECTOR,
                                filter=query_filter, limit=k * 4),
                models.Prefetch(query=to_sparse(sparse[0]), using=SPARSE_VECTOR,
                                filter=query_filter, limit=k * 4),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter, limit=k, with_payload=True,
        )
        return result.points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load the parsed corpus into Qdrant")
    parser.add_argument("--limit", type=int, help="index only N documents")
    parser.add_argument("--force", action="store_true", help="re-index documents already done")
    parser.add_argument("--recreate", action="store_true", help="drop the collection first")
    parser.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--search", metavar="QUERY", help="run a hybrid search and exit")
    parser.add_argument("--kind", choices=["text", "figure", "table"],
                        help="restrict --search to one row kind")
    args = parser.parse_args()

    populator = Populator(args.collection)
    if args.search:
        for hit in populator.search(args.search, kind=args.kind):
            payload = hit.payload
            print(f"\n[{hit.score:.4f}] {payload['kind']:6} {payload['doc_id']} "
                  f"§{payload['section_id']}  {str(payload.get('heading'))[:50]}")
            print(f"   {payload['text'][:200].replace(chr(10), ' ')}")
    else:
        populator.run(args.limit, args.force, args.recreate,
                      args.chunk_tokens, args.overlap_tokens)
