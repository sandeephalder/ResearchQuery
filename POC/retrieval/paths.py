"""Where indexing put things, resolved from anywhere.

Indexing runs from `Ingestion/` and its constants are relative to that
directory; querying is done from anywhere, including a notebook two directories
away and a uvicorn worker started at the repo root. So every path the query side
uses is resolved through here, and the indexing constants are imported rather
than restated — a chunk size that disagreed with the one used at index time
would build a BM25 index whose rows are not the rows in Qdrant.
"""

import os
import sys

from .constants import BM25_DIR_NAME, INGESTION_DIR

# `db_populate` imports its siblings flat (`from constants import ...`), because
# it is run as a script from inside Ingestion/. Importing it as a module means
# putting that directory on the path rather than rewriting working code.
if INGESTION_DIR not in sys.path:
    sys.path.insert(0, INGESTION_DIR)

import constants as indexing                                          # noqa: E402
import db_populate                                                    # noqa: E402

CHUNK_CHARS = indexing.CHUNK_TOKENS * indexing.CHARS_PER_TOKEN
OVERLAP_CHARS = indexing.CHUNK_OVERLAP_TOKENS * indexing.CHARS_PER_TOKEN


class RetrievalError(RuntimeError):
    """A store is missing, locked, or out of step with the other."""


def absolute(path):
    """Ingestion's constants are relative to Ingestion/; resolve them from anywhere."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(INGESTION_DIR, path))


def qdrant_path():
    return absolute(indexing.QDRANT_PATH)


def docs_dir():
    return absolute(os.path.join(indexing.LOCAL_PROCESSED_DIR,
                                 indexing.PROCESSED_DOCS_DIR_NAME))


def bm25_dir():
    return absolute(os.path.join(indexing.LOCAL_PROCESSED_DIR, BM25_DIR_NAME))
