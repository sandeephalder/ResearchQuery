"""Knobs for the query side of the pipeline.

Indexing constants live in `Ingestion/constants.py` and are imported from there,
never restated: a chunk size that disagrees with the one used at index time
would build a BM25 index whose rows are not the rows in Qdrant.
"""

import os

# Paths are resolved against the package, not the working directory. Indexing
# runs from `Ingestion/` and its constants are relative to that; querying is done
# from anywhere, including a notebook two directories away.
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INGESTION_DIR = os.path.join(POC_DIR, "Ingestion")

# --- BM25 index -------------------------------------------------------------- #
BM25_DIR_NAME = "bm25"
BM25_MATRIX_NAME = "postings.npz"       # the inverted index, term-major
BM25_META_NAME = "meta.json"            # vocabulary, row ids, corpus statistics

BM25_K1 = 1.2                  # Lucene's defaults. Nothing here has been tuned against
BM25_B = 0.75                  # the benchmark yet — that belongs in the eval harness.
MIN_TOKEN_CHARS = 2            # drops "a", "x"; keeps "3d", "co", "ml"

# Terms carrying no discrimination at all. Deliberately short: BM25's own IDF
# already demotes common words, and a long list is how domain terms get lost
# ("control", "state", "significant" are content words in this corpus).
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
without into over under is are was were be been being am do does did doing have has had
having it its as we our us they them their he she his her you your i not no nor so such
can could may might must shall should will would
""".split())

# --- Retrieval --------------------------------------------------------------- #
VECTOR_TOP_K = 10              # candidates from Qdrant
BM25_TOP_K = 10                # candidates from BM25
RERANK_TOP_N = 5               # survivors sent to the answer model
RRF_K = 60                     # rank-fusion damping; 60 is the constant from the paper

# What the vector leg searches with. "dense" keeps the two legs complementary —
# BGE-M3's own sparse vector is a learned lexical match, so pairing it with BM25
# spends both legs on lexical overlap. "hybrid" fuses dense+sparse server-side
# first, which is what `db_populate.py --search` does; it costs nothing extra,
# since one BGE-M3 forward pass emits both.
VECTOR_MODE = os.getenv("VECTOR_MODE", "dense")
VECTOR_MODES = ("dense", "hybrid")

# Sent to the answer model. Long enough for a full chunk — unlike reranking,
# answering needs the passage, not the gist of it.
ANSWER_MAX_CHARS = 4000

# --- Reranking --------------------------------------------------------------- #
# "llm" is the default because it is what the pipeline was specified around, and
# because a listwise call weighs candidates against each other in a way that
# scoring independent pairs cannot. "cross-encoder" is local, ~270x faster and
# free, and removes ~73% of the token cost of an evaluation sweep. Which one is
# actually better on this corpus is unmeasured — that is what the eval is for.
# Local by default. The listwise LLM call may or may not rank better — that is
# still unmeasured — but it is an API call on the query path, and reranking is
# ~73% of the token bill of a sweep. `--reranker llm` opts back in.
RERANKER = os.getenv("RERANKER", "cross-encoder")
RERANKERS = ("llm", "cross-encoder", "none")

# Trained alongside bge-m3, the model that produced the vectors being reranked,
# and reads 8192 tokens — more than MAX_ROW_TOKENS, so no row is truncated
# before it is judged. ms-marco-MiniLM-L-6-v2 is the 88 MB alternative, but its
# 512-token limit cuts the p90 table row in half.
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
# Rows are capped at MAX_ROW_TOKENS (2048) when indexed; this leaves room for
# the question alongside the longest of them.
CROSS_ENCODER_MAX_LENGTH = 2560
