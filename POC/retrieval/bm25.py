"""The lexical leg, as a LangChain `BM25Retriever`.

    uv run python -m retrieval.bm25 --build
    uv run python -m retrieval.bm25 --search "how does detection probability vary with range"

This replaced a hand-written inverted index — a term-major sparse matrix, its
own IDF, its own length normalisation and its own on-disk format. `rank_bm25`
does all of that, and `BM25Retriever` puts a LangChain interface on it, which is
what lets the fusion downstream be `EnsembleRetriever` instead of more bespoke
code.

Two things did not come from a library, because both are properties of *this*
corpus rather than of BM25:

**The tokenizer.** It is passed in as `preprocess_func`, which is the hook
`BM25Retriever` provides for exactly this, and it is unchanged from the index it
replaced — compound identifiers indexed whole and in parts, a deliberately short
stopword list, no stemming.

**The kind filter.** `BM25Retriever` has no notion of metadata filters, and the
router needs one: the corpus holds prose, figure descriptions and table
transcriptions in one index, and a query routed to figures should be able to ask
for figures. `CorpusBM25Retriever` scores everything and then keeps the rows of
the requested kind, which is the honest way round — filtering before scoring
would change the IDF statistics the scores mean.

The rows are the rows Qdrant holds, produced by `store.iter_rows` from the same
`db_populate` code with the same chunk parameters. That identity is what lets a
BM25 hit be an id Qdrant can resolve.
"""

import argparse
import os
import pickle
import re
import sys
import time

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .constants import BM25_B, BM25_K1, BM25_TOP_K, MIN_TOKEN_CHARS, STOPWORDS
from .paths import RetrievalError, bm25_dir
from .store import iter_rows, label

# Bump when `tokenize` changes: an index built by a different tokenizer scores
# correctly and matches nothing, which is the failure that does not announce itself.
TOKENIZER_VERSION = 2
INDEX_NAME = "bm25.pkl"

KINDS = ("text", "figure", "table")

# A token is a run of letters and digits, optionally joined by the separators
# that hold identifiers together — "bge-m3", "gpt-4.1", "mmol/l", "f1_score".
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-_/][a-z0-9]+)*")
SEPARATOR_RE = re.compile(r"[.\-_/]")


def tokenize(text):
    """Lowercase tokens, with compound identifiers indexed whole *and* in parts.

    "CIFAR-100" is indexed as `cifar-100`, `cifar` and `100`, so a query matches
    whichever form the author used. Parts of a purely numeric token ("0.5" ->
    "0", "5") are dropped: they are noise, and they are common enough to distort
    the document lengths that BM25 normalises by.

    No stemming. It costs more than it returns on this corpus — the vocabulary
    is dominated by symbols, units and model names, where "GAs" and "GA" are
    different things, and BGE-M3's sparse leg already covers the morphological
    variants that a stemmer would catch.
    """
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) >= MIN_TOKEN_CHARS and token not in STOPWORDS:
            tokens.append(token)
        if SEPARATOR_RE.search(token):
            for part in SEPARATOR_RE.split(token):
                if (len(part) >= MIN_TOKEN_CHARS and part not in STOPWORDS
                        and not part.isdigit()):
                    tokens.append(part)
    return tokens


class CorpusBM25Retriever(BM25Retriever):
    """`BM25Retriever` that can be asked for one kind of row.

    `kind` is set on the retriever rather than passed per call because that is
    what `EnsembleRetriever` can hold: fusion takes retrievers, not arguments,
    so a kind-filtered leg is a second retriever, which `for_kind` returns
    without rebuilding the index behind it.
    """

    kind: str | None = None

    def for_kind(self, kind):
        """A view of this index restricted to one kind. Shares the vectorizer."""
        if not kind:
            return self
        return CorpusBM25Retriever(vectorizer=self.vectorizer, docs=self.docs, k=self.k,
                                   preprocess_func=self.preprocess_func, kind=kind)

    def scored(self, query, k=None, kind=None):
        """[(point_id, score)] best first — what the fusion in `pipeline` reads.

        Rows scoring zero are dropped. BM25 gives a zero to a row sharing no
        query term at all, and a rank in the fused list is a claim that the leg
        found something; padding the list to `k` with rows it did not find makes
        that claim falsely.
        """
        scores = self.vectorizer.get_scores(self.preprocess_func(query))
        kind = kind or self.kind
        order = sorted(range(len(self.docs)), key=lambda i: -scores[i])

        hits = []
        for index in order:
            if scores[index] <= 0:
                break                       # sorted, so nothing below this scores either
            document = self.docs[index]
            if kind and document.metadata.get("kind") != kind:
                continue
            hits.append((document.metadata["_id"], float(scores[index])))
            if len(hits) >= (k or self.k):
                break
        return hits

    def _get_relevant_documents(self, query, *, run_manager=None, **kwargs):
        by_id = {document.metadata["_id"]: document for document in self.docs}
        return [by_id[point_id] for point_id, _ in self.scored(query)]


# --------------------------------------------------------------------------- #
# Building and loading
# --------------------------------------------------------------------------- #

def documents(limit=None, rows=None):
    """The corpus as `Document`s, carrying the point id Qdrant knows them by."""
    for position, (point_id, kind, text) in enumerate(rows or iter_rows()):
        if limit is not None and position >= limit:
            return
        yield Document(page_content=text, metadata={"_id": point_id, "kind": kind})


def build(k1=BM25_K1, b=BM25_B, limit=None, rows=None, k=BM25_TOP_K, progress=True):
    """Tokenize every row and fit BM25 over it. ~6 s for this corpus."""
    started = time.time()
    docs = list(documents(limit, rows))
    if not docs:
        raise RetrievalError(
            "No rows to index. Parse the corpus first:\n"
            "    cd Ingestion && uv run data_process.py")
    if progress:
        print(f"indexing {len(docs)} rows ...", file=sys.stderr, flush=True)

    retriever = CorpusBM25Retriever.from_documents(
        docs, k=k, preprocess_func=tokenize,
        # Lucene's defaults. Nothing here has been tuned against the benchmark
        # yet — that belongs in the eval harness.
        bm25_params={"k1": k1, "b": b})
    if progress:
        print(f"built in {time.time() - started:.1f}s", file=sys.stderr)
    return retriever


def index_path():
    return os.path.join(bm25_dir(), INDEX_NAME)


def save(retriever):
    """Written `.part` then renamed, so a kill cannot leave a file the next run trusts."""
    os.makedirs(bm25_dir(), exist_ok=True)
    path = index_path()
    with open(f"{path}.part", "wb") as handle:
        pickle.dump({"version": TOKENIZER_VERSION, "retriever": retriever},
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(f"{path}.part", path)
    return path


def load():
    """The saved index, or a failure that says how to make one."""
    path = index_path()
    if not os.path.exists(path):
        raise RetrievalError(
            f"No BM25 index at {path}. Build it — it takes about six seconds:\n"
            f"    uv run python -m retrieval.bm25 --build")
    with open(path, "rb") as handle:
        saved = pickle.load(handle)
    if saved.get("version") != TOKENIZER_VERSION:
        raise RetrievalError(
            f"The BM25 index at {path} was built by tokenizer v{saved.get('version')}, "
            f"and this is v{TOKENIZER_VERSION}. An index scored by a different tokenizer "
            f"matches nothing without erroring, so rebuild it:\n"
            f"    uv run python -m retrieval.bm25 --build")
    return saved["retriever"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    """The CLI.

    Reached through a re-import below rather than run in place, and that is not
    a style choice. Running this file as `python -m retrieval.bm25` makes its
    module `__main__`, so a `CorpusBM25Retriever` built here would be pickled as
    `__main__.CorpusBM25Retriever` and would not unpickle anywhere else. Calling
    `main` from the imported module gives the class its real name.
    """
    parser = argparse.ArgumentParser(description="The BM25 leg of hybrid retrieval")
    parser.add_argument("--build", action="store_true", help="build the index and save it")
    parser.add_argument("--limit", type=int, help="build from the first N rows only")
    parser.add_argument("--k1", type=float, default=BM25_K1)
    parser.add_argument("--b", type=float, default=BM25_B)
    parser.add_argument("--search", metavar="QUERY", help="search the saved index")
    parser.add_argument("--kind", choices=list(KINDS), help="restrict to one row kind")
    parser.add_argument("-k", type=int, default=10, help="results to show")
    parser.add_argument("--stats", action="store_true", help="describe the saved index")
    arguments = parser.parse_args()

    try:
        if arguments.build:
            print(f"saved {save(build(arguments.k1, arguments.b, arguments.limit))}")
        if arguments.stats or arguments.search:
            index = load()
            if arguments.stats:
                kinds = {}
                for document in index.docs:
                    kinds[document.metadata["kind"]] = kinds.get(document.metadata["kind"], 0) + 1
                print(f"{len(index.docs)} rows, tokenizer v{TOKENIZER_VERSION}")
                print("  " + "  ".join(f"{kind} {count}" for kind, count in sorted(kinds.items())))
            if arguments.search:
                from .store import connect, indexing
                client, _ = connect()
                hits = index.scored(arguments.search, arguments.k, arguments.kind)
                payloads = {str(record.id): record.payload for record in client.retrieve(
                    collection_name=indexing.QDRANT_COLLECTION, ids=[i for i, _ in hits],
                    with_payload=True)} if hits else {}
                for position, (point_id, score) in enumerate(hits, start=1):
                    print(f"{position:>3} {score:7.3f}  {label(payloads.get(point_id, {}))}")
        if not (arguments.build or arguments.stats or arguments.search):
            parser.print_help()
    except RetrievalError as error:
        sys.exit(f"FAILED: {error}")


if __name__ == "__main__":
    from retrieval.bm25 import main as _main

    _main()
