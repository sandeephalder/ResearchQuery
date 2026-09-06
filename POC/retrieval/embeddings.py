"""BGE-M3 behind LangChain's `Embeddings` interface.

The collection in Qdrant was built by `Ingestion/db_populate.py` with
FlagEmbedding's `BGEM3FlagModel`, and its 80,365 dense vectors are what a query
vector has to land among. So this wraps that same model rather than reaching for
`HuggingFaceEmbeddings`: a second implementation of "BGE-M3" that pooled or
normalised even slightly differently would be a silent, corpus-wide accuracy
loss with no error to read.

Implementing `Embeddings` is what makes the rest of the query side ordinary
LangChain. `QdrantVectorStore` takes one of these, and so does anything else
that would ever want to embed against this corpus — which is the point of doing
it here instead of passing raw vectors around.

BGE-M3 emits a dense and a sparse vector from one forward pass, so
`SparseEmbeddings` comes free alongside the dense one and the two cannot
disagree about what they encoded.
"""

import sys

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector

from .paths import indexing


class BGEM3Embeddings(Embeddings):
    """The model that built the index, loaded once per process.

    Loading costs ~15 s, which is why the model is built on first use and the
    object is meant to be held rather than remade. `Pipeline` holds one; so does
    the FastAPI runtime.
    """

    def __init__(self, model_name=None, use_fp16=True, batch_size=8):
        self.model_name = model_name or indexing.EMBED_MODEL
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._model = None
        # The last text and its output. A single question is encoded more than
        # once whenever retrieval runs twice over it — the agent flow does, to
        # add a kind-filtered pass to an unfiltered one — and a second forward
        # pass through BGE-M3 to produce the vectors just produced is the most
        # expensive no-op in the query path.
        self._cached = None

    # Named `encoder`, not `model`. RAGAS's `LangchainEmbeddingsWrapper` reads
    # `embeddings.model` when it builds its telemetry event and requires a
    # string; a property returning the loaded FlagEmbedding object made every
    # `answer_relevancy` score fail with a pydantic ValidationError on a field
    # that has nothing to do with embedding. `model_name` is the string.
    @property
    def encoder(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            print(f"loading {self.model_name} ...", file=sys.stderr, flush=True)
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return self._model

    # -- the one forward pass both interfaces share -------------------------- #

    def encode(self, texts):
        """{"dense_vecs": ..., "lexical_weights": ...} for a list of texts."""
        if len(texts) == 1 and self._cached is not None and self._cached[0] == texts[0]:
            return self._cached[1]
        out = self.encoder.encode(list(texts), batch_size=self.batch_size,
                                  return_dense=True, return_sparse=True,
                                  return_colbert_vecs=False)
        if len(texts) == 1:
            self._cached = (texts[0], out)
        return out

    # -- Embeddings ---------------------------------------------------------- #

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.encode(list(texts))["dense_vecs"]]

    def embed_query(self, text):
        return self.encode([text])["dense_vecs"][0].tolist()


class BGEM3SparseEmbeddings(SparseEmbeddings):
    """The lexical half of the same forward pass, as Qdrant sparse vectors.

    Shares the `BGEM3Embeddings` it is given rather than loading a second copy
    of a 2 GB model — and shares its cache, so asking for both halves of one
    question costs one pass.
    """

    def __init__(self, dense: BGEM3Embeddings):
        self.dense = dense

    @staticmethod
    def _sparse(weights):
        """FlagEmbedding's {token_id: weight} as a Qdrant SparseVector.

        Token ids arrive as strings and weights as numpy floats; both have to be
        native types before qdrant-client will serialise them.
        """
        items = [(int(token), float(weight)) for token, weight in weights.items()
                 if float(weight) > 0]
        return SparseVector(indices=[i for i, _ in items], values=[w for _, w in items])

    def embed_documents(self, texts):
        return [self._sparse(w) for w in self.encode_sparse(list(texts))]

    def embed_query(self, text):
        return self._sparse(self.encode_sparse([text])[0])

    def encode_sparse(self, texts):
        return self.dense.encode(texts)["lexical_weights"]
