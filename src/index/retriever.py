"""Stage 3: query interface over the Chroma index.

Import-safe by design: no notebook state, no side effects on import, no global
client construction at module load time. The MCP server runs in its own OS
process and cannot reach notebook memory, so this module must work identically
whether it's imported from a notebook, a script, or `python -m mcp.server`.

BGE models expect a short instruction prefix on QUERIES but not on documents.
Getting this backwards silently degrades retrieval — it doesn't error, it just
retrieves worse passages, which is exactly the kind of bug nobody notices.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parents[2] / "data" / "chroma")
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "polars_docs")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# BGE-family instruction prefix for asymmetric search (query != document encoding).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME)


def embed_query(query: str) -> list[float]:
    embedder = _get_embedder()
    vec = embedder.encode(
        QUERY_INSTRUCTION + query,
        normalize_embeddings=True,  # inner product == cosine similarity
    )
    return vec.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    embedder = _get_embedder()
    vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


def search(query: str, k: int = 5) -> list[dict]:
    """Semantic search over the Polars docs. Returns [{text, source, breadcrumb, score}, ...]."""
    collection = _get_collection()
    query_vec = embed_query(query)
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source"),
                "breadcrumb": meta.get("breadcrumb"),
                "score": 1 - dist,  # cosine distance -> similarity
            }
        )
    return hits


_SIGNATURE_HINT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)")


def lookup_signature(symbol: str) -> list[dict]:
    """Targeted lookup for a specific API symbol (e.g. 'DataFrame.group_by', 'pl.col').

    There is no separate signature database — we bias the same vector search
    toward code-bearing chunks and rerank by literal symbol occurrence, since
    the docs corpus doesn't ship a structured API reference distinct from prose.
    """
    m = _SIGNATURE_HINT_RE.match(symbol.strip())
    bare_symbol = m.group(1) if m else symbol.strip()
    short_name = bare_symbol.split(".")[-1]

    candidates = search(f"{symbol} function signature parameters", k=15)
    code_hits = [c for c in candidates if "```" in c["text"]]
    pool = code_hits or candidates

    def relevance(hit: dict) -> tuple[int, float]:
        occurrences = hit["text"].count(short_name)
        return (occurrences, hit["score"])

    pool.sort(key=relevance, reverse=True)
    return pool[:5]
