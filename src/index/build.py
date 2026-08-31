"""Stage 3: embed chunks.json and write them into a persistent Chroma collection."""

import json
from pathlib import Path

import chromadb

from src.index.retriever import (
    COLLECTION_NAME,
    PERSIST_DIR,
    embed_documents,
)

CHUNKS_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks.json"
BATCH_SIZE = 64


def build_index(chunks_path: Path = CHUNKS_PATH) -> None:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"[index] embedding {len(chunks)} chunks with bge-small-en-v1.5")

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_documents(texts)  # no instruction prefix on documents
        collection.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": c["source"],
                    "breadcrumb": c["breadcrumb"],
                    "has_code": c["has_code"],
                }
                for c in batch
            ],
        )
        print(f"[index] wrote {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"[index] done. collection '{COLLECTION_NAME}' at {PERSIST_DIR}, count={collection.count()}")


if __name__ == "__main__":
    build_index()
