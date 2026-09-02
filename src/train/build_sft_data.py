"""Stage 9a: build the SFT dataset for the QLoRA adapter.

The adapter's job (see src/generator/model.py's module docstring and the
README's "Design decisions") is not to teach the 1.5B model new Polars facts —
it's to teach it the *answer format* the RAG system prompt asks for
(grounded-in-context, cites the source doc, includes a code example when
relevant) since small models imitate a target format far more reliably than
they follow it from instructions alone.

Target answers are distilled from `llama3.1:8b` (the same model already used
as the agent's reasoning brain, run locally via Ollama) conditioned on a
single doc chunk as context, using the exact RAG_SYSTEM_PROMPT the generator
uses at inference time. A synthetic question is generated per chunk first,
also by the teacher, since the eval set's real questions must stay untouched.

Leakage guard: every chunk used here comes from a source doc that does NOT
appear in eval/eval_set.jsonl. Training and evaluation are disjoint at the
document level, not just the question level.
"""

import json
import re
from pathlib import Path

import ollama

from src.generator.model import RAG_SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.json"
EVAL_SET_PATH = REPO_ROOT / "eval" / "eval_set.jsonl"
OUT_PATH = REPO_ROOT / "data" / "sft_train.jsonl"

TEACHER_MODEL = "llama3.1:8b"

QUESTION_PROMPT = (
    "Below is a passage from the Polars documentation. Write ONE realistic "
    "question a Polars user would ask that this passage directly answers. "
    "Output only the question, no preamble, no quotes.\n\n"
    "Passage:\n{chunk}"
)


def _eval_source_docs() -> set[str]:
    docs = set()
    for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line).get("source_doc")
        if doc:
            docs.add(doc)
    return docs


def _load_train_chunks() -> list[dict]:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    eval_docs = _eval_source_docs()
    train_chunks = [c for c in chunks if c["id"].split("::")[0] not in eval_docs]
    assert not ({c["id"].split("::")[0] for c in train_chunks} & eval_docs)
    return train_chunks


def _chat(system: str, user: str) -> str:
    response = ollama.chat(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()


def _clean_question(raw: str) -> str:
    text = raw.strip().strip('"').strip()
    text = re.sub(r"^(question:|q:)\s*", "", text, flags=re.IGNORECASE)
    return text.splitlines()[0].strip()


def _already_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["chunk_id"])
    return done


def build(limit: int | None = None) -> None:
    chunks = _load_train_chunks()
    if limit:
        chunks = chunks[:limit]

    done = _already_done_ids(OUT_PATH)
    print(f"[sft] {len(chunks)} candidate chunks, {len(done)} already done")

    with OUT_PATH.open("a", encoding="utf-8") as out:
        for i, chunk in enumerate(chunks):
            if chunk["id"] in done:
                continue
            try:
                question = _clean_question(
                    _chat(
                        "You write concise, natural questions for a documentation "
                        "QA dataset.",
                        QUESTION_PROMPT.format(chunk=chunk["text"][:2000]),
                    )
                )
                answer = _chat(
                    RAG_SYSTEM_PROMPT,
                    f"Context:\n{chunk['text'][:2000]}\n\nQuestion: {question}",
                )
            except Exception as exc:  # ollama connection hiccup, retry-able
                print(f"[sft] chunk {chunk['id']} failed: {exc}")
                continue

            record = {
                "chunk_id": chunk["id"],
                "source_doc": chunk["id"].split("::")[0],
                "context": chunk["text"],
                "question": question,
                "answer": answer,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[sft] {i + 1}/{len(chunks)} {chunk['id']}")

    print(f"[sft] wrote {OUT_PATH}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build(limit=n)
