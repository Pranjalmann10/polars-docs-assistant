"""Stage 2: structural chunking on ## header boundaries.

Fixed-size windows slice code examples away from the prose that explains them.
These docs have real header structure (## sections, ### subsections), so we
split on that instead.

Rules:
- Split primarily on ## headers.
- A section over ~4000 chars is split further on ### headers.
- A section under ~200 chars merges into the previous chunk from the same file
  (too short to stand alone, usually a stub or a lone intro sentence).
- Every chunk is prefixed with a breadcrumb: "file path > section title".
  A chunk that says "it accepts three arguments" is meaningless without
  knowing what "it" is.
- source and breadcrumb are kept as metadata fields, not just baked into text.
"""

import json
import re
from pathlib import Path

INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "resolved_docs.json"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks.json"

MAX_SECTION_CHARS = 4000
MIN_CHUNK_CHARS = 200

H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
H3_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)


def split_by_heading(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Return [(heading_title, section_text_including_heading), ...].
    Any text before the first match of `pattern` is returned with heading ''."""
    matches = list(pattern.finditer(text))
    if not matches:
        return [("", text)]
    sections = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((title, text[start:end]))
    return sections


def title_of_doc(doc_text: str, source: str) -> str:
    m = H1_RE.search(doc_text)
    if m:
        return m.group(1).strip()
    return Path(source).stem.replace("-", " ").title()


def chunk_document(doc: dict) -> list[dict]:
    source = doc["source"]
    text = doc["text"]
    doc_title = title_of_doc(text, source)

    h2_sections = split_by_heading(text, H2_RE)

    raw_chunks: list[dict] = []
    for h2_title, h2_text in h2_sections:
        section_title = h2_title or doc_title
        if len(h2_text) > MAX_SECTION_CHARS:
            h3_sections = split_by_heading(h2_text, H3_RE)
            if len(h3_sections) > 1:
                for h3_title, h3_text in h3_sections:
                    title = f"{section_title} > {h3_title}" if h3_title else section_title
                    raw_chunks.append({"title": title, "text": h3_text})
                continue
        raw_chunks.append({"title": section_title, "text": h2_text})

    # merge chunks under MIN_CHUNK_CHARS into the previous chunk (same doc);
    # a too-short chunk with nothing before it (e.g. a bare H1 stub before the
    # first ## heading) merges forward into the next one instead of being lost
    merged: list[dict] = []
    pending_prefix: dict | None = None
    for c in raw_chunks:
        body = c["text"].strip()
        if not body:
            continue
        if len(body) < MIN_CHUNK_CHARS:
            if merged:
                merged[-1]["text"] = merged[-1]["text"].rstrip() + "\n\n" + c["text"]
            elif pending_prefix is None:
                pending_prefix = c
            else:
                pending_prefix["text"] = pending_prefix["text"].rstrip() + "\n\n" + c["text"]
            continue
        if pending_prefix is not None:
            c = dict(c)
            c["text"] = pending_prefix["text"].rstrip() + "\n\n" + c["text"]
            pending_prefix = None
        merged.append(c)
    if pending_prefix is not None:
        merged.append(pending_prefix)

    chunks = []
    for i, c in enumerate(merged):
        breadcrumb = f"{source} > {c['title']}" if c["title"] else source
        body = c["text"].strip()
        full_text = f"{breadcrumb}\n\n{body}"
        chunks.append(
            {
                "id": f"{source}::{i}",
                "text": full_text,
                "source": source,
                "breadcrumb": breadcrumb,
                "has_code": "```" in body,
                "char_len": len(full_text),
            }
        )
    return chunks


def chunk_all(input_path: Path = INPUT_PATH) -> list[dict]:
    docs = json.loads(input_path.read_text(encoding="utf-8"))
    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all()
    lengths = sorted(c["char_len"] for c in chunks)
    median = lengths[len(lengths) // 2]
    n_code = sum(1 for c in chunks if c["has_code"])
    print(f"[chunk] {len(chunks)} chunks (target 250-350)")
    print(f"[chunk] median length {median} chars, min {lengths[0]}, max {lengths[-1]}")
    print(f"[chunk] {n_code}/{len(chunks)} chunks contain fenced code")

    OUTPUT_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"[chunk] wrote {OUTPUT_PATH}")
