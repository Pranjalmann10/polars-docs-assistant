"""Stage 1b: resolve --8<-- transclusion directives and strip code_block() macros.

The user-guide markdown contains no code. Code lives in separate .py files under
docs/source/src/<lang>/user-guide/... and is pulled in at doc-build time via:

    --8<-- "python/user-guide/<path>.py:<tag>"

referencing paired markers in the snippet file:

    # --8<-- [start:tag]
    <code>
    # --8<-- [end:tag]

We resolve every directive into a real fenced code block so the indexed text has
syntax, not just prose. We also strip `{{ code_block(...) }}` macros (website-only
language-tab widgets, no textual content) and validate fence balance per document.
"""

import json
import re
from pathlib import Path

from src.ingest.fetch import RAW_DIR, list_snippet_files, list_user_guide_md

SNIPPET_ROOT = "docs/source/src"  # directive paths are relative to this
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "resolved_docs.json"

TRANSCLUSION_RE = re.compile(r'--8<--\s*"([^":]+):([^"]+)"')
CODE_BLOCK_MACRO_RE = re.compile(r"^\s*\{\{\s*code_block\(.*\)\s*\}\}\s*$", re.MULTILINE)
TAG_RE = re.compile(r"#\s*--8<--\s*\[(start|end):([^\]]+)\]")


def build_snippet_lookup(repo_dir: Path = RAW_DIR) -> dict[tuple[str, str], str]:
    """Map (directive_path, tag) -> code, e.g. ("python/user-guide/expressions/aggregation.py", "basic")."""
    lookup: dict[tuple[str, str], str] = {}
    for py_file in list_snippet_files(repo_dir):
        directive_path = py_file.relative_to(repo_dir / SNIPPET_ROOT).as_posix()
        text = py_file.read_text(encoding="utf-8")
        lines = text.split("\n")
        open_tag = None
        buf: list[str] = []
        for line in lines:
            m = TAG_RE.search(line)
            if m:
                kind, tag = m.group(1), m.group(2)
                if kind == "start":
                    open_tag = tag
                    buf = []
                elif kind == "end" and open_tag is not None:
                    lookup[(directive_path, open_tag)] = "\n".join(buf).strip("\n")
                    open_tag = None
                continue
            if open_tag is not None:
                buf.append(line)
    return lookup


def resolve_document(md_text: str, lookup: dict[tuple[str, str], str], source: str) -> str:
    md_text = CODE_BLOCK_MACRO_RE.sub("", md_text)

    lines = md_text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```python exec=") or stripped.startswith("```py exec="):
            j = i + 1
            directives: list[str] = []
            while j < len(lines) and lines[j].strip() != "```":
                m = TRANSCLUSION_RE.search(lines[j])
                if m:
                    directives.append((m.group(1), m.group(2)))
                j += 1
            # j now points at the closing ``` (or EOF if unterminated)
            resolved_chunks = []
            for path, tag in directives:
                code = lookup.get((path, tag))
                if code is None:
                    print(f"[resolve] WARNING: missing snippet {path}:{tag} in {source}")
                    continue
                resolved_chunks.append(code)
            if resolved_chunks:
                out.append("```python")
                out.append("\n".join(resolved_chunks))
                out.append("```")
            # if nothing resolved (empty/orphaned exec block), drop the whole
            # fence pair instead of leaving a dangling opener or closer
            i = j + 1 if j < len(lines) else j
            continue
        out.append(line)
        i += 1

    return "\n".join(out)


def validate_fence_balance(text: str, source: str) -> bool:
    fence_count = len(re.findall(r"^```", text, flags=re.MULTILINE))
    if fence_count % 2 != 0:
        print(f"[resolve] FENCE IMBALANCE in {source}: {fence_count} fence lines")
        return False
    if re.search(r"^\s*python\s*$", text, flags=re.MULTILINE):
        # a bare "python" line with no opening ``` before it is the orphan trap
        for m in re.finditer(r"^\s*python\s*$", text, flags=re.MULTILINE):
            preceding = text[: m.start()].rstrip().splitlines()
            if not preceding or not preceding[-1].strip().startswith("```"):
                print(f"[resolve] BARE 'python' LINE in {source}")
                return False
    return True


def resolve_all(repo_dir: Path = RAW_DIR) -> list[dict]:
    lookup = build_snippet_lookup(repo_dir)
    print(f"[resolve] built lookup with {len(lookup)} tagged snippets")

    docs = []
    md_root = repo_dir / "docs" / "source" / "user-guide"
    for md_file in list_user_guide_md(repo_dir):
        rel_source = md_file.relative_to(repo_dir / "docs" / "source").as_posix()
        raw = md_file.read_text(encoding="utf-8")
        resolved = resolve_document(raw, lookup, rel_source)
        ok = validate_fence_balance(resolved, rel_source)
        docs.append({"source": rel_source, "text": resolved, "fence_ok": ok})

    n_bad = sum(1 for d in docs if not d["fence_ok"])
    print(f"[resolve] {len(docs)} docs resolved, {n_bad} with fence issues")
    return docs


if __name__ == "__main__":
    docs = resolve_all()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(f"[resolve] wrote {OUTPUT_PATH}")
