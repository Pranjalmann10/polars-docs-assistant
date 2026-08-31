"""Stage 1a: clone the Polars repo and keep only the user-guide markdown."""

import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/pola-rs/polars.git"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
USER_GUIDE_MD = "docs/source/user-guide"
# code referenced by --8<-- directives lives under docs/source/src/python/user-guide,
# NOT top-level python/user-guide (that's a common but wrong assumption).
SNIPPET_DIRS = ["docs/source/src/python/user-guide"]


def clone_repo(dest: Path = RAW_DIR) -> Path:
    """Sparse, blobless, depth-1 clone: only the user-guide docs + their code snippets."""
    if dest.exists() and any(dest.iterdir()):
        print(f"[fetch] {dest} already populated, skipping clone")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "--filter=blob:none", "--sparse",
            REPO_URL, str(dest),
        ],
        check=True,
    )
    sparse_file = dest / ".git" / "info" / "sparse-checkout"
    sparse_file.write_text(
        f"/{USER_GUIDE_MD}/**\n" + "".join(f"/{d}/**\n" for d in SNIPPET_DIRS)
    )
    subprocess.run(["git", "sparse-checkout", "reapply"], cwd=dest, check=True)
    return dest


def list_user_guide_md(repo_dir: Path = RAW_DIR) -> list[Path]:
    root = repo_dir / USER_GUIDE_MD
    if not root.exists():
        raise FileNotFoundError(f"{root} not found — did the clone succeed?")
    return sorted(root.rglob("*.md"))


def list_snippet_files(repo_dir: Path = RAW_DIR) -> list[Path]:
    files: list[Path] = []
    for sub in SNIPPET_DIRS:
        root = repo_dir / sub
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


if __name__ == "__main__":
    repo_dir = clone_repo()
    md_files = list_user_guide_md(repo_dir)
    py_files = list_snippet_files(repo_dir)
    print(f"[fetch] {len(md_files)} markdown files under {USER_GUIDE_MD}")
    print(f"[fetch] {len(py_files)} snippet .py files under {SNIPPET_DIRS}")
