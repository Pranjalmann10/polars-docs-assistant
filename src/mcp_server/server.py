"""Stage 5: expose retrieval as an MCP server over stdio.

Two distinct tools, not one. With a single tool there is no routing decision
for the agent to make and the "agent" reduces to a retrieval call with extra
latency. The docstrings below are the agent's only guidance on when to use
each tool — they matter as much as a prompt would.
"""

import sys
from pathlib import Path

# Allow running as `python .../src/mcp_server/server.py` (per the spec's client
# config) in addition to `python -m src.mcp_server.server`: put the project
# root on sys.path so `from src...` resolves regardless of invocation style.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from src.index.retriever import lookup_signature, search

mcp = FastMCP("polars-docs")


def _format_hits(hits: list[dict]) -> str:
    if not hits:
        return "No relevant passages found in the Polars user guide."
    parts = []
    for h in hits:
        parts.append(f"### Source: {h['breadcrumb']} (score={h['score']:.3f})\n{h['text']}")
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def search_docs(query: str, k: int = 5) -> str:
    """Search the Polars user guide for passages relevant to a natural-language question.

    Use this for conceptual, how-to, or "why does X behave this way" questions
    about Polars — e.g. "how do I filter rows", "difference between lazy and
    eager", "how does Polars handle missing data". Returns the top-k most
    semantically similar documentation passages, each with its source file and
    a relevance score. Prefer this over get_api_signature when the user is
    asking about a concept, workflow, or behavior rather than a single named
    function.
    """
    hits = search(query, k=k)
    return _format_hits(hits)


@mcp.tool()
def get_api_signature(symbol: str) -> str:
    """Look up the exact signature, parameters, and usage of a specific Polars API symbol.

    Use this when the user names a specific function, method, or class — e.g.
    "DataFrame.group_by", "pl.col", "join_asof", "cast" — and wants to know its
    parameters, default values, or exact usage, rather than a general
    conceptual explanation. Pass the bare or dotted symbol name as `symbol`.
    Returns the documentation passages most likely to contain that symbol's
    signature and usage examples, ranked by literal symbol occurrence.
    """
    hits = lookup_signature(symbol)
    return _format_hits(hits)


if __name__ == "__main__":
    mcp.run(transport="stdio")
