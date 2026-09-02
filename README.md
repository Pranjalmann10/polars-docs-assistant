# Polars Docs Assistant

An agentic RAG question-answering system over the [Polars](https://pola.rs) user guide.
A LangGraph agent decides which of two MCP tools to call — semantic search or a
targeted API-signature lookup — over a Chroma vector index of the docs, then a
small generator model writes the final answer from the retrieved context. The
point of the project isn't the chatbot: it's the **measurement**. Three
architectural configurations (no retrieval / plain RAG / agentic RAG) are
scored on one Ragas harness so the value each added can actually be quantified,
not just claimed.

## Architecture

```
                 ┌─────────────────────────────┐
   user query →  │   LangGraph ReAct agent      │
                 │   (llama3.1:8b via Ollama)   │
                 └──────────┬──────────────────┘
                            │  MCP / stdio
                 ┌──────────▼──────────────────┐
                 │   polars-docs MCP server     │
                 │   • search_docs(query, k)    │
                 │   • get_api_signature(name)  │
                 └──────────┬──────────────────┘
                            │
                 ┌──────────▼──────────────────┐
                 │   Chroma (persistent)        │
                 │   294 chunks, 384-dim        │
                 │   bge-small-en-v1.5          │
                 └─────────────────────────────┘
                            │
                 ┌──────────▼──────────────────┐
                 │   Generator                  │
                 │   Qwen2.5-1.5B-Instruct       │
                 │   (+ QLoRA adapter, if set)  │
                 └─────────────────────────────┘

   observability: LangSmith traces every node + tool call (optional, needs a key)
   evaluation:    Ragas over a 45-question held-out set
```

**Honest scoping note.** A vector database isn't technically necessary at
~300 chunks — a numpy dot product retrieves fine. Chroma is used because it
solves process separation (the MCP server runs in its own OS process and
can't reach notebook/script memory) and because it's the standard tool a
reviewer expects to see, not because it was a performance decision.

## Repository layout

```
polars-docs-assistant/
├── data/
│   ├── raw/                  # sparse clone of pola-rs/polars (gitignored)
│   ├── resolved_docs.json    # markdown with --8<-- transclusions resolved
│   └── chunks.json           # structurally chunked corpus with metadata
├── eval/
│   ├── eval_set.jsonl        # 45 question/ground-truth pairs
│   ├── eval_set_run.jsonl    # 15-question subset actually scored (see below)
│   └── results/              # scored output, committed
├── src/
│   ├── ingest/                # fetch.py, resolve.py, chunk.py — Stages 1-2
│   ├── index/                  # build.py, retriever.py — Stage 3
│   ├── mcp_server/server.py    # Stage 5
│   ├── agent/graph.py          # Stage 6
│   ├── generator/model.py      # Stage 7
│   └── evaluate/               # run_eval.py, metrics.py — Stage 8
└── notebooks/                 # exploration only, not the source of truth
```

`src/index/retriever.py` is importable with no notebook state and no side
effects on import — verified by running it from a fresh process (see Stage 3
below). The MCP server is a separate OS process; if the retriever depended on
globals, nothing downstream would work.

## Setup (from a clean clone)

```bash
pip install -e .
cp .env.example .env        # fill in LANGCHAIN_API_KEY if you want tracing

# Stage 1-2: fetch + resolve + chunk the docs
python -m src.ingest.fetch
python -m src.ingest.resolve
python -m src.ingest.chunk

# Stage 3: embed and build the Chroma index
python -m src.index.build

# Stage 5 sanity check: run the MCP server standalone
python -m src.mcp_server.server

# Stage 6: ask the agent something
python -m src.agent.graph "how do I do a groupby in polars"

# Stage 8: run the evaluation
python -m src.evaluate.run_eval
```

The agent's reasoning-brain model defaults to a **local Ollama model**
(`llama3.1:8b`, no API key needed) — see [Design decisions](#design-decisions)
for why `qwen2.5-coder:7b` was tried first and rejected. Pull it once with
`ollama pull llama3.1:8b`.

## Design decisions

**Why `llama3.1:8b`, not the fine-tuned Qwen2.5-1.5B, drives the agent.**
A 1.5B fine-tuned model does not reliably emit well-formed tool calls — small
models are bad at structured tool invocation and good at imitating a target
output format. So the agent uses a separate, competent tool-calling model to
decide *what to retrieve*, and hands the retrieved context to the fine-tuned
Qwen generator to decide *how to answer*. The agent itself never writes the
user-facing answer.

**Why `llama3.1:8b`, not `qwen2.5-coder:7b` (which was already pulled
locally).** Tested first since it required no download. Its Ollama chat
template asks the model to wrap a tool call in `<tool_call>...</tool_call>`
tags so Ollama's parser can extract it; in testing it reliably emitted the
correct JSON but *without* those tags, so Ollama couldn't parse it out and the
call leaked into plain-text `content` instead of `tool_calls` — the agent
never actually called a tool, it just wrote text that looked like one.
`llama3.1:8b` followed Ollama's tool-calling protocol correctly and
consistently in testing.

**A subtler failure found the same way: a prompt can talk a tool-calling
model out of calling tools.** An early system prompt for the agent described
tool-calling mechanics in prose (*"call exactly one of the available tools...
after calling a tool, write..."*). That was enough to make `llama3.1:8b`
abandon native tool-calling and narrate a fake `"Tool Response:"` block as
plain text — same failure mode as above, different cause. The fix was to
strip the prompt down to tool *choice* only (`search_docs` vs
`get_api_signature`) and let `bind_tools` communicate the calling mechanics,
since that's what it's for. Verified with a minimal `create_react_agent` +
one dummy tool, with and without the prompt, before touching the real MCP
tools — worth doing that isolation before assuming the retrieval side is
broken.

**No QLoRA adapter exists yet for this build.** `src/generator/model.py` is
written so arm C (agent + adapter) activates automatically the moment
`ADAPTER_PATH` points at a real checkpoint trained against
`Qwen/Qwen2.5-1.5B-Instruct` — no code changes required. Until then, arm C
uses the base model, which still isolates the *agent's retrieval-routing*
contribution from the *adapter's* contribution; they're evaluated separately by
construction. `_verify_adapter_base_match` refuses to load an adapter trained
against a different base model, because a mismatch either fails on shape or
silently produces fluent-looking garbage.

**Ragas judge is a local model, not a hosted API.** No `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` was available in this environment, so `src/evaluate/metrics.py`
defaults the Ragas judge to the same local `llama3.1:8b` used as the agent
brain (`RAGAS_JUDGE_PROVIDER=ollama`). This is a real limitation, not a
detail to gloss over: an 8B local judge is weaker and noisier than GPT-4o or
Claude as a judge. Treat the Ragas numbers below as useful for comparing arms
A/B/C *against each other* on this harness, not as absolute scores comparable
to published Ragas benchmarks. Set `RAGAS_JUDGE_PROVIDER=anthropic` (with a
key) to swap in a stronger judge.

**The eval run committed here scored 15 of the 45 questions, not all 45.**
Generation alone took ~30s/question on a single 4GB consumer GPU, and Ragas'
four metrics require one local-LLM judge call each — at `max_workers=1`
(serialized, since running the judge and the generator concurrently on a
4GB GPU risks OOM) that's `4 × N` sequential judge calls per arm. Scaling that
to all 45 questions × 3 arms would take multiple hours on this hardware. The
15-question subset (`eval/eval_set_run.jsonl`, every 3rd question from the
full set, preserving the topic/difficulty spread) keeps the run tractable
while still exercising all three arms identically. The full 45-question set
ships in `eval/eval_set.jsonl` and `python -m src.evaluate.run_eval` scores it
end-to-end on better hardware or with a hosted judge — just unset
`EVAL_SET_PATH`/`EVAL_RESULTS_DIR` or point them elsewhere.

## Observability

LangSmith tracing is wired up (`LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` /
`LANGCHAIN_PROJECT` in `.env.example`, loaded automatically via
`src/__init__.py`) but **no LangSmith API key was available in this
environment**, so no trace was actually recorded and no screenshot is
included here. Set those three variables and rerun the agent or the eval to
get traces — LangChain auto-instruments every node and tool call once the
env vars are present, no code changes needed.

## Results

See `eval/results/summary.md` / `summary.json` for the table this run
produced, and `eval/results/*_raw.jsonl` for the per-question answers,
retrieved contexts, and latencies behind it.

<!-- RESULTS_TABLE_PLACEHOLDER -->

**Interpretation:** *(filled in after the run completes — see below)*

## What I would do differently

- Use a hosted judge model (Claude Haiku or GPT-4o-mini) for Ragas scoring
  instead of a local 8B model — the numbers below are directionally useful for
  comparing arms against each other, but I would not trust their absolute
  values, and a stronger judge would make the faithfulness/relevancy scores
  much more trustworthy on their own.
- Train the actual QLoRA adapter this project was scoped around. Right now
  arm C measures only the agent's retrieval-routing contribution, not the
  fine-tuning contribution the project's title promises — that's the biggest
  gap between what this repo does and what the original spec asked for.
- Run the full 45-question set instead of the 15-question subset, on
  better/rented GPU hardware, to get tighter confidence in the per-arm scores.
- Add a small set of "trap" questions to the eval set that specifically probe
  whether the fine-tuned adapter's answer-format training *overrides* good
  retrieved context (spec's stage 7 flags this as a real failure mode worth
  measuring, not just retrieval quality).
