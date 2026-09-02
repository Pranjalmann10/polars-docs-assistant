# Polars Docs Assistant — Interview Prep & Resume Notes

Everything below is grounded in the actual code in this repo as of 2026-09-02.
Read top to bottom once; the "likely questions" sections at the end are the
fast-review pass the night before.

---

## 1. The one-sentence pitch

An agentic RAG system that answers Polars (Python dataframe library) questions
by routing through a LangGraph agent to an MCP tool server backed by a Chroma
vector index, then generating the final answer with a fine-tuned Qwen2.5-1.5B
model — and a 3-arm evaluation harness (no-retrieval / plain-RAG / agentic-RAG)
that measures how much each layer actually adds, using Ragas metrics.

The point of the project is **the measurement**, not the chatbot. That framing
is the thing to lead with in an interview — it signals you understand that
"add RAG" and "add an agent" are hypotheses that need evidence, not defaults.

---

## 2. End-to-end data/control flow

```
docs (GitHub)  →  fetch.py  →  resolve.py  →  chunk.py  →  build.py (Chroma)
                                                                  │
                                                          retriever.py
                                                        (search / lookup_signature)
                                                                  │
                                                         mcp_server/server.py
                                                     (search_docs, get_api_signature)
                                                                  │
  user question → agent/graph.py (LangGraph ReAct, llama3.1:8b via Ollama)
                                                                  │
                                            tool-call results (retrieved passages)
                                                                  │
                                        generator/model.py (Qwen2.5-1.5B [+QLoRA])
                                                                  │
                                                            final answer
```

Separately: `train/build_sft_data.py` → `train/sft_train.jsonl` →
`train/train_qlora.py` → `models/qwen2.5-1.5b-polars-qlora/` (the adapter),
and `evaluate/run_eval.py` → `eval/results/*` (the scoring harness that ties
all three arms together).

---

## 3. File-by-file walkthrough

### `src/ingest/fetch.py` — Stage 1a: get the raw docs
- Does a **sparse, blobless, depth-1 git clone** of `pola-rs/polars` — only
  `docs/source/user-guide/**` and the Python code-snippet directory are
  checked out, not the whole (huge) monorepo.
- Uses `git clone --filter=blob:none --sparse` then writes a
  `.git/info/sparse-checkout` file and reapplies it. This is a real git
  feature worth being able to explain: blobless clone fetches trees/commits
  eagerly but blobs (file contents) on demand, and sparse-checkout further
  restricts the working tree to matching paths.
- Non-obvious detail worth mentioning: the code-snippet files referenced by
  the docs live at `docs/source/src/python/user-guide/...`, **not** the
  more "obvious" top-level `python/user-guide/...` path — a wrong assumption
  the comment explicitly calls out avoiding.

### `src/ingest/resolve.py` — Stage 1b: resolve transclusions
- The user-guide markdown files contain **no code** — code lives in separate
  `.py` files and is pulled into the docs at build time via
  `--8<-- "path:tag"` transclusion directives, referencing
  `# --8<-- [start:tag]` / `[end:tag]` markers in the snippet files (this is
  the `pymdownx.snippets` MkDocs convention).
- `build_snippet_lookup()` parses every snippet file once into a
  `{(path, tag): code}` dict.
- `resolve_document()` finds fenced blocks like ` ```python exec=... ` that
  contain transclusion directives and replaces them with a real fenced code
  block containing the actual resolved code — this is what turns prose-only
  markdown into markdown with real code inside it (important because you're
  about to embed/retrieve this text, and "there's an example below" with no
  actual example is useless to a retriever).
- Also strips `{{ code_block(...) }}` Jinja macros (website-only UI widgets
  for language tabs — no textual content to keep).
- `validate_fence_balance()` is a sanity check: after all this
  find/replace, are the triple-backtick fences still balanced, and is there a
  stray bare `python` line that indicates a broken close? This is the kind
  of "did my regex surgery accidentally break the document" check that's easy
  to skip and easy to regret skipping.

### `src/ingest/chunk.py` — Stage 2: structural chunking
- **Deliberately not** fixed-size/sliding-window chunking. Splits on real
  Markdown structure: primarily `##` headers, with `###` sub-splitting for
  any section over ~4000 chars, because fixed windows would slice a code
  example away from the prose explaining it.
- Sections under ~200 chars get merged into a neighboring chunk (too short to
  be useful/retrievable on their own — usually a stub intro sentence).
- Every chunk is prefixed with a **breadcrumb**: `"file path > section title"`.
  Rationale stated directly in the docstring: a chunk that says "it accepts
  three arguments" is meaningless without knowing what "it" is once it's been
  pulled out of its document context by a retriever.
- Output: 294 chunks (README's "~300" claim), each with `id`, `text`
  (breadcrumb + body), `source`, `breadcrumb`, `has_code` (bool), `char_len`.

### `src/index/build.py` — Stage 3: embed + index
- Deletes and recreates a Chroma **persistent** collection (`polars_docs`)
  configured for cosine distance (`hnsw:space: cosine`).
- Batches embedding in groups of 64, writes `ids`, `documents`, `embeddings`,
  and a slim `metadatas` dict (source/breadcrumb/has_code) per chunk.
- Documents are embedded with **no instruction prefix** — asymmetric
  encoding, see retriever.py below for why that matters.

### `src/index/retriever.py` — Stage 3: the query interface
- **Import-safety is the design constraint of this file**, called out in its
  own docstring: no notebook globals, no client built at import time, because
  the MCP server runs in a *separate OS process* and can't reach whatever
  state a notebook has in memory. Everything is built lazily via
  `@lru_cache(maxsize=1)` accessor functions (`_get_embedder`,
  `_get_collection`) — cached after first call, but constructed on demand,
  not at import.
- **BGE asymmetric search**: BGE-family embedding models expect a short
  instruction prefix on the *query* side only
  (`"Represent this sentence for searching relevant passages: "`), not on
  documents. Get this backwards and nothing crashes — retrieval just gets
  quietly worse. This is a great "subtle bug that would never throw an
  exception" story if asked about tricky debugging.
- `search(query, k)`: embeds the query (normalized so inner product = cosine
  similarity), queries Chroma, converts Chroma's cosine *distance* back to a
  similarity score (`1 - dist`).
- `lookup_signature(symbol)`: there's no separate structured API-reference
  database in this docs corpus — so this doesn't do a real "signature
  lookup." It biases the *same* vector search toward code-bearing chunks
  (`"{symbol} function signature parameters"` as the query, k=15, then
  filters to chunks containing a fenced code block) and **reranks by literal
  symbol occurrence** in the text, falling back to the raw candidate pool if
  no code-bearing chunk exists. This is an honest, pragmatic hack over
  "there's no ground-truth signature data, so approximate it" — worth
  narrating plainly if asked, rather than implying it's a real API reference.

### `src/mcp_server/server.py` — Stage 5: MCP tool server
- Uses `FastMCP` (from the `mcp` SDK) to expose exactly **two** tools over
  stdio: `search_docs` and `get_api_signature`.
- Explicit design point in the docstring: two tools, not one — with a single
  tool there's no *routing decision* left for the agent to make, and the
  "agent" degenerates into a retrieval call with extra latency bolted on. The
  two tools' docstrings are written as the actual decision boundary the agent
  sees (conceptual/how-to → `search_docs`; named function/method →
  `get_api_signature`) — those docstrings are functionally part of the
  prompt, since `bind_tools`/MCP tool schemas surface them to the model.
- Handles being invoked either as `python -m src.mcp_server.server` or as a
  bare script path (`python .../server.py`) by manually inserting the project
  root onto `sys.path` — needed because a bare script invocation doesn't get
  package-relative imports for free.

### `src/agent/graph.py` — Stage 6: the LangGraph agent
- Builds a `create_react_agent` (LangGraph's prebuilt ReAct loop) wired to
  the MCP server via `MultiServerMCPClient`, which spawns the server as a
  **child process** over stdio and exposes its tools as LangChain tools.
- **Reasoning-brain model is deliberately not the fine-tuned Qwen.** Explicit
  design rationale (also in the README): a 1.5B fine-tuned model won't
  reliably emit well-formed tool calls — small models are good at imitating a
  target output *format* (which is what the SFT trained it to do) but bad at
  general structured tool invocation. So a separate, more capable model
  (`llama3.1:8b` via local Ollama, no API cost) does routing/tool-calling, and
  hands retrieved context to the fine-tuned generator only for the final
  answer text. **The agent itself never writes the user-facing answer.**
- Two documented, real debugging stories baked into comments here — good
  interview material (see section 5 below):
  1. `qwen2.5-coder:7b` was tried first (already local, no download needed)
     but Ollama's chat template for it expects tool calls wrapped in
     `<tool_call>...</tool_call>` tags for its parser to extract; the model
     reliably produced correct JSON but *without* the tags, so Ollama's
     parser missed it and the "tool call" leaked into plain-text `content`
     instead of the `tool_calls` field — meaning the agent silently never
     called a real tool, it just wrote text that looked like one.
  2. An earlier system prompt described tool-calling *mechanics* in prose
     ("call exactly one of the tools... after calling a tool, write...").
     That alone was enough to make `llama3.1:8b` abandon native tool-calling
     and narrate a fake `"Tool Response:"` block as plain text — a different
     root cause, same symptom. Fix: strip the prompt down to tool *choice*
     only, and trust `bind_tools` to communicate calling mechanics (that's
     what the tool schema is for).
- `AGENT_MODEL_PROVIDER` env var can swap `ollama` → `anthropic` if you want
  a hosted model instead, with no code change.

### `src/generator/model.py` — Stage 7: the answer generator
- Loads base `Qwen/Qwen2.5-1.5B-Instruct`, and **optionally** layers a QLoRA
  adapter on top via PEFT if `ADAPTER_PATH` is set.
- `_verify_adapter_base_match()`: reads the adapter's own
  `adapter_config.json` and refuses to load it if `base_model_name_or_path`
  doesn't match the base model currently configured. Rationale stated
  directly: a LoRA adapter is a small set of low-rank weight *deltas* that
  only mean anything relative to the exact base it was trained against — a
  mismatch can fail on a shape error, or worse, load fine and silently
  produce fluent-sounding garbage. This is a "fail loud and early instead of
  silently corrupting output" design choice worth citing generally.
- Two separate system prompts: `RAG_SYSTEM_PROMPT` (answer only from provided
  context, cite the source doc, include code when relevant, say so if the
  context doesn't answer it) vs `NO_RAG_SYSTEM_PROMPT` (answer from
  parametric knowledge, used only in arm A). Same `generate()` function
  serves all three arms — behavior branches on whether `context` is passed.
- Device resolution: CUDA → MPS (Apple Silicon) → CPU fallback.
- `@lru_cache` on `_load()` means the (large) model+tokenizer load happens
  once per process even across repeated `generate()` calls — important given
  eval calls it once per question.

### `src/train/build_sft_data.py` — Stage 9a: build the fine-tuning dataset
- The adapter's job is **not** to teach the model new Polars facts — it's to
  teach the 1.5B model the specific *answer format* the RAG system prompt
  demands (grounded in context, cites source, includes code when relevant).
  Explicit rationale: small models imitate a target format far more reliably
  than they follow it purely from instructions.
- **Knowledge distillation setup**: `llama3.1:8b` (the same Ollama model
  already used as the agent's brain) is the "teacher." For each training
  chunk it (1) generates a synthetic question that chunk would answer, then
  (2) answers that question using the exact `RAG_SYSTEM_PROMPT` the real
  generator uses at inference time, conditioned on that chunk as context.
  That produces `(context, question, answer)` triples that exactly mirror
  the real inference-time input/output shape.
- **Train/eval leakage guard, enforced with an assertion, not just a
  comment**: every chunk used for training comes from a source document that
  does **not** appear in `eval/eval_set.jsonl`. This is document-level
  separation, stricter than question-level — worth explaining precisely if
  asked, since "we held out some questions" and "we held out entire source
  documents" are different (and the latter is the stronger claim).
- Resumable: `_already_done_ids()` lets you re-run the script after an
  interruption (e.g. an Ollama hiccup) without regenerating already-done
  chunks — it appends to the existing JSONL and skips IDs already present.
- Result: **150 training examples** in `data/sft_train.jsonl`.

### `src/train/train_qlora.py` — Stage 9b: train the adapter
- Loads the base model in **4-bit NF4 quantization** via `bitsandbytes`
  (`BitsAndBytesConfig`), then `prepare_model_for_kbit_training` (this is the
  "Q" in QLoRA — quantized base, full-precision LoRA deltas trained on top).
- LoRA config actually used (confirmed from `adapter_config.json` on disk):
  `r=16`, `alpha=32`, `dropout=0.05`, targeting **all** attention and MLP
  projection matrices (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj`) — not just attention, which is a common LoRA-scope choice worth
  being able to justify (broader coverage, more trainable params, still tiny
  relative to full fine-tuning).
- **Prompt masking**: builds each example as
  `chat_template(system, user) + answer_tokens`, then sets `labels = [-100]*len(prompt) + answer_ids`.
  `-100` is the PyTorch/HF convention for "ignore this token in the loss."
  This means the model is trained to predict the *answer* only — it is never
  rewarded for memorizing the context or the question, which would otherwise
  waste capacity and risk it learning to regurgitate context verbatim instead
  of reasoning over it.
- Training args: batch size 1 with gradient accumulation 8 (effective batch
  size 8) — a hardware-constrained choice for a small/consumer GPU — 3
  epochs, lr 2e-4, `paged_adamw_8bit` optimizer (memory-efficient optimizer
  variant designed to pair with 4-bit quantized training), fp16.
- After training, asserts the saved adapter's `base_model_name_or_path`
  matches `BASE_MODEL_NAME` — closing the loop with the safety check in
  `generator/model.py`.
- **Confirmed on disk**: this adapter has actually been trained — `models/qwen2.5-1.5b-polars-qlora/adapter_model.safetensors` exists (~74MB). It is currently **not committed to git**.

### `src/evaluate/run_eval.py` — Stage 8: the 3-arm evaluation harness
This is the centerpiece of "the point of the project is the measurement."
- **Arm A — no retrieval**: base Qwen answers from parametric knowledge
  alone. Explicitly forces `adapter_path=None` regardless of the
  `ADAPTER_PATH` env var — a nice detail: the harness *hardcodes* which arms
  are and aren't allowed to see the adapter, so a stray env var can't quietly
  contaminate a result you're about to compare against another arm.
- **Arm B — plain top-k RAG**: calls `retriever.search()` directly (no
  agent, no tool-routing decision), joins the top-5 chunks as context, base
  model answers. Also forces no adapter — isolates "does retrieval help" from
  "does fine-tuning help."
- **Arm C — agentic RAG**: the real LangGraph agent runs, decides which
  tool(s) to call, and *this* arm uses `ADAPTER_PATH` if set — isolating
  "does agentic tool-routing help" and "does the fine-tuned adapter help" as
  the last two variables introduced, one arm at a time. This is a controlled
  experiment design: each arm changes exactly one variable versus the
  previous one (no context → shared context → agent-selected context; and
  base model → base model → base+adapter model).
- Same question set and same `k=5` across arms B/C for comparability.
- Records latency (median + p95) and tool-call counts (arm C) alongside the
  four Ragas metrics.
- Output: per-arm raw JSONL (every question, answer, retrieved context,
  latency — for manual inspection) plus a combined `summary.json` and a
  markdown table meant to drop straight into the README.
- **Current real state**: only `A_no_retrieval_raw.jsonl` and
  `B_plain_rag_raw.jsonl` exist in `eval/results/`; arm C hasn't been scored
  yet, and `summary.json`/`summary.md` don't exist. Be ready to say this
  plainly if asked to show final numbers — it's a real, known gap, not
  something to gloss over.

### `src/evaluate/metrics.py` — Ragas scoring configuration
- Four Ragas metrics: `faithfulness`, `answer_relevancy`, `context_precision`,
  `context_recall`. Know what each measures if asked:
  - **Faithfulness**: does the answer's content stay factually consistent with
    (derivable from) the retrieved context — a proxy for hallucination.
  - **Answer relevancy**: does the answer actually address the question asked.
  - **Context precision**: of the chunks retrieved, how many were actually
    relevant/useful (signal-to-noise of retrieval).
  - **Context recall**: of what the ground-truth answer needed, how much did
    the retrieved context actually contain (did retrieval miss something).
- **Judge model honesty**: no hosted judge API key (Anthropic/OpenAI) was
  available in this environment, so the Ragas judge defaults to the *same*
  local `llama3.1:8b` used as the agent's brain. Explicitly documented as a
  real limitation: an 8B local judge is weaker/noisier than GPT-4o or Claude
  as a judge, so these scores are useful for comparing arms A/B/C *against
  each other* on this one harness, not as absolute numbers comparable to
  published Ragas benchmarks. `RAGAS_JUDGE_PROVIDER=anthropic` swaps in a
  stronger hosted judge if a key is available.
- A concrete dependency-hell fix worth knowing how to explain: `ragas` 0.2.x
  unconditionally imports `langchain_community.chat_models.vertexai` at
  module load time even when Vertex AI is never used; that submodule doesn't
  exist in the `langchain-community` version this project's LangGraph/MCP
  stack needs. Rather than downgrade langchain-community (which would break
  the agent) the file **stubs the unused import** by registering a fake
  module in `sys.modules` before importing ragas. This is a legitimate,
  narrow technique for resolving genuinely incompatible transitive
  dependencies — not something to reach for casually, but a real tool.
- `RunConfig(timeout=600, max_workers=1)`: Ragas's defaults (180s timeout,
  concurrent judge calls) cause spurious timeouts against one local Ollama
  instance under load — serializing judge calls (`max_workers=1`) and raising
  the timeout fixes silent NaN-metric drops.

---

## 4. Key numbers to have ready

| Fact | Value |
|---|---|
| Source docs | Polars user guide, sparse-cloned from `pola-rs/polars` |
| Chunks indexed | 294 |
| Embedding model | `BAAI/bge-small-en-v1.5`, 384-dim, cosine distance |
| Vector store | Chroma, persistent client |
| Agent reasoning model | `llama3.1:8b` via local Ollama |
| Generator base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| QLoRA config | r=16, alpha=32, dropout=0.05, all 7 attn+MLP proj matrices |
| SFT training examples | 150, distilled from `llama3.1:8b` |
| Eval set size | 45 questions (full), 15-question subset actually scored (A/B only so far) |
| Ragas metrics | faithfulness, answer_relevancy, context_precision, context_recall |
| Evaluation arms | A: no retrieval — B: plain top-5 RAG — C: agent + MCP tools (+ optional adapter) |

---

## 5. Likely interview questions and how to answer them

**"Walk me through the architecture."**
Use the diagram in section 2. Emphasize: docs → chunks → vector index →
two MCP tools → LangGraph agent picks a tool → generator writes the answer
from retrieved context. Then pivot immediately to "but the actual point was
measuring whether each layer earns its complexity" — that's your strongest
differentiator versus a generic "I built a RAG chatbot" answer.

**"Why did you use an agent instead of just doing RAG?"**
Be honest to the design: the hypothesis is that letting a model choose
*which* retrieval tool to call (broad semantic search vs. targeted signature
lookup) produces better answers than always doing the same fixed retrieval —
and arm B vs. arm C exists specifically to test that hypothesis rather than
assume it.

**"Why is the fine-tuned model not the one making tool-call decisions?"**
Small models are unreliable at structured tool invocation but good at
imitating a target output format — so a 1.5B fine-tuned model is well suited
to *writing the final answer in a specific format* (which SFT taught it) but
poorly suited to *deciding which of two tools to call and with what
arguments*. Splitting those two responsibilities across two models is the
core design decision of Stage 6/7.

**"Tell me about a bug you had to debug."**
Two strong ones, both non-obvious because neither crashed:
1. BGE query/document instruction-prefix asymmetry — retrieval silently gets
   worse, no exception (retriever.py).
2. Tool-calling silently not happening because either (a) the model's Ollama
   chat template needs a tag the model didn't reliably emit, or (b) the
   system prompt over-specified calling mechanics and talked the model out
   of using native tool-calling at all (agent/graph.py). Mention that you
   isolated cause (b) using a minimal `create_react_agent` + one dummy tool,
   with/without the prompt — i.e., you reduced the reproduction before
   touching the real MCP tools.

**"How do you know your fine-tuning actually helped?"**
Honestly: right now you don't have that number yet — arm C hasn't been fully
scored against arm B. Say what the harness is designed to show (isolating
adapter contribution from agent-routing contribution, since both are
introduced across A→B→C one at a time) and that finishing that run is the
immediate next step. Don't overclaim results you don't have.

**"What would you do differently / what's unfinished?"** (from the README,
still accurate)
- Use a hosted judge (Claude Haiku / GPT-4o-mini) instead of the local 8B
  Ollama judge for trustworthier absolute Ragas scores.
- Finish scoring arm C — right now the QLoRA adapter is trained but its
  contribution to answer quality hasn't been measured yet.
- Run the full 45-question eval set instead of the 15-question subset,
  ideally on better/rented GPU hardware.
- Add "trap" questions that test whether the fine-tuned adapter's format
  training overrides good retrieved context — a specific, named failure mode
  worth measuring, not just generic retrieval quality.

**"Why Chroma if you only have 294 chunks — a numpy dot product would work?"**
This is directly addressed in the README's "honest scoping note": at this
scale a vector database isn't a performance necessity. Chroma is used because
the MCP server runs in a separate OS process from anything that builds the
index, so you need some form of process-independent persistence — and
because it's the tool a reviewer expects to see. Say this plainly if asked;
it demonstrates you understand tool choice should be justified, not assumed.

---

## 6. Resume bullet options

Pick 2–4 depending on space; don't use all of them. **Do not add a number to
any of these** (e.g. "improved faithfulness by X%") until arm C is actually
scored and `eval/results/summary.json` exists with real values — a bullet
that names a metric invites "show me," and right now there's nothing to show
for arm C. The bullets below deliberately describe what the harness was
*designed to measure*, not a result it produced. That distinction is true and
defensible; a fabricated or guessed number is not.

- Built an agentic RAG system over technical documentation (Polars) using
  LangGraph, MCP (Model Context Protocol), and Chroma; designed a controlled
  3-arm evaluation (no-retrieval / plain RAG / agentic RAG) to isolate and
  quantify each architectural layer's contribution using Ragas metrics
  (faithfulness, relevancy, context precision/recall).
- Fine-tuned Qwen2.5-1.5B via QLoRA (4-bit NF4 quantization, PEFT/LoRA,
  bitsandbytes) on a 150-example instruction dataset distilled from a larger
  teacher model, with document-level train/eval leakage prevention enforced
  by assertion.
- Built an MCP server exposing two purpose-scoped retrieval tools (semantic
  search + targeted API lookup) consumed by a LangGraph ReAct agent running
  as a separate OS process, communicating over stdio.
- Diagnosed and fixed two distinct silent tool-calling failures in a local
  LLM agent (Ollama + LangGraph) — a chat-template/parser mismatch and a
  prompt that inadvertently suppressed native tool-calling — by isolating
  root cause with a minimal reproduction before touching production code.
- Built a document-structure-aware chunking pipeline (header-boundary
  splitting with breadcrumb metadata) and resolved custom Markdown
  transclusion directives to recover code examples missing from raw
  documentation text before indexing.

**Bullet 2 (the fine-tuning one) makes a specific claim: an adapter exists.**
Before that bullet goes on a resume with a GitHub link attached, the adapter
it describes has to actually be reachable from that link — see section 7.

## 7. Known gaps — fix before this goes on a resume with a repo link

These aren't just "be upfront if asked" items anymore — for a resume (as
opposed to a private portfolio conversation), a couple of these are blocking,
because a reviewer will click the link and check.

- **Arm C is not scored. Do not write a results number anywhere until it is.**
  Only arms A and B have raw results in `eval/results/`; `summary.json` /
  `summary.md` don't exist. Finish the arm C run (or explicitly scope any
  results section to "arms A/B only, arm C in progress") before claiming any
  metric on faithfulness, relevancy, or precision/recall in a resume bullet,
  README, or interview answer with a number attached.
- **The trained adapter (~74MB) is not committed to git.** Bullet 2 above
  describes a fine-tuned model as a real, done thing — if that bullet is on a
  resume with a repo link, someone will open the repo expecting to find it.
  Right now they'd find the training *code* but no trained artifact, which
  reads as "claimed but not delivered." Fix one of two ways before publishing
  the resume:
  1. Commit the adapter (Git LFS, since it's ~74MB — plain git will work but
     LFS is the correct tool for a binary this size), or
  2. Push it to the Hugging Face Hub and link it from the README, and add a
     one-line README note stating plainly what's tracked in git vs. hosted
     externally.
  Either is fine; leaving it silently absent while the resume implies it
  exists is not.
- LangSmith tracing is wired up but was never actually exercised (no API key
  in this environment) — no trace screenshot exists. Low priority, but don't
  claim "full observability" without a screenshot or a real trace link.

## 8. Where this sits relative to other projects on the resume

This is a strong depth-of-engineering piece (multi-stage pipeline, real
debugging, a genuine controlled-experiment design) — but it's an LLM/RAG
project, not embedded/edge work. If the KWS (keyword-spotting) project is
also on this resume and the target is Renesas, **KWS goes first** — it's the
directly relevant headline for an embedded/edge role. This project is the
supporting proof of general engineering depth and rigor, not the lead.
