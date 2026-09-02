"""Stage 8: three-arm evaluation over the held-out eval set.

Arm A: no retrieval,        base Qwen2.5-1.5B-Instruct
Arm B: plain top-k RAG,     base Qwen2.5-1.5B-Instruct
Arm C: agent (MCP tools),   base Qwen2.5-1.5B-Instruct (+ QLoRA adapter if
                            ADAPTER_PATH is set — see src/generator/model.py)

Same question set, same retrieval settings (top-k=5) across arms B and C.
Records Ragas faithfulness / answer_relevancy / context_precision /
context_recall, plus median and p95 latency per arm, and tool-call counts
for arm C. Results are written to eval/results/ as both raw per-question
JSONL (for inspection) and a summary JSON/Markdown table (for the README).
"""

import asyncio
import json
import statistics
import time
from pathlib import Path

from datasets import Dataset

import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SET_PATH = Path(os.environ.get("EVAL_SET_PATH", PROJECT_ROOT / "eval" / "eval_set.jsonl"))
RESULTS_DIR = Path(os.environ.get("EVAL_RESULTS_DIR", PROJECT_ROOT / "eval" / "results"))


def load_eval_set() -> list[dict]:
    items = []
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


async def run_arm_a(items: list[dict]) -> list[dict]:
    """No retrieval, base model answers from parametric knowledge alone.

    Forces adapter_path=None regardless of the ADAPTER_PATH env var: this arm
    exists to measure the base model with no context at all, and must stay
    identical whether or not arm C's adapter happens to be configured for
    this run.
    """
    from src.generator.model import GeneratorConfig, generate

    base_config = GeneratorConfig(adapter_path=None)
    rows = []
    for item in items:
        t0 = time.perf_counter()
        answer = generate(item["question"], context=None, config=base_config)
        latency = time.perf_counter() - t0
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": answer,
                "contexts": [],
                "latency_s": latency,
                "tool_calls": 0,
            }
        )
    return rows


async def run_arm_b(items: list[dict], k: int = 5) -> list[dict]:
    """Plain top-k RAG: retriever.search() directly, then base model.

    Forces adapter_path=None for the same reason as arm A: arm B isolates
    plain retrieval's contribution from the base model, not the adapter's.
    """
    from src.generator.model import GeneratorConfig, generate
    from src.index.retriever import search

    base_config = GeneratorConfig(adapter_path=None)
    rows = []
    for item in items:
        t0 = time.perf_counter()
        hits = search(item["question"], k=k)
        context = "\n\n---\n\n".join(h["text"] for h in hits)
        answer = generate(item["question"], context=context, config=base_config)
        latency = time.perf_counter() - t0
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": answer,
                "contexts": [h["text"] for h in hits],
                "latency_s": latency,
                "tool_calls": 0,
            }
        )
    return rows


async def run_arm_c(items: list[dict]) -> list[dict]:
    """Agent (MCP tools) decides what to retrieve, generator writes the answer
    from whatever the agent's tool calls returned."""
    from src.agent.graph import build_agent
    from src.generator.model import ADAPTER_PATH, GeneratorConfig, generate

    adapter_config = GeneratorConfig(adapter_path=ADAPTER_PATH)
    agent = await build_agent()
    rows = []
    for item in items:
        t0 = time.perf_counter()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": item["question"]}]}
        )
        tool_call_names = []
        contexts = []
        for msg in result["messages"]:
            calls = getattr(msg, "tool_calls", None)
            if calls:
                tool_call_names.extend(c["name"] for c in calls)
            if type(msg).__name__ == "ToolMessage":
                contexts.append(str(msg.content))

        context = "\n\n---\n\n".join(contexts)
        answer = generate(item["question"], context=context or None, config=adapter_config)
        latency = time.perf_counter() - t0
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": answer,
                "contexts": contexts or [""],
                "latency_s": latency,
                "tool_calls": len(tool_call_names),
                "tool_call_names": tool_call_names,
            }
        )
    return rows


def score_arm(arm_name: str, rows: list[dict]) -> dict:
    from src.evaluate.metrics import score_dataset

    ds = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"] or [""],
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ]
    )
    ragas_result = score_dataset(ds)
    ragas_df = ragas_result.to_pandas()

    latencies = [r["latency_s"] for r in rows]
    summary = {
        "arm": arm_name,
        "n_questions": len(rows),
        "median_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(_percentile(latencies, 0.95), 3),
        "mean_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2),
    }
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in ragas_df.columns:
            values = [v for v in ragas_df[metric].tolist() if v == v]  # drop NaN
            summary[metric] = round(statistics.mean(values), 3) if values else None
    return summary


async def main():
    items = load_eval_set()
    print(f"[eval] loaded {len(items)} questions")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for arm_name, runner in [("A_no_retrieval", run_arm_a), ("B_plain_rag", run_arm_b), ("C_agent_mcp", run_arm_c)]:
        print(f"[eval] running arm {arm_name} ...")
        rows = await runner(items)
        raw_path = RESULTS_DIR / f"{arm_name}_raw.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] wrote {raw_path}")

        print(f"[eval] scoring arm {arm_name} with Ragas ...")
        summary = score_arm(arm_name, rows)
        summaries.append(summary)
        print(f"[eval] {arm_name}: {summary}")

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[eval] wrote {summary_path}")

    md_lines = [
        "| Arm | n | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Median Latency (s) | P95 Latency (s) | Mean Tool Calls |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        md_lines.append(
            f"| {s['arm']} | {s['n_questions']} | {s.get('faithfulness')} | "
            f"{s.get('answer_relevancy')} | {s.get('context_precision')} | "
            f"{s.get('context_recall')} | {s['median_latency_s']} | {s['p95_latency_s']} | "
            f"{s['mean_tool_calls']} |"
        )
    md_path = RESULTS_DIR / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[eval] wrote {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
