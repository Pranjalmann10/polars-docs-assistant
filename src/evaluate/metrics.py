"""Stage 8: Ragas metric configuration.

No paid judge-model API key is assumed to be available for this evaluation
run, so the Ragas judge LLM defaults to the same local Ollama model used as
the agent's reasoning brain (AGENT_MODEL_NAME). This is an explicit, honest
tradeoff worth stating in the README: a local 8B judge is weaker and noisier
than GPT-4o/Claude as a judge, so treat Ragas scores here as directionally
useful for comparing arms A/B/C against each other, not as absolute quality
numbers comparable to published Ragas benchmarks.

Set RAGAS_JUDGE_PROVIDER=anthropic (with ANTHROPIC_API_KEY set) to use a
stronger hosted judge instead, if available.
"""

import os
import sys
import types

# ragas 0.2.x unconditionally imports langchain_community.chat_models.vertexai
# at module load time even when Vertex AI is never used. That submodule was
# removed from langchain-community in the version this project's agent stack
# needs (langchain-core 1.x, for langgraph/langchain-mcp-adapters). Rather than
# downgrade langchain-community and break the agent, stub the unused import.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _UnusedChatVertexAI:  # pragma: no cover - never instantiated
        pass

    _stub.ChatVertexAI = _UnusedChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from ragas import evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# A local 8B judge on CPU/consumer-GPU is much slower per call than a hosted
# API judge, and Ragas's default timeout (180s) plus default concurrency
# (multiple parallel judge calls hammering one local Ollama instance) causes
# spurious TimeoutErrors that silently drop metrics to NaN. Give it more time
# per call and serialize judge calls instead of fanning them out.
JUDGE_RUN_CONFIG = RunConfig(timeout=600, max_workers=1)


def _build_judge_llm():
    provider = os.environ.get("RAGAS_JUDGE_PROVIDER", "ollama")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model_name = os.environ.get("AGENT_MODEL_NAME", "llama3.1:8b")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return LangchainLLMWrapper(ChatOllama(model=model_name, base_url=base_url, temperature=0))
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = os.environ.get("RAGAS_JUDGE_MODEL", "claude-haiku-4-5")
        return LangchainLLMWrapper(ChatAnthropic(model=model_name, temperature=0))
    else:
        raise ValueError(f"Unknown RAGAS_JUDGE_PROVIDER: {provider}")


def _build_judge_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=model_name))


def score_dataset(dataset):
    """dataset: a ragas-compatible Dataset with columns
    question, answer, contexts, ground_truth. Returns a ragas Result object."""
    judge_llm = _build_judge_llm()
    judge_embeddings = _build_judge_embeddings()
    return ragas_evaluate(
        dataset,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=JUDGE_RUN_CONFIG,
    )
