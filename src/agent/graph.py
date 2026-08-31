"""Stage 6: LangGraph ReAct agent consuming the polars-docs MCP server.

Design decision (spec section 7.4 / Stage 6): a 1.5B fine-tuned Qwen model
will not reliably emit well-formed tool calls — small models are bad at
structured tool invocation but good at imitating a target output format. So
this agent uses a separate, competent tool-calling model as its reasoning
brain (default: a local Ollama model, no API cost) and hands the *retrieved
context* off to the fine-tuned Qwen generator (src/generator/model.py) for
the final answer. The agent itself never generates the user-facing answer.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = PROJECT_ROOT / "src" / "mcp_server" / "server.py"

SYSTEM_PROMPT = (
    "You are a Polars documentation assistant. Use search_docs for conceptual or "
    "how-to questions, and get_api_signature when the user names a specific function, "
    "method, or class. Answer using only the retrieved passages; say so explicitly if "
    "they don't contain the answer."
)
# NOTE: keep this short and purely about tool CHOICE, not tool-call mechanics.
# An earlier version described the call/response format in prose (e.g. "call
# exactly one of the tools", "Tool Response:") and that was enough to make
# llama3.1:8b abandon native tool-calling and instead narrate a fake JSON
# "Tool Response" as plain text, which never reaches the real MCP tool at all.
# The model already gets the tool schemas via bind_tools; don't re-describe
# the protocol in the prompt.


def _build_chat_model():
    """Select the agent's reasoning-brain model from env vars.

    AGENT_MODEL_PROVIDER=ollama (default) uses a local model via langchain-ollama,
    no API key required. Set to 'anthropic' to swap in a hosted model instead.
    """
    provider = os.environ.get("AGENT_MODEL_PROVIDER", "ollama")
    # NOTE: qwen2.5-coder:7b was tried first (it's already local) but its Ollama
    # template asks the model to wrap tool calls in <tool_call></tool_call> tags
    # and it unreliably omits them, so Ollama can't parse the call and the JSON
    # leaks into `content` as plain text instead of `tool_calls`. llama3.1:8b
    # follows Ollama's tool-calling protocol reliably in testing; swap back via
    # AGENT_MODEL_NAME if a future qwen2.5-coder build fixes this.
    model_name = os.environ.get("AGENT_MODEL_NAME", "llama3.1:8b")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unknown AGENT_MODEL_PROVIDER: {provider}")


async def build_agent():
    """Construct the LangGraph ReAct agent wired to the polars-docs MCP server."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent

    client = MultiServerMCPClient(
        {
            "polars": {
                "command": sys.executable,
                "args": ["-m", "src.mcp_server.server"],
                "cwd": str(PROJECT_ROOT),
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    model = _build_chat_model()
    agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
    return agent


async def ask(question: str) -> dict:
    """Run the agent on one question, returning the final messages and tool calls made."""
    agent = await build_agent()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    tool_calls = []
    for msg in result["messages"]:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            tool_calls.extend(c["name"] for c in calls)
    return {"messages": result["messages"], "tool_calls": tool_calls}


if __name__ == "__main__":
    import asyncio

    async def _main():
        question = sys.argv[1] if len(sys.argv) > 1 else "how do I do a groupby in polars"
        result = await ask(question)
        print("tool_calls:", result["tool_calls"])
        print("final answer:\n", result["messages"][-1].content)

    asyncio.run(_main())
