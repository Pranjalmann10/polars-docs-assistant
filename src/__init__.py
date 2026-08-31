from pathlib import Path

from dotenv import load_dotenv

# Load .env once, from wherever `import src...` first happens (script, notebook,
# or the MCP server subprocess) so LANGCHAIN_TRACING_V2 / API keys / model
# config are picked up without every entry point repeating this call.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
