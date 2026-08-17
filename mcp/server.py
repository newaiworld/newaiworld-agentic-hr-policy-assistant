"""MCP server entry point for the Agentic HR Policy Assistant."""

from mcp.server.fastmcp import FastMCP

# Resolve the official MCP SDK before adding the repository root.
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


mcp = FastMCP(
    "Agentic HR Policy Assistant"
)


def main() -> None:
    """Run the MCP server over the frozen V1 stdio transport."""

    mcp.run(
        transport="stdio",
    )


if __name__ == "__main__":
    main()
