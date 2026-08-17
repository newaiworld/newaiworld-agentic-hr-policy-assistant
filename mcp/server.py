"""MCP server entry point for the Agentic HR Policy Assistant."""

from mcp.server.fastmcp import FastMCP


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
