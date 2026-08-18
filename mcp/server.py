"""MCP server entry point for the Agentic HR Policy Assistant."""

from __future__ import annotations

# Resolve the official MCP SDK before adding the repository root.
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_POLICY_PATH = (
    PROJECT_ROOT
    / "mcp"
    / "tools_policy.py"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


def _load_tools_policy_module() -> ModuleType:
    """Load project policy tools without shadowing the MCP SDK package."""

    if not TOOLS_POLICY_PATH.is_file():
        raise RuntimeError(
            "Project MCP policy tools file was not found: "
            f"{TOOLS_POLICY_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "project_mcp_tools_policy",
        TOOLS_POLICY_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Project MCP policy tools module could not be loaded."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def _load_search_policy_documents() -> Callable[..., object]:
    """Return the existing policy-search implementation for registration."""

    module = _load_tools_policy_module()

    function = getattr(
        module,
        "search_policy_documents",
        None,
    )

    if function is None:
        raise RuntimeError(
            "Project MCP policy tools module does not expose "
            "search_policy_documents."
        )

    if not callable(function):
        raise RuntimeError(
            "search_policy_documents must be callable."
        )

    return function


mcp = FastMCP(
    "Agentic HR Policy Assistant"
)


search_policy_documents = _load_search_policy_documents()


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    search_policy_documents
)


def main() -> None:
    """Run the MCP server over the frozen V1 stdio transport."""

    mcp.run(
        transport="stdio",
    )


if __name__ == "__main__":
    main()
