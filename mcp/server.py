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

TOOLS_DATA_PATH = (
    PROJECT_ROOT
    / "mcp"
    / "tools_data.py"
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


def _load_policy_tool(
    name: str,
) -> Callable[..., object]:
    """Return one existing project policy-tool implementation.

    The project ``mcp/`` directory intentionally remains a non-package
    so that it cannot shadow the installed MCP SDK. Tool implementations
    are therefore loaded dynamically from ``tools_policy.py``.

    Args:
        name:
            Public policy-tool function name to load.

    Returns:
        The existing callable implementation.

    Raises:
        TypeError:
            If ``name`` is not a string.
        ValueError:
            If ``name`` is empty or whitespace-only.
        RuntimeError:
            If the project module does not expose the requested callable.
    """

    if not isinstance(
        name,
        str,
    ):
        raise TypeError(
            "name must be a string."
        )

    name = name.strip()

    if not name:
        raise ValueError(
            "name must be a non-empty string."
        )

    module = _load_tools_policy_module()

    function = getattr(
        module,
        name,
        None,
    )

    if function is None:
        raise RuntimeError(
            "Project MCP policy tools module does not expose "
            f"{name}."
        )

    if not callable(
        function
    ):
        raise RuntimeError(
            f"{name} must be callable."
        )

    return function


def _load_tools_data_module() -> ModuleType:
    """Load project data tools without shadowing the MCP SDK package."""

    if not TOOLS_DATA_PATH.is_file():
        raise RuntimeError(
            "Project MCP data tools file was not found: "
            f"{TOOLS_DATA_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "project_mcp_tools_data",
        TOOLS_DATA_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Project MCP data tools module could not be loaded."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def _load_data_tool(
    name: str,
) -> Callable[..., object]:
    """Return one existing project data-tool implementation."""

    if not isinstance(
        name,
        str,
    ):
        raise TypeError(
            "name must be a string."
        )

    name = name.strip()

    if not name:
        raise ValueError(
            "name must be a non-empty string."
        )

    module = _load_tools_data_module()

    function = getattr(
        module,
        name,
        None,
    )

    if function is None:
        raise RuntimeError(
            "Project MCP data tools module does not expose "
            f"{name}."
        )

    if not callable(
        function
    ):
        raise RuntimeError(
            f"{name} must be callable."
        )

    return function


mcp = FastMCP(
    "Agentic HR Policy Assistant"
)


search_policy_documents = _load_policy_tool(
    "search_policy_documents"
)

get_policy_section = _load_policy_tool(
    "get_policy_section"
)

lookup_employee_profile = _load_data_tool(
    "lookup_employee_profile"
)


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    search_policy_documents
)


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    get_policy_section
)

mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    lookup_employee_profile
)


def main() -> None:
    """Run the MCP server over the frozen V1 stdio transport."""

    mcp.run(
        transport="stdio",
    )


if __name__ == "__main__":
    main()
