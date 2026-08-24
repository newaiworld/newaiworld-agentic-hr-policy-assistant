"""Agent orchestration primitives for the HR Policy Assistant.

S6 begins by owning the MCP client boundary here. The agent discovers
production tools dynamically over MCP stdio and never imports RAG or
mock-data business implementations directly.
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import anyio
from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)
from mcp.types import CallToolResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = PROJECT_ROOT / "mcp" / "server.py"

MCP_TOOL_TIMEOUT_SECONDS = 10


class AgentMCPError(RuntimeError):
    """Raised when the agent cannot safely use the MCP tool layer."""


@dataclass(
    frozen=True,
    slots=True,
)
class DiscoveredTool:
    """Stable agent-side representation of one discovered MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool | None

    def to_llm_schema(
        self,
    ) -> dict[str, Any]:
        """Return the OpenAI-compatible function-tool representation."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class AgentMCPClient:
    """Own one stdio MCP subprocess/session for agent tool execution."""

    def __init__(
        self,
        *,
        server_path: Path = MCP_SERVER_PATH,
    ) -> None:
        """Create a disconnected client in the safe degraded state."""

        self._server_path = Path(
            server_path
        )

        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

        self._tools: tuple[
            DiscoveredTool,
            ...,
        ] = ()

        self._status: Literal[
            "connected",
            "degraded",
        ] = "degraded"

        self._last_error: str | None = None

    @property
    def status(
        self,
    ) -> Literal[
        "connected",
        "degraded",
    ]:
        """Return the current MCP connectivity state."""

        return self._status

    @property
    def tools(
        self,
    ) -> tuple[
        DiscoveredTool,
        ...,
    ]:
        """Return the immutable discovered-tool snapshot."""

        return self._tools

    @property
    def llm_tools(
        self,
    ) -> list[
        dict[str, Any]
    ]:
        """Return discovered tools in OpenAI-compatible function format."""

        return [
            tool.to_llm_schema()
            for tool in self._tools
        ]

    @property
    def last_error(
        self,
    ) -> str | None:
        """Return the most recent startup/tool-layer failure summary."""

        return self._last_error

    async def start(
        self,
    ) -> tuple[
        DiscoveredTool,
        ...,
    ]:
        """Start the MCP subprocess, initialize it, and discover tools.

        Discovery failure is a degraded-mode condition rather than an
        application crash. In degraded mode the discovered-tool set is empty,
        preventing the agent from pretending unavailable tools exist.
        """

        await self.close()

        self._status = "degraded"
        self._tools = ()
        self._last_error = None

        if not self._server_path.is_file():
            self._last_error = (
                "MCP server entry point was not found."
            )
            return self._tools

        stack = AsyncExitStack()

        try:
            server = StdioServerParameters(
                command=sys.executable,
                args=[
                    str(
                        self._server_path
                    ),
                ],
            )

            read_stream, write_stream = (
                await stack.enter_async_context(
                    stdio_client(
                        server
                    )
                )
            )

            session = (
                await stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(
                            seconds=MCP_TOOL_TIMEOUT_SECONDS
                        ),
                    )
                )
            )

            with anyio.fail_after(
                MCP_TOOL_TIMEOUT_SECONDS
            ):
                await session.initialize()

                response = (
                    await session.list_tools()
                )

            tools = tuple(
                self._convert_tool(
                    tool
                )
                for tool in response.tools
            )

            if not tools:
                raise AgentMCPError(
                    "MCP discovery returned no tools."
                )

        except BaseException as exc:
            await stack.aclose()

            if isinstance(
                exc,
                (
                    KeyboardInterrupt,
                    SystemExit,
                ),
            ):
                raise

            self._last_error = (
                self._clean_error(
                    exc
                )
            )

            self._status = "degraded"
            self._tools = ()
            self._session = None
            self._stack = None

            return self._tools

        self._stack = stack
        self._session = session
        self._tools = tools
        self._status = "connected"
        self._last_error = None

        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[
            str,
            Any,
        ],
    ) -> CallToolResult:
        """Call one previously discovered tool through the MCP session."""

        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "Tool name must be a string."
            )

        name = name.strip()

        if not name:
            raise ValueError(
                "Tool name must be a non-empty string."
            )

        if not isinstance(
            arguments,
            dict,
        ):
            raise TypeError(
                "Tool arguments must be a dictionary."
            )

        if (
            self._status
            != "connected"
            or self._session
            is None
        ):
            raise AgentMCPError(
                "MCP tools are unavailable because "
                "the client is degraded."
            )

        discovered_names = {
            tool.name
            for tool in self._tools
        }

        if (
            name
            not in discovered_names
        ):
            raise AgentMCPError(
                f"MCP tool {name!r} was not discovered."
            )

        try:
            with anyio.fail_after(
                MCP_TOOL_TIMEOUT_SECONDS
            ):
                return (
                    await self._session.call_tool(
                        name,
                        arguments,
                    )
                )

        except TimeoutError as exc:
            self._last_error = (
                "MCP tool call timed out."
            )

            raise AgentMCPError(
                self._last_error
            ) from exc

        except Exception as exc:
            self._last_error = (
                self._clean_error(
                    exc
                )
            )

            raise AgentMCPError(
                "MCP tool call failed: "
                f"{self._last_error}"
            ) from exc

    async def close(
        self,
    ) -> None:
        """Close the current MCP session and owned subprocess safely."""

        stack = self._stack

        self._stack = None
        self._session = None
        self._tools = ()
        self._status = "degraded"

        if stack is not None:
            await stack.aclose()

    @staticmethod
    def _convert_tool(
        tool: Any,
    ) -> DiscoveredTool:
        """Validate and convert one MCP SDK Tool result."""

        name = getattr(
            tool,
            "name",
            None,
        )

        if (
            not isinstance(
                name,
                str,
            )
            or not name.strip()
        ):
            raise AgentMCPError(
                "Discovered MCP tool has an invalid name."
            )

        description = getattr(
            tool,
            "description",
            None,
        )

        if description is None:
            description = ""

        if not isinstance(
            description,
            str,
        ):
            raise AgentMCPError(
                f"MCP tool {name!r} has an invalid description."
            )

        input_schema = getattr(
            tool,
            "inputSchema",
            None,
        )

        if not isinstance(
            input_schema,
            dict,
        ):
            raise AgentMCPError(
                f"MCP tool {name!r} has an invalid input schema."
            )

        annotations = getattr(
            tool,
            "annotations",
            None,
        )

        read_only: bool | None = None

        if annotations is not None:
            value = getattr(
                annotations,
                "readOnlyHint",
                None,
            )

            if value is not None and not isinstance(
                value,
                bool,
            ):
                raise AgentMCPError(
                    f"MCP tool {name!r} has an invalid "
                    "readOnlyHint annotation."
                )

            read_only = value

        return DiscoveredTool(
            name=name,
            description=description,
            input_schema=input_schema,
            read_only=read_only,
        )

    @staticmethod
    def _clean_error(
        exc: BaseException,
    ) -> str:
        """Return a compact public-safe failure summary."""

        text = str(
            exc
        ).strip()

        if text:
            return text

        return type(
            exc
        ).__name__
