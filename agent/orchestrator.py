"""Agent orchestration primitives for the HR Policy Assistant.

S6 begins by owning the MCP client boundary here. The agent discovers
production tools dynamically over MCP stdio and never imports RAG or
mock-data business implementations directly.
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence
from uuid import uuid4

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


MAX_AGENT_ITERATIONS = 6


class AgentLLM(Protocol):
    """Small LLM interface required by the orchestration loop."""

    async def chat(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
    ) -> Any:
        """Return a normalized LLM response."""


@dataclass(
    frozen=True,
    slots=True,
)
class PendingConfirmation:
    """One exact ACTION proposal awaiting explicit user confirmation."""

    confirmation_id: str
    tool: str
    arguments: dict[str, Any]
    preview: str
    _bound_arguments_json: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        """Validate and snapshot the externally visible pending action."""

        if (
            not isinstance(self.confirmation_id, str)
            or not self.confirmation_id.strip()
        ):
            raise ValueError(
                "confirmation_id must be a non-empty string."
            )

        if (
            not isinstance(self.tool, str)
            or not self.tool.strip()
        ):
            raise ValueError(
                "tool must be a non-empty string."
            )

        if not isinstance(
            self.arguments,
            dict,
        ):
            raise TypeError(
                "arguments must be a dictionary."
            )

        if (
            not isinstance(self.preview, str)
            or not self.preview.strip()
        ):
            raise ValueError(
                "preview must be a non-empty string."
            )

        copied_arguments = deepcopy(
            self.arguments
        )

        object.__setattr__(
            self,
            "arguments",
            copied_arguments,
        )

        object.__setattr__(
            self,
            "_bound_arguments_json",
            json.dumps(
                copied_arguments,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    def bound_arguments(
        self,
    ) -> dict[str, Any]:
        """Return the exact business arguments bound to this preview."""

        arguments = json.loads(
            self._bound_arguments_json
        )

        if not isinstance(
            arguments,
            dict,
        ):
            raise RuntimeError(
                "Bound confirmation arguments are invalid."
            )

        return arguments

    def as_dict(
        self,
    ) -> dict[str, Any]:
        """Return the API-compatible pending-confirmation representation."""

        return {
            "confirmation_id": self.confirmation_id,
            "tool": self.tool,
            "arguments": deepcopy(self.arguments),
            "preview": self.preview,
        }


@dataclass(
    frozen=True,
    slots=True,
)
class AgentResult:
    """Externally visible result of one agent turn."""

    answer: str
    citations: tuple[dict[str, str], ...]
    trace: tuple[Any, ...]
    exhausted: bool = False
    pending_confirmation: PendingConfirmation | None = None


async def run_turn(
    *,
    message: str,
    mcp_client: AgentMCPClient,
    llm: AgentLLM,
) -> AgentResult:
    """Run one bounded agent turn using discovered MCP tools.

    This function owns orchestration only. It does not call RAG or mock-data
    implementations directly; every tool execution crosses the MCP client.
    """

    from agent.prompts import SYSTEM_PROMPT
    from agent.trace import TraceItem

    if not isinstance(message, str):
        raise TypeError(
            "message must be a string."
        )

    message = message.strip()

    if not message:
        raise ValueError(
            "message must be a non-empty string."
        )

    if mcp_client.status != "connected":
        return AgentResult(
            answer=(
                "The HR tools are currently unavailable. "
                "Please try again later or contact HR if the matter is urgent."
            ),
            citations=(),
            trace=(
                TraceItem(
                    step=1,
                    tool=None,
                    arguments={},
                    result_summary=(
                        mcp_client.last_error
                        or "MCP client is degraded."
                    ),
                    sources=(),
                    decision="mcp_degraded",
                ),
            ),
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    trace: list[TraceItem] = []
    citations: list[dict[str, str]] = []

    for iteration in range(
        1,
        MAX_AGENT_ITERATIONS + 1,
    ):
        response = await llm.chat(
            messages=messages,
            tools=mcp_client.llm_tools,
        )

        content = getattr(
            response,
            "content",
            None,
        )

        tool_calls = getattr(
            response,
            "tool_calls",
            (),
        )

        if content is not None and not tool_calls:
            trace.append(
                TraceItem(
                    step=iteration,
                    tool=None,
                    arguments={},
                    result_summary=content,
                    sources=tuple(citations),
                    decision="answer",
                )
            )

            return AgentResult(
                answer=content,
                citations=tuple(citations),
                trace=tuple(trace),
            )

        if not tool_calls:
            trace.append(
                TraceItem(
                    step=iteration,
                    tool=None,
                    arguments={},
                    result_summary=(
                        "LLM returned neither a usable answer "
                        "nor a tool call."
                    ),
                    sources=tuple(citations),
                    decision="invalid_llm_response",
                )
            )

            return AgentResult(
                answer=(
                    "I couldn't complete that request because "
                    "the model returned an unusable response."
                ),
                citations=tuple(citations),
                trace=tuple(trace),
            )

        messages.append(
            _assistant_tool_call_message(
                content=content,
                tool_calls=tool_calls,
            )
        )

        for tool_call in tool_calls:
            tool_name = getattr(
                tool_call,
                "name",
                None,
            )
            arguments = getattr(
                tool_call,
                "arguments",
                None,
            )
            call_id = getattr(
                tool_call,
                "call_id",
                None,
            )

            if (
                not isinstance(tool_name, str)
                or not isinstance(arguments, dict)
                or not isinstance(call_id, str)
            ):
                raise AgentMCPError(
                    "LLM returned an invalid normalized tool call."
                )

            if _requires_confirmation(
                mcp_client,
                tool_name,
            ):
                pending = _create_pending_confirmation(
                    tool=tool_name,
                    arguments=arguments,
                )

                trace.append(
                    TraceItem(
                        step=iteration,
                        tool=tool_name,
                        arguments=deepcopy(arguments),
                        result_summary=pending.preview,
                        sources=tuple(citations),
                        decision="confirmation_required",
                    )
                )

                return AgentResult(
                    answer=(
                        "This action requires your explicit confirmation "
                        "before it can be executed."
                    ),
                    citations=tuple(citations),
                    trace=tuple(trace),
                    pending_confirmation=pending,
                )

            try:
                tool_result = await mcp_client.call_tool(
                    tool_name,
                    arguments,
                )

            except AgentMCPError as exc:
                trace.append(
                    TraceItem(
                        step=iteration,
                        tool=tool_name,
                        arguments=arguments,
                        result_summary=str(exc),
                        sources=tuple(citations),
                        decision="tool_error",
                    )
                )

                return AgentResult(
                    answer=(
                        "I couldn't complete the requested HR tool operation. "
                        "Please try again or contact HR."
                    ),
                    citations=tuple(citations),
                    trace=tuple(trace),
                )

            structured = (
                tool_result.structuredContent
            )

            new_sources = _extract_citations(
                structured
            )

            _append_unique_citations(
                citations,
                new_sources,
            )

            summary = _summarize_tool_result(
                structured
            )

            trace.append(
                TraceItem(
                    step=iteration,
                    tool=tool_name,
                    arguments=arguments,
                    result_summary=summary,
                    sources=tuple(new_sources),
                    decision="tool_result",
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": summary,
                }
            )

    trace.append(
        TraceItem(
            step=MAX_AGENT_ITERATIONS,
            tool=None,
            arguments={},
            result_summary=(
                "The agent reached the maximum iteration limit."
            ),
            sources=tuple(citations),
            decision="max_iterations",
        )
    )

    return AgentResult(
        answer=_build_exhaustion_answer(
            citations
        ),
        citations=tuple(citations),
        trace=tuple(trace),
        exhausted=True,
    )


def _assistant_tool_call_message(
    *,
    content: str | None,
    tool_calls: Sequence[Any],
) -> dict[str, Any]:
    """Build the assistant tool-call message sent back to the LLM."""

    serialized_calls: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        serialized_calls.append(
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": __import__(
                        "json"
                    ).dumps(
                        tool_call.arguments,
                        sort_keys=True,
                    ),
                },
            }
        )

    return {
        "role": "assistant",
        "content": content,
        "tool_calls": serialized_calls,
    }


def _extract_citations(
    structured: Any,
) -> list[dict[str, str]]:
    """Extract citation-shaped policy evidence from an MCP result.

    FastMCP may expose list-returning tools through structuredContent as
    {"result": [...]}. Direct lists and direct citation dictionaries are
    also accepted so orchestration remains independent of transport
    wrapping details.
    """

    citations: list[dict[str, str]] = []

    value = structured

    if (
        isinstance(value, dict)
        and set(value) == {"result"}
    ):
        value = value["result"]

    candidates: list[Any]

    if isinstance(value, list):
        candidates = value
    elif isinstance(value, dict):
        candidates = [value]
    else:
        return citations

    for candidate in candidates:
        if not isinstance(
            candidate,
            dict,
        ):
            continue

        doc_id = candidate.get(
            "doc_id"
        )
        title = candidate.get(
            "title"
        )

        section = (
            candidate.get("section")
            or candidate.get("section_path")
        )

        snippet = (
            candidate.get("snippet")
            or candidate.get("text")
        )

        if (
            isinstance(doc_id, str)
            and doc_id.strip()
            and isinstance(title, str)
            and title.strip()
        ):
            citations.append(
                {
                    "doc_id": doc_id.strip(),
                    "title": title.strip(),
                    "section": _citation_text(
                        section
                    ),
                    "snippet": _citation_text(
                        snippet
                    ),
                }
            )

    return citations


def _citation_text(
    value: Any,
) -> str:
    """Convert a citation field to compact display text."""

    if value is None:
        return ""

    if isinstance(
        value,
        (list, tuple),
    ):
        return " > ".join(
            str(item)
            for item in value
        )

    return str(
        value
    )


def _append_unique_citations(
    existing: list[dict[str, str]],
    incoming: Sequence[dict[str, str]],
) -> None:
    """Append citation dictionaries without creating duplicates."""

    seen = {
        (
            item["doc_id"],
            item["section"],
            item["snippet"],
        )
        for item in existing
    }

    for item in incoming:
        key = (
            item["doc_id"],
            item["section"],
            item["snippet"],
        )

        if key in seen:
            continue

        existing.append(
            item
        )
        seen.add(
            key
        )


def _summarize_tool_result(
    structured: Any,
) -> str:
    """Return stable compact JSON text for LLM and trace consumption."""

    import json

    try:
        return json.dumps(
            structured,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    except (TypeError, ValueError):
        return str(
            structured
        )


def _build_exhaustion_answer(
    citations: Sequence[dict[str, str]],
) -> str:
    """Return the required controlled max-iteration response."""

    if citations:
        return (
            "I gathered some policy evidence, but I could not fully "
            "complete the task within the agent's execution limit. "
            "Please rephrase the request or contact HR for assistance."
        )

    return (
        "I could not fully complete the task within the agent's "
        "execution limit and do not have enough supporting evidence "
        "to provide a reliable answer. Please rephrase the request "
        "or contact HR."
    )


def _find_discovered_tool(
    mcp_client: Any,
    name: str,
) -> DiscoveredTool | None:
    """Return discovered metadata for one tool name when available."""

    tools = getattr(
        mcp_client,
        "tools",
        (),
    )

    for tool in tools:
        if (
            isinstance(tool, DiscoveredTool)
            and tool.name == name
        ):
            return tool

    return None


def _requires_confirmation(
    mcp_client: Any,
    name: str,
) -> bool:
    """Derive ACTION status exclusively from discovered MCP metadata."""

    tool = _find_discovered_tool(
        mcp_client,
        name,
    )

    return (
        tool is not None
        and tool.read_only is False
    )


def _build_action_preview(
    *,
    tool: str,
    arguments: dict[str, Any],
) -> str:
    """Build a deterministic generic preview without tool-name branching."""

    rendered_arguments = json.dumps(
        arguments,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return (
        f"Confirm ACTION tool {tool!r} with arguments "
        f"{rendered_arguments}."
    )


def _create_pending_confirmation(
    *,
    tool: str,
    arguments: dict[str, Any],
) -> PendingConfirmation:
    """Create one server-generated confirmation binding."""

    copied_arguments = deepcopy(
        arguments
    )

    return PendingConfirmation(
        confirmation_id=uuid4().hex,
        tool=tool,
        arguments=copied_arguments,
        preview=_build_action_preview(
            tool=tool,
            arguments=copied_arguments,
        ),
    )


async def confirm_pending_action(
    *,
    pending: PendingConfirmation,
    confirmation_id: str,
    mcp_client: AgentMCPClient,
) -> AgentResult:
    """Execute exactly one previously previewed ACTION after ID validation."""

    from agent.trace import TraceItem

    if not isinstance(
        pending,
        PendingConfirmation,
    ):
        raise TypeError(
            "pending must be a PendingConfirmation."
        )

    if (
        not isinstance(confirmation_id, str)
        or not confirmation_id.strip()
    ):
        raise ValueError(
            "confirmation_id must be a non-empty string."
        )

    if confirmation_id != pending.confirmation_id:
        return AgentResult(
            answer=(
                "The confirmation did not match the pending action, "
                "so nothing was executed."
            ),
            citations=(),
            trace=(
                TraceItem(
                    step=1,
                    tool=pending.tool,
                    arguments={},
                    result_summary=(
                        "Confirmation ID did not match the pending action."
                    ),
                    sources=(),
                    decision="confirmation_rejected",
                ),
            ),
        )

    if mcp_client.status != "connected":
        return AgentResult(
            answer=(
                "The HR tools are currently unavailable, "
                "so the confirmed action was not executed."
            ),
            citations=(),
            trace=(
                TraceItem(
                    step=1,
                    tool=pending.tool,
                    arguments={},
                    result_summary=(
                        mcp_client.last_error
                        or "MCP client is degraded."
                    ),
                    sources=(),
                    decision="mcp_degraded",
                ),
            ),
        )

    try:
        result = await mcp_client.call_tool(
            pending.tool,
            pending.bound_arguments(),
        )

    except AgentMCPError as exc:
        return AgentResult(
            answer=(
                "The confirmed HR action could not be completed. "
                "Please try again or contact HR."
            ),
            citations=(),
            trace=(
                TraceItem(
                    step=1,
                    tool=pending.tool,
                    arguments=deepcopy(
                        pending.arguments
                    ),
                    result_summary=str(exc),
                    sources=(),
                    decision="tool_error",
                ),
            ),
        )

    summary = _summarize_tool_result(
        result.structuredContent
    )

    return AgentResult(
        answer=(
            "The confirmed mock HR action was executed successfully."
        ),
        citations=(),
        trace=(
            TraceItem(
                step=1,
                tool=pending.tool,
                arguments=deepcopy(
                    pending.arguments
                ),
                result_summary=summary,
                sources=(),
                decision="action_executed",
            ),
        ),
    )
