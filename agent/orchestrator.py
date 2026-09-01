"""Agent orchestration primitives for the HR Policy Assistant.

S6 begins by owning the MCP client boundary here. The agent discovers
production tools dynamically over MCP stdio and never imports RAG or
mock-data business implementations directly.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
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

MCP_STARTUP_TIMEOUT_SECONDS = 30
MCP_TOOL_TIMEOUT_SECONDS = 60


def _build_mcp_subprocess_env() -> dict[str, str] | None:
    """Return explicit runtime configuration required by the MCP child."""

    sanctioned_names = (
        "CHROMA_DIR",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )

    environment = {
        name: value
        for name in sanctioned_names
        if (
            value := os.getenv(
                name
            )
        )
        is not None
    }

    if not environment:
        return None

    return environment


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
                env=_build_mcp_subprocess_env(),
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
                MCP_STARTUP_TIMEOUT_SECONDS
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
            try:
                await stack.aclose()
            except BaseException:
                pass

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
            self._status = "degraded"

            raise AgentMCPError(
                self._last_error
            ) from exc

        except Exception as exc:
            self._last_error = (
                self._clean_error(
                    exc
                )
            )
            self._status = "degraded"

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
            try:
                await stack.aclose()
            except anyio.BrokenResourceError:
                pass
            except BaseExceptionGroup as exc:
                _, unexpected = exc.split(
                    anyio.BrokenResourceError
                )
                if unexpected is not None:
                    raise unexpected

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


def _select_wf2_policy_citation(
    citations: Sequence[dict[str, str]],
    *,
    decision: Literal[
        "manager_review",
        "insufficient_balance",
    ] = "manager_review",
) -> dict[str, str] | None:
    """Select retrieved PTO evidence that supports the WF2 decision."""

    if decision == "manager_review":
        preferred_sections = (
            "10.1",
            "8",
            "9.1",
            "5.3",
            "5.2",
            "5.5",
            "9.3",
        )
    elif decision == "insufficient_balance":
        preferred_sections = (
            "8",
            "4.4",
            "9.2",
        )
    else:
        raise ValueError(
            "Unsupported WF2 decision: "
            f"{decision}"
        )

    for preferred_section in preferred_sections:
        for citation in citations:
            if citation.get("doc_id") != "HR-POL-002":
                continue

            if (
                _citation_section_number(
                    citation.get("section", "")
                )
                == preferred_section
            ):
                return citation

    return None



def _evaluate_wf2_decision(
    *,
    requested_days: float | None,
    available_days: float | None,
    has_profile: bool,
    policy_citation: dict[str, str] | None,
) -> dict[str, Any]:
    """Evaluate the frozen WF2 business outcome from grounded evidence."""

    evidence_ready = (
        has_profile
        and requested_days is not None
        and available_days is not None
        and policy_citation is not None
    )

    if not evidence_ready:
        return {
            "evidence_ready": False,
            "balance_sufficient": None,
            "action_eligible": False,
            "decision": "insufficient_evidence",
        }

    balance_sufficient = (
        available_days >= requested_days
    )

    if balance_sufficient:
        return {
            "evidence_ready": True,
            "balance_sufficient": True,
            "action_eligible": True,
            "decision": "manager_review",
        }

    return {
        "evidence_ready": True,
        "balance_sufficient": False,
        "action_eligible": False,
        "decision": "insufficient_balance",
    }



def _build_wf2_guidance(
    *,
    section_number: str,
    available_days: Any,
    requested_days_text: str,
    policy_ref: str,
) -> str:
    """Build section-aware grounded guidance for frozen WF2."""

    balance_prefix = (
        "Your available PTO balance is "
        f"{available_days} days, and the request is for "
        f"{requested_days_text}. "
    )

    if section_number == "10.1":
        policy_guidance = (
            "The retrieved policy supports requesting the leave when "
            "sufficient PTO is available; written manager approval and "
            "operational coverage are still required "
            f"under {policy_ref}."
        )
    elif section_number == "8":
        policy_guidance = (
            "The request is within the current available balance and "
            "may proceed to manager approval "
            f"under {policy_ref}."
        )
    elif section_number == "5.2":
        policy_guidance = (
            "This is a short-notice request. The retrieved policy says "
            "a shorter-notice request may still be considered where "
            "operational coverage can be maintained; sufficient balance "
            "and written manager approval are still required "
            f"under {policy_ref}."
        )
    elif section_number == "5.3":
        policy_guidance = (
            "The retrieved policy approval conditions apply to this "
            "request, including sufficient available balance and "
            "manager approval "
            f"under {policy_ref}."
        )
    elif section_number == "5.5":
        policy_guidance = (
            "The manager must assess operational coverage for this "
            "request, and manager review remains required "
            f"under {policy_ref}."
        )
    else:
        raise ValueError(
            "Unsupported WF2 policy section: "
            f"{section_number}"
        )

    return balance_prefix + policy_guidance


def _is_wf2_action_request(
    message: str,
) -> bool:
    """Return whether the message is a sufficiently specified PTO request."""

    normalized = " ".join(
        message.lower().split()
    )

    pto_subject = (
        "pto" in normalized
        or "paid time off" in normalized
    )

    request_language = any(
        phrase in normalized
        for phrase in (
            "can i take",
            "request",
            "take ",
        )
    )

    employee_identified = (
        "employee " in normalized
        or "e001" in normalized
    )

    amount_identified = any(
        token in normalized
        for token in (
            " day ",
            " days ",
        )
    )

    period_identified = any(
        token in normalized
        for token in (
            "next week",
            "this week",
            "next month",
            "this month",
        )
    )

    return (
        pto_subject
        and request_language
        and employee_identified
        and amount_identified
        and period_identified
    )




def _build_wf2_insufficient_balance_guidance(
    *,
    available_days: Any,
    requested_days_text: str,
    policy_ref: str,
) -> str:
    """Build grounded no-action guidance for insufficient WF2 balance."""

    return (
        "Your available PTO balance is "
        f"{available_days} days, but the request is for "
        f"{requested_days_text}. "
        "The request cannot be approved as paid time off because "
        "the available balance is insufficient "
        f"under {policy_ref}. "
        "You may contact HR to discuss alternatives such as "
        "different dates or another applicable leave arrangement."
    )



def _build_wf2_pending_context(
    *,
    message: str,
    section_number: str,
    available_days: Any,
    requested_days_text: str,
    policy_ref: str,
) -> str:
    """Build grounded context for the confirmation-gated WF2 action."""

    policy_guidance = _build_wf2_guidance(
        section_number=section_number,
        available_days=available_days,
        requested_days_text=requested_days_text,
        policy_ref=policy_ref,
    )

    return f"{message} {policy_guidance}"



def _wf2_requires_action_proposal(
    message: str,
    trace: Sequence[Any],
) -> bool:
    """Return whether a sufficiently specified PTO workflow is incomplete.

    This guard is intentionally narrow. It protects the frozen WF2 action
    workflow from premature LLM termination after its required read checks,
    without converting ordinary PTO information questions into actions.
    """

    normalized = " ".join(
        message.lower().split()
    )

    pto_subject = (
        "pto" in normalized
        or "paid time off" in normalized
    )

    request_language = any(
        phrase in normalized
        for phrase in (
            "can i take",
            "request",
            "take ",
        )
    )

    employee_identified = (
        "employee " in normalized
        or "e001" in normalized
    )

    amount_identified = any(
        token in normalized
        for token in (
            " day ",
            " days ",
        )
    )

    period_identified = any(
        token in normalized
        for token in (
            "next week",
            "this week",
            "next month",
            "this month",
        )
    )

    if not (
        pto_subject
        and request_language
        and employee_identified
        and amount_identified
        and period_identified
    ):
        return False

    successful_tools = {
        getattr(item, "tool", None)
        for item in trace
        if getattr(item, "decision", None) == "tool_result"
    }

    required_reads = {
        "lookup_employee_profile",
        "check_pto_balance",
        "search_policy_documents",
    }

    return required_reads.issubset(
        successful_tools
    )


async def run_turn(
    *,
    message: str,
    mcp_client: AgentMCPClient,
    llm: AgentLLM,
    history: list[dict[str, str]] | None = None,
) -> AgentResult:
    """Run one bounded agent turn using discovered MCP tools.

    This function owns orchestration only. It does not call RAG or mock-data
    implementations directly; every tool execution crosses the MCP client.
    """

    from agent.llm import LLMError
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

    prior_history = [
        dict(item)
        for item in (history or [])
    ]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *prior_history,
        {
            "role": "user",
            "content": message,
        },
    ]

    trace: list[TraceItem] = []
    citations: list[dict[str, str]] = []
    grounded_policy_selectors: set[
        tuple[str, str]
    ] = set()
    grounded_policy_doc_ids: set[str] = set()
    known_employee_ids: set[str] = set()
    policy_search_stagnation_streak = 0
    policy_search_doc_stagnation_streak = 0
    wf2_pto_balance: dict[str, Any] | None = None
    wf2_policy_retry_used = False

    for iteration in range(
        1,
        MAX_AGENT_ITERATIONS + 1,
    ):
        try:
            response = await llm.chat(
                messages=messages,
                tools=mcp_client.llm_tools,
            )
        except LLMError as exc:
            trace.append(
                TraceItem(
                    step=iteration,
                    tool=None,
                    arguments={},
                    result_summary=str(exc),
                    sources=tuple(citations),
                    decision="llm_error",
                )
            )

            return AgentResult(
                answer=(
                    "The language model is temporarily unavailable. "
                    "Please try again later."
                ),
                citations=tuple(citations),
                trace=tuple(trace),
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
                citations=_project_answer_citations(
                    content,
                    citations,
                ),
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

            if (
                tool_name == "get_policy_section"
                and not _is_grounded_policy_section_call(
                    arguments,
                    grounded_policy_selectors,
                )
            ):
                rejection_summary = (
                    "Exact policy-section lookup rejected because "
                    "the requested document/section selector was not "
                    "established by prior policy evidence."
                )

                trace.append(
                    TraceItem(
                        step=iteration,
                        tool=tool_name,
                        arguments=deepcopy(arguments),
                        result_summary=rejection_summary,
                        sources=tuple(citations),
                        decision="section_guard_rejected",
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (
                            "The exact policy section you requested has "
                            "not been established by prior policy evidence. "
                            "Search policy documents again with a more "
                            "specific query or answer from evidence already "
                            "retrieved."
                        ),
                    }
                )

                continue

            if (
                tool_name == "draft_hr_email"
                and _is_wf2_action_request(message)
            ):
                rejection_summary = (
                    "LLM-proposed WF2 draft action rejected because "
                    "the deterministic WF2 finalizer owns the "
                    "confirmation boundary."
                )

                trace.append(
                    TraceItem(
                        step=iteration,
                        tool=tool_name,
                        arguments=deepcopy(arguments),
                        result_summary=rejection_summary,
                        sources=tuple(citations),
                        decision="workflow_guard_rejected",
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (
                            "Do not create the PTO draft action directly. "
                            "Use the required employee-profile, PTO-balance, "
                            "and policy read tools. The workflow will finalize "
                            "the action deterministically after grounded "
                            "evidence is available."
                        ),
                    }
                )

                continue

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

                confirmation_answer = (
                    "This action requires your explicit confirmation "
                    "before it can be executed."
                )

                return AgentResult(
                    answer=confirmation_answer,
                    citations=_project_answer_citations(
                        confirmation_answer,
                        citations,
                    ),
                    trace=tuple(trace),
                    pending_confirmation=pending,
                )

            try:
                tool_result = await mcp_client.call_tool(
                    tool_name,
                    arguments,
                )

            except AgentMCPError as exc:
                degraded = (
                    mcp_client.status
                    == "degraded"
                )

                decision = (
                    "mcp_degraded"
                    if degraded
                    else "tool_error"
                )

                trace.append(
                    TraceItem(
                        step=iteration,
                        tool=tool_name,
                        arguments=arguments,
                        result_summary=str(exc),
                        sources=tuple(citations),
                        decision=decision,
                    )
                )

                if (
                    degraded
                    and citations
                ):
                    answer = (
                        "I gathered some policy evidence, but the HR tools "
                        "became unavailable before I could fully complete "
                        "the request. Please use the cited policy guidance "
                        "cautiously and contact HR for assistance."
                    )

                elif degraded:
                    answer = (
                        "The HR tools became unavailable before I could "
                        "complete the request. Please try again later or "
                        "contact HR."
                    )

                else:
                    answer = (
                        "I couldn't complete the requested HR tool operation. "
                        "Please try again or contact HR."
                    )

                return AgentResult(
                    answer=answer,
                    citations=tuple(citations),
                    trace=tuple(trace),
                )

            if getattr(
                tool_result,
                "isError",
                False,
            ):
                error_text = _mcp_result_error_text(
                    tool_result
                )

                employee_id = arguments.get(
                    "employee_id"
                )

                known_employee_pto_absence = (
                    tool_name == "check_pto_balance"
                    and isinstance(employee_id, str)
                    and employee_id in known_employee_ids
                    and _is_missing_pto_record_error(
                        error_text
                    )
                )

                if known_employee_pto_absence:
                    summary = (
                        "PTO balance record is unavailable for this known "
                        "employee. Continue with applicable PTO policy "
                        "eligibility and scope evidence."
                    )

                    trace.append(
                        TraceItem(
                            step=iteration,
                            tool=tool_name,
                            arguments=deepcopy(arguments),
                            result_summary=summary,
                            sources=tuple(citations),
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

                    continue

                if _is_unknown_employee_error(
                    error_text
                ):
                    answer = (
                        "I couldn't find that employee ID. "
                        "Please check the ID and try again."
                    )
                    decision = "unknown_employee"

                else:
                    answer = (
                        "I couldn't complete the requested HR tool operation. "
                        "Please try again or contact HR."
                    )
                    decision = "tool_error"

                trace.append(
                    TraceItem(
                        step=iteration,
                        tool=tool_name,
                        arguments=arguments,
                        result_summary=error_text,
                        sources=tuple(citations),
                        decision=decision,
                    )
                )

                return AgentResult(
                    answer=answer,
                    citations=tuple(citations),
                    trace=tuple(trace),
                )

            structured = (
                tool_result.structuredContent
            )

            if (
                tool_name == "check_pto_balance"
                and isinstance(structured, dict)
            ):
                wf2_pto_balance = deepcopy(structured)

            if (
                tool_name == "lookup_employee_profile"
                and isinstance(
                    arguments.get("employee_id"),
                    str,
                )
            ):
                known_employee_ids.add(
                    arguments["employee_id"]
                )

            new_policy_selectors = (
                _extract_grounded_policy_selectors(
                    structured,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            )

            if tool_name == "search_policy_documents":
                genuinely_new_selectors = (
                    new_policy_selectors
                    - grounded_policy_selectors
                )

                if genuinely_new_selectors:
                    policy_search_stagnation_streak = 0
                else:
                    policy_search_stagnation_streak += 1

                current_policy_doc_ids = {
                    doc_id
                    for doc_id, _section
                    in new_policy_selectors
                }

                genuinely_new_doc_ids = (
                    current_policy_doc_ids
                    - grounded_policy_doc_ids
                )

                if genuinely_new_doc_ids:
                    policy_search_doc_stagnation_streak = 0
                else:
                    policy_search_doc_stagnation_streak += 1

                grounded_policy_doc_ids.update(
                    current_policy_doc_ids
                )

            grounded_policy_selectors.update(
                new_policy_selectors
            )

            new_sources = _extract_citations(
                structured,
                tool_name=tool_name,
                arguments=arguments,
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

            # WF2 deterministic evidence-completion boundary.
            #
            # Once the frozen PTO workflow has completed its required
            # employee, balance, and policy reads, determine the business
            # outcome from structured evidence before deciding whether an
            # ACTION is eligible. Policy evidence is selected for that
            # outcome, rather than assuming every completed PTO read may
            # proceed to manager review.
            if (
                tool_name == "search_policy_documents"
                and wf2_pto_balance is not None
                and known_employee_ids
                and _wf2_requires_action_proposal(
                    message,
                    trace,
                )
            ):
                available_days = wf2_pto_balance.get(
                    "available_days"
                )

                requested_days_match = re.search(
                    r"\b(\d+(?:\.\d+)?)\s+days?\b",
                    message,
                    flags=re.IGNORECASE,
                )

                requested_days = (
                    float(requested_days_match.group(1))
                    if requested_days_match is not None
                    else None
                )

                requested_days_text = (
                    requested_days_match.group(0)
                    if requested_days_match is not None
                    else "the requested PTO"
                )

                if (
                    isinstance(available_days, (int, float))
                    and not isinstance(available_days, bool)
                    and requested_days is not None
                ):
                    available_days_value = float(
                        available_days
                    )

                    selector_decision = (
                        "manager_review"
                        if available_days_value >= requested_days
                        else "insufficient_balance"
                    )

                    preferred_policy_citation = (
                        _select_wf2_policy_citation(
                            citations,
                            decision=selector_decision,
                        )
                    )

                    # S10 WF2 bounded retrieval recovery.
                    #
                    # If the model-selected first policy search did not
                    # retrieve evidence that supports the actual PTO
                    # decision, the orchestrator performs exactly one
                    # deterministic retry through the MCP boundary.
                    # No additional LLM planning call is used.
                    if (
                        preferred_policy_citation is None
                        and not wf2_policy_retry_used
                    ):
                        wf2_policy_retry_used = True

                        retry_arguments = {
                            "query": (
                                "PTO paid time off manager approval "
                                "short notice operational coverage decision"
                            ),
                            "k": 5,
                        }

                        try:
                            retry_result = await mcp_client.call_tool(
                                "search_policy_documents",
                                retry_arguments,
                            )
                        except AgentMCPError as exc:
                            trace.append(
                                TraceItem(
                                    step=iteration,
                                    tool="search_policy_documents",
                                    arguments=retry_arguments,
                                    result_summary=str(exc),
                                    sources=(),
                                    decision="tool_error",
                                )
                            )

                            answer = (
                                "I could not obtain enough supporting "
                                "Paid Time Off Policy evidence to complete "
                                "this request reliably. No HR action has "
                                "been prepared. Please try again or contact "
                                "People and Culture."
                            )

                            return AgentResult(
                                answer=answer,
                                citations=_project_answer_citations(
                                    answer,
                                    citations,
                                ),
                                trace=tuple(trace),
                                pending_confirmation=None,
                            )

                        if getattr(
                            retry_result,
                            "isError",
                            False,
                        ):
                            trace.append(
                                TraceItem(
                                    step=iteration,
                                    tool="search_policy_documents",
                                    arguments=retry_arguments,
                                    result_summary=(
                                        "The bounded WF2 policy retry "
                                        "returned a tool error."
                                    ),
                                    sources=(),
                                    decision="tool_error",
                                )
                            )

                            answer = (
                                "I could not obtain enough supporting "
                                "Paid Time Off Policy evidence to complete "
                                "this request reliably. No HR action has "
                                "been prepared. Please try again or contact "
                                "People and Culture."
                            )

                            return AgentResult(
                                answer=answer,
                                citations=_project_answer_citations(
                                    answer,
                                    citations,
                                ),
                                trace=tuple(trace),
                                pending_confirmation=None,
                            )

                        retry_structured = getattr(
                            retry_result,
                            "structuredContent",
                            None,
                        )

                        if not isinstance(
                            retry_structured,
                            dict,
                        ):
                            retry_structured = {}

                        retry_sources = _extract_citations(
                            retry_structured,
                            tool_name="search_policy_documents",
                            arguments=retry_arguments,
                        )

                        _append_unique_citations(
                            citations,
                            retry_sources,
                        )

                        trace.append(
                            TraceItem(
                                step=iteration,
                                tool="search_policy_documents",
                                arguments=retry_arguments,
                                result_summary=_summarize_tool_result(
                                    retry_structured
                                ),
                                sources=tuple(retry_sources),
                                decision="tool_result",
                            )
                        )

                        preferred_policy_citation = (
                            _select_wf2_policy_citation(
                                citations,
                                decision=selector_decision,
                            )
                        )

                    # The single permitted retry has completed. If the
                    # accumulated RAG evidence still cannot support the
                    # decision, stop here rather than returning to the
                    # LLM loop or allowing a side effect.
                    if preferred_policy_citation is None:
                        answer = (
                            "I could not find enough relevant Paid Time Off "
                            "Policy evidence to determine this request "
                            "reliably. No HR action has been prepared. "
                            "Please contact People and Culture for assistance."
                        )

                        trace.append(
                            TraceItem(
                                step=iteration,
                                tool=None,
                                arguments={},
                                result_summary=(
                                    "WF2 stopped after the single permitted "
                                    "policy-retrieval retry produced "
                                    "insufficient decision evidence."
                                ),
                                sources=tuple(citations),
                                decision="insufficient_evidence",
                            )
                        )

                        return AgentResult(
                            answer=answer,
                            citations=_project_answer_citations(
                                answer,
                                citations,
                            ),
                            trace=tuple(trace),
                            pending_confirmation=None,
                        )

                    evaluation = _evaluate_wf2_decision(
                        requested_days=requested_days,
                        available_days=available_days_value,
                        has_profile=bool(known_employee_ids),
                        policy_citation=preferred_policy_citation,
                    )

                    if (
                        evaluation["evidence_ready"]
                        and preferred_policy_citation is not None
                    ):
                        section_number = (
                            _citation_section_number(
                                preferred_policy_citation.get(
                                    "section",
                                    "",
                                )
                            )
                        )

                        policy_ref = (
                            f"[HR-POL-002 §{section_number}]"
                        )

                        if (
                            evaluation["decision"]
                            == "insufficient_balance"
                        ):
                            answer = (
                                _build_wf2_insufficient_balance_guidance(
                                    available_days=available_days_value,
                                    requested_days_text=requested_days_text,
                                    policy_ref=policy_ref,
                                )
                            )

                            trace.append(
                                TraceItem(
                                    step=iteration,
                                    tool=None,
                                    arguments={},
                                    result_summary=(
                                        "WF2 completed without an ACTION "
                                        "because the requested PTO exceeds "
                                        "the available balance."
                                    ),
                                    sources=(
                                        preferred_policy_citation,
                                    ),
                                    decision="answer",
                                )
                            )

                            return AgentResult(
                                answer=answer,
                                citations=_project_answer_citations(
                                    answer,
                                    citations,
                                ),
                                trace=tuple(trace),
                                pending_confirmation=None,
                            )

                        if (
                            evaluation["decision"]
                            == "manager_review"
                            and evaluation["action_eligible"]
                        ):
                            policy_guidance = (
                                _build_wf2_guidance(
                                    section_number=section_number,
                                    available_days=available_days_value,
                                    requested_days_text=requested_days_text,
                                    policy_ref=policy_ref,
                                )
                            )

                            confirmation_answer = (
                                f"{policy_guidance}\n\n"
                                "I can prepare a mock PTO request email "
                                "to your manager. This action requires "
                                "your explicit confirmation before it "
                                "can be executed."
                            )

                            pending_arguments = {
                                "to_role": "manager",
                                "subject": "PTO request",
                                "context": _build_wf2_pending_context(
                                    message=message,
                                    section_number=section_number,
                                    available_days=available_days_value,
                                    requested_days_text=(
                                        requested_days_text
                                    ),
                                    policy_ref=policy_ref,
                                ),
                            }

                            pending = _create_pending_confirmation(
                                tool="draft_hr_email",
                                arguments=pending_arguments,
                            )

                            trace.append(
                                TraceItem(
                                    step=iteration,
                                    tool="draft_hr_email",
                                    arguments=deepcopy(
                                        pending_arguments
                                    ),
                                    result_summary=pending.preview,
                                    sources=tuple(citations),
                                    decision="confirmation_required",
                                )
                            )

                            return AgentResult(
                                answer=confirmation_answer,
                                citations=_project_answer_citations(
                                    confirmation_answer,
                                    citations,
                                ),
                                trace=tuple(trace),
                                pending_confirmation=pending,
                            )

            if (
                tool_name == "search_policy_documents"
                and (
                    policy_search_stagnation_streak >= 2
                    or policy_search_doc_stagnation_streak >= 2
                )
            ):
                trace.append(
                    TraceItem(
                        step=iteration,
                        tool=None,
                        arguments={},
                        result_summary=(
                            "Policy search terminated after two "
                            "consecutive searches added no new "
                            "grounded policy selectors."
                        ),
                        sources=tuple(citations),
                        decision="policy_search_stagnated",
                    )
                )

                stagnation_answer = (
                    "I could not find enough supporting policy "
                    "evidence to answer this reliably. Please "
                    "contact People and Culture for the governing "
                    "policy."
                )

                return AgentResult(
                    answer=stagnation_answer,
                    citations=_project_answer_citations(
                        stagnation_answer,
                        citations,
                    ),
                    trace=tuple(trace),
                    exhausted=False,
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

    exhaustion_answer = _build_exhaustion_answer(
        citations
    )

    return AgentResult(
        answer=exhaustion_answer,
        citations=_project_answer_citations(
            exhaustion_answer,
            citations,
        ),
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


_POLICY_SECTION_PREFIX_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\b"
)

_POLICY_REF_RE = re.compile(
    r"^(HR-POL-\d{3}) §(\d+(?:\.\d+)*)$"
)


def _policy_section_selectors(
    doc_id: str,
    section: str,
) -> set[tuple[str, str]]:
    """Return exact selectors established by one policy section."""

    if not isinstance(
        doc_id,
        str,
    ):
        return set()

    if not isinstance(
        section,
        str,
    ):
        return set()

    if (
        not doc_id
        or not section
        or doc_id != doc_id.strip()
        or section != section.strip()
    ):
        return set()

    selectors = {
        (
            doc_id,
            section,
        ),
    }

    match = _POLICY_SECTION_PREFIX_RE.match(
        section
    )

    if match is not None:
        selectors.add(
            (
                doc_id,
                match.group(1),
            )
        )

    return selectors


def _extract_grounded_policy_selectors(
    structured: Any,
    *,
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
) -> set[tuple[str, str]]:
    """Extract exact policy selectors grounded by one successful tool result."""

    selectors: set[
        tuple[str, str]
    ] = set()

    if tool_name == "search_policy_documents":
        value = structured

        if (
            isinstance(value, dict)
            and set(value) == {"result"}
        ):
            value = value["result"]

        if not isinstance(
            value,
            list,
        ):
            return selectors

        for item in value:
            if not isinstance(
                item,
                dict,
            ):
                continue

            doc_id = item.get(
                "doc_id"
            )

            section = item.get(
                "section"
            )

            if (
                isinstance(doc_id, str)
                and isinstance(section, str)
            ):
                selectors.update(
                    _policy_section_selectors(
                        doc_id,
                        section,
                    )
                )

        return selectors

    if tool_name == "check_policy_compliance":
        if not isinstance(
            structured,
            dict,
        ):
            return selectors

        policy_refs = structured.get(
            "policy_refs"
        )

        if not isinstance(
            policy_refs,
            list,
        ):
            return selectors

        for item in policy_refs:
            if not isinstance(
                item,
                str,
            ):
                continue

            match = _POLICY_REF_RE.fullmatch(
                item
            )

            if match is None:
                continue

            selectors.add(
                (
                    match.group(1),
                    match.group(2),
                )
            )

        return selectors

    if tool_name == "get_policy_section":
        if not isinstance(
            structured,
            dict,
        ):
            return selectors

        if not isinstance(
            arguments,
            Mapping,
        ):
            return selectors

        doc_id = arguments.get(
            "doc_id"
        )

        section = structured.get(
            "section"
        )

        if (
            isinstance(doc_id, str)
            and isinstance(section, str)
        ):
            selectors.update(
                _policy_section_selectors(
                    doc_id,
                    section,
                )
            )

        return selectors

    return selectors


def _is_grounded_policy_section_call(
    arguments: Mapping[str, Any],
    selectors: set[tuple[str, str]],
) -> bool:
    """Return whether an exact-section call is established by prior evidence."""

    if not isinstance(
        arguments,
        Mapping,
    ):
        return False

    doc_id = arguments.get(
        "doc_id"
    )

    section = arguments.get(
        "section"
    )

    if (
        not isinstance(doc_id, str)
        or not isinstance(section, str)
    ):
        return False

    return (
        doc_id,
        section,
    ) in selectors


def _extract_citations(
    structured: Any,
    *,
    tool_name: str | None = None,
    arguments: Mapping[str, Any] | None = None,
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

        if (
            (
                not isinstance(doc_id, str)
                or not doc_id.strip()
            )
            and tool_name == "get_policy_section"
            and isinstance(arguments, Mapping)
        ):
            invocation_doc_id = arguments.get(
                "doc_id"
            )

            if (
                isinstance(invocation_doc_id, str)
                and invocation_doc_id.strip()
            ):
                doc_id = invocation_doc_id

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


_ANSWER_POLICY_REFERENCE_RE = re.compile(
    r"\b(HR-POL-\d{3})\s+§\s*"
    r"([0-9]+(?:\.[0-9]+)*)"
)


def _answer_policy_selectors(
    answer: str,
) -> set[tuple[str, str]]:
    """Return explicit policy doc/section selectors cited in an answer."""

    if not isinstance(answer, str):
        raise TypeError(
            "answer must be a string"
        )

    return {
        (
            match.group(1),
            match.group(2),
        )
        for match in _ANSWER_POLICY_REFERENCE_RE.finditer(
            answer
        )
    }


def _citation_section_number(
    section: str,
) -> str | None:
    """Return the leading numeric section selector from citation metadata."""

    if not isinstance(section, str):
        return None

    match = re.match(
        r"^\s*([0-9]+(?:\.[0-9]+)*)\b",
        section,
    )

    if match is None:
        return None

    return match.group(1)


def _project_answer_citations(
    answer: str,
    citations: Sequence[dict[str, str]],
) -> tuple[dict[str, str], ...]:
    """Project accumulated evidence to citations explicitly named in answer."""

    selectors = _answer_policy_selectors(
        answer
    )

    if not selectors:
        return ()

    projected: list[dict[str, str]] = []

    for citation in citations:
        doc_id = citation.get(
            "doc_id"
        )

        section = citation.get(
            "section",
            "",
        )

        if not isinstance(doc_id, str):
            continue

        section_number = (
            _citation_section_number(
                section
            )
        )

        if section_number is None:
            continue

        if (
            doc_id,
            section_number,
        ) not in selectors:
            continue

        projected.append(
            citation
        )

    return tuple(
        projected
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


def _build_confirmed_action_answer(
    *,
    tool: str,
    structured: Any,
) -> str:
    """Project a successful confirmed ACTION result into a user-facing answer."""

    if tool == "create_mock_hr_ticket" and isinstance(structured, dict):
        ticket_id = structured.get("ticket_id")
        status = structured.get("status")

        if (
            isinstance(ticket_id, str)
            and ticket_id
            and isinstance(status, str)
            and status
        ):
            return (
                "Mock HR ticket created successfully.\n\n"
                f"Ticket ID: **{ticket_id}**\n"
                f"Status: **{status}**"
            )

    if tool == "draft_hr_email" and isinstance(structured, dict):
        draft_text = structured.get("draft_text")
        note = structured.get("note")

        if isinstance(draft_text, str) and draft_text:
            answer = (
                "Mock HR email draft created successfully.\n\n"
                f"{draft_text}"
            )

            if isinstance(note, str) and note:
                answer += f"\n\n{note}"

            return answer

    return "The confirmed mock HR action was executed successfully."


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

    answer = _build_confirmed_action_answer(
        tool=pending.tool,
        structured=result.structuredContent,
    )

    return AgentResult(
        answer=answer,
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


def _mcp_result_error_text(
    result: Any,
) -> str:
    """Extract a clean text message from an MCP error result."""

    content = getattr(
        result,
        "content",
        (),
    )

    if not isinstance(
        content,
        (list, tuple),
    ):
        return "MCP tool returned an error."

    parts: list[str] = []

    for item in content:
        text = getattr(
            item,
            "text",
            None,
        )

        if (
            isinstance(text, str)
            and text.strip()
        ):
            parts.append(
                text.strip()
            )

    if not parts:
        return "MCP tool returned an error."

    return " ".join(
        parts
    )


def _is_unknown_employee_error(
    error_text: str,
) -> bool:
    """Recognize the frozen unknown-employee lookup failure condition."""

    lowered = " ".join(
        error_text.casefold().split()
    )

    return (
        "employee not found" in lowered
        and "pto balance record not found" not in lowered
    )


def _is_missing_pto_record_error(
    error_text: str,
) -> bool:
    """Recognize a missing PTO-balance record without implying unknown employee."""

    lowered = " ".join(
        error_text.casefold().split()
    )

    return (
        "pto balance record not found" in lowered
        and "employee" in lowered
    )
