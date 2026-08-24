"""Tests for the S6 agent orchestration layer."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_orchestrator():
    """Import the S6 orchestrator after its production file exists."""

    from agent import orchestrator

    return orchestrator


def test_discovered_tool_converts_to_llm_function_schema() -> None:
    """Discovered MCP schemas are projected without a hardcoded registry."""

    orchestrator = _load_orchestrator()

    tool = orchestrator.DiscoveredTool(
        name="example_tool",
        description="Example discovered tool.",
        input_schema={
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                },
            },
            "required": [
                "employee_id",
            ],
        },
        read_only=True,
    )

    assert tool.to_llm_schema() == {
        "type": "function",
        "function": {
            "name": "example_tool",
            "description": "Example discovered tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "string",
                    },
                },
                "required": [
                    "employee_id",
                ],
            },
        },
    }


def test_agent_mcp_client_starts_degraded() -> None:
    """Before successful discovery the client must not claim MCP is usable."""

    orchestrator = _load_orchestrator()

    client = orchestrator.AgentMCPClient()

    assert client.status == "degraded"
    assert client.tools == ()
    assert client.last_error is None


def test_agent_mcp_client_discovers_real_tools_over_stdio() -> None:
    """The S6 client discovers the production MCP surface over real stdio."""

    orchestrator = _load_orchestrator()

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient()

        try:
            tools = await client.start()

            assert client.status == "connected"
            assert len(tools) == 8

            discovered_names = {
                tool.name
                for tool in tools
            }

            # These names belong in the test because they verify the frozen
            # external MCP contract. Production agent selection must not use
            # this set as a registry.
            assert discovered_names == {
                "search_policy_documents",
                "get_policy_section",
                "lookup_employee_profile",
                "lookup_benefits_status",
                "check_pto_balance",
                "check_policy_compliance",
                "create_mock_hr_ticket",
                "draft_hr_email",
            }

            by_name = {
                tool.name: tool
                for tool in tools
            }

            assert (
                by_name["search_policy_documents"].read_only
                is True
            )
            assert (
                by_name["create_mock_hr_ticket"].read_only
                is False
            )
            assert (
                by_name["draft_hr_email"].read_only
                is False
            )

            llm_tools = client.llm_tools

            assert len(llm_tools) == 8

            assert {
                item["function"]["name"]
                for item in llm_tools
            } == discovered_names

        finally:
            await client.close()

    asyncio.run(exercise())


def test_agent_mcp_client_calls_discovered_tool_over_stdio() -> None:
    """Tool execution crosses the production MCP protocol boundary."""

    orchestrator = _load_orchestrator()

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient()

        try:
            await client.start()

            result = await client.call_tool(
                "lookup_employee_profile",
                {
                    "employee_id": "E003",
                },
            )

            assert result.isError is False
            assert result.structuredContent is not None

            profile = result.structuredContent

            assert isinstance(profile, dict)
            assert profile["name"]
            assert profile["employment_type"]

        finally:
            await client.close()

    asyncio.run(exercise())


def test_agent_mcp_client_rejects_undiscovered_tool() -> None:
    """The agent cannot call a name outside its discovered MCP surface."""

    orchestrator = _load_orchestrator()

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient()

        try:
            await client.start()

            with pytest.raises(
                orchestrator.AgentMCPError,
                match="was not discovered",
            ):
                await client.call_tool(
                    "invented_agent_tool",
                    {},
                )

        finally:
            await client.close()

    asyncio.run(exercise())


def test_agent_mcp_client_stays_degraded_when_server_is_missing() -> None:
    """Startup failure leaves the agent in a safe tool-disabled state."""

    orchestrator = _load_orchestrator()

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient(
            server_path=PROJECT_ROOT / "does-not-exist" / "server.py",
        )

        tools = await client.start()

        assert tools == ()
        assert client.tools == ()
        assert client.status == "degraded"
        assert client.last_error == (
            "MCP server entry point was not found."
        )

    asyncio.run(exercise())


def test_prompt_version_is_defined() -> None:
    """Prompt changes remain attributable during later evaluation."""

    from agent.prompts import PROMPT_VERSION

    assert PROMPT_VERSION == "1.0"


def test_trace_contains_required_operational_fields() -> None:
    """Trace output exposes execution facts without hidden reasoning."""

    from agent.trace import TraceItem

    item = TraceItem(
        step=1,
        tool="example_tool",
        arguments={
            "employee_id": "E003",
        },
        result_summary="Employee profile retrieved.",
        sources=(),
        decision="tool_result",
    )

    assert item.as_dict() == {
        "step": 1,
        "tool": "example_tool",
        "arguments": {
            "employee_id": "E003",
        },
        "result_summary": "Employee profile retrieved.",
        "sources": [],
        "decision": "tool_result",
        "prompt_version": "1.0",
    }


def test_trace_contract_contains_no_chain_of_thought_field() -> None:
    """Operational traces must never expose hidden chain-of-thought."""

    from agent.trace import TraceItem

    fields = set(
        TraceItem.__dataclass_fields__
    )

    assert "reasoning" not in fields
    assert "chain_of_thought" not in fields
    assert "thoughts" not in fields


def test_llm_client_sends_openai_compatible_payload() -> None:
    """The LLM boundary sends model, messages, tools, and temperature zero."""

    import json

    import httpx

    from agent.llm import LLMClient

    captured: dict[str, object] = {}

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        captured["url"] = str(
            request.url
        )
        captured["authorization"] = request.headers.get(
            "Authorization"
        )
        captured["payload"] = json.loads(
            request.content
        )

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Completed.",
                        },
                    },
                ],
            },
        )

    async def exercise() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://llm.example/v1",
            model="test-model",
            transport=httpx.MockTransport(
                handler
            ),
        )

        try:
            result = await client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": "Hello",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "example_tool",
                            "description": "Example.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    },
                ],
            )

            assert result.content == "Completed."
            assert result.tool_calls == ()

        finally:
            await client.close()

    asyncio.run(exercise())

    assert captured["url"] == (
        "https://llm.example/v1/chat/completions"
    )
    assert captured["authorization"] == (
        "Bearer test-key"
    )

    payload = captured["payload"]

    assert isinstance(
        payload,
        dict,
    )
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0
    assert payload["tool_choice"] == "auto"
    assert len(payload["tools"]) == 1


def test_llm_client_normalizes_function_tool_call() -> None:
    """Tool-call JSON is normalized for later orchestration."""

    import httpx

    from agent.llm import LLMClient

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        del request

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "example_tool",
                                        "arguments": (
                                            '{"employee_id":"E003"}'
                                        ),
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
        )

    async def exercise() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://llm.example/v1",
            model="test-model",
            transport=httpx.MockTransport(
                handler
            ),
        )

        try:
            result = await client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": "Find E003",
                    },
                ],
            )

            assert result.content is None
            assert len(result.tool_calls) == 1

            tool_call = result.tool_calls[0]

            assert tool_call.call_id == "call-1"
            assert tool_call.name == "example_tool"
            assert tool_call.arguments == {
                "employee_id": "E003",
            }

        finally:
            await client.close()

    asyncio.run(exercise())


def test_llm_client_converts_timeout_to_clean_error() -> None:
    """Provider timeouts become controlled LLM failures."""

    import httpx
    import pytest

    from agent.llm import (
        LLMClient,
        LLMError,
        LLM_TIMEOUT_SECONDS,
    )

    assert LLM_TIMEOUT_SECONDS == 30

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ReadTimeout(
            "simulated timeout",
            request=request,
        )

    async def exercise() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://llm.example/v1",
            model="test-model",
            transport=httpx.MockTransport(
                handler
            ),
        )

        try:
            with pytest.raises(
                LLMError,
                match="LLM request timed out",
            ):
                await client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": "Hello",
                        },
                    ],
                )

        finally:
            await client.close()

    asyncio.run(exercise())


def test_llm_client_rejects_invalid_tool_arguments() -> None:
    """Malformed provider tool arguments cannot enter the agent loop."""

    import httpx
    import pytest

    from agent.llm import (
        LLMClient,
        LLMError,
    )

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        del request

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "example_tool",
                                        "arguments": "{not-json}",
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
        )

    async def exercise() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://llm.example/v1",
            model="test-model",
            transport=httpx.MockTransport(
                handler
            ),
        )

        try:
            with pytest.raises(
                LLMError,
                match="arguments are not valid JSON",
            ):
                await client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": "Hello",
                        },
                    ],
                )

        finally:
            await client.close()

    asyncio.run(exercise())


def test_run_turn_returns_direct_llm_answer() -> None:
    """A terminal LLM answer completes the turn without tool execution."""

    from agent.llm import LLMResponse
    from agent.orchestrator import AgentResult, run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

    class FakeLLM:
        async def chat(self, *, messages, tools=()):
            del messages, tools

            return LLMResponse(
                content="Direct answer.",
                tool_calls=(),
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Hello",
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert isinstance(result, AgentResult)
        assert result.answer == "Direct answer."
        assert result.citations == ()
        assert result.exhausted is False
        assert result.trace[-1].decision == "answer"

    asyncio.run(exercise())


def test_run_turn_calls_tool_then_returns_final_answer() -> None:
    """The loop executes a requested MCP tool and returns to the LLM."""

    from types import SimpleNamespace

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": "example_tool",
                    "description": "Example.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ]

        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append(
                (name, arguments)
            )

            return SimpleNamespace(
                structuredContent={
                    "value": "ok",
                }
            )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, *, messages, tools=()):
            del messages, tools

            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="call-1",
                            name="example_tool",
                            arguments={
                                "employee_id": "E003",
                            },
                        ),
                    ),
                )

            return LLMResponse(
                content="Completed after tool use.",
                tool_calls=(),
            )

    async def exercise() -> None:
        mcp = FakeMCP()
        llm = FakeLLM()

        result = await run_turn(
            message="Use the tool.",
            mcp_client=mcp,
            llm=llm,
        )

        assert mcp.calls == [
            (
                "example_tool",
                {
                    "employee_id": "E003",
                },
            ),
        ]

        assert llm.calls == 2
        assert result.answer == "Completed after tool use."
        assert result.trace[0].decision == "tool_result"
        assert result.trace[-1].decision == "answer"

    asyncio.run(exercise())


def test_run_turn_feeds_tool_result_back_to_llm() -> None:
    """Observed MCP results become tool messages in the next LLM request."""

    from types import SimpleNamespace

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        async def call_tool(self, name, arguments):
            del name, arguments

            return SimpleNamespace(
                structuredContent={
                    "available_days": 12,
                }
            )

    class FakeLLM:
        def __init__(self):
            self.message_snapshots = []

        async def chat(self, *, messages, tools=()):
            del tools

            self.message_snapshots.append(
                list(messages)
            )

            if len(self.message_snapshots) == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="call-1",
                            name="example_tool",
                            arguments={},
                        ),
                    ),
                )

            return LLMResponse(
                content="You have 12 days.",
                tool_calls=(),
            )

    async def exercise() -> None:
        llm = FakeLLM()

        await run_turn(
            message="How much PTO?",
            mcp_client=FakeMCP(),
            llm=llm,
        )

        second_messages = (
            llm.message_snapshots[1]
        )

        tool_messages = [
            item
            for item in second_messages
            if item.get("role") == "tool"
        ]

        assert len(tool_messages) == 1
        assert '"available_days": 12' in (
            tool_messages[0]["content"]
        )

    asyncio.run(exercise())


def test_run_turn_collects_policy_citations() -> None:
    """Policy-shaped MCP results flow into the final agent result."""

    from types import SimpleNamespace

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        async def call_tool(self, name, arguments):
            del name, arguments

            return SimpleNamespace(
                structuredContent=[
                    {
                        "doc_id": "HR-POL-004",
                        "title": "Remote Work Policy",
                        "section": "4.4",
                        "snippet": "International remote work requires approval.",
                    },
                ]
            )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, *, messages, tools=()):
            del messages, tools

            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="call-1",
                            name="example_tool",
                            arguments={},
                        ),
                    ),
                )

            return LLMResponse(
                content="Approval is required [HR-POL-004 §4.4].",
                tool_calls=(),
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Can I work overseas?",
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert result.citations == (
            {
                "doc_id": "HR-POL-004",
                "title": "Remote Work Policy",
                "section": "4.4",
                "snippet": (
                    "International remote work requires approval."
                ),
            },
        )

        assert result.trace[0].sources == result.citations

    asyncio.run(exercise())


def test_run_turn_handles_undiscovered_tool_cleanly() -> None:
    """An LLM cannot bypass the discovered MCP tool surface."""

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import (
        AgentMCPError,
        run_turn,
    )

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        async def call_tool(self, name, arguments):
            del name, arguments

            raise AgentMCPError(
                "MCP tool 'invented' was not discovered."
            )

    class FakeLLM:
        async def chat(self, *, messages, tools=()):
            del messages, tools

            return LLMResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        call_id="call-1",
                        name="invented",
                        arguments={},
                    ),
                ),
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Do something.",
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert "couldn't complete" in (
            result.answer.lower()
        )
        assert result.trace[-1].decision == "tool_error"

    asyncio.run(exercise())


def test_run_turn_stops_after_six_iterations() -> None:
    """The orchestrator, not the model, enforces the six-step bound."""

    from types import SimpleNamespace

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import (
        MAX_AGENT_ITERATIONS,
        run_turn,
    )

    assert MAX_AGENT_ITERATIONS == 6

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        def __init__(self):
            self.call_count = 0

        async def call_tool(self, name, arguments):
            del name, arguments

            self.call_count += 1

            return SimpleNamespace(
                structuredContent={
                    "iteration": self.call_count,
                }
            )

    class FakeLLM:
        def __init__(self):
            self.call_count = 0

        async def chat(self, *, messages, tools=()):
            del messages, tools

            self.call_count += 1

            return LLMResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        call_id=f"call-{self.call_count}",
                        name="example_tool",
                        arguments={},
                    ),
                ),
            )

    async def exercise() -> None:
        mcp = FakeMCP()
        llm = FakeLLM()

        result = await run_turn(
            message="Keep going.",
            mcp_client=mcp,
            llm=llm,
        )

        assert llm.call_count == 6
        assert mcp.call_count == 6
        assert result.exhausted is True

    asyncio.run(exercise())


def test_run_turn_exhaustion_records_max_iterations() -> None:
    """Exhaustion is explicit in both the answer and operational trace."""

    from types import SimpleNamespace

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        async def call_tool(self, name, arguments):
            del name, arguments

            return SimpleNamespace(
                structuredContent={}
            )

    class FakeLLM:
        def __init__(self):
            self.call_count = 0

        async def chat(self, *, messages, tools=()):
            del messages, tools

            self.call_count += 1

            return LLMResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        call_id=f"call-{self.call_count}",
                        name="example_tool",
                        arguments={},
                    ),
                ),
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Never finish.",
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert result.exhausted is True
        assert "could not fully complete" in (
            result.answer.lower()
        )
        assert (
            result.trace[-1].decision
            == "max_iterations"
        )

    asyncio.run(exercise())


def test_action_classification_uses_discovered_readonly_false() -> None:
    """ACTION status comes from discovered MCP metadata, not tool names."""

    from agent.orchestrator import (
        DiscoveredTool,
        _requires_confirmation,
    )

    class FakeMCP:
        tools = (
            DiscoveredTool(
                name="future_action",
                description="Future action.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
                read_only=False,
            ),
        )

    assert _requires_confirmation(
        FakeMCP(),
        "future_action",
    ) is True


def test_run_turn_action_returns_preview_without_execution() -> None:
    """An unconfirmed ACTION stops before the MCP call."""

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import (
        DiscoveredTool,
        run_turn,
    )

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        tools = (
            DiscoveredTool(
                name="future_action",
                description="Future action.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
                read_only=False,
            ),
        )

        def __init__(self):
            self.call_count = 0

        async def call_tool(self, name, arguments):
            del name, arguments
            self.call_count += 1
            raise AssertionError(
                "ACTION must not execute before confirmation."
            )

    class FakeLLM:
        async def chat(self, *, messages, tools=()):
            del messages, tools

            return LLMResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        call_id="call-1",
                        name="future_action",
                        arguments={
                            "employee_id": "E001",
                        },
                    ),
                ),
            )

    async def exercise() -> None:
        mcp = FakeMCP()

        result = await run_turn(
            message="Perform the action.",
            mcp_client=mcp,
            llm=FakeLLM(),
        )

        assert mcp.call_count == 0
        assert result.pending_confirmation is not None
        assert result.trace[-1].decision == (
            "confirmation_required"
        )

    asyncio.run(exercise())


def test_pending_confirmation_contains_preview_id_tool_and_arguments() -> None:
    """The pending object binds the exact previewed action."""

    from agent.orchestrator import (
        _create_pending_confirmation,
    )

    pending = _create_pending_confirmation(
        tool="future_action",
        arguments={
            "employee_id": "E001",
            "days": 3,
        },
    )

    assert pending.confirmation_id
    assert pending.tool == "future_action"
    assert pending.arguments == {
        "employee_id": "E001",
        "days": 3,
    }
    assert "future_action" in pending.preview
    assert '"days": 3' in pending.preview


def test_pending_confirmation_snapshots_original_arguments() -> None:
    """Later mutation of the proposal dictionary cannot change the previewed action."""

    from agent.orchestrator import (
        _create_pending_confirmation,
    )

    arguments = {
        "employee_id": "E001",
        "days": 3,
    }

    pending = _create_pending_confirmation(
        tool="future_action",
        arguments=arguments,
    )

    arguments["days"] = 99

    assert pending.arguments["days"] == 3


def test_matching_confirmation_executes_exact_pending_action() -> None:
    """A matching ID executes the stored tool and stored arguments."""

    from types import SimpleNamespace

    from agent.orchestrator import (
        PendingConfirmation,
        confirm_pending_action,
    )

    class FakeMCP:
        status = "connected"
        last_error = None

        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append(
                (
                    name,
                    arguments,
                )
            )

            return SimpleNamespace(
                structuredContent={
                    "status": "MOCK",
                }
            )

    async def exercise() -> None:
        pending = PendingConfirmation(
            confirmation_id="confirm-123",
            tool="future_action",
            arguments={
                "employee_id": "E001",
                "days": 3,
            },
            preview="Preview.",
        )

        mcp = FakeMCP()

        result = await confirm_pending_action(
            pending=pending,
            confirmation_id="confirm-123",
            mcp_client=mcp,
        )

        assert mcp.calls == [
            (
                "future_action",
                {
                    "employee_id": "E001",
                    "days": 3,
                },
            ),
        ]

        assert (
            result.trace[-1].decision
            == "action_executed"
        )
        assert result.pending_confirmation is None

    asyncio.run(exercise())


def test_wrong_confirmation_id_does_not_execute() -> None:
    """A detached or incorrect confirmation cannot fire the action."""

    from agent.orchestrator import (
        PendingConfirmation,
        confirm_pending_action,
    )

    class FakeMCP:
        status = "connected"
        last_error = None

        def __init__(self):
            self.call_count = 0

        async def call_tool(self, name, arguments):
            del name, arguments
            self.call_count += 1
            raise AssertionError(
                "Wrong confirmation ID must not execute."
            )

    async def exercise() -> None:
        pending = PendingConfirmation(
            confirmation_id="confirm-correct",
            tool="future_action",
            arguments={
                "employee_id": "E001",
            },
            preview="Preview.",
        )

        mcp = FakeMCP()

        result = await confirm_pending_action(
            pending=pending,
            confirmation_id="confirm-wrong",
            mcp_client=mcp,
        )

        assert mcp.call_count == 0
        assert (
            result.trace[-1].decision
            == "confirmation_rejected"
        )

    asyncio.run(exercise())


def test_confirmation_api_cannot_replace_pending_arguments() -> None:
    """Confirmation accepts no user-supplied replacement business arguments."""

    import inspect

    from agent.orchestrator import (
        confirm_pending_action,
    )

    parameters = set(
        inspect.signature(
            confirm_pending_action
        ).parameters
    )

    assert parameters == {
        "pending",
        "confirmation_id",
        "mcp_client",
    }


def test_confirmed_execution_does_not_forward_confirmation_fields_to_mcp() -> None:
    """MCP receives business arguments only after confirmation."""

    from types import SimpleNamespace

    from agent.orchestrator import (
        PendingConfirmation,
        confirm_pending_action,
    )

    class FakeMCP:
        status = "connected"
        last_error = None

        async def call_tool(self, name, arguments):
            del name

            assert arguments == {
                "employee_id": "E001",
                "category": "PTO",
                "summary": "Request PTO guidance.",
            }

            forbidden = {
                "confirmed",
                "confirmation_id",
                "conversation_id",
                "pending_confirmation",
                "preview",
            }

            assert forbidden.isdisjoint(
                arguments
            )

            return SimpleNamespace(
                structuredContent={
                    "status": "MOCK",
                }
            )

    async def exercise() -> None:
        pending = PendingConfirmation(
            confirmation_id="confirm-123",
            tool="future_action",
            arguments={
                "employee_id": "E001",
                "category": "PTO",
                "summary": "Request PTO guidance.",
            },
            preview="Preview.",
        )

        result = await confirm_pending_action(
            pending=pending,
            confirmation_id="confirm-123",
            mcp_client=FakeMCP(),
        )

        assert (
            result.trace[-1].decision
            == "action_executed"
        )

    asyncio.run(exercise())


def test_confirmation_executes_bound_snapshot_after_pending_dict_mutation() -> None:
    """Mutation of exposed pending data cannot alter the confirmed action."""

    from types import SimpleNamespace

    from agent.orchestrator import (
        PendingConfirmation,
        confirm_pending_action,
    )

    class FakeMCP:
        status = "connected"
        last_error = None

        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append(
                (
                    name,
                    arguments,
                )
            )

            return SimpleNamespace(
                structuredContent={
                    "status": "MOCK",
                }
            )

    async def exercise() -> None:
        pending = PendingConfirmation(
            confirmation_id="confirm-123",
            tool="future_action",
            arguments={
                "employee_id": "E001",
                "days": 3,
            },
            preview="Preview.",
        )

        # Simulate accidental or hostile mutation after preview creation.
        pending.arguments["days"] = 99

        mcp = FakeMCP()

        result = await confirm_pending_action(
            pending=pending,
            confirmation_id="confirm-123",
            mcp_client=mcp,
        )

        assert mcp.calls == [
            (
                "future_action",
                {
                    "employee_id": "E001",
                    "days": 3,
                },
            ),
        ]

        assert (
            result.trace[-1].decision
            == "action_executed"
        )

    asyncio.run(exercise())
