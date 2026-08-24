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
