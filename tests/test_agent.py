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
