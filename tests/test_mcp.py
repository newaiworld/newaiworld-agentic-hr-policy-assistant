"""Tests for the MCP server foundation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import anyio
import pytest
from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)
from mcp.server.fastmcp import FastMCP

from rag.retrieve import (
    PolicySection,
    RetrievalError,
    RetrievalResult,
)


SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "mcp"
    / "server.py"
)

TOOLS_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "mcp"
    / "tools_policy.py"
)

TOOLS_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "mcp"
    / "tools_data.py"
)


CURRENT_COMPLETED_MCP_TOOL_NAMES = (
    "search_policy_documents",
    "get_policy_section",
    "lookup_employee_profile",
    "lookup_benefits_status",
    "check_pto_balance",
    "check_policy_compliance",
    "create_mock_hr_ticket",
    "draft_hr_email",
)

FINAL_REQUIRED_MCP_TOOL_NAMES = (
    "search_policy_documents",
    "get_policy_section",
    "lookup_employee_profile",
    "lookup_benefits_status",
    "check_pto_balance",
    "check_policy_compliance",
    "create_mock_hr_ticket",
    "draft_hr_email",
)


def load_project_mcp_server() -> ModuleType:
    """Load the project MCP server without shadowing the SDK package."""

    spec = importlib.util.spec_from_file_location(
        "project_mcp_server",
        SERVER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Project MCP server module could not be loaded."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_project_tools_policy() -> ModuleType:
    """Load the project policy adapter without shadowing the MCP SDK."""

    spec = importlib.util.spec_from_file_location(
        "project_tools_policy",
        TOOLS_POLICY_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Project policy tools module could not be loaded."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_project_tools_data() -> ModuleType:
    """Load the project data tools without shadowing the MCP SDK."""

    spec = importlib.util.spec_from_file_location(
        "project_tools_data",
        TOOLS_DATA_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Project data tools module could not be loaded."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def test_server_module_exposes_fastmcp_instance() -> None:
    """The project server must expose one configured FastMCP instance."""

    module = load_project_mcp_server()

    assert isinstance(
        module.mcp,
        FastMCP,
    )
    assert (
        module.mcp.name
        == "Agentic HR Policy Assistant"
    )


def test_server_main_runs_explicit_stdio_transport() -> None:
    """The production entry point must use only the frozen stdio transport."""

    module = load_project_mcp_server()

    with patch.object(
        module.mcp,
        "run",
    ) as run_mock:
        result = module.main()

    assert result is None

    run_mock.assert_called_once_with(
        transport="stdio",
    )


def test_server_registers_exactly_completed_tools() -> None:
    """Production server must expose exactly the completed MCP tools."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        assert tuple(
            tool.name
            for tool in tools
        ) == CURRENT_COMPLETED_MCP_TOOL_NAMES

    asyncio.run(
        inspect_tools()
    )


def test_final_required_mcp_tool_contract_matches_frozen_spec() -> None:
    """Final S5 MCP tool contract must match the frozen eight-tool spec."""

    assert len(
        FINAL_REQUIRED_MCP_TOOL_NAMES
    ) == 8

    assert len(
        set(
            FINAL_REQUIRED_MCP_TOOL_NAMES
        )
    ) == 8

    assert (
        FINAL_REQUIRED_MCP_TOOL_NAMES[
            :len(CURRENT_COMPLETED_MCP_TOOL_NAMES)
        ]
        == CURRENT_COMPLETED_MCP_TOOL_NAMES
    )

    assert FINAL_REQUIRED_MCP_TOOL_NAMES == (
        "search_policy_documents",
        "get_policy_section",
        "lookup_employee_profile",
        "lookup_benefits_status",
        "check_pto_balance",
        "check_policy_compliance",
        "create_mock_hr_ticket",
        "draft_hr_email",
    )


def test_search_policy_documents_discovery_is_read_only() -> None:
    """Policy search must advertise its frozen READ classification."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "search_policy_documents" in tool_by_name

        tool = tool_by_name[
            "search_policy_documents"
        ]

        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True

    asyncio.run(
        inspect_tools()
    )


def test_search_policy_documents_discovery_preserves_input_schema() -> None:
    """FastMCP discovery must preserve the frozen public input contract."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "search_policy_documents" in tool_by_name

        schema = tool_by_name[
            "search_policy_documents"
        ].inputSchema

        assert schema["type"] == "object"

        properties = schema["properties"]

        assert properties["query"]["type"] == "string"
        assert properties["k"]["type"] == "integer"
        assert properties["k"]["default"] == 5

        assert schema["required"] == [
            "query",
        ]

    asyncio.run(
        inspect_tools()
    )


def test_get_policy_section_discovery_preserves_read_contract() -> None:
    """Exact-section discovery must preserve READ classification and schema."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "get_policy_section" in tool_by_name

        tool = tool_by_name[
            "get_policy_section"
        ]

        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True

        schema = tool.inputSchema

        assert schema["type"] == "object"

        properties = schema[
            "properties"
        ]

        assert properties[
            "doc_id"
        ][
            "type"
        ] == "string"

        assert properties[
            "section"
        ][
            "type"
        ] == "string"

        assert schema[
            "required"
        ] == [
            "doc_id",
            "section",
        ]

    asyncio.run(
        inspect_tools()
    )


def test_server_registration_uses_existing_get_policy_section_implementation() -> None:
    """Server registration must reuse the existing exact-section composition."""

    server_module = load_project_mcp_server()
    tools_module = load_project_tools_policy()

    assert callable(
        server_module.get_policy_section
    )

    assert (
        server_module.get_policy_section.__name__
        == tools_module.get_policy_section.__name__
    )

    assert (
        server_module.get_policy_section.__code__.co_code
        == tools_module.get_policy_section.__code__.co_code
    )


def test_existing_search_discovery_contract_remains_intact() -> None:
    """Adding exact-section lookup must not change the search tool contract."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        search_tool = tool_by_name[
            "search_policy_documents"
        ]

        assert search_tool.annotations is not None
        assert (
            search_tool.annotations.readOnlyHint
            is True
        )

        schema = search_tool.inputSchema

        assert schema["type"] == "object"

        properties = schema[
            "properties"
        ]

        assert properties[
            "query"
        ][
            "type"
        ] == "string"

        assert properties[
            "k"
        ][
            "type"
        ] == "integer"

        assert properties[
            "k"
        ][
            "default"
        ] == 5

        assert schema[
            "required"
        ] == [
            "query",
        ]

    asyncio.run(
        inspect_tools()
    )


def test_server_registration_uses_existing_policy_search_implementation() -> None:
    """Server registration must reuse the existing policy-search composition."""

    server_module = load_project_mcp_server()
    tools_module = load_project_tools_policy()

    assert callable(
        server_module.search_policy_documents
    )

    assert (
        server_module.search_policy_documents.__name__
        == tools_module.search_policy_documents.__name__
    )

    assert (
        server_module.search_policy_documents.__code__.co_code
        == tools_module.search_policy_documents.__code__.co_code
    )


def test_stdio_client_calls_policy_search_through_mcp(
    tmp_path: Path,
) -> None:
    """A real stdio client must invoke policy search in a subprocess."""

    fixture_server = (
        tmp_path
        / "fixture_mcp_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "R6E-C6 Fixture MCP Server"
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def search_policy_documents(
    query: str,
    k: int = 5,
) -> list[dict[str, str | float]]:
    return [
        {
            "doc_id": "HR-POL-004",
            "title": "Remote and Flexible Work Policy",
            "section": "5.3 International approval",
            "snippet": (
                f"fixture-server-pid:{os.getpid()} "
                f"query={query} k={k}"
            ),
            "score": 0.75,
        }
    ]


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
"""
    )

    async def call_tool_through_stdio() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(fixture_server),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(20):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        "search_policy_documents",
                        arguments={
                            "query": "remote work abroad",
                            "k": 3,
                        },
                    )

                    assert result.isError is False
                    assert isinstance(
                        result.structuredContent,
                        dict,
                    )

                    records = result.structuredContent[
                        "result"
                    ]

                    assert isinstance(
                        records,
                        list,
                    )

                    assert len(records) == 1

                    record = records[0]

                    assert set(record) == {
                        "doc_id",
                        "title",
                        "section",
                        "snippet",
                        "score",
                    }

                    assert record["doc_id"] == "HR-POL-004"

                    assert (
                        record["title"]
                        == "Remote and Flexible Work Policy"
                    )

                    assert (
                        record["section"]
                        == "5.3 International approval"
                    )

                    assert record["score"] == 0.75

                    snippet = record["snippet"]

                    assert isinstance(
                        snippet,
                        str,
                    )

                    assert (
                        "query=remote work abroad"
                        in snippet
                    )

                    assert "k=3" in snippet

                    pid_prefix = (
                        "fixture-server-pid:"
                    )

                    assert snippet.startswith(
                        pid_prefix
                    )

                    server_pid_text = (
                        snippet[
                            len(pid_prefix):
                        ]
                        .split(
                            " ",
                            1,
                        )[0]
                    )

                    server_pid = int(
                        server_pid_text
                    )

                    assert server_pid > 0
                    assert server_pid != os.getpid()

    asyncio.run(
        call_tool_through_stdio()
    )


def test_stdio_client_receives_clean_mcp_error_result(
    tmp_path: Path,
) -> None:
    """Tool validation failures must become clean MCP error results."""

    fixture_server = (
        tmp_path
        / "fixture_mcp_error_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "R6E-C6 Error Fixture MCP Server"
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def search_policy_documents(
    query: str,
    k: int = 5,
) -> list[dict[str, str | float]]:
    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    return []


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
"""
    )

    async def call_invalid_tool() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(fixture_server),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(20):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        "search_policy_documents",
                        arguments={
                            "query": "remote work",
                            "k": 0,
                        },
                    )

                    assert result.isError is True
                    assert result.structuredContent is None
                    assert len(result.content) >= 1

                    error_text = "\n".join(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                        for item in result.content
                    )

                    assert (
                        "k must be positive."
                        in error_text
                    )

                    assert (
                        "Error executing tool "
                        "search_policy_documents"
                        in error_text
                    )

                    assert (
                        "Traceback"
                        not in error_text
                    )

    asyncio.run(
        call_invalid_tool()
    )


def test_stdio_client_session_recovers_after_tool_error(
    tmp_path: Path,
) -> None:
    """The same MCP session must remain usable after a tool error."""

    fixture_server = (
        tmp_path
        / "fixture_mcp_recovery_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "R6E-C6 Recovery Fixture MCP Server"
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def search_policy_documents(
    query: str,
    k: int = 5,
) -> list[dict[str, str | float]]:
    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    return [
        {
            "doc_id": "HR-POL-004",
            "title": "Remote and Flexible Work Policy",
            "section": "5.3 International approval",
            "snippet": (
                f"recovered-server-pid:{os.getpid()} "
                f"query={query} k={k}"
            ),
            "score": 0.75,
        }
    ]


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
"""
    )

    async def verify_recovery() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(fixture_server),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(20):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    error_result = await session.call_tool(
                        "search_policy_documents",
                        arguments={
                            "query": "remote work",
                            "k": 0,
                        },
                    )

                    assert error_result.isError is True
                    assert (
                        error_result.structuredContent
                        is None
                    )

                    valid_result = await session.call_tool(
                        "search_policy_documents",
                        arguments={
                            "query": "remote work abroad",
                            "k": 3,
                        },
                    )

                    assert valid_result.isError is False

                    assert isinstance(
                        valid_result.structuredContent,
                        dict,
                    )

                    records = (
                        valid_result
                        .structuredContent[
                            "result"
                        ]
                    )

                    assert isinstance(
                        records,
                        list,
                    )

                    assert len(records) == 1

                    record = records[0]

                    assert set(record) == {
                        "doc_id",
                        "title",
                        "section",
                        "snippet",
                        "score",
                    }

                    assert (
                        record["doc_id"]
                        == "HR-POL-004"
                    )

                    assert (
                        record["score"]
                        == 0.75
                    )

                    snippet = record["snippet"]

                    assert isinstance(
                        snippet,
                        str,
                    )

                    assert (
                        "query=remote work abroad"
                        in snippet
                    )

                    assert "k=3" in snippet

                    pid_prefix = (
                        "recovered-server-pid:"
                    )

                    assert snippet.startswith(
                        pid_prefix
                    )

                    server_pid_text = (
                        snippet[
                            len(pid_prefix):
                        ]
                        .split(
                            " ",
                            1,
                        )[0]
                    )

                    server_pid = int(
                        server_pid_text
                    )

                    assert server_pid > 0
                    assert server_pid != os.getpid()

    asyncio.run(
        verify_recovery()
    )


def test_stdio_client_calls_get_policy_section_through_mcp(
    tmp_path: Path,
) -> None:
    """A real stdio client must invoke exact-section lookup in a subprocess."""

    fixture_server = (
        tmp_path
        / "fixture_get_policy_section_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "D6 Exact Section Fixture"
)


def get_policy_section(
    doc_id: str,
    section: str,
) -> dict[str, str]:
    return {
        "title": "Remote and Flexible Work Policy",
        "section": "5.3 International approval",
        "text": (
            f"fixture-server-pid:{os.getpid()} "
            f"doc_id={doc_id} "
            f"section={section}"
        ),
    }


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    get_policy_section
)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
"""
    )

    async def call_tool_through_stdio() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(fixture_server),
            ],
        )

        with anyio.fail_after(20):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        "get_policy_section",
                        arguments={
                            "doc_id": "HR-POL-004",
                            "section": (
                                "5.3 International approval"
                            ),
                        },
                    )

                    assert result.isError is False

                    assert result.structuredContent is not None

                    payload = result.structuredContent

                    assert set(payload) == {
                        "title",
                        "section",
                        "text",
                    }

                    assert (
                        payload["title"]
                        == "Remote and Flexible Work Policy"
                    )

                    assert (
                        payload["section"]
                        == "5.3 International approval"
                    )

                    text_value = payload["text"]

                    assert isinstance(
                        text_value,
                        str,
                    )

                    assert "doc_id=HR-POL-004" in text_value

                    assert (
                        "section=5.3 International approval"
                        in text_value
                    )

                    prefix = (
                        "fixture-server-pid:"
                    )

                    assert text_value.startswith(
                        prefix
                    )

                    pid_text = (
                        text_value[
                            len(prefix):
                        ]
                        .split(
                            " ",
                            1,
                        )[0]
                    )

                    server_pid = int(
                        pid_text
                    )

                    assert server_pid > 0
                    assert server_pid != os.getpid()

    asyncio.run(
        call_tool_through_stdio()
    )


def test_stdio_client_get_policy_section_recovers_after_error(
    tmp_path: Path,
) -> None:
    """A handled exact-section MCP error must not poison the session."""

    fixture_server = (
        tmp_path
        / "fixture_get_policy_section_recovery_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "D6 Exact Section Recovery Fixture"
)


def get_policy_section(
    doc_id: str,
    section: str,
) -> dict[str, str]:
    if section == "99.99":
        raise RuntimeError(
            "Policy section not found for document "
            f"'{doc_id}': '{section}'."
        )

    return {
        "title": "Remote and Flexible Work Policy",
        "section": "5.3 International approval",
        "text": (
            f"recovered-server-pid:{os.getpid()} "
            f"doc_id={doc_id}"
        ),
    }


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    get_policy_section
)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
"""
    )

    async def exercise_error_and_recovery() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(fixture_server),
            ],
        )

        with anyio.fail_after(20):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    error_result = await session.call_tool(
                        "get_policy_section",
                        arguments={
                            "doc_id": "HR-POL-004",
                            "section": "99.99",
                        },
                    )

                    assert error_result.isError is True

                    assert (
                        error_result.structuredContent
                        is None
                    )

                    error_text = "\n".join(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                        for item in error_result.content
                    )

                    assert (
                        "Policy section not found"
                        in error_text
                    )

                    assert "Traceback" not in error_text

                    valid_result = await session.call_tool(
                        "get_policy_section",
                        arguments={
                            "doc_id": "HR-POL-004",
                            "section": (
                                "5.3 International approval"
                            ),
                        },
                    )

                    assert valid_result.isError is False

                    assert valid_result.structuredContent is not None

                    payload = valid_result.structuredContent

                    assert set(payload) == {
                        "title",
                        "section",
                        "text",
                    }

                    text_value = payload["text"]

                    assert isinstance(
                        text_value,
                        str,
                    )

                    prefix = (
                        "recovered-server-pid:"
                    )

                    assert text_value.startswith(
                        prefix
                    )

                    pid_text = (
                        text_value[
                            len(prefix):
                        ]
                        .split(
                            " ",
                            1,
                        )[0]
                    )

                    server_pid = int(
                        pid_text
                    )

                    assert server_pid > 0
                    assert server_pid != os.getpid()

    asyncio.run(
        exercise_error_and_recovery()
    )


def test_stdio_client_calls_lookup_employee_profile_through_mcp(
    tmp_path: Path,
) -> None:
    """A real stdio client must invoke employee lookup in a subprocess."""

    fixture_server = (
        tmp_path
        / "fixture_lookup_employee_profile_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "E6 Employee Profile Fixture"
)


def lookup_employee_profile(
    employee_id: str,
) -> dict[str, str | None]:
    return {
        "name": "Alex Rivera",
        "role": "Senior Data Analyst",
        "employment_type": "full_time",
        "location": "SYDNEY_HQ",
        "manager_id": "E010",
        "start_date": (
            f"fixture-server-pid:{os.getpid()} "
            f"employee_id={employee_id} "
            "date=2023-04-17"
        ),
    }


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    lookup_employee_profile
)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
"""
    )

    async def call_tool_through_stdio() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(fixture_server),
            ],
        )

        with anyio.fail_after(20):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        "lookup_employee_profile",
                        arguments={
                            "employee_id": "E001",
                        },
                    )

                    assert result.isError is False
                    assert result.structuredContent is not None

                    payload = result.structuredContent

                    assert set(payload) == {
                        "name",
                        "role",
                        "employment_type",
                        "location",
                        "manager_id",
                        "start_date",
                    }

                    assert payload["name"] == "Alex Rivera"

                    assert (
                        payload["role"]
                        == "Senior Data Analyst"
                    )

                    assert (
                        payload["employment_type"]
                        == "full_time"
                    )

                    assert (
                        payload["location"]
                        == "SYDNEY_HQ"
                    )

                    assert payload["manager_id"] == "E010"

                    start_date = payload["start_date"]

                    assert isinstance(
                        start_date,
                        str,
                    )

                    prefix = (
                        "fixture-server-pid:"
                    )

                    assert start_date.startswith(
                        prefix
                    )

                    assert (
                        "employee_id=E001"
                        in start_date
                    )

                    assert (
                        "date=2023-04-17"
                        in start_date
                    )

                    pid_text = (
                        start_date[
                            len(prefix):
                        ]
                        .split(
                            " ",
                            1,
                        )[0]
                    )

                    server_pid = int(
                        pid_text
                    )

                    assert server_pid > 0
                    assert server_pid != os.getpid()

    asyncio.run(
        call_tool_through_stdio()
    )


def test_stdio_client_calls_check_policy_compliance_through_mcp(
    tmp_path: Path,
) -> None:
    """A real stdio client must invoke compliance through a subprocess."""

    fixture_server = (
        tmp_path
        / "fixture_check_policy_compliance_server.py"
    )

    pid_path = (
        tmp_path
        / "fixture_check_policy_compliance_server.pid"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "F3.6 Compliance Fixture MCP Server"
)

PID_PATH = Path(__file__).with_suffix(".pid")


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def check_policy_compliance(
    topic: str,
    employee_id: str,
) -> dict[str, object]:
    if topic != "remote_work_international":
        raise RuntimeError(
            f"Unsupported compliance topic: {topic!r}."
        )

    if employee_id != "E003":
        raise RuntimeError(
            f"Employee not found: {employee_id!r}."
        )

    return {
        "compliant": False,
        "reasons": [
            (
                "A six-week international remote-work proposal "
                "exceeds the standard 30-calendar-day limit "
                "and requires formal exception review."
            ),
            (
                "International remote work also requires the applicable "
                "approvals, Information Security review, and overseas-access "
                "controls before approval."
            ),
        ],
        "policy_refs": [
            "HR-POL-004 §4.4",
            "HR-POL-004 §8",
            "HR-POL-005 §4.5",
        ],
    }


if __name__ == "__main__":
    PID_PATH.write_text(
        str(os.getpid()),
        encoding="utf-8",
    )

    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def call_tool_through_stdio() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        "check_policy_compliance",
                        arguments={
                            "topic": "remote_work_international",
                            "employee_id": "E003",
                        },
                    )

                    assert result.isError is False
                    assert result.structuredContent is not None

                    payload = result.structuredContent

                    assert tuple(
                        payload
                    ) == (
                        "compliant",
                        "reasons",
                        "policy_refs",
                    )

                    assert payload == {
                        "compliant": False,
                        "reasons": [
                            (
                                "A six-week international remote-work proposal "
                                "exceeds the standard 30-calendar-day limit "
                                "and requires formal exception review."
                            ),
                            (
                                "International remote work also requires the applicable "
                                "approvals, Information Security review, and overseas-access "
                                "controls before approval."
                            ),
                        ],
                        "policy_refs": [
                            "HR-POL-004 §4.4",
                            "HR-POL-004 §8",
                            "HR-POL-005 §4.5",
                        ],
                    }

        assert pid_path.exists()

        server_pid = int(
            pid_path.read_text(
                encoding="utf-8"
            )
        )

        assert server_pid > 0
        assert server_pid != os.getpid()

    asyncio.run(
        call_tool_through_stdio()
    )


def test_stdio_client_check_policy_compliance_recovers_after_error(
    tmp_path: Path,
) -> None:
    """A stdio session must remain usable after a clean compliance error."""

    fixture_server = (
        tmp_path
        / "fixture_check_policy_compliance_recovery_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "F3.6 Compliance Recovery Fixture MCP Server"
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def check_policy_compliance(
    topic: str,
    employee_id: str,
) -> dict[str, object]:
    if topic != "remote_work_international":
        raise RuntimeError(
            f"Unsupported compliance topic: {topic!r}."
        )

    if employee_id != "E003":
        raise RuntimeError(
            f"Employee not found: {employee_id!r}."
        )

    return {
        "compliant": False,
        "reasons": [
            (
                "A six-week international remote-work proposal "
                "exceeds the standard 30-calendar-day limit "
                "and requires formal exception review."
            ),
            (
                "International remote work also requires the applicable "
                "approvals, Information Security review, and overseas-access "
                "controls before approval."
            ),
        ],
        "policy_refs": [
            "HR-POL-004 §4.4",
            "HR-POL-004 §8",
            "HR-POL-005 §4.5",
        ],
    }


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def call_tool_through_stdio() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    error_result = await session.call_tool(
                        "check_policy_compliance",
                        arguments={
                            "topic": "remote_work_international",
                            "employee_id": "E999",
                        },
                    )

                    assert error_result.isError is True

                    assert error_result.structuredContent is None

                    error_text = " ".join(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                        for item in error_result.content
                    )

                    assert (
                        "Employee not found: 'E999'."
                        in error_text
                    )

                    assert "Traceback" not in error_text
                    assert "File \"" not in error_text

                    success_result = await session.call_tool(
                        "check_policy_compliance",
                        arguments={
                            "topic": "remote_work_international",
                            "employee_id": "E003",
                        },
                    )

                    assert success_result.isError is False
                    assert success_result.structuredContent is not None

                    assert success_result.structuredContent == {
                        "compliant": False,
                        "reasons": [
                            (
                                "A six-week international remote-work proposal "
                                "exceeds the standard 30-calendar-day limit "
                                "and requires formal exception review."
                            ),
                            (
                                "International remote work also requires the applicable "
                                "approvals, Information Security review, and overseas-access "
                                "controls before approval."
                            ),
                        ],
                        "policy_refs": [
                            "HR-POL-004 §4.4",
                            "HR-POL-004 §8",
                            "HR-POL-005 §4.5",
                        ],
                    }

    asyncio.run(
        call_tool_through_stdio()
    )


def test_stdio_client_calls_check_pto_balance_through_mcp(
    tmp_path: Path,
) -> None:
    """A real stdio client must invoke PTO balance through a subprocess."""

    fixture_server = (
        tmp_path
        / "fixture_check_pto_balance_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "F2.7 PTO Fixture MCP Server"
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def check_pto_balance(
    employee_id: str,
) -> dict[str, object]:
    if employee_id != "E001":
        raise RuntimeError(
            f"PTO balance record not found for employee: {employee_id!r}."
        )

    return {
        "available_days": 8.0,
        "accrual_rate": 1.6667,
        "next_accrual_date": (
            f"fixture-server-pid:{os.getpid()}"
        ),
    }


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def call_tool_through_stdio() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        "check_pto_balance",
                        arguments={
                            "employee_id": "E001",
                        },
                    )

                    assert result.isError is False
                    assert result.structuredContent is not None

                    payload = result.structuredContent

                    assert set(
                        payload
                    ) == {
                        "available_days",
                        "accrual_rate",
                        "next_accrual_date",
                    }

                    assert (
                        payload[
                            "available_days"
                        ]
                        == 8.0
                    )

                    assert (
                        payload[
                            "accrual_rate"
                        ]
                        == 1.6667
                    )

                    next_accrual_date = payload[
                        "next_accrual_date"
                    ]

                    prefix = (
                        "fixture-server-pid:"
                    )

                    assert isinstance(
                        next_accrual_date,
                        str,
                    )

                    assert next_accrual_date.startswith(
                        prefix
                    )

                    server_pid = int(
                        next_accrual_date[
                            len(prefix):
                        ]
                    )

                    assert server_pid != os.getpid()

    asyncio.run(
        call_tool_through_stdio()
    )


def test_stdio_client_check_pto_balance_recovers_after_error(
    tmp_path: Path,
) -> None:
    """The same MCP session must recover after a PTO lookup error."""

    fixture_server = (
        tmp_path
        / "fixture_check_pto_balance_recovery_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "F2.7 PTO Recovery Fixture MCP Server"
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def check_pto_balance(
    employee_id: str,
) -> dict[str, object]:
    if employee_id != "E001":
        raise RuntimeError(
            f"PTO balance record not found for employee: {employee_id!r}."
        )

    return {
        "available_days": 8.0,
        "accrual_rate": 1.6667,
        "next_accrual_date": (
            f"recovered-server-pid:{os.getpid()}"
        ),
    }


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def exercise_error_and_recovery() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    error_result = await session.call_tool(
                        "check_pto_balance",
                        arguments={
                            "employee_id": "E999",
                        },
                    )

                    assert error_result.isError is True
                    assert (
                        error_result.structuredContent
                        is None
                    )

                    error_text = "\n".join(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                        for item in error_result.content
                    )

                    assert (
                        "PTO balance record not found for employee: 'E999'."
                        in error_text
                    )

                    assert (
                        "Error executing tool "
                        "check_pto_balance"
                        in error_text
                    )

                    assert (
                        "Traceback"
                        not in error_text
                    )

                    valid_result = await session.call_tool(
                        "check_pto_balance",
                        arguments={
                            "employee_id": "E001",
                        },
                    )

                    assert valid_result.isError is False
                    assert (
                        valid_result.structuredContent
                        is not None
                    )

                    payload = valid_result.structuredContent

                    assert set(
                        payload
                    ) == {
                        "available_days",
                        "accrual_rate",
                        "next_accrual_date",
                    }

                    assert (
                        payload[
                            "available_days"
                        ]
                        == 8.0
                    )

                    assert (
                        payload[
                            "accrual_rate"
                        ]
                        == 1.6667
                    )

                    next_accrual_date = payload[
                        "next_accrual_date"
                    ]

                    prefix = (
                        "recovered-server-pid:"
                    )

                    assert isinstance(
                        next_accrual_date,
                        str,
                    )

                    assert next_accrual_date.startswith(
                        prefix
                    )

                    server_pid = int(
                        next_accrual_date[
                            len(prefix):
                        ]
                    )

                    assert server_pid != os.getpid()

    asyncio.run(
        exercise_error_and_recovery()
    )


def test_stdio_client_calls_lookup_benefits_status_through_mcp(
    tmp_path: Path,
) -> None:
    """A real stdio client must invoke benefits lookup in a subprocess."""

    fixture_server = (
        tmp_path
        / "fixture_lookup_benefits_status_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "F1.6 Benefits Fixture MCP Server"
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def lookup_benefits_status(
    employee_id: str,
) -> dict[str, object]:
    if employee_id != "E001":
        raise RuntimeError(
            f"Benefits record not found for employee: {employee_id!r}."
        )

    return {
        "elections": {
            "health_support": (
                f"fixture-server-pid:{os.getpid()}"
            ),
            "professional_development": "enrolled",
            "wellbeing_program": "enrolled",
        },
        "eligibility": "eligible",
        "coverage_start": "2023-06-01",
    }


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def call_tool_through_stdio() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        "lookup_benefits_status",
                        arguments={
                            "employee_id": "E001",
                        },
                    )

                    assert result.isError is False
                    assert result.structuredContent is not None

                    payload = result.structuredContent

                    assert set(
                        payload
                    ) == {
                        "elections",
                        "eligibility",
                        "coverage_start",
                    }

                    assert (
                        payload[
                            "eligibility"
                        ]
                        == "eligible"
                    )

                    assert (
                        payload[
                            "coverage_start"
                        ]
                        == "2023-06-01"
                    )

                    elections = payload[
                        "elections"
                    ]

                    assert isinstance(
                        elections,
                        dict,
                    )

                    assert (
                        elections[
                            "professional_development"
                        ]
                        == "enrolled"
                    )

                    assert (
                        elections[
                            "wellbeing_program"
                        ]
                        == "enrolled"
                    )

                    health_support = elections[
                        "health_support"
                    ]

                    prefix = (
                        "fixture-server-pid:"
                    )

                    assert health_support.startswith(
                        prefix
                    )

                    server_pid = int(
                        health_support[
                            len(prefix):
                        ]
                    )

                    assert server_pid > 0
                    assert server_pid != os.getpid()

    asyncio.run(
        call_tool_through_stdio()
    )


def test_stdio_client_lookup_benefits_status_recovers_after_error(
    tmp_path: Path,
) -> None:
    """The same MCP session must recover after a benefits lookup error."""

    fixture_server = (
        tmp_path
        / "fixture_lookup_benefits_status_recovery_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "F1.6 Benefits Recovery Fixture MCP Server"
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    )
)
def lookup_benefits_status(
    employee_id: str,
) -> dict[str, object]:
    if employee_id != "E001":
        raise RuntimeError(
            f"Benefits record not found for employee: {employee_id!r}."
        )

    return {
        "elections": {
            "health_support": (
                f"recovered-server-pid:{os.getpid()}"
            ),
            "professional_development": "enrolled",
            "wellbeing_program": "enrolled",
        },
        "eligibility": "eligible",
        "coverage_start": "2023-06-01",
    }


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def exercise_error_and_recovery() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    error_result = await session.call_tool(
                        "lookup_benefits_status",
                        arguments={
                            "employee_id": "E999",
                        },
                    )

                    assert error_result.isError is True
                    assert (
                        error_result.structuredContent
                        is None
                    )

                    error_text = "\n".join(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                        for item in error_result.content
                    )

                    assert (
                        "Benefits record not found for employee: 'E999'."
                        in error_text
                    )

                    assert (
                        "Error executing tool "
                        "lookup_benefits_status"
                        in error_text
                    )

                    assert (
                        "Traceback"
                        not in error_text
                    )

                    valid_result = await session.call_tool(
                        "lookup_benefits_status",
                        arguments={
                            "employee_id": "E001",
                        },
                    )

                    assert valid_result.isError is False
                    assert (
                        valid_result.structuredContent
                        is not None
                    )

                    payload = valid_result.structuredContent

                    assert set(
                        payload
                    ) == {
                        "elections",
                        "eligibility",
                        "coverage_start",
                    }

                    assert (
                        payload[
                            "eligibility"
                        ]
                        == "eligible"
                    )

                    assert (
                        payload[
                            "coverage_start"
                        ]
                        == "2023-06-01"
                    )

                    elections = payload[
                        "elections"
                    ]

                    assert isinstance(
                        elections,
                        dict,
                    )

                    assert (
                        elections[
                            "professional_development"
                        ]
                        == "enrolled"
                    )

                    assert (
                        elections[
                            "wellbeing_program"
                        ]
                        == "enrolled"
                    )

                    health_support = elections[
                        "health_support"
                    ]

                    prefix = (
                        "recovered-server-pid:"
                    )

                    assert health_support.startswith(
                        prefix
                    )

                    server_pid = int(
                        health_support[
                            len(prefix):
                        ]
                    )

                    assert server_pid > 0
                    assert server_pid != os.getpid()

    asyncio.run(
        exercise_error_and_recovery()
    )


def test_stdio_client_lookup_employee_profile_recovers_after_error(
    tmp_path: Path,
) -> None:
    """A handled employee MCP error must not poison the client session."""

    fixture_server = (
        tmp_path
        / "fixture_lookup_employee_profile_recovery_server.py"
    )

    fixture_server.write_text(
        """from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP(
    "E6 Employee Profile Recovery Fixture"
)


def lookup_employee_profile(
    employee_id: str,
) -> dict[str, str | None]:
    if employee_id == "E999":
        raise RuntimeError(
            "Employee not found: 'E999'."
        )

    return {
        "name": "Alex Rivera",
        "role": "Senior Data Analyst",
        "employment_type": "full_time",
        "location": "SYDNEY_HQ",
        "manager_id": "E010",
        "start_date": (
            f"recovered-server-pid:{os.getpid()} "
            f"employee_id={employee_id} "
            "date=2023-04-17"
        ),
    }


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    lookup_employee_profile
)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
"""
    )

    async def exercise_error_and_recovery() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(fixture_server),
            ],
        )

        with anyio.fail_after(20):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    error_result = await session.call_tool(
                        "lookup_employee_profile",
                        arguments={
                            "employee_id": "E999",
                        },
                    )

                    assert error_result.isError is True

                    assert (
                        error_result.structuredContent
                        is None
                    )

                    error_text = "\n".join(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                        for item in error_result.content
                    )

                    assert (
                        "Employee not found: 'E999'."
                        in error_text
                    )

                    assert "Traceback" not in error_text

                    valid_result = await session.call_tool(
                        "lookup_employee_profile",
                        arguments={
                            "employee_id": "E001",
                        },
                    )

                    assert valid_result.isError is False
                    assert valid_result.structuredContent is not None

                    payload = valid_result.structuredContent

                    assert set(payload) == {
                        "name",
                        "role",
                        "employment_type",
                        "location",
                        "manager_id",
                        "start_date",
                    }

                    assert payload["name"] == "Alex Rivera"

                    start_date = payload["start_date"]

                    assert isinstance(
                        start_date,
                        str,
                    )

                    prefix = (
                        "recovered-server-pid:"
                    )

                    assert start_date.startswith(
                        prefix
                    )

                    assert (
                        "employee_id=E001"
                        in start_date
                    )

                    pid_text = (
                        start_date[
                            len(prefix):
                        ]
                        .split(
                            " ",
                            1,
                        )[0]
                    )

                    server_pid = int(
                        pid_text
                    )

                    assert server_pid > 0
                    assert server_pid != os.getpid()

    asyncio.run(
        exercise_error_and_recovery()
    )


def test_local_mcp_directory_remains_non_package() -> None:
    """The project MCP directory must not shadow the installed SDK package."""

    local_init = (
        SERVER_PATH.parent
        / "__init__.py"
    )

    assert not local_init.exists()


def make_policy_section(
    *,
    text: str = (
        "Employees must obtain approval before working "
        "internationally."
    ),
) -> PolicySection:
    """Build one valid exact-section domain object for MCP adapter tests."""

    return PolicySection(
        doc_id="HR-POL-004",
        title="Remote and Flexible Work Policy",
        section="5.3 International approval",
        section_path=(
            "Remote and Flexible Work Policy",
            "5. Procedures or Application",
            "5.3 International approval",
        ),
        section_number="5.3",
        text=text,
        source_format="md",
        section_order=17,
    )


def make_retrieval_result(
    *,
    chunk_id: str = "HR-POL-004__0000__abcdef0123456789",
    doc_id: str = "HR-POL-004",
    title: str = "Remote and Flexible Work Policy",
    section: str = "5.3 International approval",
    snippet: str = (
        "International remote work requires written approval."
    ),
    distance: float = 0.25,
    similarity: float = 0.75,
) -> RetrievalResult:
    """Build one valid retrieval-domain result for adapter tests."""

    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        section=section,
        section_path=(
            title,
            section,
        ),
        snippet=snippet,
        source_format="md",
        distance=distance,
        similarity=similarity,
    )


def test_convert_policy_section_projects_frozen_schema() -> None:
    """Exact-section adapter must expose only the frozen MCP fields."""

    module = load_project_tools_policy()
    result = make_policy_section()

    payload = module._convert_policy_section(
        result
    )

    assert payload == {
        "title": result.title,
        "section": result.section,
        "text": result.text,
    }

    assert set(payload) == {
        "title",
        "section",
        "text",
    }


def test_convert_policy_section_rejects_wrong_type() -> None:
    """Exact-section adapter must reject non-PolicySection values."""

    module = load_project_tools_policy()

    with pytest.raises(
        TypeError,
        match="result must be a PolicySection instance",
    ):
        module._convert_policy_section(
            object()
        )


def test_convert_policy_section_preserves_complete_text() -> None:
    """Exact-section MCP projection must not truncate policy text."""

    module = load_project_tools_policy()

    complete_text = (
        "First complete policy paragraph.\n\n"
        "Second complete policy paragraph containing approval details."
    )

    result = make_policy_section(
        text=complete_text,
    )

    payload = module._convert_policy_section(
        result
    )

    assert payload["text"] == complete_text
    assert len(payload["text"]) == len(
        complete_text
    )


def test_convert_policy_section_does_not_mutate_source() -> None:
    """Projection must leave the immutable PolicySection unchanged."""

    module = load_project_tools_policy()
    result = make_policy_section()

    original = (
        result.doc_id,
        result.title,
        result.section,
        result.section_path,
        result.section_number,
        result.text,
        result.source_format,
        result.section_order,
    )

    payload = module._convert_policy_section(
        result
    )

    assert set(payload) == {
        "title",
        "section",
        "text",
    }

    assert (
        result.doc_id,
        result.title,
        result.section,
        result.section_path,
        result.section_number,
        result.text,
        result.source_format,
        result.section_order,
    ) == original


def test_convert_retrieval_results_projects_frozen_schema() -> None:
    """Adapter must expose only the frozen policy-search response fields."""

    module = load_project_tools_policy()

    result = make_retrieval_result()

    payload = module._convert_retrieval_results(
        (
            result,
        )
    )

    assert payload == [
        {
            "doc_id": result.doc_id,
            "title": result.title,
            "section": result.section,
            "snippet": result.snippet,
            "score": result.similarity,
        }
    ]

    assert set(
        payload[0]
    ) == {
        "doc_id",
        "title",
        "section",
        "snippet",
        "score",
    }


def test_convert_retrieval_results_preserves_order() -> None:
    """Adapter must preserve retrieval ranking without re-ranking."""

    module = load_project_tools_policy()

    first = make_retrieval_result()

    second = make_retrieval_result(
        chunk_id="HR-POL-005__0000__abcdef0123456789",
        doc_id="HR-POL-005",
        title="Information Security and Acceptable Use Policy",
        section="4.5 Overseas access controls",
        snippet="Second result.",
        distance=0.35,
        similarity=0.65,
    )

    payload = module._convert_retrieval_results(
        (
            first,
            second,
        )
    )

    assert [
        item["doc_id"]
        for item in payload
    ] == [
        "HR-POL-004",
        "HR-POL-005",
    ]

    assert [
        item["score"]
        for item in payload
    ] == [
        first.similarity,
        second.similarity,
    ]


def test_convert_retrieval_results_accepts_empty_tuple() -> None:
    """Zero retrieval results must become an empty MCP result list."""

    module = load_project_tools_policy()

    result = module._convert_retrieval_results(
        ()
    )

    assert result == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        "results",
    ],
)
def test_convert_retrieval_results_rejects_wrong_container_type(
    value: object,
) -> None:
    """Adapter input must retain the retrieval tuple contract."""

    module = load_project_tools_policy()

    with pytest.raises(
        TypeError,
        match="results must be a tuple",
    ):
        module._convert_retrieval_results(
            value  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "results",
    [
        ("result",),
        (None,),
        ({},),
    ],
)
def test_convert_retrieval_results_rejects_invalid_members(
    results: tuple[object, ...],
) -> None:
    """Every adapter input member must be a RetrievalResult."""

    module = load_project_tools_policy()

    with pytest.raises(
        TypeError,
        match=(
            "results must contain only "
            "RetrievalResult instances"
        ),
    ):
        module._convert_retrieval_results(
            results  # type: ignore[arg-type]
        )


def test_server_bootstraps_repository_root() -> None:
    """Server entry point must make repository modules importable."""

    module = load_project_mcp_server()

    expected_root = (
        Path(__file__).resolve().parents[1]
    )

    assert module.PROJECT_ROOT == expected_root
    assert str(expected_root) in sys.path


def test_server_bootstrap_preserves_official_mcp_sdk() -> None:
    """Repository bootstrap must not replace the installed MCP SDK."""

    load_project_mcp_server()

    spec = importlib.util.find_spec(
        "mcp.server.fastmcp"
    )

    assert spec is not None
    assert spec.origin is not None
    assert "site-packages/mcp/" in str(
        spec.origin
    )


def test_get_policy_section_composes_retrieval_and_adapter() -> None:
    """Exact-section MCP composition must delegate and project only."""

    module = load_project_tools_policy()

    policy_section = make_policy_section()

    expected = {
        "title": policy_section.title,
        "section": policy_section.section,
        "text": policy_section.text,
    }

    with (
        patch.object(
            module,
            "retrieve_policy_section",
            return_value=policy_section,
        ) as retrieve_mock,
        patch.object(
            module,
            "_convert_policy_section",
            return_value=expected,
        ) as convert_mock,
    ):
        result = module.get_policy_section(
            " raw-doc-id ",
            " raw-section ",
        )

    assert result is expected

    retrieve_mock.assert_called_once_with(
        " raw-doc-id ",
        " raw-section ",
    )

    convert_mock.assert_called_once_with(
        policy_section
    )


def test_get_policy_section_returns_real_wf1_section() -> None:
    """WF1 exact lookup must expose the complete remote-work section."""

    module = load_project_tools_policy()

    result = module.get_policy_section(
        "HR-POL-004",
        "5.3 International approval",
    )

    assert set(result) == {
        "title",
        "section",
        "text",
    }

    assert (
        result["title"]
        == "Remote and Flexible Work Policy"
    )

    assert (
        result["section"]
        == "5.3 International approval"
    )

    assert isinstance(
        result["text"],
        str,
    )

    assert result["text"].strip()


def test_get_policy_section_returns_real_wf2_section() -> None:
    """WF2 exact lookup must expose the three-day PTO policy section."""

    module = load_project_tools_policy()

    result = module.get_policy_section(
        "HR-POL-002",
        "9.1 Three-day request with sufficient balance",
    )

    assert set(result) == {
        "title",
        "section",
        "text",
    }

    assert (
        result["title"]
        == "Paid Time Off Policy"
    )

    assert (
        result["section"]
        == "9.1 Three-day request with sufficient balance"
    )

    assert isinstance(
        result["text"],
        str,
    )

    assert result["text"].strip()


def test_get_policy_section_propagates_type_error() -> None:
    """Exact-section MCP composition must not wrap delegated type errors."""

    module = load_project_tools_policy()

    error = TypeError(
        "doc_id must be a string."
    )

    with patch.object(
        module,
        "retrieve_policy_section",
        side_effect=error,
    ):
        with pytest.raises(
            TypeError,
        ) as exc_info:
            module.get_policy_section(
                None,  # type: ignore[arg-type]
                "5.3",
            )

    assert exc_info.value is error


def test_get_policy_section_propagates_value_error() -> None:
    """Exact-section MCP composition must not wrap delegated value errors."""

    module = load_project_tools_policy()

    error = ValueError(
        "section must be a non-empty string."
    )

    with patch.object(
        module,
        "retrieve_policy_section",
        side_effect=error,
    ):
        with pytest.raises(
            ValueError,
        ) as exc_info:
            module.get_policy_section(
                "HR-POL-004",
                "",
            )

    assert exc_info.value is error


def test_get_policy_section_propagates_retrieval_error() -> None:
    """Exact-section MCP composition must preserve retrieval failures."""

    module = load_project_tools_policy()

    error = RetrievalError(
        "Policy section not found."
    )

    with patch.object(
        module,
        "retrieve_policy_section",
        side_effect=error,
    ):
        with pytest.raises(
            RetrievalError,
        ) as exc_info:
            module.get_policy_section(
                "HR-POL-004",
                "99.99",
            )

    assert exc_info.value is error


def test_search_policy_documents_composes_retrieval_and_adapter() -> None:
    """Policy search must compose retrieval with the frozen MCP adapter."""

    module = load_project_tools_policy()

    retrieval_results = (
        make_retrieval_result(),
    )

    expected = [
        {
            "doc_id": "HR-POL-004",
            "title": "Remote and Flexible Work Policy",
            "section": "5.3 International approval",
            "snippet": (
                "International remote work requires written approval."
            ),
            "score": 0.75,
        }
    ]

    with (
        patch.object(
            module,
            "retrieve_policy",
            return_value=retrieval_results,
        ) as retrieve_mock,
        patch.object(
            module,
            "_convert_retrieval_results",
            return_value=expected,
        ) as convert_mock,
    ):
        result = module.search_policy_documents(
            "remote work abroad",
            k=3,
        )

    assert result is expected

    retrieve_mock.assert_called_once_with(
        "remote work abroad",
        k=3,
    )

    convert_mock.assert_called_once_with(
        retrieval_results
    )


def test_search_policy_documents_defaults_to_five_results() -> None:
    """The frozen MCP policy-search contract must retain k=5."""

    module = load_project_tools_policy()

    retrieval_results: tuple[RetrievalResult, ...] = ()

    with patch.object(
        module,
        "retrieve_policy",
        return_value=retrieval_results,
    ) as retrieve_mock:
        result = module.search_policy_documents(
            "annual leave policy"
        )

    assert result == []

    retrieve_mock.assert_called_once_with(
        "annual leave policy",
        k=5,
    )


@pytest.mark.parametrize(
    "error",
    [
        TypeError(
            "query must be a string."
        ),
        ValueError(
            "k must be positive."
        ),
        RetrievalError(
            "index unavailable"
        ),
    ],
)
def test_search_policy_documents_propagates_retrieval_errors(
    error: Exception,
) -> None:
    """Policy search must not wrap delegated retrieval failures."""

    module = load_project_tools_policy()

    with patch.object(
        module,
        "retrieve_policy",
        side_effect=error,
    ):
        with pytest.raises(
            type(error),
        ) as exc_info:
            module.search_policy_documents(
                "remote work",
                k=5,
            )

    assert exc_info.value is error

def test_check_policy_compliance_returns_frozen_e003_result() -> None:
    """Compliance calculation must return the frozen E003 WF1 result."""

    module = load_project_tools_data()

    result = module.check_policy_compliance(
        "remote_work_international",
        "E003",
    )

    assert result == {
        "compliant": False,
        "reasons": [
            (
                "A six-week international remote-work proposal "
                "exceeds the standard 30-calendar-day limit "
                "and requires formal exception review."
            ),
            (
                "International remote work also requires the applicable "
                "approvals, Information Security review, and overseas-access "
                "controls before approval."
            ),
        ],
        "policy_refs": [
            "HR-POL-004 §4.4",
            "HR-POL-004 §8",
            "HR-POL-005 §4.5",
        ],
    }


def test_check_policy_compliance_projects_frozen_schema() -> None:
    """Compliance result must expose exactly the frozen public schema."""

    module = load_project_tools_data()

    result = module.check_policy_compliance(
        "remote_work_international",
        "E003",
    )

    assert tuple(
        result.keys()
    ) == (
        "compliant",
        "reasons",
        "policy_refs",
    )


def test_check_policy_compliance_returns_boolean_compliant() -> None:
    """The frozen compliance decision must be a real boolean."""

    module = load_project_tools_data()

    result = module.check_policy_compliance(
        "remote_work_international",
        "E003",
    )

    assert result["compliant"] is False
    assert isinstance(
        result["compliant"],
        bool,
    )


def test_check_policy_compliance_preserves_frozen_reasons() -> None:
    """Compliance reasons must preserve the frozen deterministic order."""

    module = load_project_tools_data()

    result = module.check_policy_compliance(
        "remote_work_international",
        "E003",
    )

    assert result["reasons"] == [
        (
            "A six-week international remote-work proposal "
            "exceeds the standard 30-calendar-day limit "
            "and requires formal exception review."
        ),
        (
            "International remote work also requires the applicable "
            "approvals, Information Security review, and overseas-access "
            "controls before approval."
        ),
    ]


def test_check_policy_compliance_preserves_frozen_policy_refs() -> None:
    """Compliance policy references must preserve verified source order."""

    module = load_project_tools_data()

    result = module.check_policy_compliance(
        "remote_work_international",
        "E003",
    )

    assert result["policy_refs"] == [
        "HR-POL-004 §4.4",
        "HR-POL-004 §8",
        "HR-POL-005 §4.5",
    ]


def test_check_policy_compliance_returns_fresh_projection() -> None:
    """Repeated compliance calls must return mutation-isolated projections."""

    module = load_project_tools_data()

    first = module.check_policy_compliance(
        "remote_work_international",
        "E003",
    )

    second = module.check_policy_compliance(
        "remote_work_international",
        "E003",
    )

    assert first == second
    assert first is not second
    assert first["reasons"] is not second["reasons"]
    assert first["policy_refs"] is not second["policy_refs"]

    first["reasons"].append(
        "mutated"
    )

    first["policy_refs"].append(
        "HR-POL-999 §1"
    )

    fresh = module.check_policy_compliance(
        "remote_work_international",
        "E003",
    )

    assert fresh["reasons"] == [
        (
            "A six-week international remote-work proposal "
            "exceeds the standard 30-calendar-day limit "
            "and requires formal exception review."
        ),
        (
            "International remote work also requires the applicable "
            "approvals, Information Security review, and overseas-access "
            "controls before approval."
        ),
    ]

    assert fresh["policy_refs"] == [
        "HR-POL-004 §4.4",
        "HR-POL-004 §8",
        "HR-POL-005 §4.5",
    ]


def test_check_policy_compliance_rejects_non_string_topic() -> None:
    """Compliance topic must reject non-string values cleanly."""

    module = load_project_tools_data()

    for value in (
        123,
        None,
        [],
        {},
        1.5,
        True,
    ):
        with pytest.raises(
            TypeError,
        ) as exc_info:
            module.check_policy_compliance(
                value,
                "E003",
            )

        assert str(
            exc_info.value
        ) == (
            "topic must be a string."
        )


def test_check_policy_compliance_rejects_blank_topic() -> None:
    """Compliance topic must reject blank and padded strings."""

    module = load_project_tools_data()

    for value in (
        "",
        " ",
        "   ",
        "\t",
        "\n",
        " remote_work_international",
        "remote_work_international ",
        " remote_work_international ",
    ):
        with pytest.raises(
            ValueError,
        ) as exc_info:
            module.check_policy_compliance(
                value,
                "E003",
            )

        assert str(
            exc_info.value
        ) == (
            "topic must be a non-empty string "
            "without leading or trailing whitespace."
        )


def test_check_policy_compliance_rejects_unsupported_topic() -> None:
    """Unsupported compliance topics must fail deterministically."""

    module = load_project_tools_data()

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module.check_policy_compliance(
            "pto_carryover",
            "E003",
        )

    assert str(
        exc_info.value
    ) == (
        "Unsupported compliance topic: 'pto_carryover'."
    )


def test_check_policy_compliance_rejects_non_string_employee_id() -> None:
    """Compliance employee ID must reject non-string values cleanly."""

    module = load_project_tools_data()

    for value in (
        123,
        None,
        [],
        {},
        1.5,
        True,
    ):
        with pytest.raises(
            TypeError,
        ) as exc_info:
            module.check_policy_compliance(
                "remote_work_international",
                value,
            )

        assert str(
            exc_info.value
        ) == (
            "employee_id must be a string."
        )


def test_check_policy_compliance_rejects_blank_employee_id() -> None:
    """Compliance employee ID must reject blank and padded strings."""

    module = load_project_tools_data()

    for value in (
        "",
        " ",
        "   ",
        "\t",
        "\n",
        " E003",
        "E003 ",
        " E003 ",
    ):
        with pytest.raises(
            ValueError,
        ) as exc_info:
            module.check_policy_compliance(
                "remote_work_international",
                value,
            )

        assert str(
            exc_info.value
        ) == (
            "employee_id must be a non-empty string "
            "without leading or trailing whitespace."
        )


def test_check_policy_compliance_is_case_sensitive() -> None:
    """Compliance employee lookup must preserve exact case sensitivity."""

    module = load_project_tools_data()

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module.check_policy_compliance(
            "remote_work_international",
            "e003",
        )

    assert str(
        exc_info.value
    ) == (
        "Employee not found: 'e003'."
    )


def test_check_policy_compliance_raises_clean_error_for_unknown_employee() -> None:
    """Unknown employee lookup must surface the frozen clean domain error."""

    module = load_project_tools_data()

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module.check_policy_compliance(
            "remote_work_international",
            "E999",
        )

    assert str(
        exc_info.value
    ) == (
        "Employee not found: 'E999'."
    )


def test_check_policy_compliance_has_no_runtime_retrieval_dependency() -> None:
    """Compliance calculation must not hide policy retrieval at runtime."""

    import ast

    source = TOOLS_DATA_PATH.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    target = next(
        (
            node
            for node in tree.body
            if isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name == "check_policy_compliance"
        ),
        None,
    )

    assert target is not None

    forbidden_calls = {
        "retrieve_policy",
        "get_policy_section",
        "get_policy_section_catalogue",
        "search_policy_documents",
    }

    called = set()

    for node in ast.walk(
        target
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = node.func

        if isinstance(
            func,
            ast.Name,
        ):
            called.add(
                func.id
            )

        elif isinstance(
            func,
            ast.Attribute,
        ):
            called.add(
                func.attr
            )

    assert called.isdisjoint(
        forbidden_calls
    )

    forbidden_import_roots = {
        "rag",
        "chromadb",
        "sentence_transformers",
    }

    imported_roots = set()

    for node in tree.body:
        if isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                imported_roots.add(
                    alias.name.split(
                        ".",
                        1,
                    )[0]
                )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            module_name = (
                node.module
                or ""
            )

            imported_roots.add(
                module_name.split(
                    ".",
                    1,
                )[0]
            )

    assert imported_roots.isdisjoint(
        forbidden_import_roots
    )


def test_check_policy_compliance_remains_framework_and_environment_independent() -> None:
    """Structured compliance logic must remain MCP- and environment-independent."""

    import ast

    source = TOOLS_DATA_PATH.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    framework_imports = []

    for node in tree.body:
        if isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                if (
                    alias.name == "mcp"
                    or alias.name.startswith(
                        "mcp."
                    )
                ):
                    framework_imports.append(
                        alias.name
                    )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            module_name = (
                node.module
                or ""
            )

            if (
                module_name == "mcp"
                or module_name.startswith(
                    "mcp."
                )
            ):
                framework_imports.append(
                    module_name
                )

    assert framework_imports == []

    environment_reads = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.Call,
        ):
            func = node.func

            if (
                isinstance(
                    func,
                    ast.Attribute,
                )
                and isinstance(
                    func.value,
                    ast.Name,
                )
                and func.value.id == "os"
                and func.attr == "getenv"
            ):
                environment_reads.append(
                    "os.getenv"
                )

        elif isinstance(
            node,
            ast.Subscript,
        ):
            value = node.value

            if (
                isinstance(
                    value,
                    ast.Attribute,
                )
                and isinstance(
                    value.value,
                    ast.Name,
                )
                and value.value.id == "os"
                and value.attr == "environ"
            ):
                environment_reads.append(
                    "os.environ"
                )

    assert environment_reads == []


def test_check_pto_balance_returns_real_e001_balance() -> None:
    """PTO balance lookup must return the frozen real E001 state."""

    module = load_project_tools_data()

    result = module.check_pto_balance(
        "E001"
    )

    assert result == {
        "available_days": 8.0,
        "accrual_rate": 1.6667,
        "next_accrual_date": "2026-09-01",
    }


def test_check_pto_balance_projects_frozen_schema() -> None:
    """PTO balance lookup must expose exactly the frozen three-field schema."""

    module = load_project_tools_data()

    result = module.check_pto_balance(
        "E001"
    )

    assert list(result) == [
        "available_days",
        "accrual_rate",
        "next_accrual_date",
    ]


def test_check_pto_balance_preserves_part_time_e002_rate() -> None:
    """PTO lookup must preserve the frozen E002 part-time accrual rate."""

    module = load_project_tools_data()

    result = module.check_pto_balance(
        "E002"
    )

    assert result == {
        "available_days": 4.5,
        "accrual_rate": 1.0,
        "next_accrual_date": "2026-09-01",
    }


def test_check_pto_balance_preserves_part_time_e008_rate() -> None:
    """PTO lookup must preserve the frozen E008 part-time accrual rate."""

    module = load_project_tools_data()

    result = module.check_pto_balance(
        "E008"
    )

    assert result == {
        "available_days": 3.0,
        "accrual_rate": 0.6667,
        "next_accrual_date": "2026-09-01",
    }


def test_check_pto_balance_preserves_probation_e005_balance() -> None:
    """PTO lookup must preserve the frozen E005 probation balance."""

    module = load_project_tools_data()

    result = module.check_pto_balance(
        "E005"
    )

    assert result == {
        "available_days": 1.0,
        "accrual_rate": 1.6667,
        "next_accrual_date": "2026-09-01",
    }


def test_check_pto_balance_rejects_non_string_employee_id() -> None:
    """PTO lookup must reject non-string employee identifiers."""

    module = load_project_tools_data()

    for value in (
        123,
        None,
        [],
        {},
        1.5,
        True,
    ):
        with pytest.raises(
            TypeError,
            match=r"^employee_id must be a string\.$",
        ):
            module.check_pto_balance(
                value
            )


def test_check_pto_balance_rejects_blank_employee_id() -> None:
    """PTO lookup must reject blank or padded employee identifiers."""

    module = load_project_tools_data()

    values = (
        "",
        " ",
        "   ",
        "\t",
        "\n",
        " E001",
        "E001 ",
        " E001 ",
        "\tE001",
        "E001\n",
    )

    for value in values:
        with pytest.raises(
            ValueError,
            match=(
                r"^employee_id must be a non-empty string "
                r"without leading or trailing whitespace\.$"
            ),
        ):
            module.check_pto_balance(
                value
            )


def test_check_pto_balance_is_case_sensitive() -> None:
    """PTO balance lookup must preserve case-sensitive employee IDs."""

    module = load_project_tools_data()

    with pytest.raises(
        module.MockDataError,
        match=(
            r"^PTO balance record not found for employee: 'e001'\.$"
        ),
    ):
        module.check_pto_balance(
            "e001"
        )


def test_check_pto_balance_raises_clean_error_for_missing_record() -> None:
    """Missing PTO balance records must raise a clean MockDataError."""

    module = load_project_tools_data()

    for employee_id in (
        "E999",
        "E006",
    ):
        with pytest.raises(
            module.MockDataError,
        ) as exc_info:
            module.check_pto_balance(
                employee_id
            )

        assert str(
            exc_info.value
        ) == (
            "PTO balance record not found for employee: "
            f"{employee_id!r}."
        )


def test_check_pto_balance_returns_fresh_projection() -> None:
    """PTO lookup must return a fresh mutation-isolated projection."""

    module = load_project_tools_data()

    first = module.check_pto_balance(
        "E001"
    )

    second = module.check_pto_balance(
        "E001"
    )

    assert first == second
    assert first is not second

    first[
        "available_days"
    ] = 9999.0

    first[
        "accrual_rate"
    ] = 9999.0

    first[
        "next_accrual_date"
    ] = "2099-12-31"

    fresh = module.check_pto_balance(
        "E001"
    )

    assert fresh == {
        "available_days": 8.0,
        "accrual_rate": 1.6667,
        "next_accrual_date": "2026-09-01",
    }


def test_lookup_benefits_status_returns_real_e001_status() -> None:
    """Benefits lookup must return the frozen real E001 status."""

    module = load_project_tools_data()

    result = module.lookup_benefits_status(
        "E001"
    )

    assert result == {
        "elections": {
            "health_support": "enrolled",
            "professional_development": "enrolled",
            "wellbeing_program": "enrolled",
        },
        "eligibility": "eligible",
        "coverage_start": "2023-06-01",
    }


def test_lookup_benefits_status_projects_frozen_schema() -> None:
    """Benefits lookup must expose exactly the frozen three-field schema."""

    module = load_project_tools_data()

    result = module.lookup_benefits_status(
        "E001"
    )

    assert tuple(
        result
    ) == (
        "elections",
        "eligibility",
        "coverage_start",
    )

    assert "employee_id" not in result


def test_lookup_benefits_status_preserves_pending_e005_status() -> None:
    """Pending benefits state must remain unchanged for E005."""

    module = load_project_tools_data()

    result = module.lookup_benefits_status(
        "E005"
    )

    assert result == {
        "elections": {
            "health_support": "pending",
            "professional_development": "pending",
            "wellbeing_program": "pending",
        },
        "eligibility": "pending",
        "coverage_start": None,
    }


def test_lookup_benefits_status_preserves_ineligible_e006_status() -> None:
    """Ineligible contractor benefits state must remain unchanged."""

    module = load_project_tools_data()

    result = module.lookup_benefits_status(
        "E006"
    )

    assert result == {
        "elections": {
            "health_support": "not_available",
            "professional_development": "not_available",
            "wellbeing_program": "not_available",
        },
        "eligibility": "ineligible",
        "coverage_start": None,
    }


def test_lookup_benefits_status_rejects_non_string_employee_id() -> None:
    """Benefits lookup must reject non-string employee identifiers."""

    module = load_project_tools_data()

    with pytest.raises(
        TypeError,
        match=r"^employee_id must be a string\.$",
    ):
        module.lookup_benefits_status(
            123
        )


def test_lookup_benefits_status_rejects_blank_employee_id() -> None:
    """Benefits lookup must reject blank employee identifiers."""

    module = load_project_tools_data()

    with pytest.raises(
        ValueError,
        match=(
            r"^employee_id must be a non-empty string "
            r"without leading or trailing whitespace\.$"
        ),
    ):
        module.lookup_benefits_status(
            " "
        )


def test_lookup_benefits_status_is_case_sensitive() -> None:
    """Benefits lookup must preserve exact employee-ID matching."""

    module = load_project_tools_data()

    with pytest.raises(
        module.MockDataError,
        match=(
            r"^Benefits record not found for employee: "
            r"'e001'\.$"
        ),
    ):
        module.lookup_benefits_status(
            "e001"
        )


def test_lookup_benefits_status_raises_clean_error_for_unknown_employee() -> None:
    """Unknown employee IDs must raise the frozen clean data error."""

    module = load_project_tools_data()

    with pytest.raises(
        module.MockDataError,
        match=(
            r"^Benefits record not found for employee: "
            r"'E999'\.$"
        ),
    ):
        module.lookup_benefits_status(
            "E999"
        )


def test_lookup_benefits_status_returns_fresh_projection() -> None:
    """Benefits lookup must return fresh top-level and nested dictionaries."""

    module = load_project_tools_data()

    first = module.lookup_benefits_status(
        "E001"
    )

    second = module.lookup_benefits_status(
        "E001"
    )

    assert first == second
    assert first is not second

    assert (
        first["elections"]
        == second["elections"]
    )

    assert (
        first["elections"]
        is not second["elections"]
    )

    first[
        "elections"
    ][
        "health_support"
    ] = "mutated"

    fresh = module.lookup_benefits_status(
        "E001"
    )

    assert (
        fresh[
            "elections"
        ][
            "health_support"
        ]
        == "enrolled"
    )


def test_lookup_employee_profile_projects_frozen_schema() -> None:
    """Employee lookup must expose exactly the frozen six-field schema."""

    module = load_project_tools_data()

    result = module.lookup_employee_profile(
        "E001"
    )

    assert list(result) == [
        "name",
        "role",
        "employment_type",
        "location",
        "manager_id",
        "start_date",
    ]

    assert set(result) == {
        "name",
        "role",
        "employment_type",
        "location",
        "manager_id",
        "start_date",
    }


def test_lookup_employee_profile_returns_real_e001_profile() -> None:
    """WF2 employee E001 must resolve from the real frozen fixture."""

    module = load_project_tools_data()

    assert module.lookup_employee_profile(
        "E001"
    ) == {
        "name": "Alex Rivera",
        "role": "Senior Data Analyst",
        "employment_type": "full_time",
        "location": "SYDNEY_HQ",
        "manager_id": "E010",
        "start_date": "2023-04-17",
    }


def test_lookup_employee_profile_returns_real_e003_profile() -> None:
    """WF1 employee E003 must resolve from the real frozen fixture."""

    module = load_project_tools_data()

    assert module.lookup_employee_profile(
        "E003"
    ) == {
        "name": "Jordan Patel",
        "role": "Machine Learning Engineer",
        "employment_type": "full_time",
        "location": "SYDNEY_HQ",
        "manager_id": "E010",
        "start_date": "2022-11-07",
    }


def test_lookup_employee_profile_preserves_nullable_manager_id() -> None:
    """The top-level employee must preserve manager_id=None."""

    module = load_project_tools_data()

    result = module.lookup_employee_profile(
        "E012"
    )

    assert result["manager_id"] is None


def test_lookup_employee_profile_rejects_non_string_employee_id() -> None:
    """Employee identifiers must be strings."""

    module = load_project_tools_data()

    with pytest.raises(
        TypeError,
        match=r"^employee_id must be a string\.$",
    ):
        module.lookup_employee_profile(
            123
        )


def test_lookup_employee_profile_rejects_blank_employee_id() -> None:
    """Blank and whitespace-surrounded employee IDs must be rejected."""

    module = load_project_tools_data()

    expected = (
        "employee_id must be a non-empty string "
        "without leading or trailing whitespace."
    )

    for employee_id in (
        "",
        "   ",
        " E001",
        "E001 ",
    ):
        with pytest.raises(
            ValueError,
        ) as exc_info:
            module.lookup_employee_profile(
                employee_id
            )

        assert str(
            exc_info.value
        ) == expected


def test_lookup_employee_profile_is_case_sensitive() -> None:
    """Employee IDs must be matched exactly without case normalization."""

    module = load_project_tools_data()

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module.lookup_employee_profile(
            "e001"
        )

    assert str(
        exc_info.value
    ) == (
        "Employee not found: 'e001'."
    )


def test_lookup_employee_profile_raises_clean_error_for_unknown_employee() -> None:
    """Unknown employee IDs must surface the frozen clean data error."""

    module = load_project_tools_data()

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module.lookup_employee_profile(
            "E999"
        )

    assert str(
        exc_info.value
    ) == (
        "Employee not found: 'E999'."
    )


def test_lookup_employee_profile_returns_fresh_projection() -> None:
    """Caller mutation must not affect later employee-profile lookups."""

    module = load_project_tools_data()

    first = module.lookup_employee_profile(
        "E001"
    )

    first["name"] = "MUTATED"

    second = module.lookup_employee_profile(
        "E001"
    )

    assert first is not second

    assert second["name"] == (
        "Alex Rivera"
    )

def test_employee_data_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Missing employee data must raise the frozen data-domain error."""

    module = load_project_tools_data()

    missing_path = (
        tmp_path
        / "missing-employees.json"
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_employee_index(
            missing_path
        )

    assert str(
        exc_info.value
    ).startswith(
        "Employee data file not found:"
    )


def test_employee_data_loader_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    """Malformed employee JSON must raise the frozen data-domain error."""

    module = load_project_tools_data()

    malformed_path = (
        tmp_path
        / "malformed-employees.json"
    )

    malformed_path.write_text(
        "{ definitely not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_employee_index(
            malformed_path
        )

    assert str(
        exc_info.value
    ).startswith(
        "Employee data file is not valid JSON:"
    )


def test_employee_data_loader_rejects_duplicate_employee_ids(
    tmp_path: Path,
) -> None:
    """Duplicate employee IDs must fail deterministically."""

    module = load_project_tools_data()

    duplicate_path = (
        tmp_path
        / "duplicate-employees.json"
    )

    employee = {
        "employee_id": "E001",
        "name": "Alex Rivera",
        "role": "Senior Data Analyst",
        "employment_type": "full_time",
        "location": "SYDNEY_HQ",
        "manager_id": "E010",
        "start_date": "2023-04-17",
    }

    duplicate_path.write_text(
        json.dumps(
            {
                "employees": [
                    employee,
                    dict(
                        employee
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_employee_index(
            duplicate_path
        )

    assert str(
        exc_info.value
    ) == (
        "Duplicate employee ID: 'E001'."
    )

def test_pto_fixture_full_time_accrual_matches_frozen_policy_rate() -> None:
    """Stored full-time PTO accrual must match the frozen 20-day rule."""

    module = load_project_tools_data()

    employees = module._load_employee_index()
    pto = module._load_pto_index()

    expected_monthly_rate = round(
        20 / 12,
        4,
    )

    full_time_ids = sorted(
        employee_id
        for employee_id, employee in employees.items()
        if (
            employee[
                "employment_type"
            ] == "full_time"
            and employee_id in pto
        )
    )

    assert full_time_ids

    for employee_id in full_time_ids:
        assert (
            pto[
                employee_id
            ][
                "accrual_rate"
            ]
            == expected_monthly_rate
        )


def test_pto_fixture_part_time_accrual_matches_recorded_fte() -> None:
    """Stored part-time PTO accrual must match the recorded FTE."""

    module = load_project_tools_data()

    employees = module._load_employee_index()
    pto = module._load_pto_index()

    full_time_monthly_rate = (
        20 / 12
    )

    part_time_ids = sorted(
        employee_id
        for employee_id, employee in employees.items()
        if employee[
            "employment_type"
        ] == "part_time"
    )

    assert part_time_ids

    for employee_id in part_time_ids:
        employee = employees[
            employee_id
        ]

        assert employee_id in pto

        expected_rate = round(
            full_time_monthly_rate
            * float(
                employee[
                    "fte"
                ]
            ),
            4,
        )

        assert (
            pto[
                employee_id
            ][
                "accrual_rate"
            ]
            == expected_rate
        )


def test_pto_fixture_contractor_absence_matches_frozen_policy_rule() -> None:
    """Contractors without a written PTO grant must have no PTO balance."""

    module = load_project_tools_data()

    employees = module._load_employee_index()
    pto = module._load_pto_index()

    contractor_ids = {
        employee_id
        for employee_id, employee in employees.items()
        if employee[
            "employment_type"
        ] == "contractor"
    }

    assert contractor_ids == {
        "E006",
    }

    assert contractor_ids.isdisjoint(
        pto
    )


def test_pto_data_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """PTO loader must fail cleanly when the fixture is missing."""

    module = load_project_tools_data()

    missing_path = (
        tmp_path
        / "missing-pto.json"
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_pto_index(
            missing_path
        )

    assert str(
        exc_info.value
    ) == (
        "PTO data file not found: "
        f"{str(missing_path)!r}."
    )


def test_pto_data_loader_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    """PTO loader must fail cleanly when the fixture JSON is malformed."""

    module = load_project_tools_data()

    malformed_path = (
        tmp_path
        / "malformed-pto.json"
    )

    malformed_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_pto_index(
            malformed_path
        )

    assert str(
        exc_info.value
    ) == (
        "PTO data file is not valid JSON: "
        f"{str(malformed_path)!r}."
    )


def test_pto_data_loader_rejects_duplicate_employee_ids(
    tmp_path: Path,
) -> None:
    """PTO loader must reject duplicate employee IDs deterministically."""

    module = load_project_tools_data()

    duplicate_path = (
        tmp_path
        / "duplicate-pto.json"
    )

    duplicate_path.write_text(
        """{
  "schema_version": "1.0",
  "as_of_date": "2026-08-05",
  "balances": [
    {
      "employee_id": "E001",
      "available_days": 8.0,
      "accrual_rate": 1.6667,
      "accrual_unit": "days_per_month",
      "last_updated": "2026-08-05",
      "next_accrual_date": "2026-09-01"
    },
    {
      "employee_id": "E001",
      "available_days": 9.0,
      "accrual_rate": 1.6667,
      "accrual_unit": "days_per_month",
      "last_updated": "2026-08-05",
      "next_accrual_date": "2026-09-01"
    }
  ]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_pto_index(
            duplicate_path
        )

    assert str(
        exc_info.value
    ) == (
        "Duplicate PTO employee ID: 'E001'."
    )


def test_benefits_data_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Missing benefits data must raise the frozen data-domain error."""

    module = load_project_tools_data()

    missing_path = (
        tmp_path
        / "missing-benefits.json"
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_benefits_index(
            missing_path
        )

    assert str(
        exc_info.value
    ).startswith(
        "Benefits data file not found:"
    )


def test_benefits_data_loader_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    """Malformed benefits JSON must raise the frozen data-domain error."""

    module = load_project_tools_data()

    malformed_path = (
        tmp_path
        / "malformed-benefits.json"
    )

    malformed_path.write_text(
        "{ definitely not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_benefits_index(
            malformed_path
        )

    assert str(
        exc_info.value
    ).startswith(
        "Benefits data file is not valid JSON:"
    )


def test_benefits_data_loader_rejects_duplicate_employee_ids(
    tmp_path: Path,
) -> None:
    """Duplicate benefits employee IDs must fail deterministically."""

    module = load_project_tools_data()

    duplicate_path = (
        tmp_path
        / "duplicate-benefits.json"
    )

    benefit = {
        "employee_id": "E001",
        "elections": {
            "health_support": "enrolled",
            "professional_development": "enrolled",
            "wellbeing_program": "enrolled",
        },
        "eligibility": "eligible",
        "coverage_start": "2023-06-01",
    }

    duplicate_path.write_text(
        json.dumps(
            {
                "benefits": [
                    benefit,
                    {
                        "employee_id": benefit[
                            "employee_id"
                        ],
                        "elections": dict(
                            benefit[
                                "elections"
                            ]
                        ),
                        "eligibility": benefit[
                            "eligibility"
                        ],
                        "coverage_start": benefit[
                            "coverage_start"
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_benefits_index(
            duplicate_path
        )

    assert str(
        exc_info.value
    ) == (
        "Duplicate benefits employee ID: 'E001'."
    )


def test_check_policy_compliance_discovery_preserves_calculation_contract() -> None:
    """Compliance discovery must preserve the frozen calculation contract."""

    server_module = load_project_mcp_server()

    async def inspect() -> None:
        tools = await server_module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "check_policy_compliance" in tool_by_name

        tool = tool_by_name[
            "check_policy_compliance"
        ]

        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True

        schema = tool.inputSchema

        assert schema["type"] == "object"

        properties = schema[
            "properties"
        ]

        assert set(
            properties
        ) == {
            "topic",
            "employee_id",
        }

        assert properties[
            "topic"
        ][
            "type"
        ] == "string"

        assert properties[
            "employee_id"
        ][
            "type"
        ] == "string"

        assert set(
            schema[
                "required"
            ]
        ) == {
            "topic",
            "employee_id",
        }

    asyncio.run(
        inspect()
    )


def test_server_registration_uses_existing_check_policy_compliance_implementation() -> None:
    """Server registration must reuse the existing compliance implementation."""

    server_module = load_project_mcp_server()
    data_module = load_project_tools_data()

    assert callable(
        server_module.check_policy_compliance
    )

    assert (
        server_module.check_policy_compliance.__name__
        == data_module.check_policy_compliance.__name__
    )

    assert (
        server_module.check_policy_compliance.__code__.co_code
        == data_module.check_policy_compliance.__code__.co_code
    )


def test_check_pto_balance_discovery_preserves_calculation_contract() -> None:
    """PTO discovery must preserve read-only calculation classification and schema."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "check_pto_balance" in tool_by_name

        tool = tool_by_name[
            "check_pto_balance"
        ]

        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True

        schema = tool.inputSchema

        assert schema[
            "type"
        ] == "object"

        properties = schema[
            "properties"
        ]

        assert properties[
            "employee_id"
        ][
            "type"
        ] == "string"

        assert schema[
            "required"
        ] == [
            "employee_id",
        ]

    asyncio.run(
        inspect_tools()
    )


def test_server_registration_uses_existing_check_pto_balance_implementation() -> None:
    """Server registration must reuse the framework-agnostic PTO implementation."""

    server_module = load_project_mcp_server()
    data_module = load_project_tools_data()

    assert callable(
        server_module.check_pto_balance
    )

    assert (
        server_module.check_pto_balance.__name__
        == data_module.check_pto_balance.__name__
    )

    assert (
        server_module.check_pto_balance.__code__.co_code
        == data_module.check_pto_balance.__code__.co_code
    )


def test_lookup_benefits_status_discovery_preserves_read_contract() -> None:
    """Benefits discovery must preserve READ classification and schema."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "lookup_benefits_status" in tool_by_name

        tool = tool_by_name[
            "lookup_benefits_status"
        ]

        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True

        schema = tool.inputSchema

        assert schema[
            "type"
        ] == "object"

        properties = schema[
            "properties"
        ]

        assert properties[
            "employee_id"
        ][
            "type"
        ] == "string"

        assert schema[
            "required"
        ] == [
            "employee_id",
        ]

    asyncio.run(
        inspect_tools()
    )


def test_server_registration_uses_existing_lookup_benefits_status_implementation() -> None:
    """Server registration must reuse the framework-agnostic benefits implementation."""

    server_module = load_project_mcp_server()
    data_module = load_project_tools_data()

    assert callable(
        server_module.lookup_benefits_status
    )

    assert (
        server_module.lookup_benefits_status.__name__
        == data_module.lookup_benefits_status.__name__
    )

    assert (
        server_module.lookup_benefits_status.__code__.co_code
        == data_module.lookup_benefits_status.__code__.co_code
    )


def test_lookup_employee_profile_discovery_preserves_read_contract() -> None:
    """Employee-profile discovery must preserve READ classification and schema."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "lookup_employee_profile" in tool_by_name

        tool = tool_by_name[
            "lookup_employee_profile"
        ]

        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True

        schema = tool.inputSchema

        assert schema["type"] == "object"

        properties = schema[
            "properties"
        ]

        assert properties[
            "employee_id"
        ][
            "type"
        ] == "string"

        assert schema[
            "required"
        ] == [
            "employee_id",
        ]

    asyncio.run(
        inspect_tools()
    )


def test_server_registration_uses_existing_lookup_employee_profile_implementation() -> None:
    """Server registration must reuse the framework-agnostic data implementation."""

    server_module = load_project_mcp_server()
    data_module = load_project_tools_data()

    assert callable(
        server_module.lookup_employee_profile
    )

    assert (
        server_module.lookup_employee_profile.__name__
        == data_module.lookup_employee_profile.__name__
    )

    assert (
        server_module.lookup_employee_profile.__code__.co_code
        == data_module.lookup_employee_profile.__code__.co_code
    )


def test_stdio_client_calls_draft_hr_email_through_mcp(
    tmp_path: Path,
) -> None:
    """Real stdio must execute the production email-draft ACTION."""

    fixture_server = (
        tmp_path
        / "fixture_draft_hr_email_server.py"
    )

    fixture_server.write_text(
        f"""from __future__ import annotations

import importlib.util

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp.types import ToolAnnotations


TOOLS_DATA_PATH = Path(
    {str(TOOLS_DATA_PATH)!r}
)


spec = importlib.util.spec_from_file_location(
    "project_mcp_tools_data_f5_stdio_success",
    TOOLS_DATA_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Could not load production tools_data.py."
    )

tools_data = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    tools_data
)


draft_hr_email = (
    tools_data.draft_hr_email
)


mcp = FastMCP(
    "R6E-F5 ACTION Success Fixture"
)


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
    ),
)(
    draft_hr_email
)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def call_action() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    tools = await session.list_tools()

                    tool_by_name = {
                        tool.name: tool
                        for tool in tools.tools
                    }

                    assert (
                        "draft_hr_email"
                        in tool_by_name
                    )

                    action = tool_by_name[
                        "draft_hr_email"
                    ]

                    assert (
                        action.annotations
                        is not None
                    )

                    assert (
                        action.annotations.readOnlyHint
                        is False
                    )

                    schema = action.inputSchema

                    assert schema[
                        "type"
                    ] == "object"

                    assert set(
                        schema[
                            "properties"
                        ]
                    ) == {
                        "to_role",
                        "subject",
                        "context",
                    }

                    assert set(
                        schema[
                            "required"
                        ]
                    ) == {
                        "to_role",
                        "subject",
                        "context",
                    }

                    result = await session.call_tool(
                        "draft_hr_email",
                        arguments={
                            "to_role": "manager",
                            "subject": "PTO request",
                            "context": (
                                "Request approval for three days "
                                "of PTO next week."
                            ),
                        },
                    )

                    assert result.isError is False

                    assert result.structuredContent == {
                        "draft_text": (
                            "To: manager\n"
                            "Subject: PTO request\n"
                            "\n"
                            "Request approval for three days "
                            "of PTO next week."
                        ),
                        "note": "MOCK — not sent",
                    }

    asyncio.run(
        call_action()
    )


def test_stdio_client_calls_create_mock_hr_ticket_through_mcp(
    tmp_path: Path,
) -> None:
    """Real stdio must execute the production ACTION against isolated state."""

    production_tickets_path = (
        Path(__file__).resolve().parents[1]
        / "mock_data"
        / "tickets.json"
    )

    production_employees_path = (
        Path(__file__).resolve().parents[1]
        / "mock_data"
        / "employees.json"
    )

    isolated_mock_data = (
        tmp_path
        / "mock_data"
    )

    isolated_mock_data.mkdir(
        parents=True
    )

    isolated_tickets_path = (
        isolated_mock_data
        / "tickets.json"
    )

    isolated_employees_path = (
        isolated_mock_data
        / "employees.json"
    )

    isolated_tickets_path.write_bytes(
        production_tickets_path.read_bytes()
    )

    isolated_employees_path.write_bytes(
        production_employees_path.read_bytes()
    )

    production_ticket_bytes = (
        production_tickets_path.read_bytes()
    )

    fixture_server = (
        tmp_path
        / "fixture_create_mock_hr_ticket_server.py"
    )

    fixture_server.write_text(
        f"""from __future__ import annotations

import importlib.util
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


TOOLS_DATA_PATH = Path(
    {str(TOOLS_DATA_PATH)!r}
)

ISOLATED_TICKETS_PATH = Path(
    {str(isolated_tickets_path)!r}
)

ISOLATED_EMPLOYEES_PATH = Path(
    {str(isolated_employees_path)!r}
)


spec = importlib.util.spec_from_file_location(
    "project_mcp_tools_data_f4_stdio_success",
    TOOLS_DATA_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Could not load production tools_data.py."
    )

tools_data = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    tools_data
)


_original_load_ticket_state = (
    tools_data._load_ticket_state
)

_original_write_ticket_state = (
    tools_data._write_ticket_state
)

_original_load_employee_index = (
    tools_data._load_employee_index
)


def _load_isolated_ticket_state():
    return _original_load_ticket_state(
        ISOLATED_TICKETS_PATH
    )


def _write_isolated_ticket_state(
    state,
):
    return _original_write_ticket_state(
        state,
        ISOLATED_TICKETS_PATH,
    )


def _load_isolated_employee_index():
    return _original_load_employee_index(
        ISOLATED_EMPLOYEES_PATH
    )


tools_data._load_ticket_state = (
    _load_isolated_ticket_state
)

tools_data._write_ticket_state = (
    _write_isolated_ticket_state
)

tools_data._load_employee_index = (
    _load_isolated_employee_index
)


create_mock_hr_ticket = (
    tools_data.create_mock_hr_ticket
)


mcp = FastMCP(
    "R6E-F4.7 ACTION Success Fixture"
)


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
    ),
)(
    create_mock_hr_ticket
)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    initial_state = json.loads(
        isolated_tickets_path.read_text(
            encoding="utf-8"
        )
    )

    initial_records = [
        dict(
            ticket
        )
        for ticket in initial_state[
            "tickets"
        ]
    ]

    assert len(
        initial_records
    ) == 4

    assert initial_state[
        "next_ticket_number"
    ] == 1005

    async def call_action() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    tools = await session.list_tools()

                    tool_by_name = {
                        tool.name: tool
                        for tool in tools.tools
                    }

                    action = tool_by_name[
                        "create_mock_hr_ticket"
                    ]

                    assert (
                        action.annotations
                        is not None
                    )

                    assert (
                        action.annotations.readOnlyHint
                        is False
                    )

                    result = await session.call_tool(
                        "create_mock_hr_ticket",
                        arguments={
                            "employee_id": "E001",
                            "category": "PTO",
                            "summary": (
                                "Request for 3 days "
                                "of PTO next week."
                            ),
                        },
                    )

                    assert result.isError is False

                    assert result.structuredContent == {
                        "ticket_id": "TKT-1005",
                        "status": "MOCK",
                    }

    asyncio.run(
        call_action()
    )

    persisted = json.loads(
        isolated_tickets_path.read_text(
            encoding="utf-8"
        )
    )

    assert persisted[
        "next_ticket_number"
    ] == 1006

    assert len(
        persisted[
            "tickets"
        ]
    ) == 5

    assert persisted[
        "tickets"
    ][:-1] == initial_records

    new_ticket = persisted[
        "tickets"
    ][-1]

    assert new_ticket[
        "ticket_id"
    ] == "TKT-1005"

    assert new_ticket[
        "employee_id"
    ] == "E001"

    assert new_ticket[
        "category"
    ] == "PTO"

    assert new_ticket[
        "status"
    ] == "open"

    assert new_ticket[
        "mock"
    ] is True

    assert new_ticket[
        "created_at"
    ].endswith(
        "+00:00"
    )

    assert (
        production_tickets_path.read_bytes()
        == production_ticket_bytes
    )



def test_stdio_client_draft_hr_email_recovers_after_error(
    tmp_path: Path,
) -> None:
    """The same stdio session must recover after a clean draft ACTION error."""

    fixture_server = (
        tmp_path
        / "fixture_draft_hr_email_recovery_server.py"
    )

    fixture_server.write_text(
        f"""from __future__ import annotations

import importlib.util

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp.types import ToolAnnotations


TOOLS_DATA_PATH = Path(
    {str(TOOLS_DATA_PATH)!r}
)


spec = importlib.util.spec_from_file_location(
    "project_mcp_tools_data_f5_stdio_recovery",
    TOOLS_DATA_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Could not load production tools_data.py."
    )

tools_data = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    tools_data
)


draft_hr_email = (
    tools_data.draft_hr_email
)

lookup_employee_profile = (
    tools_data.lookup_employee_profile
)


mcp = FastMCP(
    "R6E-F5 ACTION Recovery Fixture"
)


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
    ),
)(
    draft_hr_email
)


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    lookup_employee_profile
)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def exercise_error_and_recovery() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    error_result = await session.call_tool(
                        "draft_hr_email",
                        arguments={
                            "to_role": "manager",
                            "subject": "PTO request",
                            "context": "",
                        },
                    )

                    assert error_result.isError is True

                    assert (
                        error_result.structuredContent
                        is None
                    )

                    error_text = "\n".join(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                        for item in error_result.content
                    )

                    assert (
                        "context must be a non-empty string "
                        "without leading or trailing whitespace."
                        in error_text
                    )

                    assert (
                        "Error executing tool "
                        "draft_hr_email"
                        in error_text
                    )

                    assert (
                        "Traceback"
                        not in error_text
                    )

                    valid_result = await session.call_tool(
                        "lookup_employee_profile",
                        arguments={
                            "employee_id": "E001",
                        },
                    )

                    assert valid_result.isError is False

                    assert (
                        valid_result.structuredContent
                        is not None
                    )

                    payload = (
                        valid_result.structuredContent
                    )

                    assert set(
                        payload
                    ) == {
                        "name",
                        "role",
                        "employment_type",
                        "location",
                        "manager_id",
                        "start_date",
                    }

                    assert payload[
                        "name"
                    ] == "Alex Rivera"

    asyncio.run(
        exercise_error_and_recovery()
    )


def test_stdio_client_create_mock_hr_ticket_recovers_after_error(
    tmp_path: Path,
) -> None:
    """The same stdio session must recover after a clean ACTION error."""

    production_tickets_path = (
        Path(__file__).resolve().parents[1]
        / "mock_data"
        / "tickets.json"
    )

    production_employees_path = (
        Path(__file__).resolve().parents[1]
        / "mock_data"
        / "employees.json"
    )

    isolated_mock_data = (
        tmp_path
        / "mock_data"
    )

    isolated_mock_data.mkdir(
        parents=True
    )

    isolated_tickets_path = (
        isolated_mock_data
        / "tickets.json"
    )

    isolated_employees_path = (
        isolated_mock_data
        / "employees.json"
    )

    isolated_tickets_path.write_bytes(
        production_tickets_path.read_bytes()
    )

    isolated_employees_path.write_bytes(
        production_employees_path.read_bytes()
    )

    production_ticket_bytes = (
        production_tickets_path.read_bytes()
    )

    initial_isolated_bytes = (
        isolated_tickets_path.read_bytes()
    )

    fixture_server = (
        tmp_path
        / "fixture_create_mock_hr_ticket_recovery_server.py"
    )

    fixture_server.write_text(
        f"""from __future__ import annotations

import importlib.util
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


TOOLS_DATA_PATH = Path(
    {str(TOOLS_DATA_PATH)!r}
)

ISOLATED_TICKETS_PATH = Path(
    {str(isolated_tickets_path)!r}
)

ISOLATED_EMPLOYEES_PATH = Path(
    {str(isolated_employees_path)!r}
)


spec = importlib.util.spec_from_file_location(
    "project_mcp_tools_data_f4_stdio_recovery",
    TOOLS_DATA_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Could not load production tools_data.py."
    )

tools_data = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    tools_data
)


_original_load_ticket_state = (
    tools_data._load_ticket_state
)

_original_write_ticket_state = (
    tools_data._write_ticket_state
)

_original_load_employee_index = (
    tools_data._load_employee_index
)


def _load_isolated_ticket_state():
    return _original_load_ticket_state(
        ISOLATED_TICKETS_PATH
    )


def _write_isolated_ticket_state(
    state,
):
    return _original_write_ticket_state(
        state,
        ISOLATED_TICKETS_PATH,
    )


def _load_isolated_employee_index():
    return _original_load_employee_index(
        ISOLATED_EMPLOYEES_PATH
    )


tools_data._load_ticket_state = (
    _load_isolated_ticket_state
)

tools_data._write_ticket_state = (
    _write_isolated_ticket_state
)

tools_data._load_employee_index = (
    _load_isolated_employee_index
)


create_mock_hr_ticket = (
    tools_data.create_mock_hr_ticket
)

lookup_employee_profile = (
    tools_data.lookup_employee_profile
)


mcp = FastMCP(
    "R6E-F4.7 ACTION Recovery Fixture"
)


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
    ),
)(
    create_mock_hr_ticket
)


mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
    ),
)(
    lookup_employee_profile
)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
""",
        encoding="utf-8",
    )

    async def exercise_error_and_recovery() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                str(
                    fixture_server
                ),
            ],
            cwd=tmp_path,
        )

        with anyio.fail_after(
            20
        ):
            async with stdio_client(
                server
            ) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=10
                    ),
                ) as session:
                    await session.initialize()

                    error_result = await session.call_tool(
                        "create_mock_hr_ticket",
                        arguments={
                            "employee_id": "E001",
                            "category": "PTO",
                            "summary": "",
                        },
                    )

                    assert error_result.isError is True

                    assert (
                        error_result.structuredContent
                        is None
                    )

                    error_text = "\n".join(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                        for item in error_result.content
                    )

                    assert (
                        "summary must be a non-empty string "
                        "without leading or trailing whitespace."
                        in error_text
                    )

                    assert (
                        "Error executing tool "
                        "create_mock_hr_ticket"
                        in error_text
                    )

                    assert (
                        "Traceback"
                        not in error_text
                    )

                    # Failed ACTION validation must not mutate ticket state.
                    assert (
                        isolated_tickets_path.read_bytes()
                        == initial_isolated_bytes
                    )

                    valid_result = await session.call_tool(
                        "lookup_employee_profile",
                        arguments={
                            "employee_id": "E001",
                        },
                    )

                    assert valid_result.isError is False

                    assert (
                        valid_result.structuredContent
                        is not None
                    )

                    payload = (
                        valid_result.structuredContent
                    )

                    assert set(
                        payload
                    ) == {
                        "name",
                        "role",
                        "employment_type",
                        "location",
                        "manager_id",
                        "start_date",
                    }

                    assert payload[
                        "name"
                    ] == "Alex Rivera"

    asyncio.run(
        exercise_error_and_recovery()
    )

    # The failed ACTION must never have changed isolated persistence.
    assert (
        isolated_tickets_path.read_bytes()
        == initial_isolated_bytes
    )

    # The committed repository fixture must also remain untouched.
    assert (
        production_tickets_path.read_bytes()
        == production_ticket_bytes
    )


def test_ticket_data_loader_accepts_real_fixture() -> None:
    """The frozen production ticket fixture must validate successfully."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    assert state[
        "schema_version"
    ] == "1.0"

    assert state[
        "next_ticket_number"
    ] == 1005

    assert len(
        state[
            "tickets"
        ]
    ) == 4

    assert [
        ticket[
            "ticket_id"
        ]
        for ticket in state[
            "tickets"
        ]
    ] == [
        "TKT-1001",
        "TKT-1002",
        "TKT-1003",
        "TKT-1004",
    ]

    assert all(
        ticket[
            "mock"
        ] is True
        for ticket in state[
            "tickets"
        ]
    )


def test_ticket_data_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Ticket loader must fail cleanly when the fixture is missing."""

    module = load_project_tools_data()

    missing_path = (
        tmp_path
        / "missing-tickets.json"
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_ticket_state(
            missing_path
        )

    assert str(
        exc_info.value
    ) == (
        "Ticket data file not found: "
        f"{str(missing_path)!r}."
    )


def test_ticket_data_loader_rejects_malformed_json(
    tmp_path: Path,
) -> None:
    """Malformed ticket JSON must raise the frozen data-domain error."""

    module = load_project_tools_data()

    malformed_path = (
        tmp_path
        / "malformed-tickets.json"
    )

    malformed_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_ticket_state(
            malformed_path
        )

    assert str(
        exc_info.value
    ) == (
        "Ticket data file is not valid JSON: "
        f"{str(malformed_path)!r}."
    )


def test_ticket_data_loader_rejects_invalid_top_level_schema(
    tmp_path: Path,
) -> None:
    """Ticket loader must reject incomplete top-level state."""

    module = load_project_tools_data()

    invalid_path = (
        tmp_path
        / "invalid-ticket-schema.json"
    )

    invalid_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tickets": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_ticket_state(
            invalid_path
        )

    assert str(
        exc_info.value
    ) == (
        "Ticket data must contain exactly "
        "'schema_version', 'next_ticket_number', and 'tickets'."
    )


def test_ticket_data_loader_rejects_duplicate_ticket_ids(
    tmp_path: Path,
) -> None:
    """Duplicate ticket IDs must fail deterministically."""

    module = load_project_tools_data()

    duplicate_path = (
        tmp_path
        / "duplicate-ticket-ids.json"
    )

    ticket = {
        "ticket_id": "TKT-1001",
        "employee_id": "E001",
        "category": "PTO",
        "summary": "Mock PTO request.",
        "status": "open",
        "created_at": "2026-08-21T00:00:00+00:00",
        "mock": True,
    }

    duplicate_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "next_ticket_number": 1002,
                "tickets": [
                    ticket,
                    dict(
                        ticket
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_ticket_state(
            duplicate_path
        )

    assert str(
        exc_info.value
    ) == (
        "Duplicate ticket ID: 'TKT-1001'."
    )


def test_ticket_data_loader_rejects_inconsistent_next_ticket_number(
    tmp_path: Path,
) -> None:
    """Ticket allocator state must agree with existing ticket IDs."""

    module = load_project_tools_data()

    inconsistent_path = (
        tmp_path
        / "inconsistent-next-ticket.json"
    )

    inconsistent_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "next_ticket_number": 1007,
                "tickets": [
                    {
                        "ticket_id": "TKT-1004",
                        "employee_id": "E009",
                        "category": "GENERAL_HR",
                        "summary": (
                            "Request for guidance on updating "
                            "an employment record."
                        ),
                        "status": "open",
                        "created_at": "2026-08-01T11:45:00+10:00",
                        "mock": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._load_ticket_state(
            inconsistent_path
        )

    assert str(
        exc_info.value
    ) == (
        "Ticket data field 'next_ticket_number' "
        "is inconsistent with existing ticket IDs."
    )



def test_ticket_state_transition_creates_real_e001_pto_ticket() -> None:
    """Pure transition must create the frozen WF2 ticket in memory."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    new_state, result = (
        module._build_ticket_state_transition(
            state,
            "E001",
            "PTO",
            "Request for 3 days of PTO next week.",
            "2026-08-21T02:30:00+00:00",
        )
    )

    assert new_state[
        "tickets"
    ][-1] == {
        "ticket_id": "TKT-1005",
        "employee_id": "E001",
        "category": "PTO",
        "summary": "Request for 3 days of PTO next week.",
        "status": "open",
        "created_at": "2026-08-21T02:30:00+00:00",
        "mock": True,
    }

    assert result[
        "ticket_id"
    ] == "TKT-1005"


def test_ticket_state_transition_returns_frozen_public_result() -> None:
    """Public ACTION result must remain separate from persisted lifecycle state."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    _, result = (
        module._build_ticket_state_transition(
            state,
            "E001",
            "PTO",
            "Request for 3 days of PTO next week.",
            "2026-08-21T02:30:00+00:00",
        )
    )

    assert result == {
        "ticket_id": "TKT-1005",
        "status": "MOCK",
    }

    assert set(
        result
    ) == {
        "ticket_id",
        "status",
    }


def test_ticket_state_transition_preserves_original_state() -> None:
    """Pure transition must not mutate its input state."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    original_next_number = state[
        "next_ticket_number"
    ]

    original_tickets = [
        dict(
            ticket
        )
        for ticket in state[
            "tickets"
        ]
    ]

    new_state, _ = (
        module._build_ticket_state_transition(
            state,
            "E001",
            "PTO",
            "Request for 3 days of PTO next week.",
            "2026-08-21T02:30:00+00:00",
        )
    )

    assert state[
        "next_ticket_number"
    ] == original_next_number

    assert state[
        "tickets"
    ] == original_tickets

    assert new_state is not state
    assert new_state[
        "tickets"
    ] is not state[
        "tickets"
    ]


def test_ticket_state_transition_increments_allocator() -> None:
    """One successful transition must consume exactly one ticket number."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    new_state, _ = (
        module._build_ticket_state_transition(
            state,
            "E001",
            "PTO",
            "Request for 3 days of PTO next week.",
            "2026-08-21T02:30:00+00:00",
        )
    )

    assert state[
        "next_ticket_number"
    ] == 1005

    assert new_state[
        "next_ticket_number"
    ] == 1006


def test_ticket_state_transition_persists_open_mock_ticket() -> None:
    """New persisted ticket state must use open lifecycle and mock marker."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    new_state, _ = (
        module._build_ticket_state_transition(
            state,
            "E001",
            "PTO",
            "Request for 3 days of PTO next week.",
            "2026-08-21T02:30:00+00:00",
        )
    )

    ticket = new_state[
        "tickets"
    ][-1]

    assert ticket[
        "status"
    ] == "open"

    assert ticket[
        "mock"
    ] is True


def test_ticket_state_transition_allocates_sequential_ids() -> None:
    """Repeated in-memory transitions must allocate sequential ticket IDs."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    first_state, first_result = (
        module._build_ticket_state_transition(
            state,
            "E001",
            "PTO",
            "First mock PTO request.",
            "2026-08-21T02:30:00+00:00",
        )
    )

    second_state, second_result = (
        module._build_ticket_state_transition(
            first_state,
            "E001",
            "PTO",
            "Second mock PTO request.",
            "2026-08-21T02:31:00+00:00",
        )
    )

    assert first_result[
        "ticket_id"
    ] == "TKT-1005"

    assert second_result[
        "ticket_id"
    ] == "TKT-1006"

    assert second_state[
        "next_ticket_number"
    ] == 1007

    assert len(
        second_state[
            "tickets"
        ]
    ) == 6


@pytest.mark.parametrize(
    (
        "field_name",
        "value",
        "expected_message",
    ),
    [
        (
            "employee_id",
            "",
            "employee_id must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "employee_id",
            " E001",
            "employee_id must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "category",
            "",
            "category must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "summary",
            "   ",
            "summary must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "created_at",
            "",
            "created_at must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
    ],
)
def test_ticket_state_transition_rejects_invalid_inputs(
    field_name: str,
    value: str,
    expected_message: str,
) -> None:
    """Transition must reject invalid action inputs before changing state."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    arguments = {
        "employee_id": "E001",
        "category": "PTO",
        "summary": "Request for 3 days of PTO next week.",
        "created_at": "2026-08-21T02:30:00+00:00",
    }

    arguments[
        field_name
    ] = value

    with pytest.raises(
        ValueError,
    ) as exc_info:
        module._build_ticket_state_transition(
            state,
            arguments[
                "employee_id"
            ],
            arguments[
                "category"
            ],
            arguments[
                "summary"
            ],
            arguments[
                "created_at"
            ],
        )

    assert str(
        exc_info.value
    ) == expected_message

    assert state[
        "next_ticket_number"
    ] == 1005

    assert len(
        state[
            "tickets"
        ]
    ) == 4


def test_ticket_state_transition_rejects_unknown_employee() -> None:
    """Unknown employee must fail without changing ticket state."""

    module = load_project_tools_data()

    state = module._load_ticket_state()

    with pytest.raises(
        module.MockDataError,
    ) as exc_info:
        module._build_ticket_state_transition(
            state,
            "E999",
            "PTO",
            "Request for 3 days of PTO next week.",
            "2026-08-21T02:30:00+00:00",
        )

    assert str(
        exc_info.value
    ) == (
        "Employee not found: 'E999'."
    )

    assert state[
        "next_ticket_number"
    ] == 1005

    assert len(
        state[
            "tickets"
        ]
    ) == 4



def test_ticket_state_writer_publishes_and_replaces_atomically(
    tmp_path: Path,
) -> None:
    """Writer must publish complete append-only state and replace safely."""

    module = load_project_tools_data()

    target = (
        tmp_path
        / "tickets.json"
    )

    temporary_path = target.with_name(
        f".{target.name}.tmp"
    )

    original_state = module._load_ticket_state()

    original_records = [
        dict(
            ticket
        )
        for ticket in original_state[
            "tickets"
        ]
    ]

    first_state, first_result = (
        module._build_ticket_state_transition(
            original_state,
            "E001",
            "PTO",
            "First atomic writer request.",
            "2026-08-21T03:00:00+00:00",
        )
    )

    module._write_ticket_state(
        first_state,
        target,
    )

    assert target.is_file()
    assert not temporary_path.exists()

    first_persisted = module._load_ticket_state(
        target
    )

    assert first_persisted == first_state

    assert first_result == {
        "ticket_id": "TKT-1005",
        "status": "MOCK",
    }

    assert first_persisted[
        "next_ticket_number"
    ] == 1006

    assert len(
        first_persisted[
            "tickets"
        ]
    ) == (
        len(
            original_records
        )
        + 1
    )

    # Logical append-only invariant:
    # every pre-existing ticket remains exactly unchanged.
    assert first_persisted[
        "tickets"
    ][:-1] == original_records

    assert first_persisted[
        "tickets"
    ][-1][
        "ticket_id"
    ] == "TKT-1005"

    first_bytes = target.read_bytes()

    second_state, second_result = (
        module._build_ticket_state_transition(
            first_persisted,
            "E001",
            "PTO",
            "Second atomic writer request.",
            "2026-08-21T03:01:00+00:00",
        )
    )

    module._write_ticket_state(
        second_state,
        target,
    )

    assert target.is_file()
    assert not temporary_path.exists()

    second_bytes = target.read_bytes()

    second_persisted = module._load_ticket_state(
        target
    )

    assert second_bytes != first_bytes

    assert second_persisted == second_state

    assert second_result == {
        "ticket_id": "TKT-1006",
        "status": "MOCK",
    }

    assert second_persisted[
        "next_ticket_number"
    ] == 1007

    assert len(
        second_persisted[
            "tickets"
        ]
    ) == (
        len(
            first_persisted[
                "tickets"
            ]
        )
        + 1
    )

    # The second replacement must preserve the complete first state.
    assert second_persisted[
        "tickets"
    ][:-1] == first_persisted[
        "tickets"
    ]

    assert second_persisted[
        "tickets"
    ][-1][
        "ticket_id"
    ] == "TKT-1006"


@pytest.mark.parametrize(
    (
        "case_name",
        "expected_exception_name",
        "expected_message",
    ),
    [
        (
            "invalid_path_type",
            "TypeError",
            "path must be a Path instance.",
        ),
        (
            "incomplete_state",
            "MockDataError",
            "Ticket data must contain exactly "
            "'schema_version', 'next_ticket_number', and 'tickets'.",
        ),
        (
            "invalid_allocator_type",
            "MockDataError",
            "Ticket data field 'next_ticket_number' "
            "must be a positive integer.",
        ),
        (
            "inconsistent_allocator",
            "MockDataError",
            "Ticket data field 'next_ticket_number' "
            "is inconsistent with existing ticket IDs.",
        ),
    ],
)
def test_ticket_state_writer_rejects_invalid_inputs_before_write(
    tmp_path: Path,
    case_name: str,
    expected_exception_name: str,
    expected_message: str,
) -> None:
    """Invalid writer input must fail before filesystem publication."""

    module = load_project_tools_data()

    target = (
        tmp_path
        / "tickets.json"
    )

    temporary_path = target.with_name(
        f".{target.name}.tmp"
    )

    valid_state = module._load_ticket_state()

    state = {
        "schema_version": valid_state[
            "schema_version"
        ],
        "next_ticket_number": valid_state[
            "next_ticket_number"
        ],
        "tickets": [
            dict(
                ticket
            )
            for ticket in valid_state[
                "tickets"
            ]
        ],
    }

    path_argument: object = target

    if case_name == "invalid_path_type":
        path_argument = str(
            target
        )

    elif case_name == "incomplete_state":
        state = {
            "schema_version": "1.0",
            "next_ticket_number": 1005,
        }

    elif case_name == "invalid_allocator_type":
        state[
            "next_ticket_number"
        ] = "1005"

    elif case_name == "inconsistent_allocator":
        state[
            "next_ticket_number"
        ] = 999

    else:
        raise AssertionError(
            f"Unhandled test case: {case_name!r}."
        )

    if expected_exception_name == "TypeError":
        expected_exception = TypeError

    elif expected_exception_name == "MockDataError":
        expected_exception = module.MockDataError

    else:
        raise AssertionError(
            "Unsupported expected exception: "
            f"{expected_exception_name!r}."
        )

    with pytest.raises(
        expected_exception,
    ) as exc_info:
        module._write_ticket_state(
            state,
            path_argument,
        )

    assert str(
        exc_info.value
    ) == expected_message

    # Validation failures must occur before any filesystem side effect.
    assert not target.exists()
    assert not temporary_path.exists()


def test_ticket_state_writer_replace_failure_preserves_target_and_cleans_temp(
    tmp_path: Path,
) -> None:
    """Failed atomic replace must preserve target bytes and remove temp state."""

    module = load_project_tools_data()

    target = (
        tmp_path
        / "tickets.json"
    )

    temporary_path = target.with_name(
        f".{target.name}.tmp"
    )

    original_state = module._load_ticket_state()

    module._write_ticket_state(
        original_state,
        target,
    )

    assert target.is_file()
    assert not temporary_path.exists()

    before_bytes = target.read_bytes()

    new_state, _ = (
        module._build_ticket_state_transition(
            original_state,
            "E001",
            "PTO",
            "Atomic replace failure probe.",
            "2026-08-21T03:02:00+00:00",
        )
    )

    with patch.object(
        module.os,
        "replace",
        side_effect=OSError(
            "simulated atomic replace failure"
        ),
    ) as replace_mock:
        with pytest.raises(
            module.MockDataError,
        ) as exc_info:
            module._write_ticket_state(
                new_state,
                target,
            )

    assert replace_mock.call_count == 1

    assert str(
        exc_info.value
    ) == (
        "Ticket data could not be written safely: "
        f"{str(target)!r}."
    )

    # Publication failure must leave the authoritative target unchanged.
    assert target.read_bytes() == before_bytes

    # The failed temporary publication must not leak residual state.
    assert not temporary_path.exists()

    persisted = module._load_ticket_state(
        target
    )

    assert persisted == original_state



def test_create_mock_hr_ticket_composes_frozen_action_contract() -> None:
    """Public ACTION must compose real state logic with frozen UTC time."""

    module = load_project_tools_data()

    frozen_created_at = (
        "2026-08-21T05:00:00+00:00"
    )

    captured_state: dict[str, object] = {}

    def capture_write(
        state: dict[str, object],
    ) -> None:
        captured_state.update(
            state
        )

    with (
        patch.object(
            module,
            "_utc_now_iso",
            return_value=frozen_created_at,
        ) as clock_mock,
        patch.object(
            module,
            "_write_ticket_state",
            side_effect=capture_write,
        ) as writer_mock,
    ):
        result = module.create_mock_hr_ticket(
            "E001",
            "PTO",
            "Request for 3 days of PTO next week.",
        )

    assert result == {
        "ticket_id": "TKT-1005",
        "status": "MOCK",
    }

    assert set(
        result
    ) == {
        "ticket_id",
        "status",
    }

    clock_mock.assert_called_once_with()
    writer_mock.assert_called_once()

    assert captured_state[
        "schema_version"
    ] == "1.0"

    assert captured_state[
        "next_ticket_number"
    ] == 1006

    tickets = captured_state[
        "tickets"
    ]

    assert isinstance(
        tickets,
        list,
    )

    assert len(
        tickets
    ) == 5

    assert tickets[
        -1
    ] == {
        "ticket_id": "TKT-1005",
        "employee_id": "E001",
        "category": "PTO",
        "summary": "Request for 3 days of PTO next week.",
        "status": "open",
        "created_at": frozen_created_at,
        "mock": True,
    }


@pytest.mark.parametrize(
    (
        "employee_id",
        "category",
        "summary",
        "expected_message",
    ),
    [
        (
            "",
            "PTO",
            "Valid summary.",
            "employee_id must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "E001",
            " PTO",
            "Valid summary.",
            "category must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "E001",
            "PTO",
            "   ",
            "summary must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
    ],
)
def test_create_mock_hr_ticket_rejects_invalid_public_inputs_before_action(
    employee_id: str,
    category: str,
    summary: str,
    expected_message: str,
) -> None:
    """Invalid public arguments must fail before loading or writing state."""

    module = load_project_tools_data()

    with (
        patch.object(
            module,
            "_load_ticket_state",
        ) as loader_mock,
        patch.object(
            module,
            "_utc_now_iso",
        ) as clock_mock,
        patch.object(
            module,
            "_write_ticket_state",
        ) as writer_mock,
    ):
        with pytest.raises(
            ValueError,
        ) as exc_info:
            module.create_mock_hr_ticket(
                employee_id,
                category,
                summary,
            )

    assert str(
        exc_info.value
    ) == expected_message

    loader_mock.assert_not_called()
    clock_mock.assert_not_called()
    writer_mock.assert_not_called()


def test_create_mock_hr_ticket_propagates_writer_failure_without_success(
) -> None:
    """Writer failure must propagate and prevent a successful ACTION result."""

    module = load_project_tools_data()

    frozen_created_at = (
        "2026-08-21T05:01:00+00:00"
    )

    writer_error = module.MockDataError(
        "simulated ticket persistence failure"
    )

    with (
        patch.object(
            module,
            "_utc_now_iso",
            return_value=frozen_created_at,
        ) as clock_mock,
        patch.object(
            module,
            "_write_ticket_state",
            side_effect=writer_error,
        ) as writer_mock,
    ):
        with pytest.raises(
            module.MockDataError,
        ) as exc_info:
            module.create_mock_hr_ticket(
                "E001",
                "PTO",
                "Request for 3 days of PTO next week.",
            )

    assert str(
        exc_info.value
    ) == (
        "simulated ticket persistence failure"
    )

    clock_mock.assert_called_once_with()
    writer_mock.assert_called_once()

    written_state = writer_mock.call_args.args[
        0
    ]

    assert written_state[
        "next_ticket_number"
    ] == 1006

    assert written_state[
        "tickets"
    ][-1][
        "ticket_id"
    ] == "TKT-1005"

    assert written_state[
        "tickets"
    ][-1][
        "created_at"
    ] == frozen_created_at





def test_draft_hr_email_returns_frozen_mock_draft_contract() -> None:
    """Draft ACTION must return the exact deterministic mock contract."""

    module = load_project_tools_data()

    result = module.draft_hr_email(
        "manager",
        "PTO request",
        "Request approval for three days of PTO next week.",
    )

    assert result == {
        "draft_text": (
            "To: manager\n"
            "Subject: PTO request\n"
            "\n"
            "Request approval for three days of PTO next week."
        ),
        "note": "MOCK — not sent",
    }

    assert set(
        result
    ) == {
        "draft_text",
        "note",
    }


def test_draft_hr_email_preserves_exact_valid_inputs() -> None:
    """Valid draft inputs must not be normalized or rewritten."""

    module = load_project_tools_data()

    result = module.draft_hr_email(
        "HR Business Partner",
        "Remote Work — Exception Review",
        "Please review E003's six-week proposal.",
    )

    assert result[
        "draft_text"
    ] == (
        "To: HR Business Partner\n"
        "Subject: Remote Work — Exception Review\n"
        "\n"
        "Please review E003's six-week proposal."
    )

    assert result[
        "note"
    ] == "MOCK — not sent"


@pytest.mark.parametrize(
    (
        "to_role",
        "subject",
        "context",
        "expected_exception",
        "expected_message",
    ),
    [
        (
            None,
            "Subject",
            "Context",
            TypeError,
            "to_role must be a string.",
        ),
        (
            "",
            "Subject",
            "Context",
            ValueError,
            "to_role must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            " manager",
            "Subject",
            "Context",
            ValueError,
            "to_role must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "manager",
            None,
            "Context",
            TypeError,
            "subject must be a string.",
        ),
        (
            "manager",
            "   ",
            "Context",
            ValueError,
            "subject must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "manager",
            "Subject ",
            "Context",
            ValueError,
            "subject must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "manager",
            "Subject",
            None,
            TypeError,
            "context must be a string.",
        ),
        (
            "manager",
            "Subject",
            "",
            ValueError,
            "context must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
        (
            "manager",
            "Subject",
            " Context",
            ValueError,
            "context must be a non-empty string "
            "without leading or trailing whitespace.",
        ),
    ],
)
def test_draft_hr_email_rejects_invalid_public_inputs(
    to_role: object,
    subject: object,
    context: object,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """Invalid public draft arguments must fail with exact clean errors."""

    module = load_project_tools_data()

    with pytest.raises(
        expected_exception,
    ) as exc_info:
        module.draft_hr_email(
            to_role,
            subject,
            context,
        )

    assert str(
        exc_info.value
    ) == expected_message


def test_draft_hr_email_discovery_preserves_action_contract() -> None:
    """Email drafting discovery must preserve ACTION classification and schema."""

    server_module = load_project_mcp_server()

    async def inspect() -> None:
        tools = await server_module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "draft_hr_email" in tool_by_name

        tool = tool_by_name[
            "draft_hr_email"
        ]

        assert tool.annotations is not None

        assert (
            tool.annotations.readOnlyHint
            is False
        )

        schema = tool.inputSchema

        assert schema[
            "type"
        ] == "object"

        properties = schema[
            "properties"
        ]

        assert set(
            properties
        ) == {
            "to_role",
            "subject",
            "context",
        }

        assert properties[
            "to_role"
        ][
            "type"
        ] == "string"

        assert properties[
            "subject"
        ][
            "type"
        ] == "string"

        assert properties[
            "context"
        ][
            "type"
        ] == "string"

        assert set(
            schema[
                "required"
            ]
        ) == {
            "to_role",
            "subject",
            "context",
        }

    asyncio.run(
        inspect()
    )


def test_server_registration_uses_existing_draft_hr_email_implementation(
) -> None:
    """Server registration must reuse the framework-agnostic draft implementation."""

    server_module = load_project_mcp_server()
    data_module = load_project_tools_data()

    assert callable(
        server_module.draft_hr_email
    )

    assert (
        server_module.draft_hr_email.__name__
        == data_module.draft_hr_email.__name__
    )

    assert (
        server_module.draft_hr_email.__code__.co_code
        == data_module.draft_hr_email.__code__.co_code
    )


def test_create_mock_hr_ticket_discovery_preserves_action_contract() -> None:
    """Ticket creation discovery must preserve ACTION classification and schema."""

    server_module = load_project_mcp_server()

    async def inspect() -> None:
        tools = await server_module.mcp.list_tools()

        tool_by_name = {
            tool.name: tool
            for tool in tools
        }

        assert "create_mock_hr_ticket" in tool_by_name

        tool = tool_by_name[
            "create_mock_hr_ticket"
        ]

        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False

        schema = tool.inputSchema

        assert schema[
            "type"
        ] == "object"

        properties = schema[
            "properties"
        ]

        assert set(
            properties
        ) == {
            "employee_id",
            "category",
            "summary",
        }

        assert properties[
            "employee_id"
        ][
            "type"
        ] == "string"

        assert properties[
            "category"
        ][
            "type"
        ] == "string"

        assert properties[
            "summary"
        ][
            "type"
        ] == "string"

        assert set(
            schema[
                "required"
            ]
        ) == {
            "employee_id",
            "category",
            "summary",
        }

    asyncio.run(
        inspect()
    )


def test_server_registration_uses_existing_create_mock_hr_ticket_implementation(
) -> None:
    """Server registration must reuse the framework-agnostic ACTION implementation."""

    server_module = load_project_mcp_server()

    data_module = load_project_tools_data()

    assert callable(
        server_module.create_mock_hr_ticket
    )

    assert (
        server_module.create_mock_hr_ticket.__name__
        == data_module.create_mock_hr_ticket.__name__
    )

    assert (
        server_module.create_mock_hr_ticket.__code__.co_code
        == data_module.create_mock_hr_ticket.__code__.co_code
    )


def test_check_policy_compliance_discovery_constrains_frozen_topic() -> None:
    """Discovery exposes only the frozen compliance topic to tool callers."""

    import asyncio

    from mcp import server as _unused_mcp_server  # noqa: F401
    from mcp import types as _unused_mcp_types  # noqa: F401

    import mcp.server as _unused_server_package  # noqa: F401
    import mcp.server.fastmcp as _unused_fastmcp_package  # noqa: F401

    from mcp.server.fastmcp import FastMCP  # noqa: F401

    import importlib.util
    from pathlib import Path

    server_path = Path(__file__).resolve().parents[1] / "mcp" / "server.py"
    spec = importlib.util.spec_from_file_location(
        "project_mcp_server_schema_contract",
        server_path,
    )
    assert spec is not None
    assert spec.loader is not None

    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)

    async def exercise() -> None:
        tools = await server_module.mcp.list_tools()
        tool_by_name = {tool.name: tool for tool in tools}

        assert "check_policy_compliance" in tool_by_name

        schema = tool_by_name["check_policy_compliance"].inputSchema
        topic_schema = schema["properties"]["topic"]

        assert topic_schema.get("type") == "string"
        assert topic_schema.get("const") == (
            "remote_work_international"
        )

    asyncio.run(exercise())
