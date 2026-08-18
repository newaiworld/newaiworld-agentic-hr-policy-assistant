"""Tests for the MCP server foundation."""

from __future__ import annotations

import asyncio
import importlib.util
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


def test_server_registers_exactly_search_policy_documents() -> None:
    """Production server must expose only the completed policy-search tool."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        assert [
            tool.name
            for tool in tools
        ] == [
            "search_policy_documents",
        ]

    asyncio.run(
        inspect_tools()
    )


def test_search_policy_documents_discovery_is_read_only() -> None:
    """Policy search must advertise its frozen READ classification."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        assert len(tools) == 1

        tool = tools[0]

        assert tool.name == "search_policy_documents"
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

        assert len(tools) == 1

        schema = tools[0].inputSchema

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


def test_local_mcp_directory_remains_non_package() -> None:
    """The project MCP directory must not shadow the installed SDK package."""

    local_init = (
        SERVER_PATH.parent
        / "__init__.py"
    )

    assert not local_init.exists()


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
