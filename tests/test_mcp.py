"""Tests for the MCP server foundation."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from mcp.server.fastmcp import FastMCP

from rag.retrieve import RetrievalResult


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


def test_server_foundation_has_no_registered_tools() -> None:
    """Foundation checkpoint must not pre-register unfinished business tools."""

    module = load_project_mcp_server()

    async def inspect_tools() -> None:
        tools = await module.mcp.list_tools()

        assert tools == []

    asyncio.run(
        inspect_tools()
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
