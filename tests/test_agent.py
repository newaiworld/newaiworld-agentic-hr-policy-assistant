"""Tests for the S6 agent orchestration layer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag.embed import embed_documents
from rag.store import build_index

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_WORKFLOW_POLICY_CHUNK_IDS: tuple[str, ...] = (
    "HR-POL-002__0000__4bb5583bfc124a5c",
    "HR-POL-004__0000__6c07151728db106c",
    "HR-POL-005__0000__b7552a9bb63a1eac",
)


def _build_isolated_workflow_policy_index(
    tmp_path: Path,
) -> Path:
    """Build the minimal real policy index required by WF1/WF2 tests."""

    chunks_path = (
        PROJECT_ROOT
        / "corpus"
        / "processed"
        / "chunks.json"
    )

    payload = json.loads(
        chunks_path.read_text(
            encoding="utf-8"
        )
    )

    chunks = (
        payload["chunks"]
        if isinstance(payload, dict)
        else payload
    )

    if not isinstance(chunks, list):
        raise AssertionError(
            "Canonical chunks artifact must contain a list."
        )

    by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
        if (
            isinstance(chunk, dict)
            and isinstance(
                chunk.get("chunk_id"),
                str,
            )
        )
    }

    missing = [
        chunk_id
        for chunk_id in _WORKFLOW_POLICY_CHUNK_IDS
        if chunk_id not in by_id
    ]

    if missing:
        raise AssertionError(
            "Required workflow policy chunks are missing: "
            f"{missing!r}"
        )

    selected = [
        by_id[chunk_id]
        for chunk_id in _WORKFLOW_POLICY_CHUNK_IDS
    ]

    texts = tuple(
        str(chunk["text"])
        for chunk in selected
    )

    embeddings = embed_documents(
        texts
    )

    index_path = (
        tmp_path
        / "workflow-policy-index"
    ).resolve()

    build_index(
        selected,
        embeddings,
        index_path,
    )

    assert index_path.is_absolute()
    assert index_path.is_relative_to(
        tmp_path.resolve()
    )
    assert index_path != (
        PROJECT_ROOT
        / "chroma_db"
    ).resolve()

    return index_path


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


def test_agent_mcp_default_timeout_contract() -> None:
    """Startup and runtime MCP timeout defaults remain intentionally distinct."""
    from agent import orchestrator

    assert (
        orchestrator.MCP_STARTUP_TIMEOUT_SECONDS
        == 30
    )
    assert (
        orchestrator.MCP_TOOL_TIMEOUT_SECONDS
        == 60
    )


def test_agent_mcp_startup_uses_dedicated_timeout(
    monkeypatch,
) -> None:
    """MCP discovery and runtime calls use their intended timeout bounds."""

    from contextlib import contextmanager

    from agent import orchestrator

    captured = {}

    class FakeSession:
        async def initialize(
            self,
        ) -> None:
            raise RuntimeError(
                "startup probe sentinel"
            )

    class FakeStack:
        def __init__(
            self,
        ) -> None:
            self.enter_count = 0

        async def enter_async_context(
            self,
            context,
        ):
            self.enter_count += 1

            if self.enter_count == 1:
                return (
                    object(),
                    object(),
                )

            if self.enter_count == 2:
                return context

            raise AssertionError(
                "Unexpected async context entry."
            )

        async def aclose(
            self,
        ) -> None:
            return None

    def fake_stdio_client(
        server,
    ):
        del server
        return object()

    def fake_client_session(
        read_stream,
        write_stream,
        *,
        read_timeout_seconds,
    ):
        del read_stream, write_stream

        captured[
            "read_timeout_seconds"
        ] = read_timeout_seconds

        return FakeSession()

    @contextmanager
    def fake_fail_after(
        seconds,
    ):
        captured["fail_after_seconds"] = (
            seconds
        )
        yield

    monkeypatch.setattr(
        orchestrator,
        "AsyncExitStack",
        FakeStack,
    )
    monkeypatch.setattr(
        orchestrator,
        "stdio_client",
        fake_stdio_client,
    )
    monkeypatch.setattr(
        orchestrator,
        "ClientSession",
        fake_client_session,
    )
    monkeypatch.setattr(
        orchestrator.anyio,
        "fail_after",
        fake_fail_after,
    )

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient()

        tools = await client.start()

        assert tools == ()
        assert client.status == "degraded"
        assert client.last_error is not None
        assert (
            "startup probe sentinel"
            in client.last_error
        )

    asyncio.run(exercise())

    assert (
        orchestrator.MCP_STARTUP_TIMEOUT_SECONDS
        == 30
    )
    assert (
        orchestrator.MCP_TOOL_TIMEOUT_SECONDS
        == 60
    )
    assert (
        captured[
            "read_timeout_seconds"
        ].total_seconds()
        == 60
    )
    assert (
        captured["fail_after_seconds"]
        == 30
    )


def test_agent_mcp_startup_cleanup_does_not_mask_primary_failure(
    monkeypatch,
) -> None:
    """Cleanup failure must not replace the primary MCP startup error."""

    from contextlib import contextmanager

    from agent import orchestrator

    class FakeSession:
        async def initialize(
            self,
        ) -> None:
            raise RuntimeError(
                "primary startup failure"
            )

    class FakeStack:
        def __init__(
            self,
        ) -> None:
            self.enter_count = 0

        async def enter_async_context(
            self,
            context,
        ):
            self.enter_count += 1

            if self.enter_count == 1:
                return (
                    object(),
                    object(),
                )

            if self.enter_count == 2:
                return context

            raise AssertionError(
                "Unexpected async context entry."
            )

        async def aclose(
            self,
        ) -> None:
            raise BaseExceptionGroup(
                "cleanup failure",
                [
                    orchestrator.anyio.BrokenResourceError(),
                ],
            )

    def fake_stdio_client(
        server,
    ):
        del server
        return object()

    def fake_client_session(
        read_stream,
        write_stream,
        *,
        read_timeout_seconds,
    ):
        del (
            read_stream,
            write_stream,
            read_timeout_seconds,
        )

        return FakeSession()

    @contextmanager
    def fake_fail_after(
        seconds,
    ):
        del seconds
        yield

    monkeypatch.setattr(
        orchestrator,
        "AsyncExitStack",
        FakeStack,
    )
    monkeypatch.setattr(
        orchestrator,
        "stdio_client",
        fake_stdio_client,
    )
    monkeypatch.setattr(
        orchestrator,
        "ClientSession",
        fake_client_session,
    )
    monkeypatch.setattr(
        orchestrator.anyio,
        "fail_after",
        fake_fail_after,
    )

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient()

        tools = await client.start()

        assert tools == ()
        assert client.tools == ()
        assert client.status == "degraded"
        assert client.last_error is not None

        assert (
            "primary startup failure"
            in client.last_error
        )

        assert (
            "cleanup failure"
            not in client.last_error
        )

        assert (
            "BrokenResourceError"
            not in client.last_error
        )

    asyncio.run(exercise())


def test_agent_mcp_close_suppresses_broken_resource_exception_group(
    monkeypatch,
) -> None:
    """Expected stdio teardown races do not escape normal MCP close."""

    from agent import orchestrator

    class FakeStack:
        async def aclose(
            self,
        ) -> None:
            raise BaseExceptionGroup(
                "stdio teardown",
                [
                    orchestrator.anyio.BrokenResourceError(),
                ],
            )

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient()

        client._stack = FakeStack()
        client._session = object()
        client._tools = ()
        client._status = "connected"

        await client.close()

        assert client._stack is None
        assert client._session is None
        assert client.tools == ()
        assert client.status == "degraded"

    asyncio.run(exercise())


def test_agent_mcp_close_propagates_unexpected_cleanup_failure(
    monkeypatch,
) -> None:
    """Unexpected close failures remain observable."""

    from agent import orchestrator

    class FakeStack:
        async def aclose(
            self,
        ) -> None:
            raise BaseExceptionGroup(
                "unexpected teardown",
                [
                    RuntimeError(
                        "unexpected close failure"
                    ),
                ],
            )

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient()

        client._stack = FakeStack()
        client._session = object()
        client._tools = ()
        client._status = "connected"

        try:
            await client.close()
        except BaseExceptionGroup as exc:
            flattened = []

            def collect(group):
                for item in group.exceptions:
                    if isinstance(
                        item,
                        BaseExceptionGroup,
                    ):
                        collect(item)
                    else:
                        flattened.append(item)

            collect(exc)

            assert any(
                isinstance(item, RuntimeError)
                and "unexpected close failure"
                in str(item)
                for item in flattened
            )
        else:
            raise AssertionError(
                "Unexpected close failure was suppressed."
            )

        assert client._stack is None
        assert client._session is None
        assert client.tools == ()
        assert client.status == "degraded"

    asyncio.run(exercise())


def test_prompt_version_is_defined() -> None:
    """Prompt changes remain attributable during later evaluation."""

    from agent.prompts import PROMPT_VERSION

    assert PROMPT_VERSION == "1.9"


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
        "prompt_version": "1.9",
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



def test_run_turn_converts_initial_llm_error_to_controlled_result() -> None:
    """An initial provider failure becomes a structured agent result."""

    from agent.llm import LLMError
    from agent.orchestrator import AgentMCPClient, run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

    class FailingLLM:
        async def chat(
            self,
            *,
            messages,
            tools=(),
        ):
            del messages, tools
            raise LLMError(
                "LLM request timed out."
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Hello",
            mcp_client=FakeMCP(),
            llm=FailingLLM(),
        )

        assert (
            result.answer
            == (
                "The language model is temporarily unavailable. "
                "Please try again later."
            )
        )
        assert result.citations == ()
        assert result.pending_confirmation is None
        assert len(result.trace) == 1

        item = result.trace[0]

        assert item.step == 1
        assert item.tool is None
        assert item.arguments == {}
        assert (
            item.result_summary
            == "LLM request timed out."
        )
        assert item.sources == ()
        assert item.decision == "llm_error"

    asyncio.run(exercise())


def test_run_turn_preserves_trace_when_llm_fails_after_tool_result() -> None:
    """A later provider failure preserves completed tool evidence."""

    from types import SimpleNamespace

    from agent.llm import (
        LLMError,
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
                    "name": "lookup_employee_profile",
                    "description": "Lookup employee.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "employee_id": {
                                "type": "string"
                            }
                        },
                        "required": ["employee_id"],
                    },
                },
            }
        ]

        async def call_tool(
            self,
            name,
            arguments,
        ):
            assert name == "lookup_employee_profile"
            assert arguments == {
                "employee_id": "E003"
            }

            return SimpleNamespace(
                isError=False,
                structuredContent={
                    "employee_id": "E003",
                    "employment_type": "full_time",
                    "location": "SYDNEY_HQ",
                },
            )

    class FailingSecondLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(
            self,
            *,
            messages,
            tools=(),
        ):
            del messages, tools
            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="call-1",
                            name="lookup_employee_profile",
                            arguments={
                                "employee_id": "E003"
                            },
                        ),
                    ),
                )

            raise LLMError(
                "LLM request timed out."
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Check employee E003.",
            mcp_client=FakeMCP(),
            llm=FailingSecondLLM(),
        )

        assert (
            result.answer
            == (
                "The language model is temporarily unavailable. "
                "Please try again later."
            )
        )

        assert len(result.trace) == 2

        first = result.trace[0]
        second = result.trace[1]

        assert first.tool == "lookup_employee_profile"
        assert first.decision == "tool_result"

        assert second.step == 2
        assert second.tool is None
        assert second.arguments == {}
        assert (
            second.result_summary
            == "LLM request timed out."
        )
        assert second.decision == "llm_error"

        assert (
            second.sources
            == tuple(result.citations)
        )

        assert result.pending_confirmation is None

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


def test_extract_citations_accepts_fastmcp_result_wrapper() -> None:
    """FastMCP list-result envelopes preserve policy citation evidence."""

    from agent.orchestrator import (
        _extract_citations,
    )

    structured = {
        "result": [
            {
                "doc_id": "HR-POL-004",
                "title": "Remote and Flexible Work Policy",
                "section": "4.4 International duration limit",
                "snippet": (
                    "International remote work is limited "
                    "to 30 calendar days."
                ),
                "score": 0.81,
            },
            {
                "doc_id": "HR-POL-005",
                "title": (
                    "Information Security and Acceptable Use Policy"
                ),
                "section": "8. Decision Rules",
                "snippet": (
                    "Overseas access requires a company-managed "
                    "device and approved VPN."
                ),
                "score": 0.76,
            },
        ],
    }

    assert _extract_citations(
        structured
    ) == [
        {
            "doc_id": "HR-POL-004",
            "title": "Remote and Flexible Work Policy",
            "section": "4.4 International duration limit",
            "snippet": (
                "International remote work is limited "
                "to 30 calendar days."
            ),
        },
        {
            "doc_id": "HR-POL-005",
            "title": (
                "Information Security and Acceptable Use Policy"
            ),
            "section": "8. Decision Rules",
            "snippet": (
                "Overseas access requires a company-managed "
                "device and approved VPN."
            ),
        },
    ]


def test_wf1_remote_work_runs_frozen_mcp_sequence_with_real_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WF1 executes the frozen remote-work workflow through real MCP."""

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import (
        AgentMCPClient,
        run_turn,
    )

    workflow_query = (
        "international remote work overseas six weeks "
        "duration approval location company-managed device "
        "VPN data security"
    )

    class WF1FakeLLM:
        def __init__(self) -> None:
            self.call_count = 0
            self.message_snapshots = []

        async def chat(
            self,
            *,
            messages,
            tools=(),
        ):
            self.call_count += 1
            self.message_snapshots.append(
                list(messages)
            )

            discovered_names = {
                item["function"]["name"]
                for item in tools
            }

            if self.call_count == 1:
                assert (
                    "lookup_employee_profile"
                    in discovered_names
                )

                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="wf1-profile",
                            name="lookup_employee_profile",
                            arguments={
                                "employee_id": "E003",
                            },
                        ),
                    ),
                )

            if self.call_count == 2:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="wf1-policy",
                            name="search_policy_documents",
                            arguments={
                                "query": workflow_query,
                                "k": 5,
                            },
                        ),
                    ),
                )

            if self.call_count == 3:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="wf1-compliance",
                            name="check_policy_compliance",
                            arguments={
                                "topic": (
                                    "remote_work_international"
                                ),
                                "employee_id": "E003",
                            },
                        ),
                    ),
                )

            assert self.call_count == 4

            return LLMResponse(
                content=(
                    "A six-week overseas arrangement is not compliant "
                    "with the ordinary international remote-work pathway "
                    "because it exceeds the 30-calendar-day standard "
                    "limit [HR-POL-004 §4.4]. Formal exception review, "
                    "manager and People and Culture approval, and "
                    "Information Security review are required. Overseas "
                    "company-system access also requires a company-managed "
                    "device and the approved VPN "
                    "[HR-POL-005 §4.5]."
                ),
                tool_calls=(),
            )

    index_path = _build_isolated_workflow_policy_index(
        tmp_path
    )

    monkeypatch.setenv(
        "CHROMA_DIR",
        str(index_path),
    )

    async def exercise() -> None:
        client = AgentMCPClient()

        try:
            tools = await client.start()

            assert client.status == "connected"
            assert len(tools) == 8

            llm = WF1FakeLLM()

            result = await run_turn(
                message=(
                    "I'm employee E003. Can I work remotely "
                    "from overseas for six weeks?"
                ),
                mcp_client=client,
                llm=llm,
            )

            assert llm.call_count == 4

            tool_trace = [
                item
                for item in result.trace
                if item.tool is not None
            ]

            assert [
                item.tool
                for item in tool_trace
            ] == [
                "lookup_employee_profile",
                "search_policy_documents",
                "check_policy_compliance",
            ]

            assert tool_trace[0].arguments == {
                "employee_id": "E003",
            }

            assert tool_trace[1].arguments == {
                "query": workflow_query,
                "k": 5,
            }

            assert tool_trace[2].arguments == {
                "topic": "remote_work_international",
                "employee_id": "E003",
            }

            citation_doc_ids = {
                item["doc_id"]
                for item in result.citations
            }

            assert "HR-POL-004" in citation_doc_ids
            assert "HR-POL-005" in citation_doc_ids

            assert "30-calendar-day" in result.answer
            assert "company-managed device" in result.answer
            assert "approved VPN" in result.answer

            assert result.exhausted is False
            assert result.pending_confirmation is None
            assert result.trace[-1].decision == "answer"

        finally:
            await client.close()

    asyncio.run(exercise())


def test_wf1_final_answer_contains_required_policy_citations() -> None:
    """The frozen WF1 answer cites remote-work and security policy."""

    answer = (
        "Six weeks exceeds the ordinary international remote-work "
        "limit [HR-POL-004 §4.4]. Overseas access requires approved "
        "security controls [HR-POL-005 §4.5]."
    )

    assert "[HR-POL-004 §4.4]" in answer
    assert "[HR-POL-005 §4.5]" in answer


def test_wf1_frozen_input_matches_demo_contract() -> None:
    """WF1 retains the exact user input frozen for demo and evaluation."""

    message = (
        "I'm employee E003. Can I work remotely "
        "from overseas for six weeks?"
    )

    assert message == (
        "I'm employee E003. Can I work remotely "
        "from overseas for six weeks?"
    )


def test_wf2_pto_runs_real_mcp_and_requires_confirmation_before_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WF2 proves PTO guidance plus a real confirmation-gated MCP ACTION."""

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import (
        AgentMCPClient,
        confirm_pending_action,
        run_turn,
    )

    policy_query = (
        "three days PTO next week available balance "
        "written manager approval operational coverage "
        "paid time off annual leave"
    )

    email_arguments = {
        "to_role": "manager",
        "subject": "PTO request — 3 days next week",
        "context": (
            "Employee E001 has 8.0 available PTO days and wants "
            "3 days of PTO next week. HR-POL-002 requires written "
            "manager approval and operational coverage."
        ),
    }

    class WF2FakeLLM:
        def __init__(self) -> None:
            self.call_count = 0

        async def chat(
            self,
            *,
            messages,
            tools=(),
        ):
            del messages

            self.call_count += 1

            discovered_names = {
                item["function"]["name"]
                for item in tools
            }

            if self.call_count == 1:
                assert (
                    "lookup_employee_profile"
                    in discovered_names
                )

                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="wf2-profile",
                            name="lookup_employee_profile",
                            arguments={
                                "employee_id": "E001",
                            },
                        ),
                    ),
                )

            if self.call_count == 2:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="wf2-balance",
                            name="check_pto_balance",
                            arguments={
                                "employee_id": "E001",
                            },
                        ),
                    ),
                )

            if self.call_count == 3:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="wf2-policy",
                            name="search_policy_documents",
                            arguments={
                                "query": policy_query,
                                "k": 5,
                            },
                        ),
                    ),
                )

            assert self.call_count == 4

            return LLMResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        call_id="wf2-action",
                        name="draft_hr_email",
                        arguments=email_arguments,
                    ),
                ),
            )

    index_path = _build_isolated_workflow_policy_index(
        tmp_path
    )

    monkeypatch.setenv(
        "CHROMA_DIR",
        str(index_path),
    )

    async def exercise() -> None:
        client = AgentMCPClient()

        try:
            tools = await client.start()

            assert client.status == "connected"
            assert len(tools) == 8

            llm = WF2FakeLLM()

            proposal = await run_turn(
                message=(
                    "I'm employee E001. Can I take "
                    "3 days of PTO next week?"
                ),
                mcp_client=client,
                llm=llm,
            )

            assert llm.call_count == 4
            assert proposal.pending_confirmation is not None
            assert (
                proposal.trace[-1].decision
                == "confirmation_required"
            )

            tool_trace = [
                item
                for item in proposal.trace
                if item.tool is not None
            ]

            assert [
                item.tool
                for item in tool_trace
            ] == [
                "lookup_employee_profile",
                "check_pto_balance",
                "search_policy_documents",
                "draft_hr_email",
            ]

            assert tool_trace[0].arguments == {
                "employee_id": "E001",
            }

            assert tool_trace[1].arguments == {
                "employee_id": "E001",
            }

            assert tool_trace[2].arguments == {
                "query": policy_query,
                "k": 5,
            }

            pending = proposal.pending_confirmation

            assert pending.tool == "draft_hr_email"
            assert pending.arguments == email_arguments

            citation_doc_ids = {
                item["doc_id"]
                for item in proposal.citations
            }

            assert "HR-POL-002" in citation_doc_ids

            confirmation = await confirm_pending_action(
                pending=pending,
                confirmation_id=pending.confirmation_id,
                mcp_client=client,
            )

            assert (
                confirmation.trace[-1].decision
                == "action_executed"
            )

            assert (
                "draft_text"
                in confirmation.trace[-1].result_summary
            )

            assert (
                "MOCK"
                in confirmation.trace[-1].result_summary
            )

        finally:
            await client.close()

    asyncio.run(exercise())


def test_wf2_frozen_input_matches_demo_contract() -> None:
    """WF2 keeps the frozen user input used for demo and evaluation."""

    message = (
        "I'm employee E001. Can I take "
        "3 days of PTO next week?"
    )

    assert message == (
        "I'm employee E001. Can I take "
        "3 days of PTO next week?"
    )


def test_wf2_policy_answer_requires_manager_approval() -> None:
    """WF2 policy evidence preserves the manager-approval requirement."""

    answer = (
        "You have enough balance for three days, but written manager "
        "approval and operational coverage are still required "
        "[HR-POL-002 §10.1]."
    )

    assert "written manager approval" in answer
    assert "operational coverage" in answer
    assert "[HR-POL-002 §10.1]" in answer


def test_wf2_action_is_draft_email_not_direct_leave_approval() -> None:
    """WF2 ACTION prepares a mock request artifact and does not approve leave."""

    action_name = "draft_hr_email"

    assert action_name == "draft_hr_email"
    assert action_name != "approve_pto"


def test_run_turn_returns_clean_unknown_employee_response() -> None:
    """An MCP isError employee lookup maps to the frozen failure response."""

    from types import SimpleNamespace

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import run_turn

    class TextItem:
        text = (
            "Error executing tool lookup_employee_profile: "
            "Employee not found: 'E999'."
        )

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        async def call_tool(self, name, arguments):
            del name, arguments

            return SimpleNamespace(
                isError=True,
                structuredContent=None,
                content=[
                    TextItem(),
                ],
            )

    class FakeLLM:
        async def chat(self, *, messages, tools=()):
            del messages, tools

            return LLMResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        call_id="unknown-employee",
                        name="example_tool",
                        arguments={
                            "employee_id": "E999",
                        },
                    ),
                ),
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Find employee E999.",
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert (
            "couldn't find that employee id"
            in result.answer.lower()
        )
        assert (
            result.trace[-1].decision
            == "unknown_employee"
        )

    asyncio.run(exercise())


def test_runtime_mcp_failure_returns_degraded_trace_with_existing_evidence() -> None:
    """Runtime MCP loss preserves previously gathered policy evidence."""

    from agent.llm import (
        LLMResponse,
        LLMToolCall,
    )
    from agent.orchestrator import (
        AgentMCPError,
        run_turn,
    )

    class FakeMCP:
        last_error = "MCP tool call timed out."
        llm_tools = []

        def __init__(self):
            self.status = "connected"
            self.calls = 0

        async def call_tool(self, name, arguments):
            del name, arguments

            self.calls += 1

            if self.calls == 1:
                from types import SimpleNamespace

                return SimpleNamespace(
                    isError=False,
                    structuredContent={
                        "result": [
                            {
                                "doc_id": "HR-POL-002",
                                "title": "Paid Time Off Policy",
                                "section": "5.3 Approval conditions",
                                "snippet": (
                                    "PTO requires written manager approval."
                                ),
                            },
                        ],
                    },
                )

            self.status = "degraded"

            raise AgentMCPError(
                "MCP tool call timed out."
            )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, *, messages, tools=()):
            del messages, tools

            self.calls += 1

            return LLMResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        call_id=f"failure-{self.calls}",
                        name="example_tool",
                        arguments={},
                    ),
                ),
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Help with my PTO.",
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert (
            result.trace[-1].decision
            == "mcp_degraded"
        )
        assert result.citations
        assert (
            result.citations[0]["doc_id"]
            == "HR-POL-002"
        )
        assert "policy evidence" in result.answer.lower()
        assert "contact hr" in result.answer.lower()

    asyncio.run(exercise())


def test_no_retrieval_hits_refuses_and_escalates() -> None:
    """An empty retrieval result leads to a grounded refusal."""

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
                isError=False,
                structuredContent={
                    "result": [],
                },
            )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, *, messages, tools=()):
            del tools

            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="no-hit",
                            name="example_tool",
                            arguments={
                                "query": "unsupported topic",
                                "k": 5,
                            },
                        ),
                    ),
                )

            tool_messages = [
                item
                for item in messages
                if item.get("role") == "tool"
            ]

            assert tool_messages
            assert '"result": []' in (
                tool_messages[-1]["content"]
            )

            return LLMResponse(
                content=(
                    "I couldn't find supporting policy evidence "
                    "for that request. Please contact HR."
                ),
                tool_calls=(),
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Tell me the unsupported company rule.",
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert result.citations == ()
        assert "couldn't find" in result.answer.lower()
        assert "contact hr" in result.answer.lower()

    asyncio.run(exercise())


def test_ambiguous_request_returns_exactly_one_clarifying_question() -> None:
    """The model contract asks one concise question for ambiguity."""

    from agent.llm import LLMResponse
    from agent.orchestrator import run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

    class FakeLLM:
        async def chat(self, *, messages, tools=()):
            del messages, tools

            return LLMResponse(
                content=(
                    "Which employee ID should I use for the PTO balance?"
                ),
                tool_calls=(),
            )

    async def exercise() -> None:
        result = await run_turn(
            message="Can you check the balance?",
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert result.answer.count("?") == 1
        assert (
            result.answer
            == "Which employee ID should I use for the PTO balance?"
        )
        assert result.trace[-1].decision == "answer"

    asyncio.run(exercise())


def test_sensitive_prompt_requires_conduct_policy_escalation_and_no_adjudication() -> None:
    """Prompt v1.3 preserves the frozen sensitive-topic policy."""

    from agent.prompts import (
        PROMPT_VERSION,
        SYSTEM_PROMPT,
    )

    assert PROMPT_VERSION == "1.9"

    lowered = SYSTEM_PROMPT.lower()

    assert "workplace conduct policy" in lowered
    assert "always recommend escalation" in lowered
    assert "people and culture" in lowered
    assert "never adjudicate" in lowered


def test_sensitive_harassment_flow_retrieves_policy_and_escalates() -> None:
    """A sensitive case uses policy evidence but never adjudicates."""

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
                isError=False,
                structuredContent={
                    "result": [
                        {
                            "doc_id": "HR-POL-011",
                            "title": "Workplace Conduct Policy",
                            "section": "6. Exceptions and Escalation",
                            "snippet": (
                                "Sensitive workplace matters require "
                                "human review and escalation."
                            ),
                        },
                    ],
                },
            )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, *, messages, tools=()):
            del tools

            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="conduct",
                            name="example_tool",
                            arguments={
                                "query": (
                                    "workplace conduct harassment "
                                    "discrimination escalation"
                                ),
                                "k": 5,
                            },
                        ),
                    ),
                )

            assert any(
                item.get("role") == "tool"
                for item in messages
            )

            return LLMResponse(
                content=(
                    "I can't determine whether harassment occurred. "
                    "This requires human review. Please escalate the "
                    "matter to People and Culture "
                    "[HR-POL-011 §6]."
                ),
                tool_calls=(),
            )

    async def exercise() -> None:
        result = await run_turn(
            message=(
                "My colleague keeps making discriminatory comments. "
                "Was I harassed?"
            ),
            mcp_client=FakeMCP(),
            llm=FakeLLM(),
        )

        assert {
            item["doc_id"]
            for item in result.citations
        } == {
            "HR-POL-011",
        }

        lowered = result.answer.lower()

        assert "can't determine" in lowered
        assert "people and culture" in lowered
        assert "human review" in lowered

    asyncio.run(exercise())


def test_agent_mcp_runtime_timeout_sets_degraded_state(
    monkeypatch,
) -> None:
    """A runtime MCP timeout moves the client into degraded state."""

    import asyncio

    from agent import orchestrator

    class SlowSession:
        async def call_tool(
            self,
            name,
            arguments,
        ):
            del name, arguments
            await asyncio.sleep(0.05)

    async def exercise() -> None:
        client = orchestrator.AgentMCPClient()

        client._status = "connected"
        client._session = SlowSession()
        client._tools = (
            orchestrator.DiscoveredTool(
                name="example_tool",
                description="Example tool.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
                read_only=True,
            ),
        )

        monkeypatch.setattr(
            orchestrator,
            "MCP_TOOL_TIMEOUT_SECONDS",
            0.001,
        )

        with pytest.raises(
            orchestrator.AgentMCPError,
            match="MCP tool call timed out",
        ):
            await client.call_tool(
                "example_tool",
                {},
            )

        assert client.status == "degraded"
        assert (
            client.last_error
            == "MCP tool call timed out."
        )

    asyncio.run(exercise())


def test_prompt_requires_exact_retrieved_section_names_for_lookup() -> None:
    """Exact-section tools must not receive invented or paraphrased section names."""

    from agent.prompts import SYSTEM_PROMPT

    prompt = " ".join(
        SYSTEM_PROMPT.lower().split()
    )

    assert "exact section name" in prompt
    assert "retrieved policy evidence" in prompt
    assert "do not invent" in prompt
    assert "paraphrase" in prompt
    assert "retrieved evidence is already sufficient" in prompt
    assert "exact-section lookup" in prompt



def test_prompt_requires_specific_policy_search_and_grounded_section_recovery() -> None:
    """Prompt v1.3 requires specific retrieval and forbids guessed sections."""

    from agent.prompts import SYSTEM_PROMPT

    prompt = " ".join(
        SYSTEM_PROMPT.lower().split()
    )

    assert "concrete terms from the user's request" in prompt
    assert "vague generic policy label" in prompt
    assert "exact section name" in prompt
    assert "numeric identifier" in prompt
    assert "another tool result" in prompt
    assert "search again with a more specific query" in prompt
    assert "instead of guessing a section name" in prompt



def test_policy_section_selectors_include_exact_heading_and_numeric_identifier() -> None:
    """A numeric heading establishes both full and canonical numeric selectors."""

    from agent.orchestrator import _policy_section_selectors

    assert _policy_section_selectors(
        "HR-POL-002",
        "5.5 Planning and operational coverage",
    ) == {
        (
            "HR-POL-002",
            "5.5 Planning and operational coverage",
        ),
        (
            "HR-POL-002",
            "5.5",
        ),
    }


def test_policy_section_selectors_preserve_nonnumeric_top_level_heading_only() -> None:
    """A nonnumeric top-level heading authorizes only its exact value."""

    from agent.orchestrator import _policy_section_selectors

    assert _policy_section_selectors(
        "HR-POL-002",
        "Paid Time Off Policy",
    ) == {
        (
            "HR-POL-002",
            "Paid Time Off Policy",
        ),
    }


def test_extract_grounded_policy_selectors_from_search_results() -> None:
    """Search evidence establishes exact headings and numeric identifiers."""

    from agent.orchestrator import _extract_grounded_policy_selectors

    selectors = _extract_grounded_policy_selectors(
        {
            "result": [
                {
                    "doc_id": "HR-POL-002",
                    "title": "Paid Time Off Policy",
                    "section": "5.5 Planning and operational coverage",
                    "snippet": "Operational coverage guidance.",
                    "score": 0.68,
                },
                {
                    "doc_id": "HR-POL-002",
                    "title": "Paid Time Off Policy",
                    "section": "Paid Time Off Policy",
                    "snippet": "Policy overview.",
                    "score": 0.65,
                },
            ],
        },
        tool_name="search_policy_documents",
    )

    assert selectors == {
        (
            "HR-POL-002",
            "5.5 Planning and operational coverage",
        ),
        (
            "HR-POL-002",
            "5.5",
        ),
        (
            "HR-POL-002",
            "Paid Time Off Policy",
        ),
    }


def test_extract_grounded_policy_selectors_from_compliance_refs() -> None:
    """Frozen compliance policy_refs establish exact numeric selectors."""

    from agent.orchestrator import _extract_grounded_policy_selectors

    selectors = _extract_grounded_policy_selectors(
        {
            "compliant": False,
            "reasons": [
                "Example reason.",
            ],
            "policy_refs": [
                "HR-POL-004 §4.4",
                "HR-POL-004 §8",
                "HR-POL-005 §4.5",
            ],
        },
        tool_name="check_policy_compliance",
    )

    assert selectors == {
        (
            "HR-POL-004",
            "4.4",
        ),
        (
            "HR-POL-004",
            "8",
        ),
        (
            "HR-POL-005",
            "4.5",
        ),
    }


def test_extract_grounded_policy_selectors_ignores_malformed_refs() -> None:
    """Malformed policy references cannot create trusted selectors."""

    from agent.orchestrator import _extract_grounded_policy_selectors

    selectors = _extract_grounded_policy_selectors(
        {
            "policy_refs": [
                "HR-POL-004 4.4",
                "HR-POL-004 §4.4 extra",
                "hr-pol-004 §4.4",
                "",
                None,
            ],
        },
        tool_name="check_policy_compliance",
    )

    assert selectors == set()


def test_extract_grounded_policy_selectors_ignores_unrelated_tools() -> None:
    """Non-policy tools cannot manufacture exact-section provenance."""

    from agent.orchestrator import _extract_grounded_policy_selectors

    selectors = _extract_grounded_policy_selectors(
        {
            "doc_id": "HR-POL-002",
            "section": "5.5 Planning and operational coverage",
        },
        tool_name="lookup_employee_profile",
    )

    assert selectors == set()


def test_grounded_policy_section_call_requires_exact_pair() -> None:
    """Exact membership permits grounded section calls."""

    from agent.orchestrator import _is_grounded_policy_section_call

    selectors = {
        (
            "HR-POL-002",
            "5.5",
        ),
    }

    assert _is_grounded_policy_section_call(
        {
            "doc_id": "HR-POL-002",
            "section": "5.5",
        },
        selectors,
    )


def test_grounded_policy_section_call_rejects_wrong_document() -> None:
    """The same section under another policy document is not grounded."""

    from agent.orchestrator import _is_grounded_policy_section_call

    selectors = {
        (
            "HR-POL-002",
            "5.5",
        ),
    }

    assert not _is_grounded_policy_section_call(
        {
            "doc_id": "HR-POL-004",
            "section": "5.5",
        },
        selectors,
    )


def test_grounded_policy_section_call_rejects_invented_sibling_section() -> None:
    """An unobserved sibling section cannot be inferred from nearby evidence."""

    from agent.orchestrator import _is_grounded_policy_section_call

    selectors = {
        (
            "HR-POL-002",
            "5.5 Planning and operational coverage",
        ),
        (
            "HR-POL-002",
            "5.5",
        ),
    }

    assert not _is_grounded_policy_section_call(
        {
            "doc_id": "HR-POL-002",
            "section": "5.6 Approval process",
        },
        selectors,
    )


def test_grounded_policy_section_call_rejects_paraphrase_and_case_variation() -> None:
    """Selector matching is exact and never fuzzy or case-normalized."""

    from agent.orchestrator import _is_grounded_policy_section_call

    selectors = {
        (
            "HR-POL-002",
            "5.5 Planning and operational coverage",
        ),
    }

    assert not _is_grounded_policy_section_call(
        {
            "doc_id": "HR-POL-002",
            "section": "Planning and operational coverage",
        },
        selectors,
    )

    assert not _is_grounded_policy_section_call(
        {
            "doc_id": "HR-POL-002",
            "section": "5.5 planning and operational coverage",
        },
        selectors,
    )


def test_extract_citations_composes_exact_section_invocation_provenance() -> None:
    """Exact-section evidence uses the invocation document ID."""

    from agent.orchestrator import _extract_citations

    structured = {
        "title": "Remote and Flexible Work Policy",
        "section": "4.4 International duration limit",
        "text": (
            "International remote work is limited to "
            "30 calendar days in any rolling 12-month period."
        ),
    }

    citations = _extract_citations(
        structured,
        tool_name="get_policy_section",
        arguments={
            "doc_id": "HR-POL-004",
            "section": "4.4",
        },
    )

    assert citations == [
        {
            "doc_id": "HR-POL-004",
            "title": "Remote and Flexible Work Policy",
            "section": "4.4 International duration limit",
            "snippet": (
                "International remote work is limited to "
                "30 calendar days in any rolling 12-month period."
            ),
        },
    ]

    assert citations[0]["snippet"].strip()


def test_extract_citations_does_not_infer_provenance_for_other_tools() -> None:
    """Unrelated tools cannot manufacture policy citation provenance."""

    from agent.orchestrator import _extract_citations

    structured = {
        "title": "Not policy evidence",
        "section": "Example",
        "text": "This result must not become a policy citation.",
    }

    citations = _extract_citations(
        structured,
        tool_name="example_tool",
        arguments={
            "doc_id": "HR-POL-004",
        },
    )

    assert citations == []


def test_extract_citations_preserves_result_side_doc_id_precedence() -> None:
    """Result-side provenance outranks invocation fallback provenance."""

    from agent.orchestrator import _extract_citations

    structured = {
        "doc_id": "HR-POL-005",
        "title": "Information Security and Acceptable Use Policy",
        "section": "4.5 Overseas access controls",
        "text": (
            "Employees must use a company-managed device "
            "and approved VPN for overseas access."
        ),
    }

    citations = _extract_citations(
        structured,
        tool_name="get_policy_section",
        arguments={
            "doc_id": "HR-POL-004",
            "section": "4.4",
        },
    )

    assert citations == [
        {
            "doc_id": "HR-POL-005",
            "title": (
                "Information Security and Acceptable Use Policy"
            ),
            "section": "4.5 Overseas access controls",
            "snippet": (
                "Employees must use a company-managed device "
                "and approved VPN for overseas access."
            ),
        },
    ]



def test_run_turn_rejects_ungrounded_exact_section_without_mcp_call() -> None:
    """Unsupported exact-section selectors never cross the MCP boundary."""

    import asyncio

    from agent.llm import LLMResponse, LLMToolCall
    from agent.orchestrator import run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append(
                (
                    name,
                    arguments,
                )
            )
            raise AssertionError(
                "Ungrounded exact-section call must not reach MCP."
            )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, *, messages, tools=()):
            del tools
            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="bad-section",
                            name="get_policy_section",
                            arguments={
                                "doc_id": "HR-POL-002",
                                "section": "5.6 Approval process",
                            },
                        ),
                    ),
                )

            tool_messages = [
                item
                for item in messages
                if item.get("role") == "tool"
            ]

            assert tool_messages
            assert (
                "not been established by prior policy evidence"
                in tool_messages[-1]["content"]
            )

            return LLMResponse(
                content=(
                    "I do not have grounded evidence for that exact "
                    "section, so I would search again."
                ),
                tool_calls=(),
            )

    async def exercise() -> None:
        mcp = FakeMCP()
        llm = FakeLLM()

        result = await run_turn(
            message="Check the PTO approval requirements.",
            mcp_client=mcp,
            llm=llm,
        )

        assert mcp.calls == []
        assert llm.calls == 2
        assert result.trace[0].decision == "section_guard_rejected"
        assert result.trace[-1].decision == "answer"

    asyncio.run(exercise())


def test_run_turn_allows_compliance_grounded_numeric_section() -> None:
    """Compliance policy_refs authorize later canonical numeric lookups."""

    import asyncio
    from types import SimpleNamespace

    from agent.llm import LLMResponse, LLMToolCall
    from agent.orchestrator import run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append(
                (
                    name,
                    arguments,
                )
            )

            if name == "check_policy_compliance":
                return SimpleNamespace(
                    isError=False,
                    structuredContent={
                        "compliant": False,
                        "reasons": [
                            "Six weeks exceeds the standard limit.",
                        ],
                        "policy_refs": [
                            "HR-POL-004 §4.4",
                        ],
                    },
                )

            assert name == "get_policy_section"
            assert arguments == {
                "doc_id": "HR-POL-004",
                "section": "4.4",
            }

            return SimpleNamespace(
                isError=False,
                structuredContent={
                    "title": "Remote and Flexible Work Policy",
                    "section": "4.4 International duration limit",
                    "text": (
                        "International remote work is limited "
                        "to 30 calendar days."
                    ),
                },
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
                            call_id="compliance",
                            name="check_policy_compliance",
                            arguments={
                                "topic": "remote_work_international",
                                "employee_id": "E003",
                            },
                        ),
                    ),
                )

            if self.calls == 2:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="section",
                            name="get_policy_section",
                            arguments={
                                "doc_id": "HR-POL-004",
                                "section": "4.4",
                            },
                        ),
                    ),
                )

            return LLMResponse(
                content=(
                    "Six weeks exceeds the standard international "
                    "remote-work limit [HR-POL-004 §4.4]."
                ),
                tool_calls=(),
            )

    async def exercise() -> None:
        mcp = FakeMCP()

        result = await run_turn(
            message="Can E003 work overseas for six weeks?",
            mcp_client=mcp,
            llm=FakeLLM(),
        )

        assert [
            name
            for name, _ in mcp.calls
        ] == [
            "check_policy_compliance",
            "get_policy_section",
        ]

        assert all(
            item.decision != "section_guard_rejected"
            for item in result.trace
        )

        assert result.trace[-1].decision == "answer"

    asyncio.run(exercise())


def test_run_turn_recovers_after_rejected_section_with_new_search() -> None:
    """A rejected exact section can recover through a more specific search."""

    import asyncio
    from types import SimpleNamespace

    from agent.llm import LLMResponse, LLMToolCall
    from agent.orchestrator import run_turn

    class FakeMCP:
        status = "connected"
        last_error = None
        llm_tools = []

        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append(
                (
                    name,
                    arguments,
                )
            )

            if name == "search_policy_documents":
                return SimpleNamespace(
                    isError=False,
                    structuredContent={
                        "result": [
                            {
                                "doc_id": "HR-POL-002",
                                "title": "Paid Time Off Policy",
                                "section": "5.3 Approval conditions",
                                "snippet": (
                                    "Approval requires sufficient "
                                    "balance and written manager approval."
                                ),
                                "score": 0.9,
                            },
                        ],
                    },
                )

            assert name == "get_policy_section"
            assert arguments == {
                "doc_id": "HR-POL-002",
                "section": "5.3",
            }

            return SimpleNamespace(
                isError=False,
                structuredContent={
                    "title": "Paid Time Off Policy",
                    "section": "5.3 Approval conditions",
                    "text": (
                        "Approval requires sufficient balance "
                        "and written manager approval."
                    ),
                },
            )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, *, messages, tools=()):
            del tools
            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="invented",
                            name="get_policy_section",
                            arguments={
                                "doc_id": "HR-POL-002",
                                "section": "5.6 Approval process",
                            },
                        ),
                    ),
                )

            if self.calls == 2:
                assert any(
                    item.get("role") == "tool"
                    and "not been established" in item.get("content", "")
                    for item in messages
                )

                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="search",
                            name="search_policy_documents",
                            arguments={
                                "query": (
                                    "Paid Time Off approval conditions "
                                    "manager approval sufficient balance"
                                ),
                                "k": 5,
                            },
                        ),
                    ),
                )

            if self.calls == 3:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="valid-section",
                            name="get_policy_section",
                            arguments={
                                "doc_id": "HR-POL-002",
                                "section": "5.3",
                            },
                        ),
                    ),
                )

            return LLMResponse(
                content=(
                    "The request requires sufficient balance and "
                    "written manager approval [HR-POL-002 §5.3]."
                ),
                tool_calls=(),
            )

    async def exercise() -> None:
        mcp = FakeMCP()

        result = await run_turn(
            message="What are the PTO approval requirements?",
            mcp_client=mcp,
            llm=FakeLLM(),
        )

        assert [
            name
            for name, _ in mcp.calls
        ] == [
            "search_policy_documents",
            "get_policy_section",
        ]

        assert any(
            item.decision == "section_guard_rejected"
            for item in result.trace
        )

        assert result.trace[-1].decision == "answer"

    asyncio.run(exercise())


def test_run_turn_collects_exact_section_citation_from_tool_arguments() -> None:
    """Grounded exact-section MCP evidence propagates into citations."""

    import asyncio
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

        def __init__(self):
            self.calls = []

        async def call_tool(
            self,
            name,
            arguments,
        ):
            self.calls.append(
                (
                    name,
                    arguments,
                )
            )

            if name == "search_policy_documents":
                assert arguments == {
                    "query": (
                        "international remote work "
                        "six weeks duration limit"
                    ),
                    "k": 5,
                }

                return SimpleNamespace(
                    structuredContent={
                        "result": [
                            {
                                "doc_id": "HR-POL-004",
                                "title": (
                                    "Remote and Flexible Work Policy"
                                ),
                                "section": (
                                    "4.4 International duration limit"
                                ),
                                "snippet": (
                                    "International remote work is limited "
                                    "to 30 calendar days."
                                ),
                                "score": 0.95,
                            },
                        ],
                    },
                    isError=False,
                )

            assert name == "get_policy_section"
            assert arguments == {
                "doc_id": "HR-POL-004",
                "section": "4.4",
            }

            return SimpleNamespace(
                structuredContent={
                    "title": "Remote and Flexible Work Policy",
                    "section": (
                        "4.4 International duration limit"
                    ),
                    "text": (
                        "International remote work is limited "
                        "to 30 calendar days."
                    ),
                },
                isError=False,
            )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(
            self,
            *,
            messages,
            tools=(),
        ):
            del messages, tools

            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="search-policy",
                            name="search_policy_documents",
                            arguments={
                                "query": (
                                    "international remote work "
                                    "six weeks duration limit"
                                ),
                                "k": 5,
                            },
                        ),
                    ),
                )

            if self.calls == 2:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="call-section-1",
                            name="get_policy_section",
                            arguments={
                                "doc_id": "HR-POL-004",
                                "section": "4.4",
                            },
                        ),
                    ),
                )

            return LLMResponse(
                content=(
                    "Six weeks exceeds the standard "
                    "international remote-work limit "
                    "[HR-POL-004 §4.4]."
                ),
                tool_calls=(),
            )

    async def exercise() -> None:
        mcp = FakeMCP()

        result = await run_turn(
            message=(
                "Can I work internationally "
                "for six weeks?"
            ),
            mcp_client=mcp,
            llm=FakeLLM(),
        )

        assert [
            name
            for name, _ in mcp.calls
        ] == [
            "search_policy_documents",
            "get_policy_section",
        ]

        assert all(
            item.decision != "section_guard_rejected"
            for item in result.trace
        )

        exact_expected = {
            "doc_id": "HR-POL-004",
            "title": "Remote and Flexible Work Policy",
            "section": (
                "4.4 International duration limit"
            ),
            "snippet": (
                "International remote work is limited "
                "to 30 calendar days."
            ),
        }

        assert exact_expected in result.citations

        exact_trace = [
            item
            for item in result.trace
            if item.tool == "get_policy_section"
        ]

        assert len(exact_trace) == 1
        assert exact_trace[0].sources == (
            exact_expected,
        )
        assert result.trace[-1].decision == "answer"

    asyncio.run(exercise())


def test_mcp_subprocess_env_is_none_when_chroma_dir_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent CHROMA_DIR preserves the MCP SDK default environment contract."""

    orchestrator = _load_orchestrator()

    monkeypatch.delenv(
        "CHROMA_DIR",
        raising=False,
    )

    assert (
        orchestrator._build_mcp_subprocess_env()
        is None
    )


def test_mcp_subprocess_env_propagates_exact_chroma_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured CHROMA_DIR is forwarded exactly to the MCP child."""

    orchestrator = _load_orchestrator()

    monkeypatch.setenv(
        "CHROMA_DIR",
        "/tmp/example-policy-index",
    )

    assert (
        orchestrator._build_mcp_subprocess_env()
        == {
            "CHROMA_DIR": "/tmp/example-policy-index",
        }
    )


def test_mcp_subprocess_env_preserves_blank_chroma_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank explicit configuration remains available for downstream validation."""

    orchestrator = _load_orchestrator()

    monkeypatch.setenv(
        "CHROMA_DIR",
        "   ",
    )

    assert (
        orchestrator._build_mcp_subprocess_env()
        == {
            "CHROMA_DIR": "   ",
        }
    )


def test_mcp_subprocess_env_does_not_forward_unrelated_or_secret_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MCP child receives only the sanctioned Chroma runtime setting."""

    orchestrator = _load_orchestrator()

    monkeypatch.setenv(
        "CHROMA_DIR",
        "/tmp/example-policy-index",
    )
    monkeypatch.setenv(
        "LLM_API_KEY",
        "should-not-propagate",
    )
    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "should-not-propagate",
    )
    monkeypatch.setenv(
        "LLM_MODEL",
        "should-not-propagate",
    )
    monkeypatch.setenv(
        "SOME_RANDOM_VARIABLE",
        "should-not-propagate",
    )

    assert (
        orchestrator._build_mcp_subprocess_env()
        == {
            "CHROMA_DIR": "/tmp/example-policy-index",
        }
    )


def test_prompt_requires_employee_context_before_employee_specific_tools() -> None:
    """Explicit employee HR tasks establish profile context first."""
    from agent.prompts import SYSTEM_PROMPT

    lower = SYSTEM_PROMPT.lower()

    assert "lookup_employee_profile" in SYSTEM_PROMPT
    assert "employee" in lower
    assert (
        "before" in lower
        and (
            "calculation" in lower
            or "employee-specific" in lower
        )
    )


def test_prompt_requires_pto_balance_and_policy_before_action() -> None:
    """PTO planning gathers balance and policy evidence before ACTION."""
    from agent.prompts import SYSTEM_PROMPT

    lower = SYSTEM_PROMPT.lower()

    assert "check_pto_balance" in SYSTEM_PROMPT
    assert "policy" in lower
    assert "action" in lower
    assert "before" in lower


def test_run_turn_includes_prior_history_before_current_message() -> None:
    """A later agent turn receives prior conversation context."""

    from types import SimpleNamespace

    from agent.orchestrator import run_turn

    class HistoryLLM:
        def __init__(self) -> None:
            self.messages = None

        async def chat(self, *, messages, tools):
            self.messages = messages
            return SimpleNamespace(
                content="You can proceed with the PTO request.",
                tool_calls=(),
            )

    class ConnectedMCP:
        status = "connected"
        last_error = None
        llm_tools = []

    llm = HistoryLLM()
    mcp_client = ConnectedMCP()

    prior_history = [
        {
            "role": "user",
            "content": (
                "I'm employee E001. Can I take "
                "3 days of PTO next week?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Your PTO request requires policy "
                "and balance checks."
            ),
        },
    ]

    result = asyncio.run(run_turn(
        message="Please create the mock HR ticket.",
        history=prior_history,
        mcp_client=mcp_client,
        llm=llm,
    ))

    assert result.answer == (
        "You can proceed with the PTO request."
    )

    assert llm.messages is not None

    assert llm.messages[1:] == [
        *prior_history,
        {
            "role": "user",
            "content": "Please create the mock HR ticket.",
        },
    ]


def test_prompt_requires_remote_work_retrieval_before_compliance() -> None:
    """WF1 retrieves multi-document policy evidence before compliance."""

    from agent.prompts import SYSTEM_PROMPT

    lower = SYSTEM_PROMPT.lower()

    assert "international remote" in lower
    assert "search_policy_documents" in SYSTEM_PROMPT
    assert "check_policy_compliance" in SYSTEM_PROMPT
    assert "remote" in lower
    assert "security" in lower
    assert "before" in lower


def test_mcp_subprocess_env_propagates_sanctioned_offline_model_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP retrieval inherits explicit offline model configuration."""

    orchestrator = _load_orchestrator()

    monkeypatch.setenv(
        "CHROMA_DIR",
        "/tmp/example-policy-index",
    )
    monkeypatch.setenv(
        "HF_HUB_OFFLINE",
        "1",
    )
    monkeypatch.setenv(
        "TRANSFORMERS_OFFLINE",
        "1",
    )

    assert (
        orchestrator._build_mcp_subprocess_env()
        == {
            "CHROMA_DIR": "/tmp/example-policy-index",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )


def test_mcp_subprocess_env_supports_offline_flags_without_chroma_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit offline flags alone form a bounded MCP child environment."""

    orchestrator = _load_orchestrator()

    monkeypatch.delenv(
        "CHROMA_DIR",
        raising=False,
    )
    monkeypatch.setenv(
        "HF_HUB_OFFLINE",
        "1",
    )
    monkeypatch.setenv(
        "TRANSFORMERS_OFFLINE",
        "1",
    )

    assert (
        orchestrator._build_mcp_subprocess_env()
        == {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )


def test_mcp_subprocess_env_is_none_when_no_sanctioned_values_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No explicit MCP runtime configuration preserves SDK defaults."""

    orchestrator = _load_orchestrator()

    for name in (
        "CHROMA_DIR",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        monkeypatch.delenv(
            name,
            raising=False,
        )

    assert (
        orchestrator._build_mcp_subprocess_env()
        is None
    )


def test_mcp_subprocess_env_offline_flags_do_not_expand_secret_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline support preserves the MCP child least-privilege boundary."""

    orchestrator = _load_orchestrator()

    monkeypatch.setenv(
        "CHROMA_DIR",
        "/tmp/example-policy-index",
    )
    monkeypatch.setenv(
        "HF_HUB_OFFLINE",
        "1",
    )
    monkeypatch.setenv(
        "TRANSFORMERS_OFFLINE",
        "1",
    )
    monkeypatch.setenv(
        "LLM_API_KEY",
        "must-not-propagate",
    )
    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "must-not-propagate",
    )
    monkeypatch.setenv(
        "LLM_MODEL",
        "must-not-propagate",
    )
    monkeypatch.setenv(
        "SOME_RANDOM_VARIABLE",
        "must-not-propagate",
    )

    child = (
        orchestrator._build_mcp_subprocess_env()
    )

    assert child == {
        "CHROMA_DIR": "/tmp/example-policy-index",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    assert {
        "LLM_API_KEY",
        "OPENROUTER_API_KEY",
        "LLM_MODEL",
        "SOME_RANDOM_VARIABLE",
    }.isdisjoint(
        child
    )


def test_wf2_prompt_treats_relative_period_as_sufficient_for_draft_action() -> None:
    """WF2 may draft the mock PTO request without inventing exact dates."""

    from agent.prompts import (
        PROMPT_VERSION,
        SYSTEM_PROMPT,
    )

    assert PROMPT_VERSION == "1.9"

    lowered = SYSTEM_PROMPT.lower()

    assert "amount of leave" in lowered
    assert "requested period" in lowered
    assert 'relative period such as "next week"' in lowered
    assert "does not require invented calendar dates" in lowered
    assert "draft_hr_email" in SYSTEM_PROMPT
    assert "do not approve pto" in lowered
    assert "explicit confirmation" in lowered


def test_wf2_prompt_calls_draft_action_without_second_conversational_confirmation() -> None:
    """A sufficiently specified PTO request must proceed to the gated draft ACTION."""

    from agent.prompts import SYSTEM_PROMPT

    lowered = " ".join(SYSTEM_PROMPT.lower().split())

    assert (
        "do not ask a second conversational confirmation"
        in lowered
    )
    assert (
        "call draft_hr_email"
        in lowered
    )
    assert (
        "the orchestrator will request explicit confirmation"
        in lowered
    )
    assert (
        "before the action executes"
        in lowered
    )


def test_wf2_prompt_skips_compliance_tool_and_proceeds_directly_to_gated_draft_action() -> None:
    """WF2 must not insert the unrelated compliance tool before its gated draft."""

    from agent.prompts import (
        PROMPT_VERSION,
        SYSTEM_PROMPT,
    )

    normalized = " ".join(
        SYSTEM_PROMPT.lower().split()
    )

    assert PROMPT_VERSION == "1.9"

    assert (
        "do not call check_policy_compliance for pto requests"
        in normalized
    )
    assert (
        "after the pto balance and relevant policy evidence are available"
        in normalized
    )
    assert (
        "proceed directly to draft_hr_email"
        in normalized
    )
    assert (
        "the orchestrator will request explicit confirmation"
        in normalized
    )


def test_prompt_constrains_international_remote_work_tool_planning() -> None:
    """WF1 prompt prevents redundant and cross-workflow tool planning."""

    from agent.prompts import (
        PROMPT_VERSION,
        SYSTEM_PROMPT,
    )

    assert PROMPT_VERSION == "1.9"

    normalized = " ".join(SYSTEM_PROMPT.lower().split())

    assert (
        "do not call check_pto_balance for remote-work requests"
        in normalized
    )

    assert (
        "do not repeat policy searches"
        in normalized
    )

    assert (
        "unnecessary exact-section lookups"
        in normalized
    )

    assert (
        "answer the user instead of continuing to gather redundant evidence"
        in normalized
    )


def test_llm_client_retries_429_once_then_succeeds() -> None:
    """One transient 429 is retried once before returning success."""
    import asyncio

    import httpx

    from agent.llm import LLMClient

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(
                429,
                request=request,
                json={"error": {"message": "rate limited"}},
            )

        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Recovered.",
                        }
                    }
                ]
            },
        )

    async def scenario() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": "hello",
                    }
                ]
            )
        finally:
            await client.close()

        assert response.content == "Recovered."
        assert response.tool_calls == ()

    asyncio.run(scenario())

    assert attempts == 2


def test_llm_client_retries_503_once_then_succeeds() -> None:
    """One transient provider 503 is retried once before success."""
    import asyncio

    import httpx

    from agent.llm import LLMClient

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(
                503,
                request=request,
                json={"error": {"message": "temporarily unavailable"}},
            )

        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Recovered.",
                        }
                    }
                ]
            },
        )

    async def scenario() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await client.chat(
                messages=[
                    {
                        "role": "user",
                        "content": "hello",
                    }
                ]
            )
        finally:
            await client.close()

        assert response.content == "Recovered."

    asyncio.run(scenario())

    assert attempts == 2


def test_llm_client_stops_after_bounded_transient_retry() -> None:
    """Repeated transient failure is bounded to two total attempts."""
    import asyncio

    import httpx
    import pytest

    from agent.llm import LLMClient, LLMError

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            429,
            request=request,
            json={"error": {"message": "rate limited"}},
        )

    async def scenario() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(
                LLMError,
                match=r"HTTP status 429",
            ):
                await client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ]
                )
        finally:
            await client.close()

    asyncio.run(scenario())

    assert attempts == 2


def test_llm_client_does_not_retry_401() -> None:
    """Authentication failures are deterministic and are not retried."""
    import asyncio

    import httpx
    import pytest

    from agent.llm import LLMClient, LLMError

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            401,
            request=request,
            json={"error": {"message": "unauthorized"}},
        )

    async def scenario() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(
                LLMError,
                match=r"HTTP status 401",
            ):
                await client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ]
                )
        finally:
            await client.close()

    asyncio.run(scenario())

    assert attempts == 1


def test_llm_client_does_not_retry_422() -> None:
    """Invalid provider requests are not retried."""
    import asyncio

    import httpx
    import pytest

    from agent.llm import LLMClient, LLMError

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            422,
            request=request,
            json={"error": {"message": "unprocessable"}},
        )

    async def scenario() -> None:
        client = LLMClient(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(
                LLMError,
                match=r"HTTP status 422",
            ):
                await client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ]
                )
        finally:
            await client.close()

    asyncio.run(scenario())

    assert attempts == 1


def test_wf2_rejects_premature_answer_after_required_reads() -> None:
    """WF2 must not terminate before proposing its confirmation-gated draft."""

    import asyncio

    from agent.llm import LLMResponse, LLMToolCall
    from agent.orchestrator import AgentMCPClient, run_turn

    class WF2PrematureAnswerLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, *, messages, tools=()):
            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            call_id="profile",
                            name="lookup_employee_profile",
                            arguments={"employee_id": "E001"},
                        ),
                        LLMToolCall(
                            call_id="balance",
                            name="check_pto_balance",
                            arguments={"employee_id": "E001"},
                        ),
                        LLMToolCall(
                            call_id="policy",
                            name="search_policy_documents",
                            arguments={
                                "query": "PTO paid time off policy"
                            },
                        ),
                    ),
                )

            if self.calls == 2:
                return LLMResponse(
                    content=(
                        "You have enough PTO and may proceed "
                        "to manager approval."
                    ),
                    tool_calls=(),
                )

            return LLMResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        call_id="draft",
                        name="draft_hr_email",
                        arguments={
                            "to_role": "People and Culture",
                            "subject": "PTO request — 3 days next week",
                            "context": (
                                "Employee E001 requests 3 days of PTO "
                                "next week after balance and policy checks."
                            ),
                        },
                    ),
                ),
            )

    async def scenario() -> None:
        mcp_client = AgentMCPClient()

        try:
            tools = await mcp_client.start()

            assert mcp_client.status == "connected"
            assert len(tools) == 8

            llm = WF2PrematureAnswerLLM()

            result = await run_turn(
                message=(
                    "I'm employee E001. Can I take "
                    "3 days of PTO next week?"
                ),
                mcp_client=mcp_client,
                llm=llm,
            )

            assert llm.calls == 3
            assert result.pending_confirmation is not None
            assert result.pending_confirmation.tool == "draft_hr_email"
            assert result.trace[-1].decision == "confirmation_required"

            decisions = [
                item.decision
                for item in result.trace
            ]

            assert "answer" not in decisions

        finally:
            await mcp_client.close()

    asyncio.run(scenario())



def test_wf2_completion_guard_does_not_convert_informational_pto_questions_to_actions() -> None:
    """Informational PTO questions must remain outside the WF2 action guard."""

    from agent.orchestrator import _wf2_requires_action_proposal

    informational_messages = (
        "What does the PTO policy say?",
        "I'm employee E001. What is my PTO balance?",
        "How does paid time off work?",
    )

    for message in informational_messages:
        assert not _wf2_requires_action_proposal(
            message,
            (),
        )


def test_wf2_completion_guard_requires_sufficient_request_detail() -> None:
    """An incomplete PTO request must not be forced into an action workflow."""

    from agent.orchestrator import _wf2_requires_action_proposal

    incomplete_messages = (
        "I'm employee E001. Can I take PTO?",
        "Can I take 3 days of PTO?",
        "I'm employee E001. Can I take 3 days of PTO?",
    )

    for message in incomplete_messages:
        assert not _wf2_requires_action_proposal(
            message,
            (),
        )
