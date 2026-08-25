"""Tests for the S7 FastAPI application boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient


def _resource_patches():
    """Return deterministic patches for shared S7 application resources."""

    llm = Mock()
    llm.close = AsyncMock()

    mcp_client = Mock()
    mcp_client.status = "degraded"
    mcp_client.start = AsyncMock(
        return_value=()
    )
    mcp_client.close = AsyncMock()

    return (
        llm,
        mcp_client,
        patch(
            "app.main.LLMClient",
            return_value=llm,
        ),
        patch(
            "app.main.AgentMCPClient",
            return_value=mcp_client,
        ),
    )


def test_root_serves_static_ui() -> None:
    """The root route serves the single static browser application."""

    from app.main import app

    (
        llm,
        mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    with llm_patch, mcp_patch:
        with TestClient(app) as client:
            response = client.get("/")

    assert response.status_code == 200
    assert "Agentic HR Policy Assistant" in response.text
    assert 'id="chat-form"' in response.text
    assert 'id="citations"' in response.text
    assert "Operational trace" in response.text
    assert 'id="confirmation-panel"' in response.text
    assert 'id="confirm-action"' in response.text
    assert 'id="wf1"' in response.text
    assert 'id="wf2"' in response.text
    assert 'fetch("/chat"' in response.text

    mcp_client.start.assert_awaited_once_with()
    mcp_client.close.assert_awaited_once_with()
    llm.close.assert_awaited_once_with()


def test_health_reports_application_resource_state() -> None:
    """Health reports a current 400-chunk policy index as ready."""

    from app.main import app

    (
        llm,
        mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    mcp_client.status = "connected"

    manifest = Mock()
    manifest.version = "1.2"

    chroma_dir = Mock()

    collection = Mock()
    collection.count.return_value = 400

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.load_manifest",
            return_value=manifest,
        ),
        patch(
            "app.main.resolve_chroma_dir",
            return_value=chroma_dir,
        ),
        patch(
            "app.main.get_chroma_client",
        ),
        patch(
            "app.main.get_policy_collection",
            return_value=collection,
        ),
        patch(
            "app.main.is_index_current",
            return_value=True,
        ) as current_patch,
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "mcp": "connected",
        "index": "ready",
        "index_chunks": 400,
        "corpus_version": "1.2",
        "llm": "ok",
    }

    current_patch.assert_called_once()

    call = current_patch.call_args

    assert call.args == (
        chroma_dir,
    )

    assert call.kwargs == {
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": 384,
        "chunk_tokens": 350,
        "chunk_overlap": 50,
    }

    mcp_client.start.assert_awaited_once_with()
    mcp_client.close.assert_awaited_once_with()
    llm.close.assert_awaited_once_with()


def test_health_degrades_when_policy_index_is_stale() -> None:
    """A populated but stale policy index must not report ready."""

    from app.main import app

    (
        llm,
        mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    mcp_client.status = "connected"

    manifest = Mock()
    manifest.version = "1.2"

    collection = Mock()
    collection.count.return_value = 400

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.load_manifest",
            return_value=manifest,
        ),
        patch(
            "app.main.resolve_chroma_dir",
        ),
        patch(
            "app.main.get_chroma_client",
        ),
        patch(
            "app.main.get_policy_collection",
            return_value=collection,
        ),
        patch(
            "app.main.is_index_current",
            return_value=False,
        ),
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 503

    assert response.json() == {
        "status": "degraded",
        "mcp": "connected",
        "index": "degraded",
        "index_chunks": 400,
        "corpus_version": "1.2",
        "llm": "ok",
    }


def test_health_degrades_when_policy_index_has_wrong_chunk_count() -> None:
    """A current but incomplete policy collection must not report ready."""

    from app.main import app

    (
        llm,
        mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    mcp_client.status = "connected"

    manifest = Mock()
    manifest.version = "1.2"

    collection = Mock()
    collection.count.return_value = 399

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.load_manifest",
            return_value=manifest,
        ),
        patch(
            "app.main.resolve_chroma_dir",
        ),
        patch(
            "app.main.get_chroma_client",
        ),
        patch(
            "app.main.get_policy_collection",
            return_value=collection,
        ),
        patch(
            "app.main.is_index_current",
            return_value=True,
        ),
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 503

    assert response.json() == {
        "status": "degraded",
        "mcp": "connected",
        "index": "degraded",
        "index_chunks": 399,
        "corpus_version": "1.2",
        "llm": "ok",
    }


def test_health_degrades_when_policy_index_is_unavailable() -> None:
    """Health degrades safely when the policy index cannot be opened."""

    from app.main import ChromaStoreError, app

    (
        llm,
        mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    mcp_client.status = "connected"

    manifest = Mock()
    manifest.version = "1.2"

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.load_manifest",
            return_value=manifest,
        ),
        patch(
            "app.main.resolve_chroma_dir",
        ),
        patch(
            "app.main.get_chroma_client",
        ),
        patch(
            "app.main.get_policy_collection",
            side_effect=ChromaStoreError(
                "Policy collection unavailable."
            ),
        ),
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "mcp": "connected",
        "index": "degraded",
        "index_chunks": None,
        "corpus_version": "1.2",
        "llm": "ok",
    }

    mcp_client.start.assert_awaited_once_with()
    mcp_client.close.assert_awaited_once_with()
    llm.close.assert_awaited_once_with()


def test_chat_rejects_blank_message() -> None:
    """The HTTP boundary rejects an empty user turn."""

    from app.main import app, sessions

    sessions.clear()

    (
        _llm,
        _mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    with llm_patch, mcp_patch:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": "   ",
                },
            )

    assert response.status_code == 422
    assert sessions == {}


def test_chat_creates_conversation_id() -> None:
    """A new chat receives a server-generated conversation identifier."""

    from agent.orchestrator import AgentResult
    from agent.trace import TraceItem
    from app.main import app, sessions

    sessions.clear()

    result = AgentResult(
        answer="You can review the policy guidance.",
        citations=(),
        trace=(
            TraceItem(
                step=1,
                tool=None,
                arguments={},
                result_summary="You can review the policy guidance.",
                sources=(),
                decision="answer",
            ),
        ),
    )

    (
        _llm,
        _mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.run_turn",
            new_callable=AsyncMock,
            return_value=result,
        ) as run_turn_mock,
    ):
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": "What is the remote work policy?",
                },
            )

    assert response.status_code == 200

    body = response.json()
    conversation_id = body["conversation_id"]

    assert isinstance(conversation_id, str)
    assert conversation_id
    assert conversation_id in sessions

    run_turn_mock.assert_awaited_once()


def test_chat_reuses_conversation_id() -> None:
    """A supplied conversation identifier reuses the same session."""

    from agent.orchestrator import AgentResult
    from app.main import app, sessions

    sessions.clear()

    result = AgentResult(
        answer="Response recorded.",
        citations=(),
        trace=(),
    )

    (
        _llm,
        _mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.run_turn",
            new_callable=AsyncMock,
            return_value=result,
        ),
    ):
        with TestClient(app) as client:
            first = client.post(
                "/chat",
                json={
                    "message": "First question",
                    "conversation_id": "conversation-test",
                },
            )
            second = client.post(
                "/chat",
                json={
                    "message": "Second question",
                    "conversation_id": "conversation-test",
                },
            )

    assert first.status_code == 200
    assert second.status_code == 200

    assert (
        first.json()["conversation_id"]
        == "conversation-test"
    )
    assert (
        second.json()["conversation_id"]
        == "conversation-test"
    )

    assert list(sessions) == [
        "conversation-test"
    ]
    assert len(
        sessions["conversation-test"].history
    ) == 4


def test_chat_serializes_answer_citations_and_trace() -> None:
    """The web adapter preserves grounded S6 output fields."""

    from agent.orchestrator import AgentResult
    from agent.trace import TraceItem
    from app.main import app, sessions

    sessions.clear()

    citation = {
        "doc_id": "HR-POL-004",
        "section": "Remote Work > Eligibility",
        "snippet": "Eligible employees may request remote work.",
    }

    result = AgentResult(
        answer="The policy permits eligible remote work requests.",
        citations=(
            citation,
        ),
        trace=(
            TraceItem(
                step=1,
                tool="search_policy_documents",
                arguments={
                    "query": "remote work eligibility",
                },
                result_summary="Policy evidence retrieved.",
                sources=(
                    citation,
                ),
                decision="tool_result",
            ),
        ),
    )

    (
        _llm,
        _mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.run_turn",
            new_callable=AsyncMock,
            return_value=result,
        ),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": "Can I work remotely?",
                },
            )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == result.answer
    assert body["citations"] == [
        citation
    ]
    assert body["trace"] == [
        result.trace[0].as_dict()
    ]
    assert body["pending_confirmation"] is None



def test_action_proposal_is_stored_and_returned() -> None:
    """A proposed ACTION remains bound to its server-side session."""

    from agent.orchestrator import AgentResult, PendingConfirmation
    from app.main import app, sessions

    sessions.clear()

    pending = PendingConfirmation(
        confirmation_id="confirm-123",
        tool="draft_hr_email",
        arguments={
            "to_role": "manager",
            "subject": "PTO request",
            "context": "Employee E001 requests three PTO days.",
        },
        preview="Draft a mock PTO request email to the manager.",
    )

    result = AgentResult(
        answer="Please confirm the proposed mock HR action.",
        citations=(),
        trace=(),
        pending_confirmation=pending,
    )

    (
        _llm,
        _mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.run_turn",
            new_callable=AsyncMock,
            return_value=result,
        ),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": "Draft my PTO request.",
                },
            )

    assert response.status_code == 200

    body = response.json()
    conversation_id = body["conversation_id"]

    assert body["pending_confirmation"] == pending.as_dict()
    assert (
        sessions[conversation_id].pending_confirmation
        is pending
    )


def test_confirmation_requires_pending_action() -> None:
    """A confirmation cannot execute without server-side pending state."""

    from app.main import app, sessions

    sessions.clear()

    (
        _llm,
        _mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    with llm_patch, mcp_patch:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "confirmed": True,
                    "conversation_id": "missing-session",
                    "confirmation_id": "confirm-123",
                },
            )

    assert response.status_code == 409
    assert sessions == {}


def test_wrong_confirmation_id_does_not_execute() -> None:
    """A mismatched ID preserves the pending action and does not execute."""

    from agent.orchestrator import (
        PendingConfirmation,
    )
    from app.main import (
        SessionState,
        app,
        sessions,
    )

    sessions.clear()

    pending = PendingConfirmation(
        confirmation_id="confirm-correct",
        tool="draft_hr_email",
        arguments={
            "to_role": "manager",
            "subject": "PTO request",
            "context": "Bound server-side context.",
        },
        preview="Draft the mock PTO request email.",
    )

    session = SessionState()
    session.pending_confirmation = pending
    sessions["conversation-test"] = session

    (
        _llm,
        mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    mcp_client.status = "connected"
    mcp_client.call_tool = AsyncMock()

    with llm_patch, mcp_patch:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "confirmed": True,
                    "conversation_id": "conversation-test",
                    "confirmation_id": "confirm-wrong",
                },
            )

    assert response.status_code == 200
    assert (
        response.json()["trace"][-1]["decision"]
        == "confirmation_rejected"
    )
    mcp_client.call_tool.assert_not_awaited()
    assert (
        sessions["conversation-test"].pending_confirmation
        is pending
    )


def test_valid_confirmation_executes_stored_pending_action() -> None:
    """A matching ID executes the immutable server-side action snapshot."""

    from types import SimpleNamespace

    from agent.orchestrator import PendingConfirmation
    from app.main import (
        SessionState,
        app,
        sessions,
    )

    sessions.clear()

    bound_arguments = {
        "to_role": "manager",
        "subject": "PTO request",
        "context": "Employee E001 requests three PTO days.",
    }

    pending = PendingConfirmation(
        confirmation_id="confirm-123",
        tool="draft_hr_email",
        arguments=bound_arguments,
        preview="Draft the mock PTO request email.",
    )

    session = SessionState()
    session.pending_confirmation = pending
    sessions["conversation-test"] = session

    (
        _llm,
        mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    mcp_client.status = "connected"
    mcp_client.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            structuredContent={
                "draft_text": "MOCK PTO request",
            },
        )
    )

    with llm_patch, mcp_patch:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "confirmed": True,
                    "conversation_id": "conversation-test",
                    "confirmation_id": "confirm-123",
                },
            )

    assert response.status_code == 200
    assert (
        response.json()["trace"][-1]["decision"]
        == "action_executed"
    )

    mcp_client.call_tool.assert_awaited_once_with(
        "draft_hr_email",
        bound_arguments,
    )

    assert (
        sessions["conversation-test"].pending_confirmation
        is None
    )


def test_new_turn_supersedes_stale_pending_confirmation() -> None:
    """A new normal turn removes an unresolved older proposal."""

    from agent.orchestrator import (
        AgentResult,
        PendingConfirmation,
    )
    from app.main import (
        SessionState,
        app,
        sessions,
    )

    sessions.clear()

    stale = PendingConfirmation(
        confirmation_id="old-confirmation",
        tool="draft_hr_email",
        arguments={
            "to_role": "manager",
            "subject": "Old request",
            "context": "Old context",
        },
        preview="Old proposal.",
    )

    session = SessionState()
    session.pending_confirmation = stale
    sessions["conversation-test"] = session

    result = AgentResult(
        answer="Here is the answer to your new question.",
        citations=(),
        trace=(),
    )

    (
        _llm,
        _mcp_client,
        llm_patch,
        mcp_patch,
    ) = _resource_patches()

    with (
        llm_patch,
        mcp_patch,
        patch(
            "app.main.run_turn",
            new_callable=AsyncMock,
            return_value=result,
        ),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": "What is the remote work policy?",
                    "conversation_id": "conversation-test",
                },
            )

    assert response.status_code == 200
    assert (
        sessions["conversation-test"].pending_confirmation
        is None
    )
