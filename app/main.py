"""FastAPI application boundary for the Agentic HR Policy Assistant.

S7 is intentionally a thin web layer. Agent orchestration remains in
``agent.orchestrator`` and MCP remains the only runtime boundary to HR
business tools.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator

from agent.llm import LLMClient
from agent.orchestrator import (
    AgentMCPClient,
    AgentResult,
    PendingConfirmation,
    confirm_pending_action,
    run_turn,
)
from rag.config import (
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    TARGET_CHUNK_TOKENS,
)
from rag.ingest import (
    ManifestValidationError,
    load_manifest,
)
from rag.store import (
    ChromaStoreError,
    get_chroma_client,
    get_policy_collection,
    is_index_current,
    resolve_chroma_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "static"
CORPUS_MANIFEST_PATH = PROJECT_ROOT / "corpus" / "version.json"


class ChatRequest(BaseModel):
    """One browser chat request."""

    model_config = ConfigDict(extra="forbid")

    message: str | None = None
    conversation_id: str | None = None
    employee_id: str | None = None
    confirmed: bool = False
    confirmation_id: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize an optional chat message."""

        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "message must be a non-empty string."
            )

        return value


class ChatResponse(BaseModel):
    """JSON-compatible result of one browser chat turn."""

    conversation_id: str
    answer: str
    citations: list[dict[str, str]]
    trace: list[dict[str, Any]]
    pending_confirmation: dict[str, Any] | None


class SessionState:
    """Minimal in-memory state for one browser conversation."""

    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []
        self.pending_confirmation: PendingConfirmation | None = None
        self.confirmation_in_progress = False


sessions: dict[str, SessionState] = {}


def _serialize_agent_result(
    *,
    conversation_id: str,
    result: AgentResult,
) -> ChatResponse:
    """Convert the existing S6 result into the HTTP response contract."""

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result.answer,
        citations=[
            dict(citation)
            for citation in result.citations
        ],
        trace=[
            item.as_dict()
            for item in result.trace
        ],
        pending_confirmation=(
            result.pending_confirmation.as_dict()
            if result.pending_confirmation is not None
            else None
        ),
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """Own the shared S6 LLM and MCP resources for one app process."""

    llm = LLMClient()
    mcp_client = AgentMCPClient()

    app.state.llm = llm
    app.state.mcp_client = mcp_client

    try:
        await mcp_client.start()
        yield
    finally:
        await mcp_client.close()
        await llm.close()


app = FastAPI(
    title="Agentic HR Policy Assistant",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get(
    "/",
    include_in_schema=False,
)
async def root() -> FileResponse:
    """Serve the minimal S7 browser application."""

    return FileResponse(
        STATIC_DIR / "index.html"
    )


@app.get("/health")
async def health(
    request: Request,
) -> JSONResponse:
    """Return cheap observational application health."""

    mcp_client: AgentMCPClient = (
        request.app.state.mcp_client
    )

    index_status = "degraded"
    index_chunks: int | None = None
    corpus_version: str | None = None

    try:
        manifest = load_manifest(
            CORPUS_MANIFEST_PATH
        )
        corpus_version = manifest.version

        chroma_dir = resolve_chroma_dir()

        client = get_chroma_client(
            chroma_dir
        )

        collection = get_policy_collection(
            client
        )

        index_chunks = collection.count()

        index_current = is_index_current(
            chroma_dir,
            corpus_version=manifest.version,
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
            chunk_tokens=TARGET_CHUNK_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
        )

        if (
            index_chunks == 400
            and index_current
        ):
            index_status = "ready"

    except (
        ManifestValidationError,
        ChromaStoreError,
    ):
        pass

    overall_status = (
        "ok"
        if (
            mcp_client.status == "connected"
            and index_status == "ready"
        )
        else "degraded"
    )

    status_code = (
        200
        if overall_status == "ok"
        else 503
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall_status,
            "mcp": mcp_client.status,
            "index": index_status,
            "index_chunks": index_chunks,
            "corpus_version": corpus_version,
            "llm": "ok",
        },
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Run one normal S6 agent turn for a browser conversation."""

    if payload.confirmed:
        if payload.conversation_id is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "conversation_id is required "
                    "for confirmation."
                ),
            )

        if payload.confirmation_id is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "confirmation_id is required "
                    "for confirmation."
                ),
            )

        session = sessions.get(
            payload.conversation_id
        )

        if (
            session is None
            or session.pending_confirmation is None
        ):
            raise HTTPException(
                status_code=409,
                detail="No pending action exists for this conversation.",
            )

        if session.confirmation_in_progress:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Confirmation is already in progress "
                    "for this conversation."
                ),
            )

        session.confirmation_in_progress = True

        try:
            result = await confirm_pending_action(
                pending=session.pending_confirmation,
                confirmation_id=payload.confirmation_id,
                mcp_client=request.app.state.mcp_client,
            )

            if (
                result.trace
                and result.trace[-1].decision
                != "confirmation_rejected"
            ):
                session.pending_confirmation = None

            session.history.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                }
            )

            return _serialize_agent_result(
                conversation_id=payload.conversation_id,
                result=result,
            )

        finally:
            session.confirmation_in_progress = False

    if payload.message is None:
        raise HTTPException(
            status_code=422,
            detail="message is required for a normal chat turn.",
        )

    conversation_id = (
        payload.conversation_id
        or uuid4().hex
    )

    session = sessions.setdefault(
        conversation_id,
        SessionState(),
    )

    # A new normal turn supersedes any unresolved prior proposal.
    session.pending_confirmation = None

    prior_history = [
        dict(item)
        for item in session.history
    ]

    session.history.append(
        {
            "role": "user",
            "content": payload.message,
        }
    )

    result = await run_turn(
        message=payload.message,
        mcp_client=request.app.state.mcp_client,
        llm=request.app.state.llm,
        history=prior_history,
    )

    session.history.append(
        {
            "role": "assistant",
            "content": result.answer,
        }
    )

    session.pending_confirmation = (
        result.pending_confirmation
    )

    return _serialize_agent_result(
        conversation_id=conversation_id,
        result=result,
    )
