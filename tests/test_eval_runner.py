"""Permanent contracts for the S9 live evaluation runner."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL_PATH = PROJECT_ROOT / "evaluation" / "run_eval.py"
EVAL_SET_PATH = PROJECT_ROOT / "evaluation" / "eval_set.jsonl"


def _load_run_eval() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "s9_live_run_eval",
        RUN_EVAL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    module = _load_run_eval()
    fn = getattr(module, name, None)
    assert callable(fn), (
        f"evaluation/run_eval.py must expose callable {name}()."
    )
    return fn(*args, **kwargs)


def test_load_eval_items_returns_exactly_24_in_canonical_order() -> None:
    """R01/R02: runner loads all frozen rows without reordering them."""
    items = _call("load_eval_items", EVAL_SET_PATH)

    assert len(items) == 24
    assert [item["id"] for item in items] == [
        "SP01", "SP02", "SP03", "SP04",
        "SP05", "SP06", "SP07", "SP08",
        "MD01", "MD02", "MD03", "MD04", "MD05",
        "TL01", "TL02", "TL03", "TL04", "TL05", "TL06",
        "AM01", "AM02", "AM03",
        "OOS01", "OOS02",
    ]


def test_serialize_trace_item_uses_public_fields_only() -> None:
    """R03: persisted trace never exposes private implementation state."""
    class FakeTrace:
        step = 1
        tool = "search_policy_documents"
        arguments = {"query": "PTO"}
        result_summary = "2 results"
        sources = (
            {
                "doc_id": "HR-POL-002",
                "section": "4.1 Annual entitlement",
            },
        )
        decision = "tool_result"
        prompt_version = "1.9"
        _private = "must-not-leak"

    result = _call("serialize_trace_item", FakeTrace())

    assert result == {
        "step": 1,
        "tool": "search_policy_documents",
        "arguments": {"query": "PTO"},
        "result_summary": "2 results",
        "sources": [
            {
                "doc_id": "HR-POL-002",
                "section": "4.1 Annual entitlement",
            }
        ],
        "decision": "tool_result",
        "prompt_version": "1.9",
    }
    assert "_private" not in result


def test_serialize_citations_preserves_doc_and_section() -> None:
    """R04: citation evidence survives result persistence."""
    result = _call(
        "serialize_citations",
        (
            {
                "doc_id": "HR-POL-002",
                "section": "4.1 Annual entitlement",
            },
            {
                "doc_id": "HR-POL-003",
                "section": "5.2 Public holiday during leave",
            },
        ),
    )

    assert result == [
        {
            "doc_id": "HR-POL-002",
            "section": "4.1 Annual entitlement",
        },
        {
            "doc_id": "HR-POL-003",
            "section": "5.2 Public holiday during leave",
        },
    ]


def test_extract_observed_tools_preserves_trace_order() -> None:
    """R05: tool sequence is derived from trace in execution order."""
    trace = [
        {"tool": None, "decision": "answer"},
        {"tool": "lookup_employee_profile", "decision": "tool_result"},
        {"tool": "search_policy_documents", "decision": "tool_result"},
        {"tool": "search_policy_documents", "decision": "tool_result"},
    ]

    assert _call("extract_observed_tools", trace) == [
        "lookup_employee_profile",
        "search_policy_documents",
        "search_policy_documents",
    ]


@pytest.mark.parametrize(
    ("trace", "pending", "answer", "expected"),
    [
        (
            [{"decision": "confirmation_required"}],
            {"confirmation_id": "abc"},
            "Please confirm.",
            "propose_action",
        ),
        (
            [{"decision": "answer"}],
            None,
            "The policy says...",
            "answer",
        ),
        (
            [{"decision": "llm_error"}],
            None,
            "I cannot complete that request.",
            None,
        ),
        (
            [{"decision": "max_iterations"}],
            None,
            "Unable to complete.",
            None,
        ),
    ],
)
def test_classify_observed_behavior_from_runtime_signals(
    trace: list[dict[str, object]],
    pending: dict[str, object] | None,
    answer: str,
    expected: str | None,
) -> None:
    """R06/R07: terminal behavior comes from runtime evidence."""
    result = _call(
        "classify_observed_behavior",
        trace=trace,
        pending_confirmation=pending,
        answer=answer,
    )

    assert result == expected


def test_classify_observed_behavior_detects_clarification() -> None:
    """R06: one-question clarification is recognized deterministically."""
    result = _call(
        "classify_observed_behavior",
        trace=[{"decision": "answer"}],
        pending_confirmation=None,
        answer="What is your employee ID?",
    )

    assert result == "clarify"


def test_classify_observed_behavior_detects_refusal() -> None:
    """R06: unsupported-policy refusal remains distinguishable."""
    result = _call(
        "classify_observed_behavior",
        trace=[{"decision": "answer"}],
        pending_confirmation=None,
        answer=(
            "The available policy evidence does not establish that "
            "entitlement, so I cannot provide a policy answer."
        ),
    )

    assert result == "refuse"


def test_classify_observed_behavior_detects_escalation() -> None:
    """R06: human-review escalation remains distinguishable."""
    result = _call(
        "classify_observed_behavior",
        trace=[{"decision": "answer"}],
        pending_confirmation=None,
        answer=(
            "I cannot determine whether the allegation is true. "
            "Please contact People and Culture for human review."
        ),
    )

    assert result == "escalate"


def test_classify_agent_status_maps_llm_error_to_provider_failure() -> None:
    """R08: provider failure stays separate from answer quality."""
    status = _call(
        "classify_agent_status",
        trace=[{"decision": "llm_error"}],
        exhausted=False,
    )

    assert status == "generation_provider_error"


def test_classify_agent_status_maps_exhaustion_to_agent_error() -> None:
    """R09: exhausted orchestration is not reported as completed."""
    status = _call(
        "classify_agent_status",
        trace=[{"decision": "max_iterations"}],
        exhausted=True,
    )

    assert status == "agent_error"


def test_classify_agent_status_completed_for_normal_result() -> None:
    """R09: ordinary terminal answer remains completed."""
    status = _call(
        "classify_agent_status",
        trace=[{"decision": "answer"}],
        exhausted=False,
    )

    assert status == "completed"


def test_completed_item_ids_are_loaded_from_existing_artifact(
    tmp_path: Path,
) -> None:
    """R13: resume mode can skip already completed item IDs."""
    path = tmp_path / "results.json"

    path.write_text(
        json.dumps(
            {
                "items": [
                    {"id": "SP01", "status": "completed"},
                    {
                        "id": "SP02",
                        "status": "generation_provider_error",
                    },
                    {"id": "SP03", "status": "completed"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _call("load_completed_item_ids", path) == {
        "SP01",
        "SP03",
    }


def test_missing_resume_artifact_returns_empty_completed_set(
    tmp_path: Path,
) -> None:
    """R14: first run has no completed IDs to skip."""
    assert _call(
        "load_completed_item_ids",
        tmp_path / "missing.json",
    ) == set()


def test_atomic_write_json_creates_parent_and_valid_json(
    tmp_path: Path,
) -> None:
    """R15/R16: result persistence is atomic and creates output dir."""
    path = tmp_path / "nested" / "results.json"

    _call(
        "atomic_write_json",
        path,
        {"status": "ok", "items": []},
    )

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "ok",
        "items": [],
    }

    temporary_files = list(path.parent.glob("*.tmp"))
    assert temporary_files == []


def test_default_retrieval_k_is_canonical_baseline() -> None:
    """R17: canonical baseline defaults to frozen k=5."""
    module = _load_run_eval()
    assert module.DEFAULT_RETRIEVAL_K == 5


def test_validate_retrieval_k_accepts_governed_values() -> None:
    """R18: runner accepts exactly the governed k values."""
    for k in (3, 5, 8):
        assert _call("validate_retrieval_k", k) == k


def test_validate_retrieval_k_rejects_other_values() -> None:
    """R18: unsupported retrieval configurations fail early."""
    with pytest.raises(ValueError):
        _call("validate_retrieval_k", 7)


def test_cli_parser_exposes_k_output_and_resume() -> None:
    """R19: CLI exposes the minimal governed execution controls."""
    parser = _call("build_cli_parser")

    args = parser.parse_args(
        [
            "--k",
            "8",
            "--output",
            "evaluation/results/test.json",
            "--resume",
        ]
    )

    assert args.k == 8
    assert str(args.output).endswith(
        "evaluation/results/test.json"
    )
    assert args.resume is True


def test_cli_parser_defaults_to_k5_and_no_resume() -> None:
    """R17/R19: default invocation is canonical k=5 baseline."""
    parser = _call("build_cli_parser")
    args = parser.parse_args([])

    assert args.k == 5
    assert args.resume is False


# ============================================================
# S9 async agent-runtime orchestration contracts
# ============================================================


class _FakeTraceItem:
    def __init__(
        self,
        *,
        step: int,
        tool: str | None,
        decision: str,
        arguments: dict[str, object] | None = None,
        result_summary: str = "",
        sources: tuple[dict[str, str], ...] = (),
        prompt_version: str = "1.9",
    ) -> None:
        self.step = step
        self.tool = tool
        self.arguments = arguments or {}
        self.result_summary = result_summary
        self.sources = sources
        self.decision = decision
        self.prompt_version = prompt_version


class _FakePending:
    def __init__(self) -> None:
        self.confirmation_id = "confirm-123"
        self.tool = "draft_hr_email"
        self.arguments = {"employee_id": "E001"}
        self.preview = "Draft PTO email"


class _FakeAgentResult:
    def __init__(
        self,
        *,
        answer: str,
        citations: tuple[dict[str, str], ...] = (),
        trace: tuple[object, ...] = (),
        exhausted: bool = False,
        pending_confirmation: object | None = None,
    ) -> None:
        self.answer = answer
        self.citations = citations
        self.trace = trace
        self.exhausted = exhausted
        self.pending_confirmation = pending_confirmation


class _FakeMCPClient:
    def __init__(self) -> None:
        self.start_calls = 0
        self.close_calls = 0

    async def start(self) -> tuple[object, ...]:
        self.start_calls += 1
        return ()

    async def close(self) -> None:
        self.close_calls += 1


class _FakeLLM:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


async def _async_test_run_eval_item_calls_existing_agent_turn_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R20/R21: one evaluated item delegates to production run_turn once."""
    module = _load_run_eval()

    calls: list[dict[str, object]] = []

    async def fake_run_turn(
        *,
        message: str,
        mcp_client: object,
        llm: object,
        history: object = None,
    ) -> object:
        calls.append(
            {
                "message": message,
                "mcp_client": mcp_client,
                "llm": llm,
                "history": history,
            }
        )
        return _FakeAgentResult(
            answer="Full-time employees receive 20 days of PTO.",
            citations=(
                {
                    "doc_id": "HR-POL-002",
                    "section": "4.1 Annual entitlement",
                },
            ),
            trace=(
                _FakeTraceItem(
                    step=1,
                    tool="search_policy_documents",
                    decision="tool_result",
                ),
                _FakeTraceItem(
                    step=2,
                    tool=None,
                    decision="answer",
                ),
            ),
        )

    monkeypatch.setattr(
        module,
        "_agent_run_turn",
        fake_run_turn,
    )

    item = {
        "id": "SP01",
        "category": "simple_policy",
        "prompt": "How much PTO does a full-time employee receive?",
        "gold_doc_ids": ["HR-POL-002"],
        "required_tools": ["search_policy_documents"],
        "allowed_optional_tools": ["get_policy_section"],
        "forbidden_tools": [
            "create_mock_hr_ticket",
            "draft_hr_email",
        ],
        "expected_behavior": "answer",
        "requires_confirmation": False,
    }

    mcp_client = object()
    llm = object()

    result = await module.run_eval_item(
        item=item,
        mcp_client=mcp_client,
        llm=llm,
        retrieval_k=5,
    )

    assert len(calls) == 1
    assert calls[0]["message"] == item["prompt"]
    assert calls[0]["mcp_client"] is mcp_client
    assert calls[0]["llm"] is llm

    assert result["id"] == "SP01"
    assert result["status"] == "completed"
    assert result["answer"].startswith("Full-time employees")
    assert result["observed_tools"] == [
        "search_policy_documents",
    ]
    assert result["observed_behavior"] == "answer"
    assert result["scores"]["tool_selection"] is True
    assert result["scores"]["workflow_completion"] is True


async def _async_test_run_eval_item_records_positive_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R22: every live item records elapsed wall-clock milliseconds."""
    module = _load_run_eval()

    async def fake_run_turn(**_: object) -> object:
        return _FakeAgentResult(
            answer="Answer.",
            trace=(
                _FakeTraceItem(
                    step=1,
                    tool=None,
                    decision="answer",
                ),
            ),
        )

    monkeypatch.setattr(
        module,
        "_agent_run_turn",
        fake_run_turn,
    )

    item = {
        "id": "TL05",
        "category": "tool_task",
        "prompt": "Example",
        "gold_doc_ids": [],
        "required_tools": [],
        "allowed_optional_tools": [],
        "forbidden_tools": [],
        "expected_behavior": "answer",
        "requires_confirmation": False,
    }

    result = await module.run_eval_item(
        item=item,
        mcp_client=object(),
        llm=object(),
        retrieval_k=5,
    )

    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


async def _async_test_run_eval_item_preserves_pending_confirmation_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R23: ACTION proposal is recorded but never auto-confirmed."""
    module = _load_run_eval()

    async def fake_run_turn(**_: object) -> object:
        return _FakeAgentResult(
            answer="Please confirm the draft.",
            trace=(
                _FakeTraceItem(
                    step=1,
                    tool="draft_hr_email",
                    decision="confirmation_required",
                ),
            ),
            pending_confirmation=_FakePending(),
        )

    monkeypatch.setattr(
        module,
        "_agent_run_turn",
        fake_run_turn,
    )

    item = {
        "id": "TL02",
        "category": "tool_task",
        "prompt": "I'm E001. Can I take 3 days of PTO?",
        "gold_doc_ids": ["HR-POL-002"],
        "required_tools": ["draft_hr_email"],
        "allowed_optional_tools": [],
        "forbidden_tools": [],
        "expected_behavior": "propose_action",
        "requires_confirmation": True,
    }

    result = await module.run_eval_item(
        item=item,
        mcp_client=object(),
        llm=object(),
        retrieval_k=5,
    )

    assert result["status"] == "completed"
    assert result["observed_behavior"] == "propose_action"
    assert result["scores"]["workflow_completion"] is True
    assert result["scores"]["action_safety"] is True
    assert result["trace"][-1]["decision"] == "confirmation_required"


async def _async_test_run_eval_item_maps_llm_error_without_quality_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R24: provider failure persists separately from model quality."""
    module = _load_run_eval()

    async def fake_run_turn(**_: object) -> object:
        return _FakeAgentResult(
            answer="I cannot complete that request.",
            trace=(
                _FakeTraceItem(
                    step=1,
                    tool=None,
                    decision="llm_error",
                ),
            ),
        )

    monkeypatch.setattr(
        module,
        "_agent_run_turn",
        fake_run_turn,
    )

    item = {
        "id": "SP01",
        "category": "simple_policy",
        "prompt": "Example",
        "gold_doc_ids": ["HR-POL-002"],
        "required_tools": ["search_policy_documents"],
        "allowed_optional_tools": [],
        "forbidden_tools": [],
        "expected_behavior": "answer",
        "requires_confirmation": False,
    }

    result = await module.run_eval_item(
        item=item,
        mcp_client=object(),
        llm=object(),
        retrieval_k=5,
    )

    assert result["status"] == "generation_provider_error"
    assert result["observed_behavior"] is None
    assert result["scores"]["workflow_completion"] is False
    assert result["scores"]["groundedness"] is None
    assert result["scores"]["citation_accuracy"] is None


async def _async_test_run_evaluation_starts_and_closes_clients_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R25/R26: one run owns one MCP/LLM lifecycle."""
    module = _load_run_eval()

    fake_mcp = _FakeMCPClient()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(
        module,
        "_make_mcp_client",
        lambda: fake_mcp,
    )
    monkeypatch.setattr(
        module,
        "_make_llm_client",
        lambda: fake_llm,
    )

    async def fake_run_eval_item(**kwargs: object) -> dict[str, object]:
        item = kwargs["item"]
        assert isinstance(item, dict)
        return {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "status": "completed",
            "answer": "ok",
            "observed_behavior": "answer",
            "retrieved_doc_ids": [],
            "citations": [],
            "observed_tools": [],
            "trace": [],
            "latency_ms": 1.0,
            "scores": {
                "recall_at_k": None,
                "groundedness": None,
                "citation_accuracy": None,
                "tool_selection": True,
                "workflow_completion": True,
                "boundary_behavior": None,
                "action_safety": True,
            },
            "error": None,
        }

    monkeypatch.setattr(
        module,
        "run_eval_item",
        fake_run_eval_item,
    )

    monkeypatch.setattr(
        module,
        "load_eval_items",
        lambda _: [
            {
                "id": "SP01",
                "category": "simple_policy",
                "prompt": "Example",
            }
        ],
    )

    output = tmp_path / "result.json"

    await module.run_evaluation(
        eval_set_path=Path("ignored.jsonl"),
        output_path=output,
        retrieval_k=5,
        resume=False,
    )

    assert fake_mcp.start_calls == 1
    assert fake_mcp.close_calls == 1
    assert fake_llm.close_calls == 1
    assert output.is_file()


async def _async_test_run_evaluation_closes_clients_after_item_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R26: resource cleanup occurs even when item execution raises."""
    module = _load_run_eval()

    fake_mcp = _FakeMCPClient()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(
        module,
        "_make_mcp_client",
        lambda: fake_mcp,
    )
    monkeypatch.setattr(
        module,
        "_make_llm_client",
        lambda: fake_llm,
    )

    monkeypatch.setattr(
        module,
        "load_eval_items",
        lambda _: [
            {
                "id": "SP01",
                "category": "simple_policy",
                "prompt": "Example",
            }
        ],
    )

    async def exploding_item(**_: object) -> dict[str, object]:
        raise RuntimeError("simulated evaluator failure")

    monkeypatch.setattr(
        module,
        "run_eval_item",
        exploding_item,
    )

    with pytest.raises(RuntimeError, match="simulated evaluator failure"):
        await module.run_evaluation(
            eval_set_path=Path("ignored.jsonl"),
            output_path=tmp_path / "result.json",
            retrieval_k=5,
            resume=False,
        )

    assert fake_mcp.start_calls == 1
    assert fake_mcp.close_calls == 1
    assert fake_llm.close_calls == 1


async def _async_test_run_evaluation_resume_skips_completed_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R27: resume skips completed rows and reruns incomplete rows."""
    module = _load_run_eval()

    fake_mcp = _FakeMCPClient()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(
        module,
        "_make_mcp_client",
        lambda: fake_mcp,
    )
    monkeypatch.setattr(
        module,
        "_make_llm_client",
        lambda: fake_llm,
    )

    items = [
        {
            "id": "SP01",
            "category": "simple_policy",
            "prompt": "One",
        },
        {
            "id": "SP02",
            "category": "simple_policy",
            "prompt": "Two",
        },
    ]

    monkeypatch.setattr(
        module,
        "load_eval_items",
        lambda _: items,
    )

    executed: list[str] = []

    async def fake_run_eval_item(**kwargs: object) -> dict[str, object]:
        item = kwargs["item"]
        assert isinstance(item, dict)
        item_id = str(item["id"])
        executed.append(item_id)
        return {
            "id": item_id,
            "category": item["category"],
            "prompt": item["prompt"],
            "status": "completed",
            "answer": "ok",
            "observed_behavior": "answer",
            "retrieved_doc_ids": [],
            "citations": [],
            "observed_tools": [],
            "trace": [],
            "latency_ms": 1.0,
            "scores": {
                "recall_at_k": None,
                "groundedness": None,
                "citation_accuracy": None,
                "tool_selection": True,
                "workflow_completion": True,
                "boundary_behavior": None,
                "action_safety": True,
            },
            "error": None,
        }

    monkeypatch.setattr(
        module,
        "run_eval_item",
        fake_run_eval_item,
    )

    output = tmp_path / "result.json"

    output.write_text(
        json.dumps(
            {
                "metadata": {},
                "items": [
                    {
                        "id": "SP01",
                        "status": "completed",
                    }
                ],
                "metrics": {},
                "latency": {},
                "failure_recovery": {},
            }
        ),
        encoding="utf-8",
    )

    await module.run_evaluation(
        eval_set_path=Path("ignored.jsonl"),
        output_path=output,
        retrieval_k=5,
        resume=True,
    )

    assert executed == ["SP02"]


async def _async_test_run_evaluation_checkpoints_after_each_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R28: completed evidence is checkpointed incrementally."""
    module = _load_run_eval()

    fake_mcp = _FakeMCPClient()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(
        module,
        "_make_mcp_client",
        lambda: fake_mcp,
    )
    monkeypatch.setattr(
        module,
        "_make_llm_client",
        lambda: fake_llm,
    )

    monkeypatch.setattr(
        module,
        "load_eval_items",
        lambda _: [
            {
                "id": "SP01",
                "category": "simple_policy",
                "prompt": "One",
            },
            {
                "id": "SP02",
                "category": "simple_policy",
                "prompt": "Two",
            },
        ],
    )

    async def fake_run_eval_item(**kwargs: object) -> dict[str, object]:
        item = kwargs["item"]
        assert isinstance(item, dict)
        return {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "status": "completed",
            "answer": "ok",
            "observed_behavior": "answer",
            "retrieved_doc_ids": [],
            "citations": [],
            "observed_tools": [],
            "trace": [],
            "latency_ms": 1.0,
            "scores": {
                "recall_at_k": None,
                "groundedness": None,
                "citation_accuracy": None,
                "tool_selection": True,
                "workflow_completion": True,
                "boundary_behavior": None,
                "action_safety": True,
            },
            "error": None,
        }

    monkeypatch.setattr(
        module,
        "run_eval_item",
        fake_run_eval_item,
    )

    writes: list[dict[str, object]] = []

    real_atomic_write_json = module.atomic_write_json

    def recording_write(
        path: Path,
        payload: object,
    ) -> None:
        assert isinstance(payload, dict)
        writes.append(payload)
        real_atomic_write_json(path, payload)

    monkeypatch.setattr(
        module,
        "atomic_write_json",
        recording_write,
    )

    await module.run_evaluation(
        eval_set_path=Path("ignored.jsonl"),
        output_path=tmp_path / "result.json",
        retrieval_k=5,
        resume=False,
    )

    assert len(writes) >= 2
    assert len(writes[0]["items"]) == 1
    assert len(writes[-1]["items"]) == 2



# Standard-library async execution wrappers.
# Deliberately avoids adding pytest-asyncio to the frozen dependency set.

def test_run_eval_item_calls_existing_agent_turn_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _async_test_run_eval_item_calls_existing_agent_turn_once(
            monkeypatch
        )
    )


def test_run_eval_item_records_positive_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _async_test_run_eval_item_records_positive_latency(
            monkeypatch
        )
    )


def test_run_eval_item_preserves_pending_confirmation_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _async_test_run_eval_item_preserves_pending_confirmation_without_execution(
            monkeypatch
        )
    )


def test_run_eval_item_maps_llm_error_without_quality_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _async_test_run_eval_item_maps_llm_error_without_quality_scoring(
            monkeypatch
        )
    )


def test_run_evaluation_starts_and_closes_clients_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _async_test_run_evaluation_starts_and_closes_clients_once(
            monkeypatch,
            tmp_path,
        )
    )


def test_run_evaluation_closes_clients_after_item_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _async_test_run_evaluation_closes_clients_after_item_failure(
            monkeypatch,
            tmp_path,
        )
    )


def test_run_evaluation_resume_skips_completed_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _async_test_run_evaluation_resume_skips_completed_items(
            monkeypatch,
            tmp_path,
        )
    )


def test_run_evaluation_checkpoints_after_each_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _async_test_run_evaluation_checkpoints_after_each_item(
            monkeypatch,
            tmp_path,
        )
    )


# ============================================================
# S9 live judge + full provenance contracts
# ============================================================


class _FakeJudgeResponse:
    def __init__(
        self,
        content: str | None,
        tool_calls: tuple[object, ...] = (),
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeJudgeClient:
    def __init__(
        self,
        *,
        content: str | None,
    ) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []
        self.close_calls = 0

    async def chat(
        self,
        *,
        messages: object,
        tools: object = (),
    ) -> object:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
            }
        )
        return _FakeJudgeResponse(
            content=self.content,
        )

    async def close(self) -> None:
        self.close_calls += 1


async def _async_test_run_judge_uses_no_tools_and_strict_json() -> None:
    module = _load_run_eval()

    client = _FakeJudgeClient(
        content=json.dumps(
            {
                "groundedness_score": 2,
                "supported_citations": 1,
                "total_citations": 1,
                "rationale": "The supplied evidence supports the answer.",
            }
        )
    )

    result = await module.run_judge(
        judge_client=client,
        answer="Employees receive 20 days of PTO.",
        citations=[
            {
                "doc_id": "HR-POL-002",
                "section": "4.1 Annual entitlement",
            }
        ],
        evidence=[
            {
                "doc_id": "HR-POL-002",
                "section": "4.1 Annual entitlement",
                "text": "Full-time employees receive 20 days per year.",
            }
        ],
    )

    assert result["groundedness"] == 1.0
    assert result["supported_citations"] == 1
    assert result["total_citations"] == 1

    assert len(client.calls) == 1
    assert client.calls[0]["tools"] == ()

    messages = client.calls[0]["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert module.JUDGE_SYSTEM_PROMPT in messages[0]["content"]


def test_run_judge_uses_no_tools_and_strict_json() -> None:
    asyncio.run(
        _async_test_run_judge_uses_no_tools_and_strict_json()
    )


async def _async_test_run_judge_rejects_missing_content() -> None:
    module = _load_run_eval()

    client = _FakeJudgeClient(
        content=None,
    )

    with pytest.raises(ValueError):
        await module.run_judge(
            judge_client=client,
            answer="Answer",
            citations=[],
            evidence=[],
        )


def test_run_judge_rejects_missing_content() -> None:
    asyncio.run(
        _async_test_run_judge_rejects_missing_content()
    )


async def _async_test_run_judge_rejects_invalid_json() -> None:
    module = _load_run_eval()

    client = _FakeJudgeClient(
        content="not-json",
    )

    with pytest.raises(ValueError):
        await module.run_judge(
            judge_client=client,
            answer="Answer",
            citations=[],
            evidence=[],
        )


def test_run_judge_rejects_invalid_json() -> None:
    asyncio.run(
        _async_test_run_judge_rejects_invalid_json()
    )


def test_build_live_run_metadata_derives_repository_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_eval()

    monkeypatch.setattr(
        module,
        "_current_git_commit",
        lambda: "bbbe3b7-test",
    )
    monkeypatch.setattr(
        module,
        "_gold_set_sha256",
        lambda _: (
            "96de38969c324cc8c578076479e501f7"
            "bff8fb9767134b96cd2183684030fb98"
        ),
    )
    monkeypatch.setattr(
        module,
        "_utc_timestamp",
        lambda: "2026-08-29T06:30:00Z",
    )

    metadata = module.build_live_run_metadata(
        generation_model="generation-model",
        judge_model="judge-model",
        llm_base_url="https://example.test/v1",
        retrieval_k=5,
        run_type="canonical_baseline",
        gold_set_path=EVAL_SET_PATH,
        seed=0,
    )

    assert metadata["generation_model"] == "generation-model"
    assert metadata["judge_model"] == "judge-model"
    assert metadata["prompt_version"] == "1.9"
    assert metadata["judge_prompt_version"] == "1.0"
    assert metadata["corpus_version"] == "1.2"
    assert metadata["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert metadata["retrieval_k"] == 5
    assert metadata["git_commit"] == "bbbe3b7-test"
    assert metadata["temperature"] == 0
    assert metadata["seed"] == 0
    assert metadata["run_type"] == "canonical_baseline"


def test_full_checkpoint_uses_governed_metadata(
    tmp_path: Path,
) -> None:
    module = _load_run_eval()

    metadata = module.build_run_metadata(
        generation_model="generation-model",
        judge_model="judge-model",
        llm_base_url="https://example.test/v1",
        prompt_version="1.9",
        corpus_version="1.2",
        embedding_model="BAAI/bge-small-en-v1.5",
        retrieval_k=5,
        timestamp="2026-08-29T06:30:00Z",
        git_commit="abc123",
        gold_set_sha256="goldhash",
        temperature=0,
        seed=0,
        run_type="canonical_baseline",
    )

    output = tmp_path / "result.json"

    module._checkpoint_artifact(
        output_path=output,
        metadata=metadata,
        items=[],
        partial=True,
    )

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert payload["metadata"]["generation_model"] == "generation-model"
    assert payload["metadata"]["judge_model"] == "judge-model"
    assert payload["metadata"]["judge_prompt_version"] == "1.0"
    assert payload["metadata"]["partial"] is True


async def _async_test_run_eval_item_applies_judge_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_eval()

    async def fake_run_turn(**_: object) -> object:
        return _FakeAgentResult(
            answer="Full-time employees receive 20 days of PTO.",
            citations=(
                {
                    "doc_id": "HR-POL-002",
                    "section": "4.1 Annual entitlement",
                },
            ),
            trace=(
                _FakeTraceItem(
                    step=1,
                    tool="search_policy_documents",
                    decision="tool_result",
                    sources=(
                        {
                            "doc_id": "HR-POL-002",
                            "section": "4.1 Annual entitlement",
                        },
                    ),
                ),
                _FakeTraceItem(
                    step=2,
                    tool=None,
                    decision="answer",
                ),
            ),
        )

    async def fake_run_judge(**_: object) -> dict[str, object]:
        return {
            "groundedness_score": 2,
            "groundedness": 1.0,
            "supported_citations": 1,
            "total_citations": 1,
            "rationale": "Supported.",
        }

    monkeypatch.setattr(
        module,
        "_agent_run_turn",
        fake_run_turn,
    )
    monkeypatch.setattr(
        module,
        "run_judge",
        fake_run_judge,
    )

    item = {
        "id": "SP01",
        "category": "simple_policy",
        "prompt": "How much PTO?",
        "gold_doc_ids": ["HR-POL-002"],
        "required_tools": ["search_policy_documents"],
        "allowed_optional_tools": [],
        "forbidden_tools": [],
        "expected_behavior": "answer",
        "requires_confirmation": False,
    }

    result = await module.run_eval_item(
        item=item,
        mcp_client=object(),
        llm=object(),
        judge_client=object(),
        retrieval_k=5,
    )

    assert result["scores"]["groundedness"] == 1.0
    assert result["scores"]["citation_accuracy"] == 1.0


def test_run_eval_item_applies_judge_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _async_test_run_eval_item_applies_judge_scores(
            monkeypatch
        )
    )


async def _async_test_judge_failure_preserves_generation_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_eval()

    async def fake_run_turn(**_: object) -> object:
        return _FakeAgentResult(
            answer="Generated answer.",
            trace=(
                _FakeTraceItem(
                    step=1,
                    tool=None,
                    decision="answer",
                ),
            ),
        )

    async def failing_judge(**_: object) -> dict[str, object]:
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(
        module,
        "_agent_run_turn",
        fake_run_turn,
    )
    monkeypatch.setattr(
        module,
        "run_judge",
        failing_judge,
    )

    item = {
        "id": "SP01",
        "category": "simple_policy",
        "prompt": "Example",
        "gold_doc_ids": ["HR-POL-002"],
        "required_tools": [],
        "allowed_optional_tools": [],
        "forbidden_tools": [],
        "expected_behavior": "answer",
        "requires_confirmation": False,
    }

    result = await module.run_eval_item(
        item=item,
        mcp_client=object(),
        llm=object(),
        judge_client=object(),
        retrieval_k=5,
    )

    assert result["answer"] == "Generated answer."
    assert result["status"] == "judge_provider_error"
    assert result["scores"]["groundedness"] is None
    assert result["scores"]["citation_accuracy"] is None


def test_judge_failure_preserves_generation_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _async_test_judge_failure_preserves_generation_result(
            monkeypatch
        )
    )


# ============================================================
# S9 canonical live-run lifecycle contracts
# ============================================================


class _FakeConfiguredClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


async def _async_test_run_evaluation_owns_separate_judge_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_run_eval()

    fake_mcp = _FakeMCPClient()
    fake_generation = _FakeConfiguredClient()
    fake_judge = _FakeConfiguredClient()

    monkeypatch.setattr(
        module,
        "_make_mcp_client",
        lambda: fake_mcp,
    )
    monkeypatch.setattr(
        module,
        "_make_llm_client",
        lambda: fake_generation,
    )
    monkeypatch.setattr(
        module,
        "_make_judge_client",
        lambda *, model: fake_judge,
    )

    monkeypatch.setattr(
        module,
        "load_eval_items",
        lambda _: [
            {
                "id": "SP01",
                "category": "simple_policy",
                "prompt": "Example",
            }
        ],
    )

    seen: list[dict[str, object]] = []

    async def fake_run_eval_item(**kwargs: object) -> dict[str, object]:
        seen.append(dict(kwargs))
        item = kwargs["item"]
        assert isinstance(item, dict)

        return {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "status": "completed",
            "answer": "ok",
            "observed_behavior": "answer",
            "retrieved_doc_ids": [],
            "citations": [],
            "observed_tools": [],
            "trace": [],
            "latency_ms": 1.0,
            "scores": {
                "recall_at_k": None,
                "groundedness": 1.0,
                "citation_accuracy": 1.0,
                "tool_selection": True,
                "workflow_completion": True,
                "boundary_behavior": None,
                "action_safety": True,
            },
            "error": None,
        }

    monkeypatch.setattr(
        module,
        "run_eval_item",
        fake_run_eval_item,
    )

    metadata = {
        field: "value"
        for field in module.REQUIRED_RUN_METADATA_FIELDS
    }
    metadata["retrieval_k"] = 5
    metadata["temperature"] = 0
    metadata["judge_prompt_version"] = "1.0"

    monkeypatch.setattr(
        module,
        "build_live_run_metadata",
        lambda **_: metadata,
    )

    output = tmp_path / "result.json"

    await module.run_evaluation(
        eval_set_path=Path("ignored.jsonl"),
        output_path=output,
        retrieval_k=5,
        resume=False,
        generation_model="generation-model",
        judge_model="judge-model",
        llm_base_url="https://example.test/v1",
        run_type="smoke",
        seed=0,
    )

    assert fake_mcp.start_calls == 1
    assert fake_mcp.close_calls == 1
    assert fake_generation.close_calls == 1
    assert fake_judge.close_calls == 1

    assert len(seen) == 1
    assert seen[0]["llm"] is fake_generation
    assert seen[0]["judge_client"] is fake_judge


def test_run_evaluation_owns_separate_judge_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _async_test_run_evaluation_owns_separate_judge_client(
            monkeypatch,
            tmp_path,
        )
    )


async def _async_test_run_evaluation_persists_full_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_run_eval()

    fake_mcp = _FakeMCPClient()
    fake_generation = _FakeConfiguredClient()
    fake_judge = _FakeConfiguredClient()

    monkeypatch.setattr(
        module,
        "_make_mcp_client",
        lambda: fake_mcp,
    )
    monkeypatch.setattr(
        module,
        "_make_llm_client",
        lambda: fake_generation,
    )
    monkeypatch.setattr(
        module,
        "_make_judge_client",
        lambda *, model: fake_judge,
    )

    monkeypatch.setattr(
        module,
        "load_eval_items",
        lambda _: [],
    )

    metadata = {
        "generation_model": "generation-model",
        "judge_model": "judge-model",
        "judge_prompt_version": "1.0",
        "llm_base_url": "https://example.test/v1",
        "prompt_version": "1.9",
        "corpus_version": "1.2",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "retrieval_k": 5,
        "timestamp": "2026-08-29T06:40:00Z",
        "git_commit": "abc123",
        "gold_set_sha256": "goldhash",
        "temperature": 0,
        "seed": 0,
        "run_type": "smoke",
    }

    monkeypatch.setattr(
        module,
        "build_live_run_metadata",
        lambda **_: metadata,
    )

    output = tmp_path / "result.json"

    result = await module.run_evaluation(
        eval_set_path=Path("ignored.jsonl"),
        output_path=output,
        retrieval_k=5,
        resume=False,
        generation_model="generation-model",
        judge_model="judge-model",
        llm_base_url="https://example.test/v1",
        run_type="smoke",
        seed=0,
    )

    assert result["metadata"]["generation_model"] == "generation-model"
    assert result["metadata"]["judge_model"] == "judge-model"
    assert result["metadata"]["partial"] is False

    persisted = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert persisted["metadata"]["prompt_version"] == "1.9"
    assert persisted["metadata"]["judge_prompt_version"] == "1.0"
    assert persisted["metadata"]["partial"] is False


def test_run_evaluation_persists_full_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _async_test_run_evaluation_persists_full_metadata(
            monkeypatch,
            tmp_path,
        )
    )


async def _async_test_run_evaluation_closes_judge_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_run_eval()

    fake_mcp = _FakeMCPClient()
    fake_generation = _FakeConfiguredClient()
    fake_judge = _FakeConfiguredClient()

    monkeypatch.setattr(
        module,
        "_make_mcp_client",
        lambda: fake_mcp,
    )
    monkeypatch.setattr(
        module,
        "_make_llm_client",
        lambda: fake_generation,
    )
    monkeypatch.setattr(
        module,
        "_make_judge_client",
        lambda *, model: fake_judge,
    )

    monkeypatch.setattr(
        module,
        "load_eval_items",
        lambda _: [
            {
                "id": "SP01",
                "category": "simple_policy",
                "prompt": "Example",
            }
        ],
    )

    monkeypatch.setattr(
        module,
        "build_live_run_metadata",
        lambda **_: {
            field: "value"
            for field in module.REQUIRED_RUN_METADATA_FIELDS
        },
    )

    async def exploding_item(**_: object) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        module,
        "run_eval_item",
        exploding_item,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await module.run_evaluation(
            eval_set_path=Path("ignored.jsonl"),
            output_path=tmp_path / "result.json",
            retrieval_k=5,
            resume=False,
            generation_model="generation-model",
            judge_model="judge-model",
            llm_base_url="https://example.test/v1",
            run_type="smoke",
            seed=0,
        )

    assert fake_mcp.close_calls == 1
    assert fake_generation.close_calls == 1
    assert fake_judge.close_calls == 1


def test_run_evaluation_closes_judge_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _async_test_run_evaluation_closes_judge_after_failure(
            monkeypatch,
            tmp_path,
        )
    )


def test_make_judge_client_uses_explicit_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_eval()

    captured: dict[str, object] = {}

    class FakeLLMClient:
        def __init__(
            self,
            *,
            model: str | None = None,
        ) -> None:
            captured["model"] = model

    monkeypatch.setattr(
        module,
        "LLMClient",
        FakeLLMClient,
    )

    module._make_judge_client(
        model="judge-model",
    )

    assert captured["model"] == "judge-model"
