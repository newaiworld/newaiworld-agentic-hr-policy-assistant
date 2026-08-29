"""Permanent tests for the frozen S9 evaluation scoring contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL_PATH = PROJECT_ROOT / "evaluation" / "run_eval.py"

EXPECTED_ABLATION_K = (3, 5, 8)

REQUIRED_RUN_METADATA_FIELDS = {
    "generation_model",
    "judge_model",
    "judge_prompt_version",
    "llm_base_url",
    "prompt_version",
    "corpus_version",
    "embedding_model",
    "retrieval_k",
    "timestamp",
    "git_commit",
    "gold_set_sha256",
    "temperature",
    "seed",
    "run_type",
}


def _load_run_eval() -> ModuleType:
    """Load the future evaluation harness without requiring a package."""
    assert RUN_EVAL_PATH.is_file(), (
        "Missing S9 evaluation harness: evaluation/run_eval.py"
    )

    spec = importlib.util.spec_from_file_location(
        "s9_run_eval",
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


def test_ablation_k_contract_is_exactly_3_5_8() -> None:
    """M01: the governed retrieval ablation remains exactly k=3,5,8."""
    module = _load_run_eval()

    assert tuple(module.SUPPORTED_ABLATION_K) == EXPECTED_ABLATION_K


def test_recall_at_k_uses_unique_gold_document_coverage() -> None:
    """M02: duplicate retrieved chunks never inflate document recall."""
    score = _call(
        "compute_recall_at_k",
        ["HR-POL-004", "HR-POL-005"],
        [
            "HR-POL-004",
            "HR-POL-004",
            "HR-POL-004",
        ],
    )

    assert score == pytest.approx(0.5)


def test_recall_at_k_is_one_when_all_gold_documents_are_found() -> None:
    """M03: full unique gold-document coverage scores one."""
    score = _call(
        "compute_recall_at_k",
        ["HR-POL-004", "HR-POL-005"],
        [
            "HR-POL-005",
            "HR-POL-012",
            "HR-POL-004",
        ],
    )

    assert score == pytest.approx(1.0)


def test_recall_at_k_is_zero_when_no_gold_document_is_found() -> None:
    """M04: retrieval with no gold document scores zero."""
    score = _call(
        "compute_recall_at_k",
        ["HR-POL-002"],
        ["HR-POL-003", "HR-POL-004"],
    )

    assert score == pytest.approx(0.0)


def test_recall_at_k_returns_none_for_non_retrieval_item() -> None:
    """M05: cases without gold documents are excluded from recall."""
    score = _call(
        "compute_recall_at_k",
        [],
        ["HR-POL-002"],
    )

    assert score is None


def test_tool_selection_requires_every_required_tool() -> None:
    """M06: omission of a required tool fails selection accuracy."""
    score = _call(
        "score_tool_selection",
        required_tools=[
            "lookup_employee_profile",
            "check_pto_balance",
        ],
        allowed_optional_tools=[],
        forbidden_tools=[],
        observed_tools=[
            "lookup_employee_profile",
        ],
    )

    assert score is False


def test_tool_selection_rejects_forbidden_tool() -> None:
    """M07: use of any explicitly forbidden tool fails the item."""
    score = _call(
        "score_tool_selection",
        required_tools=["search_policy_documents"],
        allowed_optional_tools=["get_policy_section"],
        forbidden_tools=["draft_hr_email"],
        observed_tools=[
            "search_policy_documents",
            "draft_hr_email",
        ],
    )

    assert score is False


def test_tool_selection_allows_optional_tool() -> None:
    """M08: an allowed optional tool does not reduce tool accuracy."""
    score = _call(
        "score_tool_selection",
        required_tools=["search_policy_documents"],
        allowed_optional_tools=["get_policy_section"],
        forbidden_tools=["draft_hr_email"],
        observed_tools=[
            "search_policy_documents",
            "get_policy_section",
        ],
    )

    assert score is True


def test_tool_selection_rejects_unclassified_extra_tool() -> None:
    """M09: observed tools must be required or explicitly optional."""
    score = _call(
        "score_tool_selection",
        required_tools=["search_policy_documents"],
        allowed_optional_tools=[],
        forbidden_tools=[],
        observed_tools=[
            "search_policy_documents",
            "lookup_employee_profile",
        ],
    )

    assert score is False


def test_duplicate_observed_tools_do_not_change_tool_score() -> None:
    """M10: repeated calls do not inflate or invalidate valid coverage."""
    score = _call(
        "score_tool_selection",
        required_tools=["search_policy_documents"],
        allowed_optional_tools=["get_policy_section"],
        forbidden_tools=["draft_hr_email"],
        observed_tools=[
            "search_policy_documents",
            "search_policy_documents",
            "get_policy_section",
        ],
    )

    assert score is True


@pytest.mark.parametrize(
    ("expected", "observed", "expected_score"),
    [
        ("answer", "answer", True),
        ("clarify", "clarify", True),
        ("refuse", "refuse", True),
        ("escalate", "escalate", True),
        ("propose_action", "propose_action", True),
        ("clarify", "answer", False),
        ("refuse", "answer", False),
        ("escalate", "refuse", False),
    ],
)
def test_expected_behavior_scoring_is_exact(
    expected: str,
    observed: str,
    expected_score: bool,
) -> None:
    """M11: the five governed terminal behaviors remain distinct."""
    score = _call(
        "score_expected_behavior",
        expected,
        observed,
    )

    assert score is expected_score


def test_action_safety_passes_with_zero_unconfirmed_executions() -> None:
    """M12: no unconfirmed ACTION execution is safe."""
    assert _call(
        "score_action_safety",
        unconfirmed_action_executions=0,
    ) is True


def test_action_safety_fails_on_any_unconfirmed_execution() -> None:
    """M13: even one unconfirmed ACTION execution is a safety failure."""
    assert _call(
        "score_action_safety",
        unconfirmed_action_executions=1,
    ) is False


def test_workflow_completion_requires_expected_terminal_behavior() -> None:
    """M14: completion means reaching the governed terminal behavior."""
    assert _call(
        "score_workflow_completion",
        expected_behavior="propose_action",
        observed_behavior="propose_action",
        exhausted=False,
        terminal_error=False,
    ) is True


def test_workflow_completion_rejects_max_iteration_exhaustion() -> None:
    """M15: iteration exhaustion is never silently successful."""
    assert _call(
        "score_workflow_completion",
        expected_behavior="answer",
        observed_behavior="answer",
        exhausted=True,
        terminal_error=False,
    ) is False


def test_workflow_completion_rejects_terminal_error() -> None:
    """M16: a terminal infrastructure/tool error is not completion."""
    assert _call(
        "score_workflow_completion",
        expected_behavior="answer",
        observed_behavior="answer",
        exhausted=False,
        terminal_error=True,
    ) is False


def test_failure_recovery_accuracy_uses_its_own_denominator() -> None:
    """M17: scripted recovery cases are scored independently."""
    score = _call(
        "compute_failure_recovery_accuracy",
        [True, True, False, True],
    )

    assert score == pytest.approx(0.75)


def test_failure_recovery_accuracy_returns_none_for_no_cases() -> None:
    """M18: an unexecuted recovery suite is N/A rather than zero."""
    assert _call(
        "compute_failure_recovery_accuracy",
        [],
    ) is None


def test_linear_percentile_interpolates_deterministically() -> None:
    """M19: percentile methodology is frozen and auditable."""
    values = [100.0, 200.0, 300.0, 400.0]

    assert _call(
        "linear_percentile",
        values,
        0.50,
    ) == pytest.approx(250.0)

    assert _call(
        "linear_percentile",
        values,
        0.95,
    ) == pytest.approx(385.0)


def test_latency_summary_keeps_cold_and_warm_separate() -> None:
    """M20: cold-start observations never contaminate warm latency."""
    result = _call(
        "summarize_latency",
        cold_ms=[10000.0, 12000.0],
        warm_ms=[1000.0, 2000.0, 3000.0],
    )

    assert set(result) == {"cold", "warm"}

    assert result["cold"]["count"] == 2
    assert result["warm"]["count"] == 3

    assert result["cold"]["p50_ms"] == pytest.approx(11000.0)
    assert result["warm"]["p50_ms"] == pytest.approx(2000.0)


def test_judge_score_normalizes_zero_to_two_scale() -> None:
    """M21: semantic judge score 0..2 normalizes deterministically."""
    assert _call("normalize_judge_score", 0) == pytest.approx(0.0)
    assert _call("normalize_judge_score", 1) == pytest.approx(0.5)
    assert _call("normalize_judge_score", 2) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [-1, 3, 1.5, "2", None])
def test_judge_score_rejects_invalid_values(value: object) -> None:
    """M22: judge output outside the frozen ordinal scale is invalid."""
    with pytest.raises((TypeError, ValueError)):
        _call("normalize_judge_score", value)


def test_citation_accuracy_returns_none_when_citations_not_required() -> None:
    """M23: citation N/A remains distinct from citation failure."""
    score = _call(
        "compute_citation_accuracy",
        supported_citations=0,
        total_citations=0,
        citation_required=False,
    )

    assert score is None


def test_citation_accuracy_scores_missing_required_citation_as_zero() -> None:
    """M24: missing required policy citation cannot vanish from denominator."""
    score = _call(
        "compute_citation_accuracy",
        supported_citations=0,
        total_citations=0,
        citation_required=True,
    )

    assert score == pytest.approx(0.0)


def test_citation_accuracy_uses_supported_over_total() -> None:
    """M25: citation correctness uses supported/evaluated citations."""
    score = _call(
        "compute_citation_accuracy",
        supported_citations=2,
        total_citations=3,
        citation_required=True,
    )

    assert score == pytest.approx(2 / 3)


def test_required_run_metadata_contract_is_frozen() -> None:
    """M26: result artifacts expose required reproducibility metadata."""
    module = _load_run_eval()

    fields = set(module.REQUIRED_RUN_METADATA_FIELDS)

    assert REQUIRED_RUN_METADATA_FIELDS <= fields


def test_provider_failure_is_not_a_quality_score() -> None:
    """M27: infrastructure failure remains separate from answer quality."""
    result = _call(
        "classify_run_status",
        generation_succeeded=False,
        judge_succeeded=False,
        harness_error=False,
    )

    assert result == "generation_provider_error"


def test_judge_failure_is_not_a_generation_quality_failure() -> None:
    """M28: judge-provider failure remains separately classified."""
    result = _call(
        "classify_run_status",
        generation_succeeded=True,
        judge_succeeded=False,
        harness_error=False,
    )

    assert result == "judge_provider_error"


def test_harness_error_has_distinct_status() -> None:
    """M29: evaluator defects remain distinct from provider/model failures."""
    result = _call(
        "classify_run_status",
        generation_succeeded=False,
        judge_succeeded=False,
        harness_error=True,
    )

    assert result == "harness_error"


def test_successful_run_status_is_completed() -> None:
    """M30: successful generation and judgment classify as completed."""
    result = _call(
        "classify_run_status",
        generation_succeeded=True,
        judge_succeeded=True,
        harness_error=False,
    )

    assert result == "completed"


# ============================================================
# S9 semantic-judge + persisted-result contracts
# ============================================================


def test_judge_prompt_version_is_explicit_and_frozen() -> None:
    """M31: semantic judge behavior has an attributable prompt version."""
    module = _load_run_eval()

    assert module.JUDGE_PROMPT_VERSION == "1.0"


def test_judge_system_prompt_requires_evidence_only_scoring() -> None:
    """M32: judge must evaluate supplied evidence, not outside knowledge."""
    module = _load_run_eval()

    prompt = module.JUDGE_SYSTEM_PROMPT.lower()

    assert "only" in prompt
    assert "evidence" in prompt
    assert "outside knowledge" in prompt
    assert "groundedness" in prompt
    assert "citation" in prompt


def test_parse_judge_result_accepts_strict_valid_object() -> None:
    """M33: valid semantic-judge output parses deterministically."""
    parsed = _call(
        "parse_judge_result",
        {
            "groundedness_score": 2,
            "supported_citations": 2,
            "total_citations": 2,
            "rationale": "All material claims are supported.",
        },
    )

    assert parsed == {
        "groundedness_score": 2,
        "groundedness": 1.0,
        "supported_citations": 2,
        "total_citations": 2,
        "rationale": "All material claims are supported.",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "groundedness_score": 2,
            "supported_citations": 1,
            "total_citations": 1,
        },
        {
            "groundedness_score": 3,
            "supported_citations": 1,
            "total_citations": 1,
            "rationale": "bad score",
        },
        {
            "groundedness_score": 2,
            "supported_citations": 2,
            "total_citations": 1,
            "rationale": "bad citation counts",
        },
        {
            "groundedness_score": 2,
            "supported_citations": 1,
            "total_citations": 1,
            "rationale": "",
        },
        {
            "groundedness_score": 2,
            "supported_citations": 1,
            "total_citations": 1,
            "rationale": "ok",
            "unexpected": True,
        },
    ],
)
def test_parse_judge_result_rejects_malformed_payload(
    payload: dict[str, object],
) -> None:
    """M34: malformed judge output cannot silently become a score."""
    with pytest.raises((TypeError, ValueError)):
        _call("parse_judge_result", payload)


def test_build_run_metadata_requires_all_reproducibility_fields() -> None:
    """M35: persisted evaluation results carry complete provenance."""
    metadata = _call(
        "build_run_metadata",
        generation_model="generation-model",
        judge_model="judge-model",
        llm_base_url="https://example.test/v1",
        prompt_version="1.9",
        corpus_version="1.2",
        embedding_model="BAAI/bge-small-en-v1.5",
        retrieval_k=5,
        timestamp="2026-08-29T00:00:00Z",
        git_commit="6891a02",
        gold_set_sha256=(
            "96de38969c324cc8c578076479e501f7"
            "bff8fb9767134b96cd2183684030fb98"
        ),
        temperature=0,
        seed=0,
        run_type="canonical_baseline",
    )

    assert set(metadata) >= REQUIRED_RUN_METADATA_FIELDS
    assert metadata["judge_prompt_version"] == "1.0"
    assert metadata["retrieval_k"] == 5
    assert metadata["temperature"] == 0


def test_build_run_metadata_rejects_unsupported_k() -> None:
    """M36: governed result artifacts cannot claim unsupported retrieval k."""
    with pytest.raises(ValueError):
        _call(
            "build_run_metadata",
            generation_model="generation-model",
            judge_model="judge-model",
            llm_base_url="https://example.test/v1",
            prompt_version="1.9",
            corpus_version="1.2",
            embedding_model="BAAI/bge-small-en-v1.5",
            retrieval_k=7,
            timestamp="2026-08-29T00:00:00Z",
            git_commit="6891a02",
            gold_set_sha256="abc",
            temperature=0,
            seed=0,
            run_type="canonical_baseline",
        )


def test_build_run_metadata_rejects_nonzero_temperature() -> None:
    """M37: governed S9 generation/judging remains deterministic."""
    with pytest.raises(ValueError):
        _call(
            "build_run_metadata",
            generation_model="generation-model",
            judge_model="judge-model",
            llm_base_url="https://example.test/v1",
            prompt_version="1.9",
            corpus_version="1.2",
            embedding_model="BAAI/bge-small-en-v1.5",
            retrieval_k=5,
            timestamp="2026-08-29T00:00:00Z",
            git_commit="6891a02",
            gold_set_sha256="abc",
            temperature=0.5,
            seed=0,
            run_type="canonical_baseline",
        )


def test_build_item_result_uses_frozen_schema() -> None:
    """M38: every evaluated case persists auditable raw + scored evidence."""
    item = _call(
        "build_item_result",
        item_id="TL01",
        category="tool_task",
        prompt="Example",
        status="completed",
        answer="Example answer",
        observed_behavior="answer",
        retrieved_doc_ids=["HR-POL-004", "HR-POL-005"],
        citations=[
            {
                "doc_id": "HR-POL-004",
                "section": "4.4 International duration limit",
            }
        ],
        observed_tools=[
            "lookup_employee_profile",
            "search_policy_documents",
            "check_policy_compliance",
        ],
        trace=[
            {
                "step": 1,
                "tool": "lookup_employee_profile",
                "decision": "tool_result",
            }
        ],
        latency_ms=1234.5,
        recall_at_k=1.0,
        groundedness=1.0,
        citation_accuracy=1.0,
        tool_selection=True,
        workflow_completion=True,
        boundary_behavior=None,
        action_safety=True,
        error=None,
    )

    assert set(item) == {
        "id",
        "category",
        "prompt",
        "status",
        "answer",
        "observed_behavior",
        "retrieved_doc_ids",
        "citations",
        "observed_tools",
        "trace",
        "latency_ms",
        "scores",
        "error",
    }

    assert item["scores"] == {
        "recall_at_k": 1.0,
        "groundedness": 1.0,
        "citation_accuracy": 1.0,
        "tool_selection": True,
        "workflow_completion": True,
        "boundary_behavior": None,
        "action_safety": True,
    }


def test_build_item_result_preserves_na_scores() -> None:
    """M39: N/A metrics remain null rather than being silently zeroed."""
    item = _call(
        "build_item_result",
        item_id="AM01",
        category="ambiguous",
        prompt="How much PTO do I have?",
        status="completed",
        answer="What is your employee ID?",
        observed_behavior="clarify",
        retrieved_doc_ids=[],
        citations=[],
        observed_tools=[],
        trace=[],
        latency_ms=100.0,
        recall_at_k=None,
        groundedness=None,
        citation_accuracy=None,
        tool_selection=True,
        workflow_completion=True,
        boundary_behavior=True,
        action_safety=True,
        error=None,
    )

    assert item["scores"]["recall_at_k"] is None
    assert item["scores"]["groundedness"] is None
    assert item["scores"]["citation_accuracy"] is None


def test_build_run_artifact_uses_frozen_top_level_schema() -> None:
    """M40: committed result file has one stable auditable envelope."""
    metadata = {
        field: "value"
        for field in REQUIRED_RUN_METADATA_FIELDS
    }

    artifact = _call(
        "build_run_artifact",
        metadata=metadata,
        items=[],
        metrics={
            "retrieval_recall": 0.9,
            "groundedness": 0.8,
        },
        latency={
            "cold": {
                "count": 0,
                "p50_ms": None,
                "p95_ms": None,
            },
            "warm": {
                "count": 0,
                "p50_ms": None,
                "p95_ms": None,
            },
        },
        failure_recovery={
            "accuracy": None,
            "cases": [],
        },
    )

    assert set(artifact) == {
        "metadata",
        "items",
        "metrics",
        "latency",
        "failure_recovery",
    }


def test_build_run_artifact_rejects_missing_metadata_field() -> None:
    """M41: incomplete provenance cannot be persisted as governed evidence."""
    metadata = {
        field: "value"
        for field in REQUIRED_RUN_METADATA_FIELDS
    }
    metadata.pop("generation_model")

    with pytest.raises(ValueError):
        _call(
            "build_run_artifact",
            metadata=metadata,
            items=[],
            metrics={},
            latency={},
            failure_recovery={},
        )
