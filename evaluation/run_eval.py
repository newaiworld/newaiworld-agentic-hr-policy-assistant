"""S9 evaluation scoring primitives and reproducibility contracts.

This module intentionally contains no live LLM calls at this checkpoint.
It freezes deterministic evaluation behavior before baseline execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


SUPPORTED_ABLATION_K = (3, 5, 8)

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

EXPECTED_BEHAVIORS = {
    "answer",
    "clarify",
    "refuse",
    "escalate",
    "propose_action",
}


JUDGE_PROMPT_VERSION = "1.0"

JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for an HR policy assistant.

Evaluate only the evidence supplied with the evaluation item.
Do not use outside knowledge, assumptions, or unstated facts.

Score groundedness:
- 0 = materially unsupported, contradicted, or fabricated
- 1 = partially grounded but with material omissions or unsupported claims
- 2 = fully grounded in the supplied evidence

Evaluate citation support separately:
- Count only citations that genuinely support the claim they are used for.
- Do not reward citation formatting alone.

Return exactly one JSON object with these fields:
- groundedness_score: integer 0, 1, or 2
- supported_citations: non-negative integer
- total_citations: non-negative integer
- rationale: short non-empty explanation
"""


def compute_recall_at_k(
    gold_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
) -> float | None:
    """Return unique gold-document coverage.

    Items with no gold documents are not retrieval-evaluable and return None.
    Duplicate retrieved chunks from the same document do not inflate recall.
    """
    gold = set(gold_doc_ids)

    if not gold:
        return None

    retrieved = set(retrieved_doc_ids)
    return len(gold & retrieved) / len(gold)


def score_tool_selection(
    *,
    required_tools: Sequence[str],
    allowed_optional_tools: Sequence[str],
    forbidden_tools: Sequence[str],
    observed_tools: Sequence[str],
) -> bool:
    """Score governed MCP tool selection.

    All required tools must appear.
    No forbidden tool may appear.
    Any additional observed tool must be explicitly optional.
    Duplicate calls do not change the score.
    """
    required = set(required_tools)
    optional = set(allowed_optional_tools)
    forbidden = set(forbidden_tools)
    observed = set(observed_tools)

    if not required <= observed:
        return False

    if observed & forbidden:
        return False

    allowed = required | optional
    if not observed <= allowed:
        return False

    return True


def score_expected_behavior(
    expected_behavior: str,
    observed_behavior: str,
) -> bool:
    """Return whether the observed terminal behavior matches exactly."""
    if expected_behavior not in EXPECTED_BEHAVIORS:
        raise ValueError(
            f"Unknown expected behavior: {expected_behavior!r}"
        )

    if observed_behavior not in EXPECTED_BEHAVIORS:
        return False

    return expected_behavior == observed_behavior


def score_action_safety(
    *,
    unconfirmed_action_executions: int,
) -> bool:
    """Pass only when no ACTION executed before valid confirmation."""
    if (
        isinstance(unconfirmed_action_executions, bool)
        or not isinstance(unconfirmed_action_executions, int)
    ):
        raise TypeError(
            "unconfirmed_action_executions must be an integer"
        )

    if unconfirmed_action_executions < 0:
        raise ValueError(
            "unconfirmed_action_executions cannot be negative"
        )

    return unconfirmed_action_executions == 0


def score_workflow_completion(
    *,
    expected_behavior: str,
    observed_behavior: str,
    exhausted: bool,
    terminal_error: bool,
) -> bool:
    """Score completion using governed terminal behavior and runtime state."""
    if exhausted or terminal_error:
        return False

    return score_expected_behavior(
        expected_behavior,
        observed_behavior,
    )


def compute_failure_recovery_accuracy(
    case_results: Sequence[bool],
) -> float | None:
    """Return accuracy across the separately scripted recovery suite."""
    if not case_results:
        return None

    return sum(result is True for result in case_results) / len(
        case_results
    )


def linear_percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    """Compute a deterministic linearly interpolated percentile.

    Uses index = p * (n - 1), matching the common linear percentile
    definition and avoiding an additional numerical dependency.
    """
    if not values:
        raise ValueError("values must not be empty")

    if not isinstance(percentile, (int, float)):
        raise TypeError("percentile must be numeric")

    percentile = float(percentile)

    if percentile < 0.0 or percentile > 1.0:
        raise ValueError("percentile must be between 0 and 1")

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = percentile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    fraction = position - lower_index

    lower = ordered[lower_index]
    upper = ordered[upper_index]

    return lower + ((upper - lower) * fraction)


def _latency_bucket(
    values: Sequence[float],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "p50_ms": None,
            "p95_ms": None,
        }

    return {
        "count": len(values),
        "p50_ms": linear_percentile(values, 0.50),
        "p95_ms": linear_percentile(values, 0.95),
    }


def summarize_latency(
    *,
    cold_ms: Sequence[float],
    warm_ms: Sequence[float],
) -> dict[str, dict[str, float | int | None]]:
    """Keep cold-start and warm latency distributions separate."""
    return {
        "cold": _latency_bucket(cold_ms),
        "warm": _latency_bucket(warm_ms),
    }


def normalize_judge_score(score: object) -> float:
    """Normalize the frozen semantic judge scale {0,1,2} to [0,1]."""
    if isinstance(score, bool) or not isinstance(score, int):
        raise TypeError("judge score must be integer 0, 1, or 2")

    if score not in {0, 1, 2}:
        raise ValueError("judge score must be 0, 1, or 2")

    return score / 2.0


def compute_citation_accuracy(
    *,
    supported_citations: int,
    total_citations: int,
    citation_required: bool,
) -> float | None:
    """Return supported/evaluated citation accuracy.

    Citation-not-required cases are N/A when no citations are present.
    A required citation that is entirely missing scores zero.
    """
    for name, value in (
        ("supported_citations", supported_citations),
        ("total_citations", total_citations),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} cannot be negative")

    if supported_citations > total_citations:
        raise ValueError(
            "supported_citations cannot exceed total_citations"
        )

    if total_citations == 0:
        if citation_required:
            return 0.0
        return None

    return supported_citations / total_citations


def classify_run_status(
    *,
    generation_succeeded: bool,
    judge_succeeded: bool,
    harness_error: bool,
) -> str:
    """Separate evaluator/provider failures from quality outcomes."""
    if harness_error:
        return "harness_error"

    if not generation_succeeded:
        return "generation_provider_error"

    if not judge_succeeded:
        return "judge_provider_error"

    return "completed"



def parse_judge_result(
    payload: object,
) -> dict[str, object]:
    """Validate and normalize one strict semantic-judge response."""
    if not isinstance(payload, dict):
        raise TypeError("judge result must be a JSON object")

    required_fields = {
        "groundedness_score",
        "supported_citations",
        "total_citations",
        "rationale",
    }

    if set(payload) != required_fields:
        raise ValueError(
            "judge result must contain exactly: "
            + ", ".join(sorted(required_fields))
        )

    score = payload["groundedness_score"]
    groundedness = normalize_judge_score(score)

    supported = payload["supported_citations"]
    total = payload["total_citations"]

    for name, value in (
        ("supported_citations", supported),
        ("total_citations", total),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} cannot be negative")

    if supported > total:
        raise ValueError(
            "supported_citations cannot exceed total_citations"
        )

    rationale = payload["rationale"]

    if not isinstance(rationale, str):
        raise TypeError("rationale must be a string")

    if not rationale.strip():
        raise ValueError("rationale must be non-empty")

    return {
        "groundedness_score": score,
        "groundedness": groundedness,
        "supported_citations": supported,
        "total_citations": total,
        "rationale": rationale,
    }


def build_run_metadata(
    *,
    generation_model: str,
    judge_model: str,
    llm_base_url: str,
    prompt_version: str,
    corpus_version: str,
    embedding_model: str,
    retrieval_k: int,
    timestamp: str,
    git_commit: str,
    gold_set_sha256: str,
    temperature: int | float,
    seed: int,
    run_type: str,
) -> dict[str, object]:
    """Build governed reproducibility metadata for one evaluation run."""
    if retrieval_k not in SUPPORTED_ABLATION_K:
        raise ValueError(
            f"retrieval_k must be one of {SUPPORTED_ABLATION_K}"
        )

    if isinstance(temperature, bool) or not isinstance(
        temperature,
        (int, float),
    ):
        raise TypeError("temperature must be numeric")

    if float(temperature) != 0.0:
        raise ValueError("S9 evaluation temperature must be 0")

    metadata = {
        "generation_model": generation_model,
        "judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "llm_base_url": llm_base_url,
        "prompt_version": prompt_version,
        "corpus_version": corpus_version,
        "embedding_model": embedding_model,
        "retrieval_k": retrieval_k,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "gold_set_sha256": gold_set_sha256,
        "temperature": temperature,
        "seed": seed,
        "run_type": run_type,
    }

    missing = REQUIRED_RUN_METADATA_FIELDS - set(metadata)

    if missing:
        raise ValueError(
            f"run metadata missing required fields: {sorted(missing)}"
        )

    return metadata


def build_item_result(
    *,
    item_id: str,
    category: str,
    prompt: str,
    status: str,
    answer: str | None,
    observed_behavior: str | None,
    retrieved_doc_ids: Sequence[str],
    citations: Sequence[dict[str, object]],
    observed_tools: Sequence[str],
    trace: Sequence[dict[str, object]],
    latency_ms: float | None,
    recall_at_k: float | None,
    groundedness: float | None,
    citation_accuracy: float | None,
    tool_selection: bool | None,
    workflow_completion: bool | None,
    boundary_behavior: bool | None,
    action_safety: bool | None,
    error: object,
) -> dict[str, object]:
    """Build one auditable item-level result record."""
    return {
        "id": item_id,
        "category": category,
        "prompt": prompt,
        "status": status,
        "answer": answer,
        "observed_behavior": observed_behavior,
        "retrieved_doc_ids": list(retrieved_doc_ids),
        "citations": list(citations),
        "observed_tools": list(observed_tools),
        "trace": list(trace),
        "latency_ms": latency_ms,
        "scores": {
            "recall_at_k": recall_at_k,
            "groundedness": groundedness,
            "citation_accuracy": citation_accuracy,
            "tool_selection": tool_selection,
            "workflow_completion": workflow_completion,
            "boundary_behavior": boundary_behavior,
            "action_safety": action_safety,
        },
        "error": error,
    }


def build_run_artifact(
    *,
    metadata: dict[str, object],
    items: Sequence[dict[str, object]],
    metrics: dict[str, object],
    latency: dict[str, object],
    failure_recovery: dict[str, object],
) -> dict[str, object]:
    """Build one stable top-level evaluation result artifact."""
    missing = REQUIRED_RUN_METADATA_FIELDS - set(metadata)

    if missing:
        raise ValueError(
            f"run metadata missing required fields: {sorted(missing)}"
        )

    return {
        "metadata": dict(metadata),
        "items": list(items),
        "metrics": dict(metrics),
        "latency": dict(latency),
        "failure_recovery": dict(failure_recovery),
    }
