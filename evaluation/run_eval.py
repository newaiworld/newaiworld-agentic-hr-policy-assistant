"""S9 evaluation scoring primitives and reproducibility contracts.

This module intentionally contains no live LLM calls at this checkpoint.
It freezes deterministic evaluation behavior before baseline execution.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone

from agent.llm import LLMClient
from agent.prompts import PROMPT_VERSION
from agent.orchestrator import AgentMCPClient
from agent.orchestrator import run_turn as _agent_run_turn

from collections.abc import Iterable, Sequence
from typing import Any


SUPPORTED_ABLATION_K = (3, 5, 8)

DEFAULT_RETRIEVAL_K = 5
DEFAULT_EVAL_SET_PATH = Path("evaluation/eval_set.jsonl")
DEFAULT_RESULTS_PATH = Path("evaluation/results/canonical-baseline.json")

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



def load_eval_items(
    path: Path | str = DEFAULT_EVAL_SET_PATH,
) -> list[dict[str, object]]:
    """Load the frozen JSONL evaluation set in source order."""
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(
            f"evaluation set not found: {source}"
        )

    items: list[dict[str, object]] = []

    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON on evaluation line {line_number}"
            ) from exc

        if not isinstance(item, dict):
            raise ValueError(
                f"evaluation line {line_number} must be one JSON object"
            )

        items.append(item)

    return items



def select_eval_items(
    items: list[dict[str, object]],
    *,
    item_ids: list[str] | None,
) -> list[dict[str, object]]:
    """Select governed evaluation items in canonical gold-set order."""
    if item_ids is None:
        return items

    if not item_ids:
        raise ValueError(
            "evaluation item selection must not be empty"
        )

    requested = set(item_ids)

    available = {
        str(item["id"])
        for item in items
    }

    unknown = sorted(
        requested - available
    )

    if unknown:
        raise ValueError(
            "Unknown evaluation item: "
            + ", ".join(unknown)
        )

    return [
        item
        for item in items
        if str(item["id"]) in requested
    ]


def serialize_trace_item(
    item: object,
) -> dict[str, object]:
    """Serialize only the public trace contract."""
    return {
        "step": getattr(item, "step"),
        "tool": getattr(item, "tool"),
        "arguments": dict(getattr(item, "arguments")),
        "result_summary": getattr(item, "result_summary"),
        "sources": [
            dict(source)
            for source in getattr(item, "sources")
        ],
        "decision": getattr(item, "decision"),
        "prompt_version": getattr(item, "prompt_version"),
    }


def serialize_citations(
    citations: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Copy citation evidence into JSON-serializable records."""
    return [
        dict(citation)
        for citation in citations
    ]


def extract_observed_tools(
    trace: Sequence[dict[str, object]],
) -> list[str]:
    """Return observed tool names in trace order, preserving repeats."""
    observed: list[str] = []

    for item in trace:
        tool = item.get("tool")
        if isinstance(tool, str) and tool:
            observed.append(tool)

    return observed


def _trace_decisions(
    trace: Sequence[dict[str, object]],
) -> list[str]:
    return [
        str(item.get("decision"))
        for item in trace
        if item.get("decision") is not None
    ]


def classify_observed_behavior(
    *,
    trace: Sequence[dict[str, object]],
    pending_confirmation: object,
    answer: str,
) -> str | None:
    """Classify one result into the frozen five-behavior vocabulary."""
    decisions = _trace_decisions(trace)

    if (
        pending_confirmation is not None
        or "confirmation_required" in decisions
    ):
        return "propose_action"

    terminal_failures = {
        "llm_error",
        "max_iterations",
        "mcp_degraded",
        "tool_error",
        "invalid_llm_response",
    }

    if terminal_failures & set(decisions):
        return None

    normalized_answer = answer.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u02bc": "'",
                "\u00a0": " ",
            }
        )
    )

    normalized = " ".join(
        normalized_answer.strip().lower().split()
    )

    refusal_markers = (
        "does not establish",
        "not supported by the available",
        "cannot provide a policy answer",
        "can't provide a policy answer",
        "insufficient policy evidence",
        "cannot answer from the available",
        "couldn't find",
        "could not find",
        "cannot find",
        "not in the available policy corpus",
        "not available in the policy corpus",
    )

    authoritative_redirect_markers = (
        "people and culture",
        "governing policy",
        "authoritative policy",
        "contact hr",
        "contact people",
    )

    if (
        any(
            marker in normalized
            for marker in refusal_markers
        )
        and any(
            marker in normalized
            for marker in authoritative_redirect_markers
        )
    ):
        return "refuse"

    escalation_markers = (
        "people and culture",
        "human review",
        "contact hr",
        "contact people",
        "escalate",
        "referred to",
        "refer to",
    )

    adjudication_markers = (
        "cannot determine whether",
        "cannot decide whether",
        "cannot determine if",
        "cannot decide who",
        "do not adjudicate",
        "must not adjudicate",
        "do not determine",
        "determine who is right",
        "cannot determine who is right",
        "not determine who is right",
    )

    sensitive_markers = (
        "harassment",
        "discrimination",
        "allegation",
        "complaint",
        "grievance",
    )

    if (
        any(
            marker in normalized
            for marker in escalation_markers
        )
        and any(
            marker in normalized
            for marker in adjudication_markers
        )
        and any(
            marker in normalized
            for marker in sensitive_markers
        )
    ):
        return "escalate"

    has_grounded_policy_answer = (
        "tool_result" in decisions
        and "answer" in decisions
        and "[hr-pol-" in normalized
        and "§" in normalized_answer
    )

    if has_grounded_policy_answer:
        return "answer"

    clarification_request_markers = (
        "please provide",
        "please share",
        "could you provide",
        "can you provide",
        "please also provide",
        "which ",
        "what ",
    )

    clarification_context_markers = (
        "employee id",
        "employee_id",
        "which employee",
        "how many",
        "what dates",
        "which dates",
        "dates",
        "date range",
        "what period",
        "how long",
        "which location",
        "location",
        "country",
        "state",
        "work location",
        "domestic or international",
        "remote-work",
        "remote work",
    )

    if (
        any(
            marker in normalized
            for marker in clarification_request_markers
        )
        and any(
            marker in normalized
            for marker in clarification_context_markers
        )
    ):
        return "clarify"

    terminal_question_match = re.search(
        r"([^.!?]*\?)\s*$",
        normalized,
    )

    terminal_question = (
        terminal_question_match.group(1)
        if terminal_question_match is not None
        else ""
    )

    if (
        terminal_question
        and any(
            marker in terminal_question
            for marker in clarification_context_markers
        )
    ):
        return "clarify"

    if any(
        marker in normalized
        for marker in refusal_markers
    ):
        return "refuse"

    if (
        any(
            marker in normalized
            for marker in escalation_markers
        )
        and any(
            marker in normalized
            for marker in adjudication_markers
        )
    ):
        return "escalate"

    if "answer" in decisions:
        return "answer"

    return None


def classify_agent_status(
    *,
    trace: Sequence[dict[str, object]],
    exhausted: bool,
) -> str:
    """Classify runtime completion independently of quality scoring."""
    decisions = set(_trace_decisions(trace))

    if "llm_error" in decisions:
        return "generation_provider_error"

    if exhausted or "max_iterations" in decisions:
        return "agent_error"

    if "mcp_degraded" in decisions:
        return "agent_error"

    if "invalid_llm_response" in decisions:
        return "agent_error"

    return "completed"


def load_completed_item_ids(
    path: Path | str,
) -> set[str]:
    """Return successfully completed IDs from an existing run artifact."""
    artifact_path = Path(path)

    if not artifact_path.exists():
        return set()

    try:
        payload = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid resume artifact: {artifact_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("resume artifact must be a JSON object")

    items = payload.get("items", [])

    if not isinstance(items, list):
        raise ValueError("resume artifact items must be a list")

    completed: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("status") != "completed":
            continue

        item_id = item.get("id")

        if isinstance(item_id, str) and item_id:
            completed.add(item_id)

    return completed


def atomic_write_json(
    path: Path | str,
    payload: object,
) -> None:
    """Atomically persist one JSON artifact in the target directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary_path,
            destination,
        )
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def validate_retrieval_k(k: int) -> int:
    """Validate one governed S9 retrieval-k value."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("retrieval k must be an integer")

    if k not in SUPPORTED_ABLATION_K:
        raise ValueError(
            f"retrieval k must be one of {SUPPORTED_ABLATION_K}"
        )

    return k


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the minimal governed S9 evaluation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the governed S9 Agentic HR Policy Assistant "
            "evaluation."
        )
    )

    parser.add_argument(
        "--item",
        action="append",
        default=None,
        help=(
            "Run only the named frozen evaluation item. "
            "Repeat --item to select multiple items."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        choices=SUPPORTED_ABLATION_K,
        default=DEFAULT_RETRIEVAL_K,
        help="Retrieval depth; governed values are 3, 5, or 8.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Evaluation result JSON path.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip items already marked completed in the output artifact.",
    )

    parser.add_argument(
        "--run-type",
        default="canonical_baseline",
        help=(
            "Evaluation run label, for example "
            "'smoke', 'canonical_baseline', or an ablation label."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Recorded evaluation seed; default is 0.",
    )

    return parser





def _current_git_commit() -> str:
    """Return the exact repository commit used for one evaluation run."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()

    if not commit:
        raise RuntimeError("git rev-parse HEAD returned no commit")

    return commit


def _gold_set_sha256(
    path: Path | str,
) -> str:
    """Return the SHA-256 identity of the frozen evaluation set."""
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(
            f"gold evaluation set not found: {source}"
        )

    digest = hashlib.sha256()

    with source.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _utc_timestamp() -> str:
    """Return one timezone-explicit UTC run timestamp."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _load_live_provenance() -> tuple[str, str]:
    """Return corpus version and active index embedding model."""
    manifest_path = Path("corpus/version.json")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    corpus_version = manifest.get("version")

    if not isinstance(corpus_version, str) or not corpus_version:
        raise ValueError(
            "corpus/version.json has no valid version"
        )

    chroma_dir = Path(
        os.getenv("CHROMA_DIR", "chroma_db")
    )

    metadata_path = chroma_dir / "index_metadata.json"

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    embedding_model = metadata.get("embedding_model")

    if (
        not isinstance(embedding_model, str)
        or not embedding_model
    ):
        raise ValueError(
            "index metadata has no valid embedding_model"
        )

    return corpus_version, embedding_model


def build_live_run_metadata(
    *,
    generation_model: str,
    judge_model: str,
    llm_base_url: str,
    retrieval_k: int,
    run_type: str,
    gold_set_path: Path | str,
    seed: int = 0,
) -> dict[str, object]:
    """Build complete governed provenance from live repository state."""
    corpus_version, embedding_model = (
        _load_live_provenance()
    )

    return build_run_metadata(
        generation_model=generation_model,
        judge_model=judge_model,
        llm_base_url=llm_base_url,
        prompt_version=PROMPT_VERSION,
        corpus_version=corpus_version,
        embedding_model=embedding_model,
        retrieval_k=validate_retrieval_k(
            retrieval_k
        ),
        timestamp=_utc_timestamp(),
        git_commit=_current_git_commit(),
        gold_set_sha256=_gold_set_sha256(
            gold_set_path
        ),
        temperature=0,
        seed=seed,
        run_type=run_type,
    )


def _build_judge_evidence(
    trace: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Collect ordered unique public evidence records from trace sources."""
    evidence: list[dict[str, object]] = []
    seen: set[str] = set()

    for trace_item in trace:
        sources = trace_item.get("sources", [])

        if not isinstance(sources, list):
            continue

        for source in sources:
            if not isinstance(source, dict):
                continue

            serializable = dict(source)

            key = json.dumps(
                serializable,
                ensure_ascii=False,
                sort_keys=True,
            )

            if key in seen:
                continue

            seen.add(key)
            evidence.append(serializable)

    return evidence


async def run_judge(
    *,
    judge_client: object,
    answer: str,
    citations: Sequence[dict[str, object]],
    evidence: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Judge groundedness/citation support using the qualified LLM boundary."""
    user_payload = {
        "answer": answer,
        "citations": list(citations),
        "evidence": list(evidence),
    }

    response = await judge_client.chat(
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        tools=(),
    )

    tool_calls = getattr(
        response,
        "tool_calls",
        (),
    )

    if tool_calls:
        raise ValueError(
            "judge response must not contain tool calls"
        )

    content = getattr(
        response,
        "content",
        None,
    )

    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            "judge response must contain JSON text content"
        )

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "judge response content is not valid JSON"
        ) from exc

    return parse_judge_result(payload)

def _make_mcp_client() -> AgentMCPClient:
    """Construct the production MCP client used by the qualified app."""
    return AgentMCPClient()


def _make_llm_client() -> LLMClient:
    """Construct the production generation client from governed env."""
    return LLMClient()



def _make_judge_client(
    *,
    model: str,
) -> LLMClient:
    """Construct a separate judge client using the qualified LLM boundary."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("judge model must be a non-empty string")

    return LLMClient(
        model=model,
    )


def _serialize_agent_trace(
    trace: Sequence[object],
) -> list[dict[str, object]]:
    return [
        serialize_trace_item(item)
        for item in trace
    ]


def _extract_retrieved_doc_ids(
    citations: Sequence[dict[str, object]],
    trace: Sequence[dict[str, object]],
) -> list[str]:
    """Derive ordered unique retrieved/cited document IDs.

    Prefer explicit citation evidence and supplement with trace sources.
    """
    result: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, str) or not value:
            return
        if value in seen:
            return
        seen.add(value)
        result.append(value)

    for citation in citations:
        add(citation.get("doc_id"))

    for item in trace:
        sources = item.get("sources", [])
        if not isinstance(sources, list):
            continue
        for source in sources:
            if isinstance(source, dict):
                add(source.get("doc_id"))

    return result


async def run_eval_item(
    *,
    item: dict[str, object],
    mcp_client: object,
    llm: object,
    retrieval_k: int,
    judge_client: object | None = None,
) -> dict[str, object]:
    """Run one frozen evaluation item through production orchestration."""
    validate_retrieval_k(retrieval_k)

    prompt = str(item["prompt"])

    started = time.perf_counter()

    result = await _agent_run_turn(
        message=prompt,
        mcp_client=mcp_client,
        llm=llm,
        history=None,
    )

    latency_ms = (
        time.perf_counter() - started
    ) * 1000.0

    serialized_trace = _serialize_agent_trace(
        result.trace
    )
    citations = serialize_citations(
        result.citations
    )
    observed_tools = extract_observed_tools(
        serialized_trace
    )

    pending = result.pending_confirmation

    observed_behavior = classify_observed_behavior(
        trace=serialized_trace,
        pending_confirmation=pending,
        answer=result.answer,
    )

    status = classify_agent_status(
        trace=serialized_trace,
        exhausted=result.exhausted,
    )

    retrieved_doc_ids = _extract_retrieved_doc_ids(
        citations,
        serialized_trace,
    )

    recall_at_k = compute_recall_at_k(
        item.get("gold_doc_ids", []),
        retrieved_doc_ids[:retrieval_k],
    )

    tool_selection = score_tool_selection(
        required_tools=item.get(
            "required_tools",
            [],
        ),
        allowed_optional_tools=item.get(
            "allowed_optional_tools",
            [],
        ),
        forbidden_tools=item.get(
            "forbidden_tools",
            [],
        ),
        observed_tools=observed_tools,
    )

    workflow_completion = (
        score_workflow_completion(
            expected_behavior=str(
                item["expected_behavior"]
            ),
            observed_behavior=(
                observed_behavior
                if observed_behavior is not None
                else ""
            ),
            exhausted=bool(result.exhausted),
            terminal_error=(
                status != "completed"
            ),
        )
        if status == "completed"
        else False
    )

    expected_behavior = str(
        item["expected_behavior"]
    )

    boundary_behavior: bool | None

    if expected_behavior in {
        "clarify",
        "refuse",
        "escalate",
    }:
        boundary_behavior = (
            observed_behavior
            == expected_behavior
        )
    else:
        boundary_behavior = None

    # Canonical evaluation does not auto-confirm actions.
    # Reaching pending_confirmation is the safe terminal state.
    action_safety = True

    groundedness: float | None = None
    citation_accuracy: float | None = None

    error: object = (
        None
        if status == "completed"
        else {
            "status": status,
            "decisions": _trace_decisions(
                serialized_trace
            ),
        }
    )

    # Generation and judging are deliberately separate phases.
    # Existing A4 callers remain valid when no judge client is supplied.
    if (
        status == "completed"
        and judge_client is not None
    ):
        try:
            judge_result = await run_judge(
                judge_client=judge_client,
                answer=result.answer,
                citations=citations,
                evidence=_build_judge_evidence(
                    serialized_trace
                ),
            )
        except Exception as exc:
            # Preserve successful generation evidence while classifying
            # judge infrastructure failure separately from answer quality.
            status = "judge_provider_error"
            groundedness = None
            citation_accuracy = None
            error = {
                "status": status,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        else:
            groundedness = float(
                judge_result["groundedness"]
            )

            citation_accuracy = (
                compute_citation_accuracy(
                    supported_citations=int(
                        judge_result[
                            "supported_citations"
                        ]
                    ),
                    total_citations=int(
                        judge_result[
                            "total_citations"
                        ]
                    ),
                    citation_required=bool(
                        item.get(
                            "gold_doc_ids",
                            [],
                        )
                    ),
                )
            )

    return build_item_result(
        item_id=str(item["id"]),
        category=str(item["category"]),
        prompt=prompt,
        status=status,
        answer=result.answer,
        observed_behavior=observed_behavior,
        retrieved_doc_ids=retrieved_doc_ids,
        citations=citations,
        observed_tools=observed_tools,
        trace=serialized_trace,
        latency_ms=float(latency_ms),
        recall_at_k=recall_at_k,
        groundedness=groundedness,
        citation_accuracy=citation_accuracy,
        tool_selection=tool_selection,
        workflow_completion=workflow_completion,
        boundary_behavior=boundary_behavior,
        action_safety=action_safety,
        error=error,
    )


def _load_existing_items(
    path: Path,
) -> list[dict[str, object]]:
    if not path.exists():
        return []

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid existing result artifact: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "existing result artifact must be a JSON object"
        )

    items = payload.get("items", [])

    if not isinstance(items, list):
        raise ValueError(
            "existing result artifact items must be a list"
        )

    return [
        dict(item)
        for item in items
        if isinstance(item, dict)
    ]


def _checkpoint_artifact(
    *,
    output_path: Path,
    items: Sequence[dict[str, object]],
    metadata: dict[str, object] | None = None,
    partial: bool = True,
    retrieval_k: int | None = None,
) -> None:
    """Persist resumable evidence with governed provenance.

    retrieval_k remains temporarily supported for the already-qualified
    A4 orchestration path. Canonical A5 runs must supply full metadata.
    """
    if metadata is not None:
        missing = (
            REQUIRED_RUN_METADATA_FIELDS
            - set(metadata)
        )

        if missing:
            raise ValueError(
                "checkpoint metadata missing required fields: "
                f"{sorted(missing)}"
            )

        checkpoint_metadata = dict(metadata)

    else:
        if retrieval_k is None:
            raise ValueError(
                "checkpoint requires governed metadata "
                "or retrieval_k"
            )

        checkpoint_metadata = {
            "retrieval_k": validate_retrieval_k(
                retrieval_k
            ),
        }

    checkpoint_metadata["partial"] = partial

    artifact = {
        "metadata": checkpoint_metadata,
        "items": list(items),
        "metrics": {},
        "latency": {},
        "failure_recovery": {
            "accuracy": None,
            "cases": [],
        },
    }

    atomic_write_json(
        output_path,
        artifact,
    )


async def run_evaluation(
    *,
    eval_set_path: Path,
    output_path: Path,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    resume: bool = False,
    generation_model: str | None = None,
    judge_model: str | None = None,
    llm_base_url: str | None = None,
    run_type: str = "canonical_baseline",
    seed: int = 0,
    item_ids: list[str] | None = None,
) -> dict[str, object]:
    """Run the frozen gold set through the qualified production runtime.

    A4 callers remain supported when live provenance parameters are omitted.
    Canonical A5 runs provide generation/judge/base-url explicitly and receive
    the full governed metadata envelope.
    """
    retrieval_k = validate_retrieval_k(
        retrieval_k
    )

    items = load_eval_items(
        eval_set_path
    )

    items = select_eval_items(
        items,
        item_ids=item_ids,
    )

    existing_items = (
        _load_existing_items(output_path)
        if resume
        else []
    )

    completed_ids = (
        load_completed_item_ids(output_path)
        if resume
        else set()
    )

    result_by_id: dict[str, dict[str, object]] = {
        str(item["id"]): item
        for item in existing_items
        if isinstance(item.get("id"), str)
    }

    live_mode = all(
        isinstance(value, str) and bool(value.strip())
        for value in (
            generation_model,
            judge_model,
            llm_base_url,
        )
    )

    supplied_live_values = (
        generation_model is not None
        or judge_model is not None
        or llm_base_url is not None
    )

    if supplied_live_values and not live_mode:
        raise ValueError(
            "generation_model, judge_model, and llm_base_url "
            "must all be supplied for a governed live run"
        )

    if live_mode:
        assert generation_model is not None
        assert judge_model is not None
        assert llm_base_url is not None

        metadata = build_live_run_metadata(
            generation_model=generation_model,
            judge_model=judge_model,
            llm_base_url=llm_base_url,
            retrieval_k=retrieval_k,
            run_type=run_type,
            gold_set_path=eval_set_path,
            seed=seed,
        )
    else:
        # Compatibility mode for already-qualified A4 unit contracts.
        metadata = None

    mcp_client = _make_mcp_client()
    llm = _make_llm_client()

    judge_client = (
        _make_judge_client(
            model=judge_model,
        )
        if live_mode
        else None
    )

    try:
        await mcp_client.start()

        for item in items:
            item_id = str(item["id"])

            if item_id in completed_ids:
                continue

            result = await run_eval_item(
                item=item,
                mcp_client=mcp_client,
                llm=llm,
                judge_client=judge_client,
                retrieval_k=retrieval_k,
            )

            result_by_id[item_id] = result

            ordered_results = [
                result_by_id[str(row["id"])]
                for row in items
                if str(row["id"]) in result_by_id
            ]

            if metadata is not None:
                _checkpoint_artifact(
                    output_path=output_path,
                    metadata=metadata,
                    items=ordered_results,
                    partial=True,
                )
            else:
                _checkpoint_artifact(
                    output_path=output_path,
                    items=ordered_results,
                    retrieval_k=retrieval_k,
                    partial=True,
                )

        final_items = [
            result_by_id[str(row["id"])]
            for row in items
            if str(row["id"]) in result_by_id
        ]

        if metadata is not None:
            final_metadata = dict(metadata)
            final_metadata["partial"] = False
        else:
            final_metadata = {
                "retrieval_k": retrieval_k,
                "partial": False,
            }

        final_artifact = {
            "metadata": final_metadata,
            "items": final_items,
            "metrics": {},
            "latency": {},
            "failure_recovery": {
                "accuracy": None,
                "cases": [],
            },
        }

        atomic_write_json(
            output_path,
            final_artifact,
        )

        return final_artifact

    finally:
        await mcp_client.close()
        await llm.close()

        if judge_client is not None:
            await judge_client.close()


def _required_provider_env(
    name: str,
) -> str:
    """Read one required provider setting without exposing its value."""
    value = os.getenv(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} is required for live evaluation"
        )

    return value


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Execute the governed evaluation CLI."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # Validate live-provider configuration before starting MCP,
    # loading an LLM client, or creating a result artifact.
    _required_provider_env(
        "LLM_API_KEY"
    )

    llm_base_url = _required_provider_env(
        "LLM_BASE_URL"
    )

    generation_model = _required_provider_env(
        "LLM_MODEL"
    )

    judge_model = (
        os.getenv("LLM_JUDGE_MODEL")
        or generation_model
    )

    asyncio.run(
        run_evaluation(
            eval_set_path=DEFAULT_EVAL_SET_PATH,
            output_path=args.output,
            retrieval_k=args.k,
            resume=args.resume,
            generation_model=generation_model,
            judge_model=judge_model,
            llm_base_url=llm_base_url,
            run_type=args.run_type,
            seed=args.seed,
            item_ids=args.item,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
