"""Permanent contract tests for the frozen S9 evaluation gold set."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET_PATH = PROJECT_ROOT / "evaluation" / "eval_set.jsonl"
CORPUS_VERSION_PATH = PROJECT_ROOT / "corpus" / "version.json"

EXPECTED_CATEGORY_IDS = {
    "simple_policy": tuple(f"SP{i:02d}" for i in range(1, 9)),
    "multi_document": tuple(f"MD{i:02d}" for i in range(1, 6)),
    "tool_task": tuple(f"TL{i:02d}" for i in range(1, 7)),
    "ambiguous": tuple(f"AM{i:02d}" for i in range(1, 4)),
    "out_of_scope": tuple(f"OOS{i:02d}" for i in range(1, 3)),
}

EXPECTED_CATEGORY_COUNTS = {
    category: len(ids)
    for category, ids in EXPECTED_CATEGORY_IDS.items()
}

EXPECTED_IDS = tuple(
    item_id
    for ids in EXPECTED_CATEGORY_IDS.values()
    for item_id in ids
)

REQUIRED_FIELDS = {
    "id",
    "category",
    "prompt",
    "gold_answer",
    "gold_doc_ids",
    "gold_sections",
    "required_tools",
    "allowed_optional_tools",
    "forbidden_tools",
    "expected_behavior",
    "requires_confirmation",
}

CANONICAL_MCP_TOOLS = {
    "search_policy_documents",
    "get_policy_section",
    "lookup_employee_profile",
    "lookup_benefits_status",
    "check_pto_balance",
    "check_policy_compliance",
    "create_mock_hr_ticket",
    "draft_hr_email",
}

ACTION_TOOLS = {
    "create_mock_hr_ticket",
    "draft_hr_email",
}

EXPECTED_BEHAVIORS = {
    "answer",
    "clarify",
    "refuse",
    "escalate",
    "propose_action",
}


def _load_eval_records() -> list[dict[str, Any]]:
    """Load one JSON object per non-blank line from the frozen gold set."""
    assert EVAL_SET_PATH.is_file(), (
        "Missing frozen S9 gold set: "
        "evaluation/eval_set.jsonl"
    )

    records: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(
        EVAL_SET_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"Invalid JSON on eval_set.jsonl line "
                f"{line_number}: {exc}"
            )

        assert isinstance(record, dict), (
            f"Evaluation row {line_number} must be one JSON object."
        )
        records.append(record)

    return records


def _canonical_doc_ids() -> set[str]:
    """Return the manifest's authoritative corpus document IDs."""
    manifest = json.loads(
        CORPUS_VERSION_PATH.read_text(encoding="utf-8")
    )
    return {
        document["doc_id"]
        for document in manifest["documents"]
    }


def _assert_string_list(
    value: object,
    *,
    field: str,
    item_id: str,
) -> list[str]:
    assert isinstance(value, list), (
        f"{item_id}.{field} must be a list."
    )
    assert all(
        isinstance(item, str) and item.strip()
        for item in value
    ), f"{item_id}.{field} must contain non-empty strings."
    return value


def test_eval_set_contains_exactly_24_records() -> None:
    """T01: the frozen S9 gold set contains exactly 24 records."""
    assert len(_load_eval_records()) == 24


def test_eval_set_rows_are_json_objects() -> None:
    """T02: every non-blank line parses to exactly one JSON object."""
    records = _load_eval_records()
    assert all(isinstance(record, dict) for record in records)


def test_eval_set_ids_match_frozen_ranges() -> None:
    """T03: IDs are unique and exactly match the frozen category ranges."""
    records = _load_eval_records()
    ids = [record.get("id") for record in records]

    assert len(ids) == len(set(ids))
    assert set(ids) == set(EXPECTED_IDS)


def test_eval_set_category_counts_match_frozen_contract() -> None:
    """T04: category counts remain exactly 8 / 5 / 6 / 3 / 2."""
    records = _load_eval_records()
    counts = Counter(record.get("category") for record in records)

    assert counts == Counter(EXPECTED_CATEGORY_COUNTS)


def test_eval_set_rows_use_exact_frozen_schema() -> None:
    """T05: every item has exactly the frozen 11-field schema."""
    for record in _load_eval_records():
        assert set(record) == REQUIRED_FIELDS, record.get("id")


def test_eval_set_prompts_and_gold_answers_are_non_empty() -> None:
    """T06: prompts and reference gold answers are meaningful strings."""
    for record in _load_eval_records():
        item_id = record["id"]

        assert isinstance(record["prompt"], str)
        assert record["prompt"].strip(), item_id

        assert isinstance(record["gold_answer"], str)
        assert record["gold_answer"].strip(), item_id


def test_eval_set_gold_doc_ids_are_canonical() -> None:
    """T07: gold document IDs come only from the canonical manifest."""
    canonical = _canonical_doc_ids()

    for record in _load_eval_records():
        doc_ids = _assert_string_list(
            record["gold_doc_ids"],
            field="gold_doc_ids",
            item_id=record["id"],
        )
        assert set(doc_ids) <= canonical, record["id"]


def test_eval_set_gold_sections_reference_gold_documents() -> None:
    """T08: each gold section belongs to one of the item's gold docs."""
    for record in _load_eval_records():
        item_id = record["id"]
        doc_ids = set(
            _assert_string_list(
                record["gold_doc_ids"],
                field="gold_doc_ids",
                item_id=item_id,
            )
        )
        sections = _assert_string_list(
            record["gold_sections"],
            field="gold_sections",
            item_id=item_id,
        )

        for section_ref in sections:
            assert any(
                section_ref.startswith(f"{doc_id} §")
                for doc_id in doc_ids
            ), (
                f"{item_id}.gold_sections contains a reference "
                f"outside gold_doc_ids: {section_ref!r}"
            )


def test_eval_set_tool_fields_use_only_canonical_mcp_tools() -> None:
    """T09: all tool expectations use the frozen MCP vocabulary."""
    for record in _load_eval_records():
        item_id = record["id"]

        for field in (
            "required_tools",
            "allowed_optional_tools",
            "forbidden_tools",
        ):
            tools = _assert_string_list(
                record[field],
                field=field,
                item_id=item_id,
            )
            assert set(tools) <= CANONICAL_MCP_TOOLS, (
                item_id,
                field,
                tools,
            )


def test_eval_set_required_and_forbidden_tools_do_not_overlap() -> None:
    """T10: an item cannot simultaneously require and forbid a tool."""
    for record in _load_eval_records():
        required = set(record["required_tools"])
        forbidden = set(record["forbidden_tools"])

        assert not required & forbidden, record["id"]


def test_eval_set_expected_behavior_uses_frozen_vocabulary() -> None:
    """T11: every case uses one governed expected behavior."""
    for record in _load_eval_records():
        assert record["expected_behavior"] in EXPECTED_BEHAVIORS, (
            record["id"],
            record["expected_behavior"],
        )


def test_eval_set_requires_confirmation_is_boolean() -> None:
    """T12: confirmation expectation is represented as a strict boolean."""
    for record in _load_eval_records():
        assert isinstance(record["requires_confirmation"], bool), (
            record["id"],
            record["requires_confirmation"],
        )


def test_confirmation_required_items_require_an_action_tool() -> None:
    """T13: confirmation-required cases must require a discovered ACTION."""
    for record in _load_eval_records():
        if not record["requires_confirmation"]:
            continue

        assert set(record["required_tools"]) & ACTION_TOOLS, (
            record["id"],
            record["required_tools"],
        )


def test_simple_policy_items_contain_policy_gold_evidence() -> None:
    """T14: simple policy cases retain policy-document evidence."""
    for record in _load_eval_records():
        if record["category"] != "simple_policy":
            continue

        assert record["gold_doc_ids"], record["id"]
        assert record["gold_sections"], record["id"]


def test_multi_document_items_require_multiple_gold_documents() -> None:
    """T15: multi-document cases genuinely require >=2 source documents."""
    for record in _load_eval_records():
        if record["category"] != "multi_document":
            continue

        assert len(set(record["gold_doc_ids"])) >= 2, record["id"]


def test_ambiguous_items_expect_clarification() -> None:
    """T16: all three ambiguous cases require one clarifying response."""
    for record in _load_eval_records():
        if record["category"] == "ambiguous":
            assert record["expected_behavior"] == "clarify", record["id"]


def test_out_of_scope_items_expect_refusal_or_escalation() -> None:
    """T17: out-of-scope cases are never governed as direct answers."""
    allowed = {"refuse", "escalate"}

    for record in _load_eval_records():
        if record["category"] == "out_of_scope":
            assert record["expected_behavior"] in allowed, record["id"]


def test_eval_set_contains_frozen_wf1_and_wf2_cases() -> None:
    """T18: the two canonical demo workflows are present as tool tasks."""
    records = {
        record["id"]: record
        for record in _load_eval_records()
    }

    wf1 = records["TL01"]
    wf2 = records["TL02"]

    assert wf1["category"] == "tool_task"
    assert wf1["prompt"] == (
        "I'm employee E003. Can I work remotely from overseas "
        "for six weeks?"
    )
    assert {
        "lookup_employee_profile",
        "search_policy_documents",
        "check_policy_compliance",
    } <= set(wf1["required_tools"])

    assert wf2["category"] == "tool_task"
    assert wf2["prompt"] == (
        "I'm employee E001. Can I take 3 days of PTO next week?"
    )
    assert {
        "lookup_employee_profile",
        "check_pto_balance",
        "search_policy_documents",
    } <= set(wf2["required_tools"])
    assert wf2["requires_confirmation"] is True
    assert set(wf2["required_tools"]) & ACTION_TOOLS
