"""Structured mock-data access for MCP data-backed tools.

The module owns deterministic repository-relative access to the frozen
mock employee fixture. Tool functions do not read environment variables
directly. V1 deliberately uses no cache because the fixture contains only
a small deterministic record set and cache state would add unnecessary
complexity.

This module is framework-agnostic: it contains no FastMCP registration.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMPLOYEES_PATH = (
    PROJECT_ROOT
    / "mock_data"
    / "employees.json"
)

BENEFITS_PATH = (
    PROJECT_ROOT
    / "mock_data"
    / "benefits.json"
)

PTO_PATH = (
    PROJECT_ROOT
    / "mock_data"
    / "pto.json"
)


class MockDataError(RuntimeError):
    """Raised when frozen structured mock data cannot be used safely."""


_REQUIRED_STRING_FIELDS = (
    "employee_id",
    "name",
    "role",
    "employment_type",
    "location",
    "start_date",
)

_REQUIRED_FIELDS = (
    *_REQUIRED_STRING_FIELDS,
    "manager_id",
)


def _record_label(
    record: dict[str, Any],
    index: int,
) -> str:
    """Return a stable identifier for employee-record validation errors."""

    employee_id = record.get(
        "employee_id"
    )

    if (
        isinstance(
            employee_id,
            str,
        )
        and employee_id
    ):
        return (
            f"employee record {employee_id!r}"
        )

    return (
        f"employee record at index {index}"
    )


def _validate_employee_record(
    record: object,
    index: int,
) -> dict[str, Any]:
    """Validate one employee fixture record.

    Args:
        record:
            Raw JSON value from the employees collection.
        index:
            Zero-based record position used for deterministic errors when
            a valid employee identifier is unavailable.

    Returns:
        The validated employee dictionary.

    Raises:
        MockDataError:
            If the record is not an object, is missing a required field,
            or contains a required field with an invalid type.
    """

    if not isinstance(
        record,
        dict,
    ):
        raise MockDataError(
            "Invalid employee record at index "
            f"{index}: expected an object."
        )

    label = _record_label(
        record,
        index,
    )

    for field in _REQUIRED_FIELDS:
        if field not in record:
            raise MockDataError(
                f"Invalid {label}: "
                f"field {field!r} is required."
            )

    for field in _REQUIRED_STRING_FIELDS:
        if not isinstance(
            record[field],
            str,
        ):
            raise MockDataError(
                f"Invalid {label}: "
                f"field {field!r} must be a string."
            )

    manager_id = record[
        "manager_id"
    ]

    if not (
        manager_id is None
        or isinstance(
            manager_id,
            str,
        )
    ):
        raise MockDataError(
            f"Invalid {label}: "
            "field 'manager_id' must be a string or null."
        )

    return record


def _project_employee_profile(
    record: dict[str, Any],
) -> dict[str, str | None]:
    """Project one validated employee record to the frozen public shape.

    Args:
        record:
            One employee record previously validated by the employee
            fixture loader.

    Returns:
        A fresh JSON-compatible dictionary containing exactly the six
        fields exposed by ``lookup_employee_profile``.

    Raises:
        TypeError:
            If ``record`` is not a dictionary.
        MockDataError:
            If the record does not satisfy the frozen employee-record
            contract.
    """

    if not isinstance(
        record,
        dict,
    ):
        raise TypeError(
            "record must be a dictionary."
        )

    validated = _validate_employee_record(
        record,
        0,
    )

    return {
        "name": validated["name"],
        "role": validated["role"],
        "employment_type": validated["employment_type"],
        "location": validated["location"],
        "manager_id": validated["manager_id"],
        "start_date": validated["start_date"],
    }


def _load_employee_index(
    path: Path = EMPLOYEES_PATH,
) -> dict[str, dict[str, Any]]:
    """Load and validate the frozen employee fixture.

    Args:
        path:
            Employee JSON file to read. The default is the frozen
            repository-relative ``mock_data/employees.json`` path.
            An explicit path is accepted to support isolated tests and
            temporary validation probes without environment variables.

    Returns:
        A newly constructed mapping from ``employee_id`` to validated
        employee records.

    Raises:
        TypeError:
            If ``path`` is not a ``Path`` instance.
        MockDataError:
            If the file cannot be read, is malformed JSON, has an invalid
            top-level structure, contains invalid employee records, or
            contains duplicate employee identifiers.
    """

    if not isinstance(
        path,
        Path,
    ):
        raise TypeError(
            "path must be a Path instance."
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise MockDataError(
            "Employee data file not found: "
            f"{str(path)!r}."
        ) from exc
    except OSError as exc:
        raise MockDataError(
            "Employee data file could not be read: "
            f"{str(path)!r}."
        ) from exc

    try:
        raw_data = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise MockDataError(
            "Employee data file is not valid JSON: "
            f"{str(path)!r}."
        ) from exc

    if not isinstance(
        raw_data,
        dict,
    ):
        raise MockDataError(
            "Employee data must be a JSON object."
        )

    employees = raw_data.get(
        "employees"
    )

    if not isinstance(
        employees,
        list,
    ):
        raise MockDataError(
            "Employee data must contain an 'employees' list."
        )

    index: dict[
        str,
        dict[str, Any],
    ] = {}

    for position, raw_record in enumerate(
        employees
    ):
        record = _validate_employee_record(
            raw_record,
            position,
        )

        employee_id = record[
            "employee_id"
        ]

        if employee_id in index:
            raise MockDataError(
                f"Duplicate employee ID: {employee_id!r}."
            )

        index[
            employee_id
        ] = record

    return index

def lookup_employee_profile(
    employee_id: str,
) -> dict[str, str | None]:
    """Return one employee profile in the frozen public response shape.

    The function performs only framework-agnostic composition over the
    validated mock-data loader and the existing profile projection.

    Args:
        employee_id:
            Exact, case-sensitive employee identifier.

    Returns:
        A fresh JSON-compatible dictionary containing exactly:
        ``name``, ``role``, ``employment_type``, ``location``,
        ``manager_id``, and ``start_date``.

    Raises:
        TypeError:
            If ``employee_id`` is not a string.
        ValueError:
            If ``employee_id`` is empty, whitespace-only, or contains
            leading or trailing whitespace.
        MockDataError:
            If the employee fixture cannot be loaded safely or if the
            requested employee does not exist.
    """

    if not isinstance(
        employee_id,
        str,
    ):
        raise TypeError(
            "employee_id must be a string."
        )

    if (
        not employee_id
        or employee_id.isspace()
        or employee_id != employee_id.strip()
    ):
        raise ValueError(
            "employee_id must be a non-empty string "
            "without leading or trailing whitespace."
        )

    employee_index = _load_employee_index()

    try:
        record = employee_index[
            employee_id
        ]
    except KeyError:
        raise MockDataError(
            f"Employee not found: {employee_id!r}."
        ) from None

    return _project_employee_profile(
        record
    )

_BENEFITS_REQUIRED_FIELDS = (
    "employee_id",
    "elections",
    "eligibility",
    "coverage_start",
)

_BENEFITS_ELECTION_KEYS = (
    "health_support",
    "professional_development",
    "wellbeing_program",
)

_ALLOWED_BENEFITS_ELIGIBILITY = frozenset(
    {
        "eligible",
        "ineligible",
        "pending",
    }
)

_ALLOWED_BENEFITS_ELECTION_VALUES = frozenset(
    {
        "enrolled",
        "declined",
        "pending",
        "not_available",
    }
)


def _benefits_record_label(
    record: dict[str, Any],
    index: int,
) -> str:
    """Return a stable label for benefits-record validation errors."""

    employee_id = record.get(
        "employee_id"
    )

    if (
        isinstance(
            employee_id,
            str,
        )
        and employee_id
    ):
        return (
            f"benefits record {employee_id!r}"
        )

    return (
        f"benefits record at index {index}"
    )


def _validate_benefits_record(
    record: object,
    index: int,
) -> dict[str, Any]:
    """Validate one frozen benefits fixture record."""

    if not isinstance(
        record,
        dict,
    ):
        raise MockDataError(
            "Invalid benefits record at index "
            f"{index}: expected an object."
        )

    label = _benefits_record_label(
        record,
        index,
    )

    for field in _BENEFITS_REQUIRED_FIELDS:
        if field not in record:
            raise MockDataError(
                f"Invalid {label}: "
                f"field {field!r} is required."
            )

    employee_id = record[
        "employee_id"
    ]

    if not isinstance(
        employee_id,
        str,
    ):
        raise MockDataError(
            f"Invalid {label}: "
            "field 'employee_id' must be a string."
        )

    elections = record[
        "elections"
    ]

    if not isinstance(
        elections,
        dict,
    ):
        raise MockDataError(
            f"Invalid {label}: "
            "field 'elections' must be an object."
        )

    actual_election_keys = tuple(
        sorted(
            elections
        )
    )

    expected_election_keys = tuple(
        sorted(
            _BENEFITS_ELECTION_KEYS
        )
    )

    if actual_election_keys != expected_election_keys:
        raise MockDataError(
            f"Invalid {label}: "
            "field 'elections' must contain exactly "
            "'health_support', 'professional_development', "
            "and 'wellbeing_program'."
        )

    for election_name in _BENEFITS_ELECTION_KEYS:
        election_value = elections[
            election_name
        ]

        if not isinstance(
            election_value,
            str,
        ):
            raise MockDataError(
                f"Invalid {label}: "
                f"election {election_name!r} must be a string."
            )

        if (
            election_value
            not in _ALLOWED_BENEFITS_ELECTION_VALUES
        ):
            raise MockDataError(
                f"Invalid {label}: "
                f"election {election_name!r} has unsupported "
                f"value {election_value!r}."
            )

    eligibility = record[
        "eligibility"
    ]

    if not isinstance(
        eligibility,
        str,
    ):
        raise MockDataError(
            f"Invalid {label}: "
            "field 'eligibility' must be a string."
        )

    if (
        eligibility
        not in _ALLOWED_BENEFITS_ELIGIBILITY
    ):
        raise MockDataError(
            f"Invalid {label}: "
            "field 'eligibility' has unsupported "
            f"value {eligibility!r}."
        )

    coverage_start = record[
        "coverage_start"
    ]

    if not (
        coverage_start is None
        or isinstance(
            coverage_start,
            str,
        )
    ):
        raise MockDataError(
            f"Invalid {label}: "
            "field 'coverage_start' must be a string or null."
        )

    return record


def _project_benefits_status(
    record: dict[str, Any],
) -> dict[str, object]:
    """Project one benefits record to the frozen public response."""

    if not isinstance(
        record,
        dict,
    ):
        raise TypeError(
            "record must be a dictionary."
        )

    validated = _validate_benefits_record(
        record,
        0,
    )

    return {
        "elections": dict(
            validated[
                "elections"
            ]
        ),
        "eligibility": validated[
            "eligibility"
        ],
        "coverage_start": validated[
            "coverage_start"
        ],
    }


def _load_benefits_index(
    path: Path = BENEFITS_PATH,
) -> dict[str, dict[str, Any]]:
    """Load and validate the frozen benefits fixture."""

    if not isinstance(
        path,
        Path,
    ):
        raise TypeError(
            "path must be a Path instance."
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise MockDataError(
            "Benefits data file not found: "
            f"{str(path)!r}."
        ) from exc
    except OSError as exc:
        raise MockDataError(
            "Benefits data file could not be read: "
            f"{str(path)!r}."
        ) from exc

    try:
        raw_data = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise MockDataError(
            "Benefits data file is not valid JSON: "
            f"{str(path)!r}."
        ) from exc

    if not isinstance(
        raw_data,
        dict,
    ):
        raise MockDataError(
            "Benefits data must be a JSON object."
        )

    benefits = raw_data.get(
        "benefits"
    )

    if not isinstance(
        benefits,
        list,
    ):
        raise MockDataError(
            "Benefits data must contain a 'benefits' list."
        )

    index: dict[
        str,
        dict[str, Any],
    ] = {}

    for position, raw_record in enumerate(
        benefits
    ):
        record = _validate_benefits_record(
            raw_record,
            position,
        )

        employee_id = record[
            "employee_id"
        ]

        if employee_id in index:
            raise MockDataError(
                f"Duplicate benefits employee ID: "
                f"{employee_id!r}."
            )

        index[
            employee_id
        ] = record

    return index


def lookup_benefits_status(
    employee_id: str,
) -> dict[str, object]:
    """Return one employee's stored benefits status.

    The function is a framework-agnostic READ over the frozen benefits
    fixture. It reports stored structured data only and does not
    recompute policy eligibility or coverage rules.

    Args:
        employee_id:
            Exact, case-sensitive employee identifier.

    Returns:
        A fresh JSON-compatible dictionary containing exactly
        ``elections``, ``eligibility``, and ``coverage_start``.

    Raises:
        TypeError:
            If ``employee_id`` is not a string.

        ValueError:
            If ``employee_id`` is empty, whitespace-only, or contains
            leading or trailing whitespace.

        MockDataError:
            If the benefits fixture cannot be loaded safely or the
            employee has no benefits record.
    """

    if not isinstance(
        employee_id,
        str,
    ):
        raise TypeError(
            "employee_id must be a string."
        )

    if (
        not employee_id
        or employee_id.isspace()
        or employee_id != employee_id.strip()
    ):
        raise ValueError(
            "employee_id must be a non-empty string "
            "without leading or trailing whitespace."
        )

    benefits_index = _load_benefits_index()

    try:
        record = benefits_index[
            employee_id
        ]
    except KeyError:
        raise MockDataError(
            f"Benefits record not found for employee: "
            f"{employee_id!r}."
        ) from None

    return _project_benefits_status(
        record
    )


_PTO_TOP_LEVEL_FIELDS = (
    "schema_version",
    "as_of_date",
    "balances",
)

_PTO_REQUIRED_FIELDS = (
    "employee_id",
    "available_days",
    "accrual_rate",
    "accrual_unit",
    "last_updated",
    "next_accrual_date",
)

_PTO_ACCRUAL_UNIT = "days_per_month"


def _pto_record_label(
    record: dict[str, Any],
    index: int,
) -> str:
    """Return a stable label for PTO-record validation errors."""

    employee_id = record.get(
        "employee_id"
    )

    if (
        isinstance(
            employee_id,
            str,
        )
        and employee_id
    ):
        return f"PTO balance record {employee_id!r}"

    return f"PTO balance record at index {index}"


def _validate_pto_record(
    record: object,
    index: int,
) -> dict[str, Any]:
    """Validate one frozen PTO balance record."""

    if not isinstance(
        record,
        dict,
    ):
        raise MockDataError(
            "Invalid PTO balance record at index "
            f"{index}: expected an object."
        )

    label = _pto_record_label(
        record,
        index,
    )

    if tuple(sorted(record)) != tuple(
        sorted(_PTO_REQUIRED_FIELDS)
    ):
        raise MockDataError(
            f"Invalid {label}: "
            "record must contain exactly "
            "'employee_id', 'available_days', "
            "'accrual_rate', 'accrual_unit', "
            "'last_updated', and 'next_accrual_date'."
        )

    employee_id = record["employee_id"]

    if (
        not isinstance(employee_id, str)
        or not employee_id
        or employee_id.isspace()
    ):
        raise MockDataError(
            f"Invalid {label}: "
            "field 'employee_id' must be a non-empty string."
        )

    for field in (
        "available_days",
        "accrual_rate",
    ):
        value = record[field]

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise MockDataError(
                f"Invalid {label}: "
                f"field {field!r} must be a number."
            )

        if not isfinite(float(value)):
            raise MockDataError(
                f"Invalid {label}: "
                f"field {field!r} must be finite."
            )

        if value < 0:
            raise MockDataError(
                f"Invalid {label}: "
                f"field {field!r} must be non-negative."
            )

    if record["accrual_unit"] != _PTO_ACCRUAL_UNIT:
        raise MockDataError(
            f"Invalid {label}: "
            "field 'accrual_unit' must be "
            f"{_PTO_ACCRUAL_UNIT!r}."
        )

    for field in (
        "last_updated",
        "next_accrual_date",
    ):
        value = record[field]

        if (
            not isinstance(value, str)
            or not value
            or value.isspace()
        ):
            raise MockDataError(
                f"Invalid {label}: "
                f"field {field!r} must be a non-empty string."
            )

    return record


def _project_pto_balance(
    record: dict[str, Any],
) -> dict[str, object]:
    """Project one PTO record to the frozen public response."""

    if not isinstance(
        record,
        dict,
    ):
        raise TypeError(
            "record must be a dictionary."
        )

    validated = _validate_pto_record(
        record,
        0,
    )

    return {
        "available_days": validated[
            "available_days"
        ],
        "accrual_rate": validated[
            "accrual_rate"
        ],
        "next_accrual_date": validated[
            "next_accrual_date"
        ],
    }


def _load_pto_index(
    path: Path = PTO_PATH,
) -> dict[str, dict[str, Any]]:
    """Load and validate the frozen PTO fixture."""

    if not isinstance(
        path,
        Path,
    ):
        raise TypeError(
            "path must be a Path instance."
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise MockDataError(
            "PTO data file not found: "
            f"{str(path)!r}."
        ) from exc
    except OSError as exc:
        raise MockDataError(
            "PTO data file could not be read: "
            f"{str(path)!r}."
        ) from exc

    try:
        raw_data = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise MockDataError(
            "PTO data file is not valid JSON: "
            f"{str(path)!r}."
        ) from exc

    if not isinstance(
        raw_data,
        dict,
    ):
        raise MockDataError(
            "PTO data must be a JSON object."
        )

    if tuple(sorted(raw_data)) != tuple(
        sorted(_PTO_TOP_LEVEL_FIELDS)
    ):
        raise MockDataError(
            "PTO data must contain exactly "
            "'schema_version', 'as_of_date', and 'balances'."
        )

    if not isinstance(
        raw_data["schema_version"],
        str,
    ):
        raise MockDataError(
            "PTO data field 'schema_version' must be a string."
        )

    if not isinstance(
        raw_data["as_of_date"],
        str,
    ):
        raise MockDataError(
            "PTO data field 'as_of_date' must be a string."
        )

    balances = raw_data["balances"]

    if not isinstance(
        balances,
        list,
    ):
        raise MockDataError(
            "PTO data field 'balances' must be a list."
        )

    index: dict[str, dict[str, Any]] = {}

    for position, raw_record in enumerate(
        balances
    ):
        record = _validate_pto_record(
            raw_record,
            position,
        )

        employee_id = record[
            "employee_id"
        ]

        if employee_id in index:
            raise MockDataError(
                "Duplicate PTO employee ID: "
                f"{employee_id!r}."
            )

        index[employee_id] = record

    return index


def check_pto_balance(
    employee_id: str,
) -> dict[str, object]:
    """Return one employee's stored PTO balance.

    This framework-agnostic CALCULATION-class capability reports
    authoritative stored PTO state without recomputing entitlement,
    FTE-adjusted accrual, or future accrual dates.
    """

    if not isinstance(
        employee_id,
        str,
    ):
        raise TypeError(
            "employee_id must be a string."
        )

    if (
        not employee_id
        or employee_id.isspace()
        or employee_id != employee_id.strip()
    ):
        raise ValueError(
            "employee_id must be a non-empty string "
            "without leading or trailing whitespace."
        )

    pto_index = _load_pto_index()

    try:
        record = pto_index[
            employee_id
        ]
    except KeyError:
        raise MockDataError(
            "PTO balance record not found for employee: "
            f"{employee_id!r}."
        ) from None

    return _project_pto_balance(
        record
    )
_SUPPORTED_COMPLIANCE_TOPIC = (
    "remote_work_international"
)

_REMOTE_WORK_INTERNATIONAL_REASONS = (
    (
        "A six-week international remote-work proposal "
        "exceeds the standard 30-calendar-day limit "
        "and requires formal exception review."
    ),
    (
        "International remote work also requires the applicable "
        "approvals, Information Security review, and overseas-access "
        "controls before approval."
    ),
)

_REMOTE_WORK_INTERNATIONAL_POLICY_REFS = (
    "HR-POL-004 §4.4",
    "HR-POL-004 §8",
    "HR-POL-005 §4.5",
)


def _project_policy_compliance() -> dict[str, object]:
    """Return a fresh projection of the frozen V1 compliance result."""

    return {
        "compliant": False,
        "reasons": list(
            _REMOTE_WORK_INTERNATIONAL_REASONS
        ),
        "policy_refs": list(
            _REMOTE_WORK_INTERNATIONAL_POLICY_REFS
        ),
    }


def check_policy_compliance(
    topic: str,
    employee_id: str,
) -> dict[str, object]:
    """Evaluate the frozen V1 policy-compliance scenario.

    The V1 implementation supports only the frozen
    ``remote_work_international`` scenario. Policy facts were verified
    against the authoritative corpus through the retrieval layer during
    engineering, but this production capability performs no runtime RAG
    or policy retrieval.

    The employee identifier is validated against authoritative mock data.
    The frozen compliance result is policy-determined and deliberately
    does not branch on employee profile attributes.

    Args:
        topic:
            Exact, case-sensitive frozen compliance-topic identifier.
        employee_id:
            Exact, case-sensitive employee identifier.

    Returns:
        A fresh JSON-compatible dictionary containing exactly
        ``compliant``, ``reasons``, and ``policy_refs``.

    Raises:
        TypeError:
            If ``topic`` or ``employee_id`` is not a string.
        ValueError:
            If either input is empty, whitespace-only, or has leading or
            trailing whitespace.
        MockDataError:
            If ``topic`` is unsupported, the employee fixture cannot be
            loaded safely, or the employee does not exist.
    """

    if not isinstance(
        topic,
        str,
    ):
        raise TypeError(
            "topic must be a string."
        )

    if (
        not topic
        or topic.isspace()
        or topic != topic.strip()
    ):
        raise ValueError(
            "topic must be a non-empty string "
            "without leading or trailing whitespace."
        )

    if topic != _SUPPORTED_COMPLIANCE_TOPIC:
        raise MockDataError(
            f"Unsupported compliance topic: {topic!r}."
        )

    if not isinstance(
        employee_id,
        str,
    ):
        raise TypeError(
            "employee_id must be a string."
        )

    if (
        not employee_id
        or employee_id.isspace()
        or employee_id != employee_id.strip()
    ):
        raise ValueError(
            "employee_id must be a non-empty string "
            "without leading or trailing whitespace."
        )

    employee_index = _load_employee_index()

    if employee_id not in employee_index:
        raise MockDataError(
            f"Employee not found: {employee_id!r}."
        )

    # V1 validates employee identity only. The frozen compliance result
    # is policy-determined and does not branch on employee attributes.
    return _project_policy_compliance()
