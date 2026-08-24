"""Operational trace structures for the S6 agent.

Trace items record observable execution facts only. They never contain
hidden chain-of-thought or private model reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.prompts import PROMPT_VERSION


@dataclass(
    frozen=True,
    slots=True,
)
class TraceItem:
    """One observable agent execution step."""

    step: int
    tool: str | None
    arguments: dict[str, Any]
    result_summary: str
    sources: tuple[dict[str, str], ...]
    decision: str
    prompt_version: str = PROMPT_VERSION

    def __post_init__(
        self,
    ) -> None:
        """Validate the small externally visible trace contract."""

        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 1
        ):
            raise ValueError(
                "Trace step must be a positive integer."
            )

        if (
            not isinstance(self.decision, str)
            or not self.decision.strip()
        ):
            raise ValueError(
                "Trace decision must be a non-empty string."
            )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        """Return a JSON-compatible trace representation."""

        return {
            "step": self.step,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "result_summary": self.result_summary,
            "sources": [
                dict(source)
                for source in self.sources
            ],
            "decision": self.decision,
            "prompt_version": self.prompt_version,
        }
