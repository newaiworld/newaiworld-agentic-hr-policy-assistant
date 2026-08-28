"""OpenAI-compatible LLM boundary for the S6 agent.

This is the only production agent module that calls the external LLM API.
Groq and the documented OpenRouter fallback both expose compatible
chat-completions endpoints, so V1 uses the already-pinned httpx client.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import monotonic
from typing import Any, Sequence

import httpx


LLM_TIMEOUT_SECONDS = 30
LLM_MAX_ATTEMPTS = 2
_LLM_RETRYABLE_STATUS_CODES = frozenset(
    {
        429,
        503,
    }
)


class LLMError(RuntimeError):
    """Raised when the external LLM cannot return a usable response."""


@dataclass(
    frozen=True,
    slots=True,
)
class LLMToolCall:
    """One normalized LLM-requested tool call."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(
    frozen=True,
    slots=True,
)
class LLMResponse:
    """Normalized response consumed later by the agent loop."""

    content: str | None
    tool_calls: tuple[LLMToolCall, ...]


class LLMClient:
    """Minimal asynchronous OpenAI-compatible chat client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create the LLM boundary from explicit values or environment."""

        self._api_key = self._require_setting(
            api_key
            if api_key is not None
            else os.getenv("LLM_API_KEY"),
            "LLM_API_KEY",
        )

        self._base_url = self._require_setting(
            base_url
            if base_url is not None
            else os.getenv("LLM_BASE_URL"),
            "LLM_BASE_URL",
        ).rstrip("/")

        self._model = self._require_setting(
            model
            if model is not None
            else os.getenv("LLM_MODEL"),
            "LLM_MODEL",
        )

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                LLM_TIMEOUT_SECONDS
            ),
            transport=transport,
        )

    async def chat(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
    ) -> LLMResponse:
        """Send one deterministic chat-completions request."""

        if not messages:
            raise ValueError(
                "messages must contain at least one message."
            )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": 0,
        }

        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        deadline = (
            monotonic()
            + LLM_TIMEOUT_SECONDS
        )

        for attempt in range(
            1,
            LLM_MAX_ATTEMPTS + 1,
        ):
            remaining_seconds = (
                deadline
                - monotonic()
            )

            if remaining_seconds <= 0:
                raise LLMError(
                    "LLM request timed out."
                )

            try:
                response = await self._client.post(
                    (
                        f"{self._base_url}"
                        "/chat/completions"
                    ),
                    headers={
                        "Authorization": (
                            f"Bearer {self._api_key}"
                        ),
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=httpx.Timeout(
                        remaining_seconds
                    ),
                )

                response.raise_for_status()

            except httpx.TimeoutException as exc:
                raise LLMError(
                    "LLM request timed out."
                ) from exc

            except httpx.HTTPStatusError as exc:
                status_code = (
                    exc.response.status_code
                )

                should_retry = (
                    status_code
                    in _LLM_RETRYABLE_STATUS_CODES
                    and attempt
                    < LLM_MAX_ATTEMPTS
                )

                if should_retry:
                    continue

                raise LLMError(
                    "LLM request failed with HTTP status "
                    f"{status_code}."
                ) from exc

            except httpx.HTTPError as exc:
                raise LLMError(
                    "LLM request failed."
                ) from exc

            break

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(
                "LLM response was not valid JSON."
            ) from exc

        return self._normalize_response(
            body
        )

    async def close(
        self,
    ) -> None:
        """Close the underlying HTTP connection pool."""

        await self._client.aclose()

    @staticmethod
    def _normalize_response(
        body: Any,
    ) -> LLMResponse:
        """Validate and normalize one OpenAI-compatible response."""

        if not isinstance(
            body,
            dict,
        ):
            raise LLMError(
                "LLM response must be a JSON object."
            )

        choices = body.get(
            "choices"
        )

        if (
            not isinstance(choices, list)
            or not choices
        ):
            raise LLMError(
                "LLM response contains no choices."
            )

        first = choices[0]

        if not isinstance(
            first,
            dict,
        ):
            raise LLMError(
                "LLM response choice is invalid."
            )

        message = first.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            raise LLMError(
                "LLM response message is invalid."
            )

        content = message.get(
            "content"
        )

        if content is not None:
            if not isinstance(
                content,
                str,
            ):
                raise LLMError(
                    "LLM response content is invalid."
                )

            content = content.strip() or None

        raw_tool_calls = message.get(
            "tool_calls",
            [],
        )

        if not isinstance(
            raw_tool_calls,
            list,
        ):
            raise LLMError(
                "LLM tool_calls field is invalid."
            )

        tool_calls = tuple(
            LLMClient._normalize_tool_call(
                item
            )
            for item in raw_tool_calls
        )

        if (
            content is None
            and not tool_calls
        ):
            raise LLMError(
                "LLM response contains neither content nor tool calls."
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _normalize_tool_call(
        item: Any,
    ) -> LLMToolCall:
        """Normalize one OpenAI-compatible function tool call."""

        if not isinstance(
            item,
            dict,
        ):
            raise LLMError(
                "LLM tool call is invalid."
            )

        call_id = item.get(
            "id",
        )

        function = item.get(
            "function",
        )

        if (
            not isinstance(call_id, str)
            or not call_id.strip()
        ):
            raise LLMError(
                "LLM tool call has an invalid id."
            )

        if not isinstance(
            function,
            dict,
        ):
            raise LLMError(
                "LLM tool call function is invalid."
            )

        name = function.get(
            "name",
        )

        raw_arguments = function.get(
            "arguments",
        )

        if (
            not isinstance(name, str)
            or not name.strip()
        ):
            raise LLMError(
                "LLM tool call has an invalid function name."
            )

        if not isinstance(
            raw_arguments,
            str,
        ):
            raise LLMError(
                "LLM tool call arguments must be JSON text."
            )

        try:
            arguments = json.loads(
                raw_arguments
            )
        except json.JSONDecodeError as exc:
            raise LLMError(
                "LLM tool call arguments are not valid JSON."
            ) from exc

        if not isinstance(
            arguments,
            dict,
        ):
            raise LLMError(
                "LLM tool call arguments must decode to an object."
            )

        return LLMToolCall(
            call_id=call_id.strip(),
            name=name.strip(),
            arguments=arguments,
        )

    @staticmethod
    def _require_setting(
        value: str | None,
        name: str,
    ) -> str:
        """Validate one required LLM configuration value."""

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise LLMError(
                f"{name} is not configured."
            )

        return value.strip()
