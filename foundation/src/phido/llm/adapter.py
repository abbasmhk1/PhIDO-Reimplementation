"""Thin LLM adapter: text in, validated pydantic object out, with validate-and-retry.

All retry/repair logic lives here, so stages never parse raw model text themselves.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str: ...


class LLMFormatError(Exception):
    """The model never produced a valid answer within the retry budget."""


def extract_json(raw: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()  # reasoning models (qwen3, r1)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the answer")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e


def _short(err: Exception, limit: int = 600) -> str:
    s = str(err)
    return s if len(s) <= limit else s[:limit] + " ..."


class StructuredLLM:
    def __init__(
        self,
        client: LLMClient,
        max_retries: int = 2,
        temperature: float = 0.0,
        include_schema: bool = True,
    ):
        self.client = client
        self.max_retries = max_retries
        self.temperature = temperature
        self.include_schema = include_schema
        self._trace: list[dict] = []

    def generate(
        self,
        system: str,
        user: str,
        schema: type[T],
        validate: Optional[Callable[[T], None]] = None,
    ) -> T:
        """Ask for a `schema` object. `validate` may raise ValueError to trigger a retry
        with the error message fed back to the model (use it for semantic checks)."""
        system_full = system
        if self.include_schema:
            system_full += (
                "\n\nReturn ONLY one JSON object, no prose and no code fences, "
                "matching this JSON schema:\n" + json.dumps(schema.model_json_schema())
            )
        feedback, last_err = "", ""
        for attempt in range(self.max_retries + 1):
            raw = self.client.complete(system_full, user + feedback, temperature=self.temperature)
            entry: dict = {"attempt": attempt, "raw": raw}
            try:
                obj = schema.model_validate(extract_json(raw))
                if validate is not None:
                    validate(obj)
                entry["ok"] = True
                self._trace.append(entry)
                return obj
            except ValueError as e:  # includes pydantic.ValidationError
                last_err = _short(e)
                entry["error"] = last_err
                self._trace.append(entry)
                feedback = (
                    f"\n\nYour previous answer was rejected: {last_err}\n"
                    "Fix the problem and return the corrected JSON object only."
                )
        raise LLMFormatError(f"no valid answer after {self.max_retries + 1} attempts: {last_err}")

    def drain_trace(self) -> list[dict]:
        trace, self._trace = self._trace, []
        return trace
