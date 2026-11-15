from __future__ import annotations

from typing import Callable, Union


class MockLLM:
    """Scripted client for tests: a list of replies (consumed in order) or a function(system, user)."""

    def __init__(self, responses: Union[list[str], Callable[[str, str], str]]):
        self._fn = responses if callable(responses) else None
        self._queue = [] if callable(responses) else list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        if self._fn is not None:
            return self._fn(system, user)
        if not self._queue:
            raise RuntimeError("MockLLM has no more scripted responses")
        return self._queue.pop(0)
