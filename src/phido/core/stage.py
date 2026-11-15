from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from pydantic import BaseModel

from .errors import StageError
from .store import RunStore

if TYPE_CHECKING:  # avoid a runtime import cycle
    from ..llm.adapter import StructuredLLM

In = TypeVar("In", bound=BaseModel)
Out = TypeVar("Out", bound=BaseModel)


class Stage(ABC, Generic[In, Out]):
    """A pipeline stage: validated input model in, validated output model out.

    - `run(inp)` is pure logic: call it directly in tests and evals.
    - `stage(inp, store)` additionally persists input/output/LLM trace to a RunStore.
    """

    name: str = "stage"
    llm: Optional["StructuredLLM"] = None

    @abstractmethod
    def run(self, inp: In) -> Out: ...

    def __call__(self, inp: In, store: Optional[RunStore] = None) -> Out:
        if store:
            store.save(self.name, "input", inp)
        try:
            out = self.run(inp)
        except StageError as e:
            if store:
                store.save_text(self.name, "error.txt", str(e))
                self._dump_trace(store)
            raise
        if store:
            store.save(self.name, "output", out)
            self._dump_trace(store)
        return out

    def _dump_trace(self, store: RunStore) -> None:
        if self.llm is not None:
            trace = self.llm.drain_trace()
            if trace:
                store.save_json(self.name, "llm_trace.json", trace)
