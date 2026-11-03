"""Component selection: map each template node to a PDK cell.

Retrieval ranks candidates; a Chooser (usually an LLM) is consulted only when the choice
is ambiguous. Each candidate is labelled exact / partial / poor, as in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

from pydantic import BaseModel

from ...core.errors import StageError
from ...core.schemas import ComponentChoice, TemplateNode
from ...knowledge.pdk import PDK, Cell
from ...llm.adapter import LLMFormatError, StructuredLLM

Label = Literal["exact", "partial", "poor"]


@dataclass(frozen=True)
class Candidate:
    cell: Cell
    score: float
    coverage: float
    ports_ok: bool
    spec_ok: bool
    label: Label


class Chooser(Protocol):
    def choose(self, node: TemplateNode, cands: list[Candidate]) -> str: ...


class _Pick(BaseModel):
    cell: str


class LLMChooser:
    SYSTEM = (
        "You are an assistant to a photonic engineer. Pick the single PDK cell that best "
        "implements the requested component. Consider the described technology, speed and port count. "
        'Return JSON: {"cell": "<one of the listed names>"}.'
    )

    def __init__(self, llm: StructuredLLM):
        self.llm = llm

    def choose(self, node: TemplateNode, cands: list[Candidate]) -> str:
        names = [c.cell.name for c in cands]
        lines = [
            f"- {c.cell.name}: {c.cell.description} (ports {c.cell.signature}, properties {c.cell.properties})"
            for c in cands
        ]
        user = (
            f"Requested component: kind={node.kind}, ports={node.ports}, "
            f"description={node.description!r}, params={node.params}\nCandidates:\n" + "\n".join(lines)
        )

        def check(p: _Pick) -> None:
            if p.cell not in names:
                raise ValueError(f"'{p.cell}' is not a candidate; choose one of {names}")

        try:
            return self.llm.generate(self.SYSTEM, user, _Pick, check).cell
        except LLMFormatError:
            return names[0]  # fall back to the retrieval winner


class ComponentSelector:
    def __init__(
        self,
        pdk: PDK,
        chooser: Optional[Chooser] = None,
        top_k: int = 4,
        margin: float = 0.15,
        min_coverage: float = 0.5,
    ):
        self.pdk, self.chooser = pdk, chooser
        self.top_k, self.margin, self.min_coverage = top_k, margin, min_coverage

    def _spec_ok(self, node: TemplateNode, cell: Cell) -> bool:
        """Node params that name a PDK capability (e.g. modulation_bandwidth_ghz) are minimums."""
        for key, want in node.params.items():
            k = key.lower()
            if k in self.pdk.property_keys and isinstance(want, (int, float)) and not isinstance(want, bool):
                have = cell.properties.get(k)
                if not isinstance(have, (int, float)) or have < want:
                    return False
        return True

    def _label(self, coverage: float, ports_ok: bool, spec_ok: bool, score: float) -> Label:
        kind_ok = coverage >= self.min_coverage
        if kind_ok and ports_ok and spec_ok:
            return "exact"
        if kind_ok or (coverage > 0 and ports_ok):
            return "partial"
        return "poor"

    def candidates(self, node: TemplateNode) -> list[Candidate]:
        query = f"{node.kind} {node.ports or ''} {node.description}"
        out = []
        for h in self.pdk.search(query, k=self.top_k * 3, focus=node.kind):
            cell = self.pdk.cell(h.id)
            ports_ok = node.ports is None or cell.signature == node.ports
            spec_ok = self._spec_ok(node, cell)
            out.append(Candidate(cell, h.score, h.coverage, ports_ok, spec_ok,
                                 self._label(h.coverage, ports_ok, spec_ok, h.score)))
        # component kind outranks port count / requirements: a wrong-kind cell must never beat the right kind
        out.sort(
            key=lambda c: (c.coverage >= self.min_coverage, c.ports_ok and c.spec_ok, c.ports_ok, c.score),
            reverse=True,
        )
        return out[: self.top_k]

    def select(self, node: TemplateNode) -> tuple[Cell, ComponentChoice]:
        cands = self.candidates(node)
        if not cands or cands[0].label == "poor":
            raise StageError("CS", f"node {node.id}: no PDK cell matches kind '{node.kind}'")
        top = cands[0]
        near_tie = (
            len(cands) > 1
            and cands[1].label == "exact"
            and (top.score - cands[1].score) < self.margin * top.score
        )
        ambiguous = top.label != "exact" or near_tie
        chosen, by = top, "retrieval"
        if ambiguous and self.chooser is not None and len(cands) > 1:
            name = self.chooser.choose(node, cands)
            chosen = next((c for c in cands if c.cell.name == name), top)
            by = "llm"
        choice = ComponentChoice(
            node=node.id,
            cell=chosen.cell.name,
            match=chosen.label,
            score=round(chosen.score, 3),
            alternatives=[c.cell.name for c in cands if c is not chosen],
            ambiguous=ambiguous,
            chosen_by=by,
        )
        return chosen.cell, choice
