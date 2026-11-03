"""Wiring edit proposers. Both produce `Connect` edits that CircuitGraph then validates.

- GreedyProposer: deterministic, ordered so that simple trees/chains come out planar.
- LLMProposer: asks an LLM; edits are validated on a clone first and errors fed back for
  retry; falls back to the greedy proposer if the model keeps failing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Protocol, Sequence

from pydantic import BaseModel, Field

from ...core.schemas import Connect, TemplateEdge
from ...llm.adapter import LLMFormatError, StructuredLLM
from .circuit import CircuitGraph

Pending = Sequence[tuple[int, TemplateEdge]]


class EditProposer(Protocol):
    def propose(self, graph: CircuitGraph, pending: Pending, feedback: Sequence[str] = ()) -> list[Connect]: ...


class GreedyProposer:
    """Per component, outgoing edges (sorted by target order) take free right-side ports top->bottom;
    incoming edges (sorted by source order) take free left-side ports top->bottom."""

    def propose(self, graph: CircuitGraph, pending: Pending, feedback: Sequence[str] = ()) -> list[Connect]:
        order = {n: i for i, n in enumerate(graph.node_ids)}
        outs: dict[str, list] = defaultdict(list)
        ins: dict[str, list] = defaultdict(list)
        for idx, e in pending:
            outs[e.src].append((order[e.dst], idx))
            ins[e.dst].append((order[e.src], idx))
        src_port: dict[int, str] = {}
        dst_port: dict[int, str] = {}
        for node, items in outs.items():
            for (_, idx), port in zip(sorted(items), graph.free_ports(node, "out")):
                src_port[idx] = port.name
        for node, items in ins.items():
            for (_, idx), port in zip(sorted(items), graph.free_ports(node, "in")):
                dst_port[idx] = port.name
        return [
            Connect(src_node=e.src, src_port=src_port[idx], dst_node=e.dst, dst_port=dst_port[idx])
            for idx, e in pending
            if idx in src_port and idx in dst_port
        ]


class EditList(BaseModel):
    edits: list[Connect] = Field(default_factory=list)


class LLMProposer:
    SYSTEM = """You are an assistant to a photonic engineer. You wire a circuit schematic by choosing which port of one component connects to which port of another.
Rules:
- Light flows from a component's output ports (right side) into the next component's input ports (left side).
- Each port can be used by at most one connection.
- Make exactly one connection for every pending edge and no others.
- Ports are listed top to bottom. When one component feeds several, give the top output port to the target listed first, so wires do not cross.
Return JSON: {"edits": [{"src_node": "...", "src_port": "...", "dst_node": "...", "dst_port": "..."}]}"""

    def __init__(self, llm: StructuredLLM, fallback: Optional[EditProposer] = None):
        self.llm = llm
        self.fallback = fallback or GreedyProposer()

    def _describe(self, graph: CircuitGraph) -> str:
        lines = []
        for n in graph.node_ids:
            outs = ", ".join(p.name for p in graph.free_ports(n, "out")) or "none"
            ins = ", ".join(p.name for p in graph.free_ports(n, "in")) or "none"
            lines.append(f"{n} = {graph.cells[n].name} | free outputs (top to bottom): {outs} | free inputs (top to bottom): {ins}")
        return "\n".join(lines)

    def propose(self, graph: CircuitGraph, pending: Pending, feedback: Sequence[str] = ()) -> list[Connect]:
        user = "Components:\n" + self._describe(graph) + "\n\nPending edges:\n" + "\n".join(
            f"{e.src} -> {e.dst}" for _, e in pending
        )
        if feedback:
            user += "\n\nProblems found earlier:\n" + "\n".join(f"- {m}" for m in feedback[-5:])

        def check(el: EditList) -> None:
            trial = graph.clone()
            for op in el.edits:
                trial.apply(op)  # raises WiringError (ValueError) -> retry with message
            if trial.remaining():
                missing = ", ".join(f"{a}->{b}" for (a, b) in trial.remaining())
                raise ValueError(f"these edges are still not connected: {missing}")

        try:
            return self.llm.generate(self.SYSTEM, user, EditList, check).edits
        except LLMFormatError:
            return self.fallback.propose(graph, pending, feedback)
