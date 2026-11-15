from __future__ import annotations

from typing import Optional

from ...core.schemas import (
    ComponentChoice,
    DesignerInput,
    DesignerOutput,
    DSLEdge,
    DSLNode,
    Issue,
    PICTemplate,
)
from ...core.stage import Stage
from ...knowledge.pdk import PDK, Cell
from ...llm.adapter import StructuredLLM
from .circuit import CircuitGraph, WiringError
from .crossings import repair_crossings
from .params import ParamConfigurator
from .proposers import EditProposer, GreedyProposer, LLMProposer
from .throughput import decisive_pair
from .selector import Chooser, ComponentSelector, LLMChooser


class DesignerStage(Stage[DesignerInput, DesignerOutput]):
    """PICTemplate -> PIC DSL: component selection, parameter configuration, port-level wiring.

    Runs fully offline by default (retrieval + greedy wiring). Pass a chooser / proposer
    (or use `with_llm`) to let an LLM resolve ambiguous choices and propose wiring edits.
    """

    name = "designer"

    def __init__(
        self,
        pdk: PDK,
        chooser: Optional[Chooser] = None,
        proposer: Optional[EditProposer] = None,
        llm: Optional[StructuredLLM] = None,
        max_wire_rounds: int = 3,
    ):
        self.pdk = pdk
        self.selector = ComponentSelector(pdk, chooser)
        self.configurator = ParamConfigurator(pdk.property_keys)
        self.proposer: EditProposer = proposer or GreedyProposer()
        self.llm = llm
        self.max_wire_rounds = max_wire_rounds

    @classmethod
    def with_llm(cls, pdk: PDK, llm: StructuredLLM, select: bool = True, wire: bool = True) -> "DesignerStage":
        return cls(
            pdk,
            chooser=LLMChooser(llm) if select else None,
            proposer=LLMProposer(llm) if wire else None,
            llm=llm,
        )

    @staticmethod
    def _enforce_through_paths(graph: CircuitGraph, pending, ops: list) -> list:
        """Whoever proposed the wiring (e.g. an LLM that only sees port positions), a chain element - one wire in,
        one wire out - keeps its through-path ports, unless that would reuse a port another connection needs."""
        ins: dict[str, int] = {}
        outs: dict[str, int] = {}
        for _, e in pending:
            outs[e.src] = outs.get(e.src, 0) + 1
            ins[e.dst] = ins.get(e.dst, 0) + 1
        ops = [op.model_copy() for op in ops]
        for node in graph.node_ids:
            if ins.get(node) != 1 or outs.get(node) != 1:
                continue
            fin = [p.name for p in graph.free_ports(node, "in")]
            fout = [p.name for p in graph.free_ports(node, "out")]
            pair = decisive_pair(graph.cells[node], graph.params.get(node, {}), fin, fout) if len(fin) > 1 or len(fout) > 1 else None
            op_in = next((o for o in ops if o.dst_node == node), None)
            op_out = next((o for o in ops if o.src_node == node), None)
            if not pair or op_in is None or op_out is None:
                continue
            taken = {(o.src_node, o.src_port) for o in ops if o not in (op_out,)} | \
                    {(o.dst_node, o.dst_port) for o in ops if o not in (op_in,)}
            if (node, pair[0]) in taken or (node, pair[1]) in taken:
                continue
            op_in.dst_port, op_out.src_port = pair
        return ops

    def _wire(self, graph: CircuitGraph, template: PICTemplate, issues: list[Issue], warnings: list[str]) -> None:
        feedback: list[str] = []
        for _ in range(self.max_wire_rounds):
            pending = graph.pending(template.edges)
            if not pending:
                break
            progress = False
            for op in self._enforce_through_paths(graph, pending, self.proposer.propose(graph, pending, feedback)):
                try:
                    graph.apply(op)
                    progress = True
                except WiringError as e:
                    feedback.append(str(e))
            if not progress:
                break
        for _, e in graph.pending(template.edges):
            issues.append(Issue(code="SG", message=f"edge {e.src} -> {e.dst} could not be wired"))
        warnings.extend(feedback)

    def run(self, inp: DesignerInput) -> DesignerOutput:
        t = inp.template
        cells: dict[str, Cell] = {}
        choices: list[ComponentChoice] = []
        params: dict[str, dict] = {}
        issues: list[Issue] = []
        warnings: list[str] = []
        n_in = {n.id: 0 for n in t.nodes}
        n_out = {n.id: 0 for n in t.nodes}
        for e in t.edges:
            n_out[e.src] += 1
            n_in[e.dst] += 1
        seen_ambiguous: set[tuple] = set()
        for node in t.nodes:
            chain = n_in[node.id] == 1 and n_out[node.id] == 1
            cell, choice = self.selector.select(node, chain)  # raises StageError("CS") when nothing fits
            p, iss = self.configurator.configure(node, cell)
            cells[node.id], params[node.id] = cell, p
            choices.append(choice)
            if choice.ambiguous and choice.match == "exact" and choice.chosen_by == "retrieval":
                key = (node.kind, node.ports, choice.cell, tuple(choice.alternatives))
                if key not in seen_ambiguous:
                    seen_ambiguous.add(key)
                    warnings.append(f"{node.id}: '{node.kind}' matched several cells about equally; picked {choice.cell} "
                                    f"(others: {', '.join(choice.alternatives)}). Name a technology or port count to choose.")
            if choice.match != "exact":
                warnings.append(
                    f"{node.id}: {cell.name} is a {choice.match} match "
                    f"(requested ports {node.ports}, cell has {cell.signature})"
                )
            issues.extend(iss)
        graph = CircuitGraph(cells, t.edges)
        graph.params = params
        self._wire(graph, t, issues, warnings)
        if graph.connections:
            left = repair_crossings(graph)  # port swaps; placement-based estimate
            if left:
                warnings.append(f"estimated {left} waveguide crossing(s) remain after port swapping")
        warnings.extend(graph.warnings())
        return DesignerOutput(
            title=t.title,
            nodes={nid: DSLNode(component=cells[nid].name, params=params[nid]) for nid in cells},
            edges={f"E{i + 1}": DSLEdge(link=l) for i, l in enumerate(graph.links())},
            choices=choices,
            issues=issues,
            warnings=warnings,
        )
