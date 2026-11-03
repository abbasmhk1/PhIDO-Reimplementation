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
from .params import ParamConfigurator
from .proposers import EditProposer, GreedyProposer, LLMProposer
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

    def _wire(self, graph: CircuitGraph, template: PICTemplate, issues: list[Issue], warnings: list[str]) -> None:
        feedback: list[str] = []
        for _ in range(self.max_wire_rounds):
            pending = graph.pending(template.edges)
            if not pending:
                break
            progress = False
            for op in self.proposer.propose(graph, pending, feedback):
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
        for node in t.nodes:
            cell, choice = self.selector.select(node)  # raises StageError("CS") when nothing fits
            p, iss = self.configurator.configure(node, cell)
            cells[node.id], params[node.id] = cell, p
            choices.append(choice)
            if choice.match != "exact":
                warnings.append(
                    f"{node.id}: {cell.name} is a {choice.match} match "
                    f"(requested ports {node.ports}, cell has {cell.signature})"
                )
            issues.extend(iss)
        graph = CircuitGraph(cells, t.edges)
        self._wire(graph, t, issues, warnings)
        warnings.extend(graph.warnings())
        return DesignerOutput(
            title=t.title,
            nodes={nid: DSLNode(component=cells[nid].name, params=params[nid]) for nid in cells},
            edges={f"E{i + 1}": DSLEdge(link=l) for i, l in enumerate(graph.links())},
            choices=choices,
            issues=issues,
            warnings=warnings,
        )
