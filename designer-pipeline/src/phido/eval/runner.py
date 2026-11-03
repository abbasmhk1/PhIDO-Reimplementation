"""Evaluate each stage on its own, with gold inputs, so upstream errors never leak in."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import yaml
from pydantic import BaseModel, Field

from ..core.errors import StageError
from ..core.schemas import DesignerInput, InterpreterInput, PICTemplate
from ..knowledge.text import canonical_kind
from ..stages.designer import DesignerStage
from ..stages.interpreter import InterpreterStage
from .metrics import summarize


# ----------------------------------------------------------------- cases
class InterpreterCase(BaseModel):
    id: str
    level: int = 0
    prompt: str
    nodes: list[dict]
    edges: list[dict] = Field(default_factory=list)

    def gold(self) -> PICTemplate:
        return PICTemplate.model_validate({"nodes": self.nodes, "edges": self.edges})


class DesignerCase(BaseModel):
    id: str
    level: int = 0
    template: PICTemplate
    expect: dict[str, list[str]] = Field(default_factory=dict)  # node id -> acceptable cell names


def load_interpreter_cases(path: str | Path) -> list[InterpreterCase]:
    return [InterpreterCase.model_validate(c) for c in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


def load_designer_cases(path: str | Path) -> list[DesignerCase]:
    return [DesignerCase.model_validate(c) for c in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


# ----------------------------------------------------------------- comparison / classification
def compare_templates(pred: PICTemplate, gold: PICTemplate) -> tuple[str, str]:
    """EE if components or connectivity differ from gold (kinds canonicalised; ports ignored in v0)."""
    pk = {n.id: canonical_kind(n.kind) for n in pred.nodes}
    gk = {n.id: canonical_kind(n.kind) for n in gold.nodes}
    from collections import Counter

    cp, cg = Counter(pk.values()), Counter(gk.values())
    if cp != cg:
        return "EE", f"component mismatch: missing={dict(cg - cp)} extra={dict(cp - cg)}"

    def graph(kinds: dict[str, str], t: PICTemplate) -> nx.DiGraph:
        g = nx.DiGraph()
        for nid, k in kinds.items():
            g.add_node(nid, kind=k)
        g.add_edges_from((e.src, e.dst) for e in t.edges)
        return g

    if not nx.is_isomorphic(graph(pk, pred), graph(gk, gold), node_match=lambda a, b: a["kind"] == b["kind"]):
        return "EE", "connectivity mismatch"
    return "S", "ok"


def classify_designer(stage: DesignerStage, case: DesignerCase) -> tuple[str, str]:
    try:
        out = stage.run(DesignerInput(template=case.template))
    except StageError as e:
        return e.code, e.message
    for ch in out.choices:
        allowed = case.expect.get(ch.node)
        if allowed and ch.cell not in allowed:
            return "CS", f"{ch.node}: got {ch.cell}, expected one of {allowed}"
    for code in ("PC", "SG"):
        for issue in out.issues:
            if issue.code == code:
                return code, issue.message
    return "S", "ok"


# ----------------------------------------------------------------- runners
@dataclass
class EvalReport:
    outcomes: dict[str, list[str]] = field(default_factory=dict)
    details: dict[str, list[str]] = field(default_factory=dict)

    def summary(self, ks=(1, 5)) -> dict:
        return summarize(self.outcomes, ks)

    def failures(self) -> list[tuple[str, str]]:
        return [(cid, d) for cid, ds in self.details.items() for o, d in zip(self.outcomes[cid], ds) if o != "S"]


def eval_interpreter(stage: InterpreterStage, cases: list[InterpreterCase], trials: int = 1) -> EvalReport:
    rep = EvalReport()
    for case in cases:
        for _ in range(trials):
            try:
                out = stage.run(InterpreterInput(prompt=case.prompt))
                code, detail = compare_templates(out.template, case.gold())
            except StageError as e:
                code, detail = e.code, e.message
            rep.outcomes.setdefault(case.id, []).append(code)
            rep.details.setdefault(case.id, []).append(detail)
    return rep


def eval_designer(stage: DesignerStage, cases: list[DesignerCase], trials: int = 1) -> EvalReport:
    rep = EvalReport()
    for case in cases:
        for _ in range(trials):
            code, detail = classify_designer(stage, case)
            rep.outcomes.setdefault(case.id, []).append(code)
            rep.details.setdefault(case.id, []).append(detail)
    return rep
