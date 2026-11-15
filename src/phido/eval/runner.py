"""Evaluate each stage on its own, with gold inputs, so upstream errors never leak in."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import yaml
from pydantic import BaseModel, Field

from ..core.errors import StageError
from ..core.schemas import (
    DesignerInput, InterpreterInput, LayoutInput, MotifSpec, PICTemplate, VerifyInput,
)
from ..knowledge.motifs import expand_motif
from ..knowledge.text import KNOWN_KINDS, canonical_kind
from ..stages.designer import DesignerStage
from ..stages.designer.params import norm_key
from ..stages.interpreter import InterpreterStage
from ..stages.layout.stage import LayoutStage
from ..stages.verify.stage import VerifyStage
from .metrics import summarize


# ----------------------------------------------------------------- cases
class InterpreterCase(BaseModel):
    id: str
    level: int = 0
    prompt: str
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    motif: Optional[MotifSpec] = None  # gold given as a motif (for circuits too big to list by hand)

    def gold(self) -> PICTemplate:
        if self.motif is not None:
            return expand_motif(self.motif)
        return PICTemplate.model_validate({"nodes": self.nodes, "edges": self.edges})


class DesignerCase(BaseModel):
    id: str
    level: int = 0
    template: PICTemplate
    expect: dict[str, list[str]] = Field(default_factory=dict)  # node id -> acceptable cell names
    expect_params: dict[str, dict[str, float]] = Field(default_factory=dict)  # node id -> PDK params


def load_interpreter_cases(path: str | Path) -> list[InterpreterCase]:
    return [InterpreterCase.model_validate(c) for c in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


def load_suite(name: str, data: Optional[Path] = None) -> list[InterpreterCase]:
    """'basic', 'hard', 'holdout' or 'all' (= basic + hard) interpreter benchmark cases from the data folder."""
    from ..paths import data_dir

    d = (Path(data) if data else data_dir()) / "testbench"
    files = {"basic": ["interpreter_gold.yaml"], "hard": ["interpreter_gold_hard.yaml"],
             "holdout": ["interpreter_gold_holdout.yaml"],  # kept out of "all" on purpose
             "all": ["interpreter_gold.yaml", "interpreter_gold_hard.yaml"]}
    if name not in files:
        raise KeyError(f"unknown suite '{name}'; use basic, hard, holdout or all")
    return [c for f in files[name] for c in load_interpreter_cases(d / f)]


def load_designer_cases(path: str | Path) -> list[DesignerCase]:
    return [DesignerCase.model_validate(c) for c in yaml.safe_load(Path(path).read_text(encoding="utf-8"))]


# ----------------------------------------------------------------- comparison / classification
def param_aliases(pdk) -> dict[str, str]:
    """normalised PDK parameter name or alias -> normalised canonical PDK name, e.g. coupling_length -> length."""
    out: dict[str, str] = {}
    for cell in pdk.cells.values():
        for name, spec in cell.params.items():
            out[norm_key(name)] = norm_key(name)
            for alias in spec.aliases:
                out[norm_key(alias)] = norm_key(name)
    return out


def _params_ok(pred: dict, gold: dict, aliases: Optional[dict[str, str]] = None) -> bool:
    canon = lambda k: (aliases or {}).get(norm_key(k), norm_key(k))
    p = {canon(k): v for k, v in pred.items()}
    for k, want in gold.items():
        got = p.get(canon(k))
        if got is None:
            return False
        if isinstance(want, (int, float)) and isinstance(got, (int, float)) and not isinstance(want, bool):
            if abs(float(want) - float(got)) > 1e-6:
                return False
        elif str(want).lower() != str(got).lower():
            return False
    return True


def _kind(node) -> str:
    """Canonical kind of a node; if its kind text is not recognised (e.g. 'phase_shifter'), use the description too."""
    k = canonical_kind(node.kind)
    if k not in KNOWN_KINDS and node.description:
        k2 = canonical_kind(f"{node.kind} {node.description}")
        return k2 if k2 in KNOWN_KINDS else k
    return k


def compare_templates(pred: PICTemplate, gold: PICTemplate, aliases: Optional[dict[str, str]] = None) -> tuple[str, str]:
    """EE if components, connectivity, (stated) ports or (stated) parameters differ from gold.

    Kinds are canonicalised. Ports/params are only checked where the gold node states them. `aliases`
    (see param_aliases) lets parameter names match through the PDK's alias table."""
    cp = Counter(_kind(n) for n in pred.nodes)
    cg = Counter(_kind(n) for n in gold.nodes)
    if cp != cg:
        return "EE", f"component mismatch: missing={dict(cg - cp)} extra={dict(cp - cg)}"

    def graph(t: PICTemplate) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in t.nodes:
            g.add_node(n.id, kind=_kind(n), ports=n.ports, params=n.params)
        g.add_edges_from((e.src, e.dst) for e in t.edges)
        return g

    gp, gg = graph(pred), graph(gold)
    by_kind = lambda a, b: a["kind"] == b["kind"]
    if not nx.is_isomorphic(gp, gg, node_match=by_kind):
        return "EE", "connectivity mismatch"
    with_ports = lambda a, b: a["kind"] == b["kind"] and (b["ports"] is None or a["ports"] == b["ports"])
    matcher = nx.isomorphism.DiGraphMatcher(gp, gg, node_match=with_ports)
    if not matcher.is_isomorphic():
        return "EE", "port mismatch (components and links match)"
    if any(n.params for n in gold.nodes):
        for k, mapping in enumerate(matcher.isomorphisms_iter()):
            if k >= 5000:
                break
            if all(_params_ok(gp.nodes[a]["params"], gg.nodes[b]["params"], aliases) for a, b in mapping.items()):
                return "S", "ok"
        shown = lambda t: sorted(f"{_kind(n)}{dict(sorted(n.params.items()))}" for n in t.nodes if n.params)
        return "EE", f"parameter mismatch: predicted {shown(pred)} vs gold {shown(gold)}"
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
    for node, want in case.expect_params.items():
        got = out.nodes[node].params
        for k, v in want.items():
            if k not in got or abs(float(got[k]) - float(v)) > 1e-6:
                return "PC", f"{node}.{k}: got {got.get(k)}, expected {v}"
    for code in ("PC", "SG"):
        for issue in out.issues:
            if issue.code == code:
                return code, issue.message
    return "S", "ok"


def classify_downstream(designer: DesignerStage, stage, case: DesignerCase) -> tuple[str, str]:
    """Run designer on a gold template, then the layout or verify stage; code = first failing stage."""
    try:
        design = designer.run(DesignerInput(template=case.template))
    except StageError as e:
        return e.code, e.message
    if design.issues:
        return design.issues[0].code, design.issues[0].message
    if isinstance(stage, LayoutStage):
        out = stage.run(LayoutInput(design=design))
    else:
        out = stage.run(VerifyInput(design=design))
    return (out.issues[0].code, out.issues[0].message) if out.issues else ("S", "ok")


# ----------------------------------------------------------------- runners
@dataclass
class EvalReport:
    outcomes: dict[str, list[str]] = field(default_factory=dict)
    details: dict[str, list[str]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)  # LLM calls / seconds / tokens, when an LLM was used

    def summary(self, ks=(1, 5)) -> dict:
        out = summarize(self.outcomes, ks)
        trials = sum(len(v) for v in self.outcomes.values()) or 1
        if self.stats.get("calls"):
            out["llm"] = {
                "calls_per_trial": round(self.stats["calls"] / trials, 2),
                "seconds_per_trial": round(self.stats["seconds"] / trials, 2),
                "prompt_tokens_per_trial": round(self.stats["prompt_tokens"] / trials, 1),
                "completion_tokens_per_trial": round(self.stats["completion_tokens"] / trials, 1),
            }
        return out

    def failures(self) -> list[tuple[str, str]]:
        return [(cid, d) for cid, ds in self.details.items() for o, d in zip(self.outcomes[cid], ds) if o != "S"]


def _record(rep: EvalReport, cid: str, code: str, detail: str) -> None:
    rep.outcomes.setdefault(cid, []).append(code)
    rep.details.setdefault(cid, []).append(detail)


def eval_interpreter(stage: InterpreterStage, cases: list[InterpreterCase], trials: int = 1,
                     aliases: Optional[dict[str, str]] = None) -> EvalReport:
    """aliases: parameter-name aliases to accept; defaults to the data folder's PDK."""
    if aliases is None:
        from ..pipeline import load_pdk

        aliases = param_aliases(load_pdk())
    rep = EvalReport()
    if stage.llm is not None:
        stage.llm.reset_stats()
    for case in cases:
        for _ in range(trials):
            try:
                out = stage.run(InterpreterInput(prompt=case.prompt))
                code, detail = compare_templates(out.template, case.gold(), aliases)
            except StageError as e:
                code, detail = e.code, e.message
            _record(rep, case.id, code, detail)
    if stage.llm is not None:
        rep.stats = stage.llm.reset_stats()
    return rep


def eval_designer(stage: DesignerStage, cases: list[DesignerCase], trials: int = 1) -> EvalReport:
    rep = EvalReport()
    if stage.llm is not None:
        stage.llm.reset_stats()
    for case in cases:
        for _ in range(trials):
            _record(rep, case.id, *classify_designer(stage, case))
    if stage.llm is not None:
        rep.stats = stage.llm.reset_stats()
    return rep


def eval_downstream(designer: DesignerStage, stage, cases: list[DesignerCase]) -> EvalReport:
    rep = EvalReport()
    for case in cases:
        _record(rep, case.id, *classify_downstream(designer, stage, case))
    return rep
