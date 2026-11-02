"""Motif generators: repeating structures expanded by code, not by the LLM.

The LLM only names a motif and its few arguments ("tree", unit=mmi, stages=4), so a
63-node circuit costs a handful of output tokens and cannot be mis-enumerated.
Register new motifs with the @motif decorator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..core.schemas import MotifSpec, PICTemplate, TemplateEdge, TemplateNode


class MotifError(ValueError):
    pass


@dataclass(frozen=True)
class Motif:
    name: str
    signature: str
    doc: str
    fn: Callable[[dict], PICTemplate]


_REGISTRY: dict[str, Motif] = {}


def motif(name: str, signature: str, doc: str):
    def deco(fn):
        _REGISTRY[name] = Motif(name, signature, doc, fn)
        return fn

    return deco


def known_motifs() -> list[str]:
    return sorted(_REGISTRY)


def describe_motifs() -> str:
    return "\n".join(f"- {m.name}({m.signature}): {m.doc}" for m in _REGISTRY.values())


def expand_motif(spec: MotifSpec) -> PICTemplate:
    m = _REGISTRY.get(spec.name.strip().lower())
    if m is None:
        raise MotifError(f"unknown motif '{spec.name}'; known motifs: {', '.join(known_motifs())}")
    return m.fn(spec.args)


# ----------------------------------------------------------------- arg helpers
def _int(args: dict, key: str, default: Optional[int] = None, lo: int = 1, hi: int = 64) -> int:
    v = args.get(key, default)
    if v is None:
        raise MotifError(f"missing required argument '{key}'")
    try:
        v = int(float(v))
    except (TypeError, ValueError):
        raise MotifError(f"argument '{key}' must be an integer, got {v!r}") from None
    if not lo <= v <= hi:
        raise MotifError(f"argument '{key}' must be between {lo} and {hi}, got {v}")
    return v


def _str(args: dict, key: str, default: Optional[str] = None) -> Optional[str]:
    v = args.get(key, default)
    return None if v in (None, "") else str(v)


def _ports(args: dict, key: str, default: Optional[str]) -> Optional[str]:
    v = _str(args, key, default)
    if v is not None and not re.fullmatch(r"\d+x\d+", v):
        raise MotifError(f"argument '{key}' must look like '1x2', got {v!r}")
    return v


# ----------------------------------------------------------------- motifs
@motif(
    "cascade",
    "unit, n, [ports], [io_kind]",
    "n identical units in series (unit feeds the next unit). io_kind adds an input and output "
    "component of that kind at the two ends, e.g. edge_coupler.",
)
def _cascade(args: dict) -> PICTemplate:
    unit = _str(args, "unit", "mzi")
    n = _int(args, "n", hi=64)
    ports = _ports(args, "ports", None)
    io = _str(args, "io_kind")
    nodes = [TemplateNode(id=f"N{i}", kind=unit, ports=ports) for i in range(1, n + 1)]
    edges = [TemplateEdge(src=f"N{i}", dst=f"N{i + 1}") for i in range(1, n)]
    if io:
        nodes = [TemplateNode(id="IN1", kind=io), *nodes, TemplateNode(id="OUT1", kind=io)]
        edges = [TemplateEdge(src="IN1", dst="N1"), *edges, TemplateEdge(src=f"N{n}", dst="OUT1")]
    return PICTemplate(title=f"cascade of {n} {unit}", nodes=nodes, edges=edges, source="motif")


@motif(
    "tree",
    "unit, stages, [fanout=2], [ports], [io_kind]",
    "balanced splitter tree: each unit feeds `fanout` units of the next stage; stages=4, fanout=2 "
    "gives 15 units and 16 outputs. io_kind adds one input component at the root and one output "
    "component per tree output.",
)
def _tree(args: dict) -> PICTemplate:
    unit = _str(args, "unit", "mmi")
    stages = _int(args, "stages", hi=8)
    fanout = _int(args, "fanout", default=2, lo=2, hi=4)
    ports = _ports(args, "ports", f"1x{fanout}")
    io = _str(args, "io_kind")
    total = (fanout**stages - 1) // (fanout - 1)
    leaves = fanout ** (stages - 1)
    if total + (1 + leaves * fanout if io else 0) > 600:
        raise MotifError(f"tree too large ({total} units); reduce stages or fanout")
    nodes = [TemplateNode(id=f"N{i}", kind=unit, ports=ports) for i in range(1, total + 1)]
    edges = []
    for parent in range(1, total - leaves + 1):  # heap indexing for a fanout-ary tree
        for c in range(fanout):
            edges.append(TemplateEdge(src=f"N{parent}", dst=f"N{fanout * (parent - 1) + 2 + c}"))
    if io:
        nodes.append(TemplateNode(id="IN1", kind=io))
        edges.append(TemplateEdge(src="IN1", dst="N1"))
        out = 0
        for leaf in range(total - leaves + 1, total + 1):
            for _ in range(fanout):
                out += 1
                nodes.append(TemplateNode(id=f"OUT{out}", kind=io))
                edges.append(TemplateEdge(src=f"N{leaf}", dst=f"OUT{out}"))
    return PICTemplate(
        title=f"{fanout}-ary tree of {unit}, {stages} stages", nodes=nodes, edges=edges, source="motif"
    )
