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
    "unit, n, [ports], [in_kind], [out_kind], [io_kind]",
    "n identical units in series (each feeds the next). in_kind adds one component of that kind at the "
    "input end only, out_kind at the output end only, io_kind (shorthand) at both ends; e.g. edge_coupler.",
)
def _cascade(args: dict) -> PICTemplate:
    unit = _str(args, "unit", "mzi")
    n = _int(args, "n", hi=64)
    ports = _ports(args, "ports", None)
    io = _str(args, "io_kind")
    in_kind, out_kind = _str(args, "in_kind", io), _str(args, "out_kind", io)
    nodes = [TemplateNode(id=f"N{i}", kind=unit, ports=ports) for i in range(1, n + 1)]
    edges = [TemplateEdge(src=f"N{i}", dst=f"N{i + 1}") for i in range(1, n)]
    if in_kind:
        nodes.insert(0, TemplateNode(id="IN1", kind=in_kind))
        edges.insert(0, TemplateEdge(src="IN1", dst="N1"))
    if out_kind:
        nodes.append(TemplateNode(id="OUT1", kind=out_kind))
        edges.append(TemplateEdge(src=f"N{n}", dst="OUT1"))
    return PICTemplate(title=f"cascade of {n} {unit}", nodes=nodes, edges=edges, source="motif")


@motif(
    "tree",
    "unit, stages, [fanout=2], [ports], [in_kind], [out_kind], [io_kind]",
    "balanced splitter tree: each unit feeds `fanout` units of the next stage; stages=4, fanout=2 "
    "gives 15 units and 16 outputs. in_kind adds one component at the root input only, out_kind one "
    "component per tree output only, io_kind (shorthand) both.",
)
def _tree(args: dict) -> PICTemplate:
    unit = _str(args, "unit", "mmi")
    stages = _int(args, "stages", hi=8)
    fanout = _int(args, "fanout", default=2, lo=2, hi=4)
    ports = _ports(args, "ports", f"1x{fanout}")
    io = _str(args, "io_kind")
    in_kind, out_kind = _str(args, "in_kind", io), _str(args, "out_kind", io)
    total = (fanout**stages - 1) // (fanout - 1)
    leaves = fanout ** (stages - 1)
    if total + (1 if in_kind else 0) + (leaves * fanout if out_kind else 0) > 600:
        raise MotifError(f"tree too large ({total} units); reduce stages or fanout")
    nodes = [TemplateNode(id=f"N{i}", kind=unit, ports=ports) for i in range(1, total + 1)]
    edges = []
    for parent in range(1, total - leaves + 1):  # heap indexing for a fanout-ary tree
        for c in range(fanout):
            edges.append(TemplateEdge(src=f"N{parent}", dst=f"N{fanout * (parent - 1) + 2 + c}"))
    if in_kind:
        nodes.append(TemplateNode(id="IN1", kind=in_kind))
        edges.append(TemplateEdge(src="IN1", dst="N1"))
    if out_kind:
        out = 0
        for leaf in range(total - leaves + 1, total + 1):
            for _ in range(fanout):
                out += 1
                nodes.append(TemplateNode(id=f"OUT{out}", kind=out_kind))
                edges.append(TemplateEdge(src=f"N{leaf}", dst=f"OUT{out}"))
    return PICTemplate(
        title=f"{fanout}-ary tree of {unit}, {stages} stages", nodes=nodes, edges=edges, source="motif"
    )


@motif(
    "mesh",
    "modes, [topology=clements|reck], [unit=mzi], [ports=2x2], [in_kind], [out_kind], [io_kind]",
    "N-mode interferometer mesh of 2x2 units (N(N-1)/2 units). clements = rectangular mesh (N columns), "
    "reck = triangular mesh. in_kind adds one component per mode at the inputs, out_kind per mode at the outputs.",
)
def _mesh(args: dict) -> PICTemplate:
    n = _int(args, "modes", lo=2, hi=12)
    topo = (_str(args, "topology", "clements") or "clements").lower()
    if topo not in ("clements", "reck"):
        raise MotifError(f"argument 'topology' must be 'clements' or 'reck', got {topo!r}")
    unit, ports = _str(args, "unit", "mzi"), _ports(args, "ports", "2x2")
    io = _str(args, "io_kind")
    in_kind, out_kind = _str(args, "in_kind", io), _str(args, "out_kind", io)
    if topo == "clements":
        pairs = [m for c in range(n) for m in range(c % 2, n - 1, 2)]  # column by column
    else:
        pairs = [i for j in range(n - 1, 0, -1) for i in range(j)]  # triangle, one diagonal at a time
    nodes: list[TemplateNode] = []
    edges: list[TemplateEdge] = []
    first: dict[int, str] = {}
    last: dict[int, str] = {}
    for k, m in enumerate(pairs, start=1):
        nid = f"N{k}"
        nodes.append(TemplateNode(id=nid, kind=unit, ports=ports))
        for mode in (m, m + 1):  # each mode continues from the previous unit that touched it
            if mode in last:
                edges.append(TemplateEdge(src=last[mode], dst=nid))
            else:
                first[mode] = nid
            last[mode] = nid
    if in_kind:
        for mode in range(n):
            nodes.append(TemplateNode(id=f"IN{mode + 1}", kind=in_kind))
            edges.append(TemplateEdge(src=f"IN{mode + 1}", dst=first[mode]))
    if out_kind:
        for mode in range(n):
            nodes.append(TemplateNode(id=f"OUT{mode + 1}", kind=out_kind))
            edges.append(TemplateEdge(src=last[mode], dst=f"OUT{mode + 1}"))
    return PICTemplate(title=f"{n}-mode {topo} mesh of {unit}", nodes=nodes, edges=edges, source="motif")
