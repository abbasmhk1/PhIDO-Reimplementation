"""Typed contracts between stages. Everything that crosses a stage boundary lives here."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

Scalar = Union[bool, int, float, str]

IDENT = r"^[A-Za-z][A-Za-z0-9_]*$"  # node / block ids (kept free of ',' and ':' so links parse)
PORTS = r"^\d+x\d+$"  # "<inputs>x<outputs>"


def _drop_none(v):
    """LLMs often write "key": null for 'not given'; treat that as absent instead of a type error."""
    return {k: x for k, x in v.items() if x is not None} if isinstance(v, dict) else v


# ----------------------------------------------------------------- interpreter: LLM output
class Block(BaseModel):
    name: str = Field(pattern=IDENT)
    kind: str = Field(min_length=1)
    ports: Optional[str] = Field(default=None, pattern=PORTS)
    description: str = ""
    params: dict[str, Scalar] = Field(default_factory=dict)

    _params_no_none = field_validator("params", mode="before")(classmethod(lambda cls, v: _drop_none(v)))


class Link(BaseModel):
    src: str
    dst: str


class MotifSpec(BaseModel):
    name: str
    args: dict[str, Scalar] = Field(default_factory=dict)

    _args_no_none = field_validator("args", mode="before")(classmethod(lambda cls, v: _drop_none(v)))


class EntityExtraction(BaseModel):
    """What the interpreter LLM must return: components (and optionally a repeating motif)."""

    blocks: list[Block] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    motif: Optional[MotifSpec] = None

    @model_validator(mode="after")
    def _consistent(self) -> "EntityExtraction":
        if not self.blocks and self.motif is None:
            raise ValueError("give at least one block, or a motif")
        names = [b.name for b in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError("block names must be unique")
        for link in self.links:
            for end in (link.src, link.dst):
                if end not in names:
                    raise ValueError(f"link endpoint '{end}' is not a block name")
            if link.src == link.dst:
                raise ValueError(f"link {link.src}->{link.dst} connects a block to itself")
        return self


# ----------------------------------------------------------------- PIC template (interpreter -> designer)
class TemplateNode(BaseModel):
    id: str = Field(pattern=IDENT)
    kind: str = Field(min_length=1)
    ports: Optional[str] = Field(default=None, pattern=PORTS)
    description: str = ""
    params: dict[str, Scalar] = Field(default_factory=dict)


class TemplateEdge(BaseModel):
    src: str
    dst: str


class PICTemplate(BaseModel):
    """Abstract circuit: components as nodes, connections as edges, no PDK cells, no port numbers."""

    title: str = ""
    nodes: list[TemplateNode]
    edges: list[TemplateEdge] = Field(default_factory=list)
    source: Literal["motif", "blocks", "library"] = "blocks"

    @model_validator(mode="after")
    def _valid_graph(self) -> "PICTemplate":
        if not self.nodes:
            raise ValueError("template needs at least one node")
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node ids must be unique")
        for e in self.edges:
            if e.src not in ids or e.dst not in ids:
                raise ValueError(f"edge {e.src}->{e.dst} refers to an unknown node")
            if e.src == e.dst:
                raise ValueError(f"edge {e.src}->{e.dst} is a self-loop")
        return self


class TemplateMatch(BaseModel):
    name: str
    score: float


class InterpreterInput(BaseModel):
    prompt: str


class InterpreterOutput(BaseModel):
    prompt: str
    extraction: Optional[EntityExtraction] = None  # None when a library template was adopted
    template: PICTemplate
    matches: list[TemplateMatch] = Field(default_factory=list)  # similar library templates
    warnings: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------- designer
class DesignerInput(BaseModel):
    template: PICTemplate


class Connect(BaseModel):
    """The one wiring edit: connect an output-side port of src to an input-side port of dst."""

    op: Literal["connect"] = "connect"
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str


class ComponentChoice(BaseModel):
    node: str
    cell: str
    match: Literal["exact", "partial", "poor"]
    score: float
    alternatives: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    chosen_by: Literal["retrieval", "llm"] = "retrieval"


class Issue(BaseModel):
    code: Literal["CS", "PC", "SG", "L", "V"]  # L = layout, V = verification
    node: Optional[str] = None
    message: str


class DSLNode(BaseModel):
    component: str
    params: dict[str, Scalar] = Field(default_factory=dict)


class DSLEdge(BaseModel):
    link: str  # "N1,o4:N2,o1"

    def parse(self) -> tuple[str, str, str, str]:
        """-> (src_node, src_port, dst_node, dst_port)"""
        a, b = self.link.replace(" ", "").split(":")
        (n1, p1), (n2, p2) = a.split(","), b.split(",")
        return n1, p1, n2, p2


class DesignerOutput(BaseModel):
    """The PIC DSL (nodes + edges) plus the decisions that led to it."""

    title: str = ""
    nodes: dict[str, DSLNode]
    edges: dict[str, DSLEdge] = Field(default_factory=dict)
    choices: list[ComponentChoice] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dsl(self) -> dict:
        return {
            "doc": {"title": self.title},
            "nodes": {k: v.model_dump(mode="json") for k, v in self.nodes.items()},
            "edges": {k: v.model_dump(mode="json") for k, v in self.edges.items()},
        }


# ----------------------------------------------------------------- layout
class LayoutInput(BaseModel):
    design: DesignerOutput


class Placement(BaseModel):
    node: str
    component: str
    x: float  # centre, um
    y: float
    width: float
    height: float
    rotation: int = 0
    backend: Literal["gdsfactory", "blackbox", "estimate"] = "estimate"


class RouteResult(BaseModel):
    edge: str
    link: str
    ok: bool
    length_um: Optional[float] = None
    message: str = ""


class LayoutOutput(BaseModel):
    placements: list[Placement]
    routes: list[RouteResult] = Field(default_factory=list)
    gds_path: Optional[str] = None
    issues: list[Issue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


# ----------------------------------------------------------------- verification
class VerifyInput(BaseModel):
    design: DesignerOutput
    wl_start_um: float = 1.50
    wl_stop_um: float = 1.60
    points: int = Field(default=201, ge=2, le=5001)


class PairSummary(BaseModel):
    src: str
    dst: str
    db_center: float  # power transmission at the centre wavelength
    peak_db: float
    peak_wavelength_um: float
    min_db: float


class VerifyOutput(BaseModel):
    inputs: list[str]
    outputs: list[str]
    wavelengths_um: list[float]
    transfer_db: dict[str, list[float]]  # "src->dst" -> power in dB per wavelength
    summary: list[PairSummary]
    unconnected: list[str] = Field(default_factory=list)  # component ports left open (normal for 2x2 cells used 1-in/1-out)
    issues: list[Issue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def lit_outputs(self) -> list[str]:
        return sorted({s.dst for s in self.summary})
