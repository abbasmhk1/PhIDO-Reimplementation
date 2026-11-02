"""PDK: parametric cells with structured descriptors, searchable through a Retriever."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.schemas import Scalar
from .retriever import Hit, KeywordRetriever, Retriever


class PortSpec(BaseModel):
    name: str
    side: Literal["left", "right", "any"]  # light enters on left, leaves on right; "any" = either
    y: float = 0.5  # vertical position 0 (bottom) .. 1 (top); used to order ports


class ParamSpec(BaseModel):
    default: Scalar = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    unit: str = ""
    aliases: list[str] = Field(default_factory=list)


def _expand_ports(sig: str) -> list[dict]:
    """'2x3' -> counter-clockwise numbering: left ports bottom->top, right ports top->bottom."""
    n_left, n_right = (int(x) for x in sig.split("x"))

    def ys(n: int) -> list[float]:
        return [0.5] if n == 1 else [i / (n - 1) for i in range(n)]

    ports, k = [], 1
    for y in ys(n_left):
        ports.append({"name": f"o{k}", "side": "left", "y": y})
        k += 1
    for y in reversed(ys(n_right)):
        ports.append({"name": f"o{k}", "side": "right", "y": y})
        k += 1
    return ports


class Cell(BaseModel):
    name: str
    description: str = ""
    functionality: list[str] = Field(default_factory=list)
    aka: list[str] = Field(default_factory=list)
    technology: str = ""
    ports: list[PortSpec]
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    # capabilities (max supported values), e.g. modulation_bandwidth_ghz; node params with the same
    # key are treated as minimum requirements during component selection
    properties: dict[str, Scalar] = Field(default_factory=dict)

    @field_validator("ports", mode="before")
    @classmethod
    def _shorthand(cls, v):
        return _expand_ports(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _unique_ports(self) -> "Cell":
        names = [p.name for p in self.ports]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"cell {self.name}: ports must be non-empty and unique")
        return self

    @property
    def signature(self) -> str:
        n_in = sum(p.side in ("left", "any") for p in self.ports)
        n_out = sum(p.side in ("right", "any") for p in self.ports)
        return f"{n_in}x{n_out}"

    def doc_text(self) -> str:
        parts = [self.name, self.name, " ".join(self.aka), " ".join(self.functionality),
                 self.technology, self.description, self.signature]
        return " ".join(parts)


class PDK:
    def __init__(self, cells: list[Cell], name: str = "pdk", retriever: Optional[Retriever] = None):
        if len({c.name for c in cells}) != len(cells):
            raise ValueError("duplicate cell names in PDK")
        self.name = name
        self.cells = {c.name: c for c in cells}
        self.retriever: Retriever = retriever or KeywordRetriever({c.name: c.doc_text() for c in cells})
        self.property_keys = {k for c in cells for k in c.properties}

    @classmethod
    def load(cls, path: str | Path) -> "PDK":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls([Cell.model_validate(c) for c in raw["cells"]], name=raw.get("name", "pdk"))

    def cell(self, name: str) -> Cell:
        try:
            return self.cells[name]
        except KeyError:
            raise KeyError(f"PDK has no cell '{name}'") from None

    def search(self, query: str, k: int = 5, focus: Optional[str] = None) -> list[Hit]:
        return self.retriever.search(query, k=k, focus=focus)
