"""Library of prior-design templates the interpreter can match a request against."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from ..core.schemas import PICTemplate, TemplateEdge, TemplateNode
from .retriever import Hit, KeywordRetriever, Retriever
from .text import canonical_kind


class LibraryEntry(BaseModel):
    name: str
    description: str = ""
    nodes: list[TemplateNode]
    edges: list[TemplateEdge] = Field(default_factory=list)

    def to_template(self) -> PICTemplate:
        return PICTemplate(title=self.name, nodes=self.nodes, edges=self.edges, source="library")

    def doc_text(self) -> str:
        kinds = " ".join(canonical_kind(n.kind) for n in self.nodes)
        ports = " ".join(n.ports for n in self.nodes if n.ports)
        return f"{self.name} {self.name} {self.description} {kinds} {ports}"


class TemplateLibrary:
    def __init__(self, entries: list[LibraryEntry], retriever: Optional[Retriever] = None):
        self.entries = {e.name: e for e in entries}
        self.retriever: Retriever = retriever or KeywordRetriever(
            {e.name: e.doc_text() for e in entries}
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemplateLibrary":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls([LibraryEntry.model_validate(e) for e in raw["templates"]])

    def get(self, name: str) -> PICTemplate:
        return self.entries[name].to_template()

    def search(self, query: str, k: int = 3) -> list[Hit]:
        return self.retriever.search(query, k=k)
