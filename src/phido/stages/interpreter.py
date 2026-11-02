"""Interpreter stage: natural language -> PICTemplate.

Design (weak-LLM friendly): the LLM only *extracts entities* (blocks, links, or a motif).
Code then builds the template graph (expanding motifs) and a retriever finds similar
library templates. Nothing else is left to the model.
"""
from __future__ import annotations

import json
from typing import Optional

from ..core.errors import StageError
from ..core.schemas import (
    EntityExtraction,
    InterpreterInput,
    InterpreterOutput,
    PICTemplate,
    TemplateEdge,
    TemplateMatch,
    TemplateNode,
)
from ..core.stage import Stage
from ..knowledge.motifs import MotifError, describe_motifs, expand_motif, known_motifs
from ..knowledge.templates import TemplateLibrary
from ..llm.adapter import LLMFormatError, StructuredLLM

# --- system prompt, in the paper's five parts: role, context, rules, examples, output format
ROLE = (
    "You are an assistant to a photonic integrated circuit (PIC) engineer. "
    "You read a circuit request written in plain language and extract its structure."
)

RULES = """Rules:
- One block per distinct physical component instance. Give each block a short unique name (letters, digits, underscore; start with a letter), e.g. mzi1.
- kind: a short lowercase component type such as mzi, mmi, ring, heater, edge_coupler, grating_coupler, photodetector, directional_coupler, y_branch, waveguide, bend.
- ports: "<inputs>x<outputs>" (for example "2x2") only if the request states or clearly implies it, otherwise null.
- params: only values the request states explicitly. Use snake_case keys with the unit as suffix, e.g. modulation_bandwidth_ghz, delta_length_um, length_um, radius_um.
- description: the user's own words about that block's details (e.g. "thermo-optic phase shifter in each arm"), or "".
- Parts inside a component (phase shifters, heaters, electrodes, arms, couplers inside an MZI) are NOT separate blocks: put them in that block's description. Make separate blocks only for components the request places in the circuit as separate stages.
- links go from the block that feeds light to the block that receives it. Never invent blocks or links the request does not imply.
- If the request is a repeating pattern of identical units (a tree, a chain of N), use a motif and leave blocks and links empty."""

EXAMPLES: list[tuple[str, dict]] = [
    (
        "Two MZIs in series, each modulating up to 10 GHz, with 2 inputs and 2 outputs.",
        {
            "blocks": [
                {"name": "mzi1", "kind": "mzi", "ports": "2x2", "params": {"modulation_bandwidth_ghz": 10}},
                {"name": "mzi2", "kind": "mzi", "ports": "2x2", "params": {"modulation_bandwidth_ghz": 10}},
            ],
            "links": [{"src": "mzi1", "dst": "mzi2"}],
        },
    ),
    (
        "A 1x8 splitter tree built from 1x2 MMIs, with an edge coupler at the input and at every output.",
        {
            "motif": {
                "name": "tree",
                "args": {"unit": "mmi", "stages": 3, "fanout": 2, "ports": "1x2", "io_kind": "edge_coupler"},
            }
        },
    ),
    (
        "A 1x2 MZI switch with heaters on both arms.",
        {
            "blocks": [
                {"name": "mzi1", "kind": "mzi", "ports": "1x2", "description": "heaters on both arms"},
            ],
        },
    ),
    (
        "A 1x2 MMI that feeds a ring resonator with 12 um radius.",
        {
            "blocks": [
                {"name": "mmi1", "kind": "mmi", "ports": "1x2"},
                {"name": "ring1", "kind": "ring", "params": {"radius_um": 12}},
            ],
            "links": [{"src": "mmi1", "dst": "ring1"}],
        },
    ),
]


def build_system_prompt() -> str:
    examples = "\n\n".join(
        f"Request: {req}\nAnswer: {json.dumps(ans)}" for req, ans in EXAMPLES
    )
    return (
        f"{ROLE}\n\nKnown repeating motifs:\n{describe_motifs()}\n\n{RULES}\n\nExamples:\n\n{examples}"
    )


class InterpreterStage(Stage[InterpreterInput, InterpreterOutput]):
    name = "interpreter"

    def __init__(
        self,
        llm: StructuredLLM,
        library: Optional[TemplateLibrary] = None,
        top_k: int = 3,
    ):
        self.llm = llm
        self.library = library
        self.top_k = top_k
        self._system = build_system_prompt()

    # --- step 1: the only LLM call
    def extract(self, prompt: str) -> EntityExtraction:
        def check(ex: EntityExtraction) -> None:  # semantic validation -> triggers LLM retry
            if ex.motif is not None:
                expand_motif(ex.motif)  # raises MotifError (a ValueError) with a helpful message

        try:
            return self.llm.generate(self._system, f"Request: {prompt}\nAnswer:", EntityExtraction, check)
        except LLMFormatError as e:
            raise StageError("EE", str(e)) from e

    # --- step 2: deterministic graph construction
    def build_template(self, ex: EntityExtraction, prompt: str) -> tuple[PICTemplate, list[str]]:
        warnings: list[str] = []
        if ex.motif is not None:
            if ex.blocks or ex.links:
                warnings.append("motif and blocks both given; blocks/links were ignored")
            try:
                return expand_motif(ex.motif), warnings
            except MotifError as e:
                raise StageError("EE", str(e)) from e
        nodes = [
            TemplateNode(id=b.name, kind=b.kind, ports=b.ports, description=b.description, params=b.params)
            for b in ex.blocks
        ]
        edges = [TemplateEdge(src=l.src, dst=l.dst) for l in ex.links]
        title = prompt.strip().splitlines()[0][:60]
        return PICTemplate(title=title, nodes=nodes, edges=edges, source="blocks"), warnings

    # --- step 3: retrieval over prior designs
    def match(self, template: PICTemplate, prompt: str) -> list[TemplateMatch]:
        if self.library is None:
            return []
        kinds = " ".join(n.kind for n in template.nodes[:50])
        hits = self.library.search(f"{prompt} {kinds}", k=self.top_k)
        return [TemplateMatch(name=h.id, score=round(h.score, 3)) for h in hits]

    def run(self, inp: InterpreterInput) -> InterpreterOutput:
        ex = self.extract(inp.prompt)
        template, warnings = self.build_template(ex, inp.prompt)
        return InterpreterOutput(
            prompt=inp.prompt,
            extraction=ex,
            template=template,
            matches=self.match(template, inp.prompt),
            warnings=warnings,
        )


__all__ = ["InterpreterStage", "build_system_prompt", "known_motifs"]
