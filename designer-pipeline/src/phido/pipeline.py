"""Wiring helpers: build stages from the data folder and chain them. Stages stay independent."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .core.schemas import DesignerInput, DesignerOutput, InterpreterInput, InterpreterOutput
from .core.store import RunStore
from .knowledge.pdk import PDK
from .knowledge.templates import TemplateLibrary
from .llm.adapter import StructuredLLM
from .paths import data_dir
from .stages.designer import DesignerStage
from .stages.interpreter import InterpreterStage


def build_interpreter(llm: StructuredLLM, data: Optional[Path] = None) -> InterpreterStage:
    d = Path(data) if data else data_dir()
    return InterpreterStage(llm, TemplateLibrary.load(d / "templates" / "templates.yaml"))


def build_designer(llm: Optional[StructuredLLM] = None, data: Optional[Path] = None,
                   use_llm: bool = False) -> DesignerStage:
    d = Path(data) if data else data_dir()
    pdk = PDK.load(d / "pdk" / "generic_pdk.yaml")
    if use_llm and llm is not None:
        return DesignerStage.with_llm(pdk, llm)
    return DesignerStage(pdk)


def run_pipeline(prompt: str, interpreter: InterpreterStage, designer: DesignerStage,
                 store: Optional[RunStore] = None) -> tuple[InterpreterOutput, DesignerOutput]:
    interp = interpreter(InterpreterInput(prompt=prompt), store)
    design = designer(DesignerInput(template=interp.template), store)
    return interp, design
