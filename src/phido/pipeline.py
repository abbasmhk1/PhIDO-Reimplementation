"""Wiring helpers: build stages from the data folder and chain them. Stages stay independent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .core.schemas import (
    DesignerInput, DesignerOutput, InterpreterInput, InterpreterOutput,
    LayoutInput, LayoutOutput, VerifyInput, VerifyOutput,
)
from .core.store import RunStore
from .knowledge.pdk import PDK
from .knowledge.templates import TemplateLibrary
from .llm.adapter import StructuredLLM
from .paths import data_dir
from .stages.designer import DesignerStage
from .stages.interpreter import InterpreterStage
from .stages.layout.stage import LayoutStage
from .stages.verify.stage import VerifyStage


def _data(data: Optional[Path]) -> Path:
    return Path(data) if data else data_dir()


def load_pdk(data: Optional[Path] = None) -> PDK:
    return PDK.load(_data(data) / "pdk" / "generic_pdk.yaml")


def build_interpreter(llm: StructuredLLM, data: Optional[Path] = None) -> InterpreterStage:
    return InterpreterStage(llm, TemplateLibrary.load(_data(data) / "templates" / "templates.yaml"))


def build_library(data: Optional[Path] = None) -> TemplateLibrary:
    return TemplateLibrary.load(_data(data) / "templates" / "templates.yaml")


def build_designer(llm: Optional[StructuredLLM] = None, data: Optional[Path] = None,
                   use_llm: bool = False) -> DesignerStage:
    pdk = load_pdk(data)
    return DesignerStage.with_llm(pdk, llm) if use_llm and llm is not None else DesignerStage(pdk)


def build_layout(data: Optional[Path] = None, out_dir: Optional[Path | str] = None) -> LayoutStage:
    return LayoutStage(load_pdk(data), out_dir=out_dir)


def build_verifier(data: Optional[Path] = None) -> VerifyStage:
    return VerifyStage(load_pdk(data))


def run_pipeline(prompt: str, interpreter: InterpreterStage, designer: DesignerStage,
                 store: Optional[RunStore] = None) -> tuple[InterpreterOutput, DesignerOutput]:
    interp = interpreter(InterpreterInput(prompt=prompt), store)
    design = designer(DesignerInput(template=interp.template), store)
    return interp, design


@dataclass
class PipelineResult:
    interpreter: InterpreterOutput
    designer: DesignerOutput
    layout: Optional[LayoutOutput] = None
    verify: Optional[VerifyOutput] = None


def run_full(interp_out: InterpreterOutput, designer: DesignerStage, layout: Optional[LayoutStage] = None,
             verifier: Optional[VerifyStage] = None, store: Optional[RunStore] = None) -> PipelineResult:
    """Designer -> (layout, verify). Layout and verify both consume the designer's DSL, independently."""
    design = designer(DesignerInput(template=interp_out.template), store)
    lay = layout(LayoutInput(design=design), store) if layout else None
    ver = verifier(VerifyInput(design=design), store) if verifier else None
    return PipelineResult(interp_out, design, lay, ver)
