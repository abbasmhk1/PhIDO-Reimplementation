"""Where does light go inside a cell at its default setting? Uses the verification models.

Used by the designer to (a) prefer cells that can pass light cleanly along a chain and (b) wire
chain elements through their strongest path, so a 2x2 switch in its cross state is entered and
left through the ports that actually connect.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...core.schemas import Scalar
from ...knowledge.pdk import Cell
from ..verify.models import MODELS, build_params

DECISIVE = 0.9  # a path is "the" through path if it carries at least this share of the light entering its input


def _matrix(cell: Cell, params: dict[str, Scalar]) -> Optional[tuple[list[str], np.ndarray]]:
    builder = MODELS.get(cell.model or "")
    if builder is None:
        return None
    m = builder(np.array([1.55]), build_params(cell, params))
    return m.ports, np.abs(m.S[0]) ** 2  # power[out, in]


def _shares(cell: Cell, params: dict[str, Scalar], in_ports: list[str], out_ports: list[str]):
    """Yield (in, out, share): fraction of the light entering `in` (among the listed outputs) that leaves at `out`."""
    mat = _matrix(cell, params)
    if mat is None:
        return
    ports, power = mat
    if not set(in_ports + out_ports) <= set(ports):
        return
    for i in in_ports:
        total = sum(power[ports.index(o), ports.index(i)] for o in out_ports)
        if total < 1e-9:
            continue
        for o in out_ports:
            yield i, o, power[ports.index(o), ports.index(i)] / total


def _sides(cell: Cell) -> tuple[list[str], list[str]]:
    return ([p.name for p in cell.ports if p.side in ("left", "any")],
            [p.name for p in cell.ports if p.side in ("right", "any")])


def directivity(cell: Cell, params: dict[str, Scalar]) -> float:
    """Best share of light that one input sends to one output: ~1 for a switch in a pure state, 0.5 for a 50/50 splitter."""
    ins, outs = _sides(cell)
    return max((s for _, _, s in _shares(cell, params, ins, outs)), default=0.0)


def decisive_pair(cell: Cell, params: dict[str, Scalar], free_in: list[str], free_out: list[str]) -> Optional[tuple[str, str]]:
    """(input port, output port) of the through path among the free ports, if one path clearly dominates."""
    best = max(_shares(cell, params, free_in, free_out), key=lambda t: t[2], default=None)
    return (best[0], best[1]) if best and best[2] >= DECISIVE else None
