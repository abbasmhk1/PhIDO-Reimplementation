"""gdsfactory backend (imported lazily): build cells, map PDK ports to gdsfactory ports.

Port mapping is geometric, so no hand-written tables: ports facing west (orientation 180) are the
cell's left ports sorted bottom->top; the remaining optical ports are the right ports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ...core.schemas import Scalar
from ...knowledge.pdk import Cell

_ACTIVATED = False


def gf_available() -> bool:
    try:
        import gdsfactory  # noqa: F401

        return True
    except Exception:
        return False


def gf():
    """Import gdsfactory and activate its generic PDK once."""
    global _ACTIVATED
    import gdsfactory as _gf

    if not _ACTIVATED:
        try:
            _gf.gpdk.PDK.activate()
        except AttributeError:  # other gdsfactory versions
            pass
        _ACTIVATED = True
    return _gf


@dataclass
class BuiltCell:
    component: Any
    backend: str  # "gdsfactory" | "blackbox"
    port_map: dict[str, str]  # PDK port name -> gdsfactory port name
    width: float
    height: float
    note: str = ""


def _blackbox(cell: Cell) -> BuiltCell:
    g = gf()
    w, h = cell.footprint
    c = g.Component()
    c.add_polygon([(0, 0), (w, 0), (w, h), (0, h)], layer=(1, 0))
    port_map: dict[str, str] = {}
    for p in cell.ports:
        y = p.y * h
        if p.side == "right":
            x, orient = w, 0
        else:  # left, and "any" (rotated later when used as a source)
            x, orient = 0, 180
        c.add_port(name=p.name, center=(x, y), width=0.5, orientation=orient, layer=(1, 0))
        port_map[p.name] = p.name
    return BuiltCell(c, "blackbox", port_map, w, h, "black box (no gdsfactory equivalent)")


def _map_ports(cell: Cell, comp) -> Optional[dict[str, str]]:
    optical = [p for p in comp.ports if p.port_type == "optical"]
    any_ports = [p for p in cell.ports if p.side == "any"]
    if any_ports:
        return {any_ports[0].name: optical[0].name} if len(optical) == 1 == len(any_ports) else None
    west = sorted((p for p in optical if round(p.orientation) % 360 == 180), key=lambda p: p.dcenter[1])
    east = sorted((p for p in optical if round(p.orientation) % 360 != 180), key=lambda p: p.dcenter[1])
    left = sorted((p for p in cell.ports if p.side == "left"), key=lambda p: p.y)
    right = sorted((p for p in cell.ports if p.side == "right"), key=lambda p: p.y)
    if len(west) != len(left) or len(east) != len(right):
        return None
    return {**{a.name: b.name for a, b in zip(left, west)}, **{a.name: b.name for a, b in zip(right, east)}}


def build_cell(cell: Cell, user_params: dict[str, Scalar]) -> BuiltCell:
    """Build the gdsfactory component for a PDK cell, falling back to a black box on any problem."""
    if cell.layout is None:
        return _blackbox(cell)
    g = gf()
    spec = cell.layout
    kwargs: dict[str, Scalar] = dict(spec.settings)
    for gf_key, pdk_name in spec.params.items():
        if pdk_name in user_params:  # only user-specified values; otherwise gdsfactory defaults
            kwargs[gf_key] = user_params[pdk_name]
    try:
        comp = getattr(g.components, spec.component)(**kwargs)
    except Exception as e:  # unknown component or invalid settings
        bb = _blackbox(cell)
        bb.note = f"{spec.component}({kwargs}) failed: {str(e)[:80]}; used black box"
        return bb
    pm = _map_ports(cell, comp)
    if pm is None:
        bb = _blackbox(cell)
        bb.note = f"{spec.component}: port layout does not match PDK cell; used black box"
        return bb
    return BuiltCell(comp, "gdsfactory", pm, comp.dxsize, comp.dysize)
