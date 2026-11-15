"""Port-level crossing estimate and repair for the designer.

Placement here uses the PDK footprints (approximate). It answers one question: given which port
each wire uses, would wires cross? If so, swap ports of the same side on one component.
"""
from __future__ import annotations

from ...geometry.placement import Pt, count_crossings, place
from .circuit import CircuitGraph


def estimate_positions(graph: CircuitGraph) -> tuple[dict[str, Pt], dict[str, tuple[float, float]]]:
    sizes = {n: tuple(c.footprint) for n, c in graph.cells.items()}
    edges = [e for e, k in graph.allowed.items() for _ in range(k)]
    return place(graph.node_ids, edges, sizes), sizes


def _port_point(graph, centers, sizes, node: str, port: str, as_source: bool) -> Pt:
    cx, cy = centers[node]
    w, h = sizes[node]
    spec = next(p for p in graph.cells[node].ports if p.name == port)
    x = cx + w / 2 if (spec.side == "right" or (spec.side == "any" and as_source)) else cx - w / 2
    return x, cy + (spec.y - 0.5) * h


def _segments(graph, centers, sizes):
    return [
        (
            _port_point(graph, centers, sizes, c.src_node, c.src_port, True),
            _port_point(graph, centers, sizes, c.dst_node, c.dst_port, False),
        )
        for c in graph.connections
    ]


def estimated_crossings(graph: CircuitGraph) -> int:
    centers, sizes = estimate_positions(graph)
    return count_crossings(_segments(graph, centers, sizes))


def repair_crossings(graph: CircuitGraph, max_passes: int = 5) -> int:
    """Greedy pairwise port swaps that reduce estimated crossings. Returns crossings left."""
    centers, sizes = estimate_positions(graph)
    best = count_crossings(_segments(graph, centers, sizes))
    for _ in range(max_passes):
        if best == 0:
            break
        improved = False
        for node in graph.node_ids:
            uses = [(i, "src") for i, c in enumerate(graph.connections) if c.src_node == node]
            uses += [(i, "dst") for i, c in enumerate(graph.connections) if c.dst_node == node]
            for role in ("src", "dst"):
                group = [u for u in uses if u[1] == role]
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        ca, cb = graph.connections[group[a][0]], graph.connections[group[b][0]]
                        fa, fb = f"{role}_port", f"{role}_port"
                        pa, pb = getattr(ca, fa), getattr(cb, fb)
                        setattr(ca, fa, pb)
                        setattr(cb, fb, pa)
                        n = count_crossings(_segments(graph, centers, sizes))
                        if n < best:
                            best, improved = n, True
                        else:
                            setattr(ca, fa, pa)
                            setattr(cb, fb, pb)
        if not improved:
            break
    return best
