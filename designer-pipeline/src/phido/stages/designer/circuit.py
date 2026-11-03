"""CircuitGraph: the designer's working state. Every wiring edit is validated on apply,
so an invalid state can never exist (the weak LLM cannot corrupt it)."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import networkx as nx

from ...core.schemas import Connect, TemplateEdge
from ...knowledge.pdk import Cell, PortSpec


class WiringError(ValueError):
    pass


class CircuitGraph:
    def __init__(self, cells: dict[str, Cell], edges: Sequence[TemplateEdge] = ()):
        self.cells = dict(cells)  # node id -> PDK cell
        self.allowed: Counter = Counter((e.src, e.dst) for e in edges)
        self.realized: Counter = Counter()
        self.connections: list[Connect] = []
        self._used: set[tuple[str, str]] = set()

    @property
    def node_ids(self) -> list[str]:
        return list(self.cells)

    def clone(self) -> "CircuitGraph":
        g = CircuitGraph(self.cells)
        g.allowed, g.realized = Counter(self.allowed), Counter(self.realized)
        g.connections, g._used = list(self.connections), set(self._used)
        return g

    # --- port queries
    def _port(self, node: str, port: str) -> PortSpec:
        if node not in self.cells:
            raise WiringError(f"unknown node '{node}'; nodes: {', '.join(self.cells)}")
        cell = self.cells[node]
        for p in cell.ports:
            if p.name == port:
                return p
        raise WiringError(
            f"node {node} ({cell.name}) has no port '{port}'; ports: {', '.join(p.name for p in cell.ports)}"
        )

    def free_ports(self, node: str, role: str) -> list[PortSpec]:
        """Unused ports usable as 'out' (right/any side) or 'in' (left/any side), top first."""
        sides = ("right", "any") if role == "out" else ("left", "any")
        ports = [p for p in self.cells[node].ports if p.side in sides and (node, p.name) not in self._used]
        return sorted(ports, key=lambda p: -p.y)

    # --- editing
    def check(self, op: Connect) -> None:
        if op.src_node == op.dst_node:
            raise WiringError(f"{op.src_node}: a component cannot connect to itself")
        sp, dp = self._port(op.src_node, op.src_port), self._port(op.dst_node, op.dst_port)
        if sp.side == "left":
            raise WiringError(f"{op.src_node}.{op.src_port} is an input-side (left) port; use a right-side port as the source")
        if dp.side == "right":
            raise WiringError(f"{op.dst_node}.{op.dst_port} is an output-side (right) port; use a left-side port as the target")
        for node, port in ((op.src_node, op.src_port), (op.dst_node, op.dst_port)):
            if (node, port) in self._used:
                raise WiringError(f"port {node}.{port} is already connected")
        key = (op.src_node, op.dst_node)
        if self.allowed[key] - self.realized[key] <= 0:
            raise WiringError(f"no pending template edge {op.src_node} -> {op.dst_node}")

    def apply(self, op: Connect) -> None:
        self.check(op)
        self.connections.append(op)
        self._used.update({(op.src_node, op.src_port), (op.dst_node, op.dst_port)})
        self.realized[(op.src_node, op.dst_node)] += 1

    # --- status
    def remaining(self) -> Counter:
        return self.allowed - self.realized  # Counter subtraction keeps only positive counts

    def pending(self, edges: Sequence[TemplateEdge]) -> list[tuple[int, TemplateEdge]]:
        left = Counter(self.realized)
        out = []
        for i, e in enumerate(edges):
            key = (e.src, e.dst)
            if left[key] > 0:
                left[key] -= 1
            else:
                out.append((i, e))
        return out

    def links(self) -> list[str]:
        return [f"{c.src_node},{c.src_port}:{c.dst_node},{c.dst_port}" for c in self.connections]

    def warnings(self) -> list[str]:
        g = nx.Graph()
        g.add_nodes_from(self.cells)
        g.add_edges_from((c.src_node, c.dst_node) for c in self.connections)
        if not nx.check_planarity(g)[0]:
            return ["connection graph is non-planar: waveguide crossings are unavoidable (router cannot do crossings)"]
        return []
