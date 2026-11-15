"""Layered left-to-right placement and crossing counting. Pure Python, no gdsfactory.

Used twice: by the designer (estimate crossings and pick port order before layout) and by the
layout stage (real component sizes). Deterministic: ties are broken by input order.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

Pt = tuple[float, float]
LONG_WIRE_WEIGHT = 50.0  # stand-in nodes for long wires resist being moved (keeps those wires straight)


def _drop_back_edges(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove edges that close a cycle (iterative DFS) so layering is well defined."""
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    color = {n: 0 for n in nodes}
    back: set[tuple[str, str]] = set()
    for root in nodes:
        if color[root]:
            continue
        color[root] = 1
        stack = [(root, iter(adj[root]))]
        while stack:
            u, it = stack[-1]
            for v in it:
                if color[v] == 1:
                    back.add((u, v))
                elif color[v] == 0:
                    color[v] = 1
                    stack.append((v, iter(adj[v])))
                    break
            else:
                color[u] = 2
                stack.pop()
    return [e for e in edges if e not in back]


def assign_layers(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, int]:
    """Longest-path layering: every edge points to a strictly higher layer."""
    dag = _drop_back_edges(nodes, edges)
    preds: dict[str, list[str]] = defaultdict(list)
    succs: dict[str, list[str]] = defaultdict(list)
    for a, b in dag:
        preds[b].append(a)
        succs[a].append(b)
    indeg = {n: len(preds[n]) for n in nodes}
    layer = {n: 0 for n in nodes}
    ready = [n for n in nodes if indeg[n] == 0]
    while ready:
        u = ready.pop(0)
        for v in succs[u]:
            layer[v] = max(layer[v], layer[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return layer


def order_layers(
    nodes: Sequence[str], edges: Sequence[tuple[str, str]], layer: dict[str, int], sweeps: int = 6
) -> dict[int, list[str]]:
    """Barycenter ordering inside each layer (index 0 = top) to reduce crossings."""
    layers: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        layers[layer[n]].append(n)
    preds: dict[str, list[str]] = defaultdict(list)
    succs: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        if layer[a] < layer[b]:
            preds[b].append(a)
            succs[a].append(b)

    def rank(n: str) -> float:
        lst = layers[layer[n]]
        return (lst.index(n) + 0.5) / len(lst)

    for _ in range(sweeps):
        for forward in (True, False):
            idxs = sorted(layers) if forward else sorted(layers, reverse=True)
            for li in idxs:
                nbr = preds if forward else succs
                keyed = []
                for pos, n in enumerate(layers[li]):
                    ns = [m for m in nbr[n] if m in layer]
                    bary = sum(rank(m) for m in ns) / len(ns) if ns else rank(n)
                    keyed.append((bary, pos, n))
                layers[li] = [n for _, _, n in sorted(keyed)]
    return dict(layers)


def place(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    sizes: dict[str, tuple[float, float]],
    gap_x: float = 60.0,
    gap_y: float = 40.0,
    port_dy: Sequence[tuple[float, float]] | None = None,
) -> dict[str, Pt]:
    """Centre coordinates (um); x grows to the right, y grows upward, first node of a layer on top.

    Edges that skip layers get invisible stand-in nodes in every layer they cross, so long wires keep
    their own track instead of running through components. They are removed from the result.

    port_dy[i] = (source port y offset, target port y offset), relative to each component's centre,
    for edges[i]. When given, components are shifted vertically so connected ports line up (straight
    wires). Order within a layer is kept, so no crossings are added."""
    nodes = list(nodes)
    base_layer = assign_layers(nodes, edges)
    all_nodes = list(nodes)
    all_edges: list[tuple[str, str]] = []
    all_dy: list[tuple[float, float]] = []
    dsizes = dict(sizes)
    for i, (a, b) in enumerate(edges):
        sdy, ddy = port_dy[i] if port_dy is not None else (0.0, 0.0)
        if base_layer[b] - base_layer[a] <= 1:
            all_edges.append((a, b))
            all_dy.append((sdy, ddy))
            continue
        prev, prev_dy = a, sdy
        for l in range(base_layer[a] + 1, base_layer[b]):
            d = f"__via{i}_{l}"
            all_nodes.append(d)
            dsizes[d] = (1.0, 2.0)  # a wire track
            all_edges.append((prev, d))
            all_dy.append((prev_dy, 0.0))
            prev, prev_dy = d, 0.0
        all_edges.append((prev, b))
        all_dy.append((prev_dy, ddy))

    layer = assign_layers(all_nodes, all_edges)
    layers = order_layers(all_nodes, all_edges, layer)
    centers: dict[str, Pt] = {}
    x = 0.0
    for li in sorted(layers):
        members = layers[li]
        width = max(dsizes[n][0] for n in members)
        total = sum(dsizes[n][1] for n in members) + gap_y * (len(members) - 1)
        y = total / 2
        for n in members:
            h = dsizes[n][1]
            centers[n] = (x + width / 2, y - h / 2)
            y -= h + gap_y
        x += width + gap_x
    if port_dy is not None:
        centers = _align_ports(layers, centers, dsizes, all_edges, all_dy, gap_y, layer)
    return {n: centers[n] for n in nodes}


def _separate(desired: list[float], seps: list[float], weights: list[float] | None = None) -> list[float]:
    """Weighted least-squares positions closest to `desired`, top to bottom, with y[i] - y[i+1] >= seps[i].
    Pool-adjacent-violators on u_i = y_i + cumulative separation, which must be non-increasing.
    Heavier items move less when two of them want the same spot."""
    n = len(desired)
    w = weights or [1.0] * n
    cum = [0.0] * n
    for i in range(1, n):
        cum[i] = cum[i - 1] + seps[i - 1]
    blocks: list[list[float]] = []  # [sum of w*u, sum of w, count]
    for i in range(n):
        blocks.append([w[i] * (desired[i] + cum[i]), w[i], 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            a2, b2, c2 = blocks.pop()
            blocks[-1][0] += a2
            blocks[-1][1] += b2
            blocks[-1][2] += c2
    u = [t / ww for t, ww, c in blocks for _ in range(c)]
    return [u[i] - cum[i] for i in range(n)]


def _align_ports(layers, centers, sizes, edges, port_dy, gap_y, layer, rounds: int = 4) -> dict[str, Pt]:
    """Alternate forward (follow predecessors) and backward (follow successors) passes."""
    incoming: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    outgoing: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for (a, b), (sdy, ddy) in zip(edges, port_dy):
        if layer.get(a, 0) < layer.get(b, 0):
            incoming[b].append((a, sdy, ddy))
            outgoing[a].append((b, sdy, ddy))
    y = {n: c[1] for n, c in centers.items()}
    order = sorted(layers)

    def sweep(li: int, forward: bool) -> None:
        members = layers[li]
        desired = []
        for n in members:
            ins = [t for t in incoming.get(n, []) if t[0].startswith("__via")] or incoming.get(n, [])
            outs = [t for t in outgoing.get(n, []) if t[0].startswith("__via")] or outgoing.get(n, [])
            if forward and ins:  # a long wire (stand-in neighbour) wins: it runs straight, the others jog
                desired.append(sum(y[a] + sdy - ddy for a, sdy, ddy in ins) / len(ins))
            elif not forward and outs:
                desired.append(sum(y[b] + ddy - sdy for b, sdy, ddy in outs) / len(outs))
            else:
                desired.append(y[n])
        seps = [sizes[members[i]][1] / 2 + sizes[members[i + 1]][1] / 2 + gap_y for i in range(len(members) - 1)]
        weights = [LONG_WIRE_WEIGHT if n.startswith("__via") else 1.0 for n in members]
        for n, v in zip(members, _separate(desired, seps, weights)):
            y[n] = v

    for _ in range(rounds):
        for li in order[1:]:
            sweep(li, True)
        for li in reversed(order[:-1]):
            sweep(li, False)
    return {n: (centers[n][0], y[n]) for n in centers}


# ----------------------------------------------------------------- crossings
def _orient(a: Pt, b: Pt, c: Pt) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(p1: Pt, p2: Pt, q1: Pt, q2: Pt) -> bool:
    """True only for a proper crossing (touching at an endpoint does not count)."""
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def segments_overlap(p1: Pt, p2: Pt, q1: Pt, q2: Pt) -> bool:
    """True if two segments are collinear and share a stretch of positive length (overlapping wires)."""
    if _orient(p1, p2, q1) != 0 or _orient(p1, p2, q2) != 0:
        return False
    axis = 0 if abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1]) else 1
    a0, a1 = sorted((p1[axis], p2[axis]))
    b0, b1 = sorted((q1[axis], q2[axis]))
    return min(a1, b1) - max(a0, b0) > 0


def count_crossings(segments: Iterable[tuple[Pt, Pt]]) -> int:
    segs = list(segments)
    return sum(
        segments_cross(*segs[i], *segs[j])
        for i in range(len(segs))
        for j in range(i + 1, len(segs))
    )


def segment_hits_box(p: Pt, q: Pt, box: tuple[float, float, float, float]) -> bool:
    """True if the segment's extent enters the interior of the box (left, bottom, right, top).
    Exact for axis-aligned (Manhattan) wires; conservative for diagonals."""
    x0, x1 = sorted((p[0], q[0]))
    y0, y1 = sorted((p[1], q[1]))
    return x0 < box[2] and x1 > box[0] and y0 < box[3] and y1 > box[1]


def boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Boxes as (left, bottom, right, top); touching edges do not overlap."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
