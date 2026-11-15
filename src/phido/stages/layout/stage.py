"""Layout stage: PIC DSL -> placement, routing, structural checks, GDSII.

Algorithmic (no LLM). Placement is the shared layered placer; routing is gdsfactory's
route_single. Checks are structural (instance overlap, routing failure, route crossings).
They are NOT a foundry DRC: there is no rule deck in this project.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ...core.schemas import Issue, LayoutInput, LayoutOutput, Placement, RouteResult
from ...core.stage import Stage
from ...geometry.placement import boxes_overlap, place, segment_hits_box, segments_cross, segments_overlap
from ...knowledge.pdk import PDK
from .gds import BuiltCell, build_cell, gf, gf_available


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:40] or "design"


class LayoutStage(Stage[LayoutInput, LayoutOutput]):
    name = "layout"

    def __init__(self, pdk: PDK, out_dir: Optional[str | Path] = None, gap_x: float = 60.0, gap_y: float = 60.0):
        # gap_y: the Manhattan router needs a vertical jog of 0 or >= ~2 bend radii (~20 um) between connected ports;
        # 60 um keeps fan-out children >= ~30 um from the parent's ports. Violations still show up as L issues.
        self.pdk, self.gap_x, self.gap_y = pdk, gap_x, gap_y
        self.out_dir = Path(out_dir) if out_dir else None

    GAP_LADDER = (1.0, 1.7, 2.7)  # multiples of gap_y tried in turn

    def run(self, inp: LayoutInput) -> LayoutOutput:
        """Layout with increasing vertical spacing until no structural issue remains (best attempt kept)."""
        best_gap, best = self.gap_y, None
        for factor in self.GAP_LADDER:
            gap = self.gap_y * factor
            out = self._attempt(inp, gap, write=False)
            if best is None or len(out.issues) < len(best.issues):
                best_gap, best = gap, out
            if not out.issues or out.gds_path is None and not out.routes:  # clean, or no gdsfactory (nothing to retry)
                break
        out = self._attempt(inp, best_gap, write=True)
        if best_gap != self.gap_y:
            out.warnings.append(f"layout needed vertical gap {best_gap:.0f} um (default {self.gap_y:.0f}) "
                                f"to {'avoid structural issues' if not out.issues else 'reduce structural issues'}")
        return out

    STUB_STEP = 6.0  # um of extra straight stub between neighbouring wires that bend the same way

    def _stagger(self, links, refs, built) -> dict[str, tuple[float, float]]:
        """Neighbouring wires leaving (or entering) one component in the same vertical direction would put
        their jogs on the same track. The outermost wire gets the longest stub so it turns last and
        cannot cut across its neighbour."""
        def py(node: str, port: str) -> float:
            return refs[node].ports[built[node].port_map[port]].dcenter[1]

        start: dict[str, float] = {}
        end: dict[str, float] = {}
        groups_s: dict[tuple, list] = {}
        groups_e: dict[tuple, list] = {}
        for eid, (n1, p1, n2, p2) in links.items():
            sy, ty = py(n1, p1), py(n2, p2)
            if abs(ty - sy) < 0.5:
                start[eid] = end[eid] = 0.0
                continue
            down = ty < sy
            groups_s.setdefault((n1, down), []).append((sy, eid))  # leaving n1 going down / up
            groups_e.setdefault((n2, down), []).append((ty, eid))  # arriving at n2 coming from above / below
        for (node, down), items in groups_s.items():
            items.sort(reverse=not down)  # down: bottom-most first; up: top-most first
            for rank, (_, eid) in enumerate(items):
                start[eid] = rank * self.STUB_STEP
        for (node, from_above), items in groups_e.items():
            items.sort(reverse=from_above)  # from above: top-most arriving port first; from below: bottom-most
            for rank, (_, eid) in enumerate(items):
                end[eid] = rank * self.STUB_STEP
        return {eid: (start.get(eid, 0.0), end.get(eid, 0.0)) for eid in links}

    def _attempt(self, inp: LayoutInput, gap_y: float, write: bool = True) -> LayoutOutput:
        d = inp.design
        links = {eid: e.parse() for eid, e in d.edges.items()}
        pairs = [(a, c) for a, _, c, _ in links.values()]
        nodes = list(d.nodes)
        warnings: list[str] = []
        issues: list[Issue] = []
        cells = {n: self.pdk.cell(v.component) for n, v in d.nodes.items()}

        if not gf_available():
            sizes = {n: tuple(c.footprint) for n, c in cells.items()}
            centers = place(nodes, pairs, sizes, self.gap_x, gap_y)
            warnings.append("gdsfactory is not installed: estimated placement only, no GDS (pip install 'phido[layout]')")
            return LayoutOutput(
                placements=[Placement(node=n, component=cells[n].name, x=centers[n][0], y=centers[n][1],
                                      width=sizes[n][0], height=sizes[n][1]) for n in nodes],
                warnings=warnings,
            )

        built: dict[str, BuiltCell] = {}
        cache: dict[tuple, BuiltCell] = {}
        for n in nodes:
            key = (d.nodes[n].component, tuple(sorted(d.nodes[n].params.items())))
            if key not in cache:
                cache[key] = build_cell(cells[n], d.nodes[n].params)
                if cache[key].note:
                    warnings.append(f"{cells[n].name}: {cache[key].note}")
            built[n] = cache[key]

        # rotate chip-I/O style cells (single "any" port) so the waveguide port faces the right way
        rotate = {n: 180 if any(a == n for a, _ in pairs) and any(p.side == "any" for p in cells[n].ports) else 0
                  for n in nodes}
        sizes = {n: (built[n].width, built[n].height) for n in nodes}

        g = gf()
        top = g.Component()
        refs = {}
        for n in nodes:
            ref = top.add_ref(built[n].component)
            if rotate[n]:
                ref.drotate(rotate[n])
            ref.dcenter = (0.0, 0.0)  # port offsets below are relative to the component centre
            refs[n] = ref

        def port_dy(node: str, port: str) -> float:
            return refs[node].ports[built[node].port_map[port]].dcenter[1] - refs[node].dcenter[1]

        offsets = [(port_dy(n1, p1), port_dy(n2, p2)) for n1, p1, n2, p2 in links.values()]
        centers = place(nodes, pairs, sizes, self.gap_x, gap_y, port_dy=offsets)
        for n in nodes:
            refs[n].dcenter = centers[n]

        boxes = {n: (r.dbbox().left, r.dbbox().bottom, r.dbbox().right, r.dbbox().top) for n, r in refs.items()}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                if boxes_overlap(boxes[a], boxes[b]):
                    issues.append(Issue(code="L", node=a, message=f"instances {a} and {b} overlap"))

        stubs = self._stagger(links, refs, built)
        routes: list[RouteResult] = []
        backbones: dict[str, list[tuple[float, float]]] = {}
        for eid, (n1, p1, n2, p2) in links.items():
            link = d.edges[eid].link
            try:
                r = g.routing.route_single(
                    top, refs[n1].ports[built[n1].port_map[p1]], refs[n2].ports[built[n2].port_map[p2]],
                    cross_section="strip", start_straight_length=stubs[eid][0], end_straight_length=stubs[eid][1],
                )
                routes.append(RouteResult(edge=eid, link=link, ok=True, length_um=round(r.length / 1000, 3)))
                backbones[eid] = [(pt.x, pt.y) for pt in r.backbone]
            except Exception as e:
                msg = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
                routes.append(RouteResult(edge=eid, link=link, ok=False, message=msg))
                issues.append(Issue(code="L", message=f"routing {link} failed: {msg}"))

        for eid, pts in backbones.items():  # a wire must not run through a component it does not connect
            n1, _, n2, _ = links[eid]
            um = [(x / 1000.0, y / 1000.0) for x, y in pts]
            for other in nodes:
                if other in (n1, n2):
                    continue
                if any(segment_hits_box(um[k], um[k + 1], boxes[other]) for k in range(len(um) - 1)):
                    issues.append(Issue(code="L", node=other, message=f"route {eid} passes through component {other}"))

        ids = list(backbones)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sa, sb = backbones[a], backbones[b]
                pairs_ab = [(sa[x], sa[x + 1], sb[y], sb[y + 1])
                            for x in range(len(sa) - 1) for y in range(len(sb) - 1)]
                if any(segments_cross(*s) for s in pairs_ab):
                    issues.append(Issue(code="L", message=f"routes {a} and {b} cross (no crossing component)"))
                elif any(segments_overlap(*s) for s in pairs_ab):
                    issues.append(Issue(code="L", message=f"routes {a} and {b} overlap on the same track (waveguides merge)"))

        gds_path = None
        if self.out_dir is not None and write:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            gds_path = str(self.out_dir / f"{_slug(d.title)}.gds")
            top.write_gds(gds_path)

        return LayoutOutput(
            placements=[Placement(node=n, component=cells[n].name, x=round(refs[n].dcenter[0], 3),
                                  y=round(refs[n].dcenter[1], 3), width=round(sizes[n][0], 3),
                                  height=round(sizes[n][1], 3), rotation=rotate[n], backend=built[n].backend)
                        for n in nodes],
            routes=routes, gds_path=gds_path, issues=issues, warnings=warnings,
        )
