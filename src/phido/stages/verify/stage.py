"""Verification stage: PIC DSL -> circuit transfer functions (schematic-level, no layout needed).

Builds one S-matrix model per node from the PDK `model` name, connects ports as the DSL links say,
solves the circuit over a wavelength sweep, and reports power transfer between external ports.
Also checks passivity (no gain) and flags outputs that receive no light.
"""
from __future__ import annotations

import numpy as np

from ...core.schemas import Issue, PairSummary, VerifyInput, VerifyOutput
from ...core.stage import Stage
from ...knowledge.pdk import PDK, Cell
from .models import MODELS, build_params
from .solver import Model, solve

FLOOR_DB = -60.0  # pairs below this peak are not reported


def _db(p: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(p, 1e-12))


class VerifyStage(Stage[VerifyInput, VerifyOutput]):
    name = "verify"

    def __init__(self, pdk: PDK):
        self.pdk = pdk

    def _params(self, cell: Cell, user: dict) -> dict[str, float]:
        return build_params(cell, user)

    def run(self, inp: VerifyInput) -> VerifyOutput:
        d = inp.design
        wl = np.linspace(inp.wl_start_um, inp.wl_stop_um, inp.points)
        issues: list[Issue] = []
        warnings: list[str] = []

        comps: dict[str, Model] = {}
        for nid, node in d.nodes.items():
            cell = self.pdk.cell(node.component)
            builder = MODELS.get(cell.model or "")
            if builder is None:
                issues.append(Issue(code="V", node=nid, message=f"{cell.name} has no simulation model"))
                continue
            comps[nid] = builder(wl, self._params(cell, node.params))
        if issues:
            return VerifyOutput(inputs=[], outputs=[], wavelengths_um=wl.tolist(), transfer_db={},
                                summary=[], issues=issues)

        links = [e.parse() for e in d.edges.values()]
        try:
            refs, S = solve(comps, [((a, p), (c, q)) for a, p, c, q in links])
        except ValueError as e:
            issues.append(Issue(code="V", message=f"circuit could not be assembled: {e}"))
            return VerifyOutput(inputs=[], outputs=[], wavelengths_um=wl.tolist(), transfer_db={},
                                summary=[], issues=issues)

        sources = {a for a, _, _, _ in links}
        is_input = []
        for node, port in refs:
            cell = self.pdk.cell(d.nodes[node].component)
            if port == "fiber":
                is_input.append(node in sources)  # chip I/O that feeds the circuit
            elif port == "pd":
                is_input.append(False)
            else:
                side = next(p.side for p in cell.ports if p.name == port)
                is_input.append(side == "left")
        names = [f"{n},{p}" for n, p in refs]
        io_names = {f"{n},{p}" for n, p in refs if p in ("fiber", "pd")}  # intended chip I/O
        inputs_all = [n for n, i in zip(names, is_input) if i]
        # driven inputs: chip fibre inputs when the design has any (dangling left ports are not driven)
        fibre_in = [n for n in inputs_all if n in io_names]
        inputs = fibre_in or inputs_all
        outputs = [n for n, i in zip(names, is_input) if not i]
        dangling = [n for n in names if n not in io_names]

        power = np.abs(S) ** 2  # (wl, out, in)
        worst = power.sum(axis=1).max()  # total power out for the worst excitation
        if worst > 1 + 1e-6:
            issues.append(Issue(code="V", message=f"circuit is not passive: {worst:.4f} x input power appears at the outputs"))

        centre = int(np.argmin(np.abs(wl - (inp.wl_start_um + inp.wl_stop_um) / 2)))
        transfer: dict[str, list[float]] = {}
        summary: list[PairSummary] = []
        reached = set()
        for i_name in inputs:
            for o_name in outputs:
                db = _db(power[:, names.index(o_name), names.index(i_name)])
                if db.max() < FLOOR_DB:
                    continue
                reached.add(o_name)
                transfer[f"{i_name}->{o_name}"] = [round(float(x), 4) for x in db]
                k = int(np.argmax(db))
                summary.append(PairSummary(src=i_name, dst=o_name, db_center=round(float(db[centre]), 3),
                                           peak_db=round(float(db[k]), 3), peak_wavelength_um=round(float(wl[k]), 5),
                                           min_db=round(float(db.min()), 3)))
        for o_name in outputs:
            if o_name in io_names and o_name not in reached:
                warnings.append(f"output {o_name} receives no light from the driven inputs (below {FLOOR_DB:.0f} dB)")
            elif o_name not in io_names:
                worst_leak = max((s.peak_db for s in summary if s.dst == o_name), default=None)
                if worst_leak is not None and worst_leak > -30.0:
                    src = next(s.src for s in summary if s.dst == o_name and s.peak_db == worst_leak)
                    warnings.append(f"light from {src} leaves through unconnected port {o_name} ({worst_leak:.1f} dB at peak)")

        if not inputs:
            warnings.append("no input ports found")
        return VerifyOutput(inputs=inputs, outputs=outputs, wavelengths_um=[round(float(x), 6) for x in wl],
                            transfer_db=transfer, summary=summary, unconnected=dangling, issues=issues, warnings=warnings)
