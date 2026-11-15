"""Analytic S-parameter models for the demo PDK. Demo constants only - not foundry data.

Every builder takes (wavelengths_um, params) and returns a solver Model. Port names follow the PDK
(2x2: o1 left-bottom, o2 left-top, o3 right-top, o4 right-bottom). Fibre / detector cells get a
virtual external port ("fiber", "pd") so light can enter or leave the circuit through them.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .solver import Model, solve

N_EFF = 2.4  # effective index at 1.55 um
N_G = 4.2  # group index
LAM0 = 1.55
LOSS_DB_PER_CM = 2.0


def neff(wl: np.ndarray) -> np.ndarray:
    return N_EFF + (N_EFF - N_G) / LAM0 * (wl - LAM0)  # gives group index N_G at LAM0


def phase(wl: np.ndarray, length_um: float) -> np.ndarray:
    return 2 * np.pi * neff(wl) * length_um / wl


def amp(length_um: float, loss_db_per_cm: float = LOSS_DB_PER_CM) -> float:
    return 10 ** (-loss_db_per_cm * length_um * 1e-4 / 20)


def _db(x: float) -> float:
    return 10 ** (-x / 20)


def _two_port(t: np.ndarray, ports=("o1", "o2")) -> Model:
    S = np.zeros((len(t), 2, 2), dtype=complex)
    S[:, 1, 0] = S[:, 0, 1] = t
    return Model(list(ports), S)


def _wg(wl, length, extra_phase=0.0, extra_loss_db=0.0) -> Model:
    return _two_port(amp(length) * _db(extra_loss_db) * np.exp(-1j * (phase(wl, length) + extra_phase)))


# ----------------------------------------------------------------- building blocks
def m_straight(wl, p) -> Model:
    return _wg(wl, p.get("length", 10.0))


def m_bend(wl, p) -> Model:
    return _wg(wl, np.pi * p.get("radius", 10.0) / 2)


def m_phase_shifter(wl, p) -> Model:
    return _wg(wl, p.get("length", 100.0), p.get("phase", 0.0), p.get("insertion_loss_db", 0.0))


def m_mmi_1x2(wl, p) -> Model:
    t = _db(p.get("excess_loss_db", 0.0)) / np.sqrt(2) * np.ones(len(wl), dtype=complex)
    S = np.zeros((len(wl), 3, 3), dtype=complex)
    for out in (1, 2):
        S[:, out, 0] = S[:, 0, out] = t
    return Model(["o1", "o2", "o3"], S)


def _coupler(wl, kappa2: float, excess_loss_db: float = 0.0) -> Model:
    k = _db(excess_loss_db)
    t, c = k * np.sqrt(1 - kappa2), 1j * k * np.sqrt(kappa2)
    S = np.zeros((len(wl), 4, 4), dtype=complex)
    # inputs o1 (bottom), o2 (top); outputs o3 (top), o4 (bottom): bar = same height, cross = opposite
    for (i, j), v in {(0, 3): t, (1, 2): t, (0, 2): c, (1, 3): c}.items():
        S[:, j, i] = S[:, i, j] = v
    return Model(["o1", "o2", "o3", "o4"], S)


def m_coupler_2x2(wl, p) -> Model:
    return _coupler(wl, p.get("kappa2", 0.5), p.get("excess_loss_db", 0.0))


def m_directional_coupler(wl, p) -> Model:
    k2 = np.sin(np.pi / 2 * p.get("length", 20.0) / p.get("full_transfer_length_um", 40.0)) ** 2
    return _coupler(wl, float(k2), p.get("excess_loss_db", 0.0))


def _arms(wl, p) -> tuple[Model, Model]:
    """Top arm, bottom arm. Delta length and the (heater/diode) phase sit on the bottom arm."""
    arm = p.get("arm_length", 100.0)
    return _wg(wl, arm), _wg(wl, arm + p.get("delta_length", 0.0), p.get("phase", 0.0))


def m_mzi_2x2(wl, p) -> Model:
    """Two 50/50 couplers joined by two arms. delta_length=0, phase=0 -> cross state (o1 -> o3)."""
    top, bot = _arms(wl, p)
    comps = {"c1": _coupler(wl, 0.5), "c2": _coupler(wl, 0.5), "top": top, "bot": bot}
    conns = [(("c1", "o3"), ("top", "o1")), (("top", "o2"), ("c2", "o2")),
             (("c1", "o4"), ("bot", "o1")), (("bot", "o2"), ("c2", "o1"))]
    refs, S = solve(comps, conns)  # externals in order: c1.o1, c1.o2, c2.o3, c2.o4
    return Model(["o1", "o2", "o3", "o4"], S)


def m_mzi_1x2(wl, p) -> Model:
    """1x2 MMI splitter, two arms, 2x2 coupler combiner; outputs are the combiner's o3 (top), o4 (bottom)."""
    top, bot = _arms(wl, p)
    comps = {"c1": m_mmi_1x2(wl, {}), "c2": _coupler(wl, 0.5), "top": top, "bot": bot}
    conns = [(("c1", "o2"), ("top", "o1")), (("top", "o2"), ("c2", "o2")),
             (("c1", "o3"), ("bot", "o1")), (("bot", "o2"), ("c2", "o1"))]
    refs, S = solve(comps, conns)  # externals in order: c1.o1, c2.o3, c2.o4
    return Model(["o1", "o2", "o3"], S)


def _ring_params(wl, p):
    r_um, gap = p.get("radius", 10.0), p.get("gap", 0.2)
    k2 = float(np.clip(0.1 * np.exp(-(gap - 0.2) / 0.08), 1e-4, 0.9))
    length = 2 * np.pi * r_um
    z = amp(length) * np.exp(-1j * phase(wl, length))
    return np.sqrt(1 - k2), k2, z


def m_ring_single(wl, p) -> Model:
    r, _, z = _ring_params(wl, p)
    return _two_port((r - z) / (1 - r * z))


def m_ring_double(wl, p) -> Model:
    r, k2, z = _ring_params(wl, p)
    through = r * (1 - z) / (1 - r * r * z)
    drop = -k2 * np.sqrt(z) / (1 - r * r * z)
    S = np.zeros((len(wl), 4, 4), dtype=complex)
    for (i, j), v in {(0, 3): through, (1, 2): through, (0, 2): drop, (1, 3): drop}.items():
        S[:, j, i] = S[:, i, j] = v
    return Model(["o1", "o2", "o3", "o4"], S)


def m_fiber_io(wl, p) -> Model:
    return _two_port(_db(p.get("coupling_loss_db", 1.5)) * np.ones(len(wl), dtype=complex), ("o1", "fiber"))


def m_detector(wl, p) -> Model:
    S = np.zeros((len(wl), 2, 2), dtype=complex)
    S[:, 1, 0] = 1.0  # all light entering o1 is absorbed and reported at the electrical port "pd"
    return Model(["o1", "pd"], S)


def build_params(cell, user: dict) -> dict[str, float]:
    """Numeric parameters for a cell model: PDK defaults, overridden by the design's values, plus model_params."""
    p: dict[str, float] = {}
    for name, spec in cell.params.items():
        if isinstance(spec.default, (int, float)) and not isinstance(spec.default, bool):
            p[name] = float(spec.default)
    for k, v in user.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            p[k] = float(v)
    p.update(cell.model_params)
    return p


MODELS: dict[str, Callable[[np.ndarray, dict], Model]] = {
    "straight": m_straight, "bend": m_bend, "phase_shifter": m_phase_shifter,
    "mmi_1x2": m_mmi_1x2, "coupler_2x2": m_coupler_2x2, "directional_coupler": m_directional_coupler,
    "mzi_2x2": m_mzi_2x2, "mzi_1x2": m_mzi_1x2, "ring_single": m_ring_single, "ring_double": m_ring_double,
    "fiber_io": m_fiber_io, "detector": m_detector,
}
