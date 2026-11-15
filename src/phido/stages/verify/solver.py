"""Frequency-domain circuit solver: component S-matrices + ideal connections -> circuit S-matrix.

With ports split into external (e) and connected (c), and Gamma the c-by-c partner permutation:
    S_ext = S_ee + S_ec Gamma (I - S_cc Gamma)^-1 S_ce
Dependency-free (numpy only). Cross-checked against SAX in the tests when SAX is installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PortRef = tuple[str, str]  # (component, port)


@dataclass
class Model:
    ports: list[str]
    S: np.ndarray  # shape (n_wavelengths, n_ports, n_ports), S[:, out, in]


def solve(
    components: dict[str, Model], connections: list[tuple[PortRef, PortRef]]
) -> tuple[list[PortRef], np.ndarray]:
    """Returns (external port refs, S_ext with shape (n_wavelengths, n_ext, n_ext))."""
    index: dict[PortRef, int] = {}
    for c, m in components.items():
        for p in m.ports:
            index[(c, p)] = len(index)
    n_wl = next(iter(components.values())).S.shape[0]
    S = np.zeros((n_wl, len(index), len(index)), dtype=complex)
    off = 0
    for m in components.values():
        k = len(m.ports)
        S[:, off : off + k, off : off + k] = m.S
        off += k

    partner: dict[PortRef, PortRef] = {}
    for a, b in connections:
        for end in (a, b):
            if end not in index:
                raise ValueError(f"unknown port {end[0]}.{end[1]}")
            if end in partner:
                raise ValueError(f"port {end[0]}.{end[1]} is connected twice")
        partner[a], partner[b] = b, a

    ext_refs = [k for k in index if k not in partner]
    con_refs = [k for k in index if k in partner]
    e = [index[k] for k in ext_refs]
    c = [index[k] for k in con_refs]
    See = S[:, e][:, :, e]
    if not c:
        return ext_refs, See
    pos = {ref: j for j, ref in enumerate(con_refs)}
    G = np.zeros((len(c), len(c)))
    for ref in con_refs:
        G[pos[ref], pos[partner[ref]]] = 1.0
    Sec, Sce, Scc = S[:, e][:, :, c], S[:, c][:, :, e], S[:, c][:, :, c]
    M = np.eye(len(c))[None] - Scc @ G[None]
    X = np.linalg.solve(M, Sce)
    return ext_refs, See + Sec @ G[None] @ X
