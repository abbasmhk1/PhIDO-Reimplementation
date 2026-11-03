"""Metrics from the paper: pass@k, absolute occurrence rate, conditional occurrence rate."""
from __future__ import annotations

from collections import Counter
from math import comb
from typing import Iterable, Mapping, Sequence

STAGE_ORDER = ("EE", "CS", "PC", "SG")  # outcome codes in pipeline order; "S" = success


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): n trials, c successes."""
    if k > n:
        raise ValueError("k cannot exceed the number of trials")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def absolute_rates(outcomes: Mapping[str, Sequence[str]]) -> dict[str, float]:
    counts = Counter(o for v in outcomes.values() for o in v)
    total = sum(counts.values()) or 1
    return {code: counts[code] / total for code in (*STAGE_ORDER, "S") if counts[code]}


def conditional_rates(outcomes: Mapping[str, Sequence[str]], order: Iterable[str] = STAGE_ORDER) -> dict[str, float]:
    """P(fail at stage | trial reached stage): normalises by trials not already stopped upstream."""
    counts = Counter(o for v in outcomes.values() for o in v)
    remaining = sum(counts.values())
    rates = {}
    for code in order:
        rates[code] = counts[code] / remaining if remaining else 0.0
        remaining -= counts[code]
    return rates


def summarize(outcomes: Mapping[str, Sequence[str]], ks: Sequence[int] = (1, 5)) -> dict:
    n_min = min((len(v) for v in outcomes.values()), default=0)
    passk = {}
    for k in ks:
        if 0 < k <= n_min:
            vals = [pass_at_k(len(v), list(v).count("S"), k) for v in outcomes.values()]
            passk[f"pass@{k}"] = sum(vals) / len(vals)
    return {
        "cases": len(outcomes),
        "trials_per_case": n_min,
        **passk,
        "absolute": absolute_rates(outcomes),
        "conditional": conditional_rates(outcomes),
    }
