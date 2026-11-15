"""Compare several LLMs on the same cases (accuracy, latency, tokens), as in the paper's study."""
from __future__ import annotations

from typing import Callable, Optional

from ..stages.interpreter import InterpreterStage
from .runner import EvalReport, InterpreterCase, eval_interpreter


def compare_models(models: list[str], make_stage: Callable[[str], InterpreterStage],
                   cases: list[InterpreterCase], trials: int = 3, ks=(1, 5),
                   on_report: Optional[Callable[[str, EvalReport], None]] = None) -> list[dict]:
    """One row per model. pass@k is reported for every k in `ks` that fits, plus k = trials.
    `on_report(model, report)` is called after each model, e.g. to print failure reasons."""
    rows = []
    for m in models:
        ks_used = tuple(sorted({1, trials} | {k for k in ks if k <= trials}))
        report = eval_interpreter(make_stage(m), cases, trials)
        if on_report:
            on_report(m, report)
        s = report.summary(ks_used)
        row = {"model": m, "cases": s["cases"], "trials": s["trials_per_case"]}
        row.update({k: round(v, 3) for k, v in s.items() if k.startswith("pass@")})
        row["EE_rate"] = round(s["conditional"].get("EE", 0.0), 3)
        row.update(s.get("llm", {}))
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
