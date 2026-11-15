from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

from .core.errors import StageError
from .core.schemas import DesignerOutput, InterpreterInput, LayoutInput, PICTemplate, DesignerInput, VerifyInput
from .core.store import RunStore
from .eval.compare import compare_models, format_table
from .eval.runner import (
    eval_designer, eval_downstream, eval_interpreter, load_designer_cases, load_interpreter_cases, load_suite,
)
from .llm.adapter import StructuredLLM
from .llm.openai_compat import OpenAICompatClient
from .paths import data_dir
from .pipeline import (
    build_designer, build_interpreter, build_layout, build_library, build_verifier, load_pdk, run_full,
)


def _llm(model: str | None = None) -> StructuredLLM:
    client = OpenAICompatClient.from_env(model)
    return StructuredLLM(client, temperature=float(os.environ.get("PHIDO_TEMPERATURE", "0")))


def _dump(obj) -> None:
    print(yaml.safe_dump(obj.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


def _store(args) -> RunStore | None:
    return None if args.no_store else RunStore(args.runs_dir)


def _report_design(design: DesignerOutput) -> None:
    for i in design.issues:
        print(f"ISSUE [{i.code}] {i.node or ''} {i.message}", file=sys.stderr)
    for w in design.warnings:
        print(f"warning: {w}", file=sys.stderr)


def _print_layout(out) -> None:
    ok = sum(r.ok for r in out.routes)
    print(f"layout: {len(out.placements)} components, {ok}/{len(out.routes)} routes ok, GDS: {out.gds_path or 'not written'}")
    for i in out.issues:
        print(f"ISSUE [{i.code}] {i.message}", file=sys.stderr)
    for w in out.warnings:
        print(f"warning: {w}", file=sys.stderr)


def _print_verify(out) -> None:
    lit = set(out.lit_outputs)
    leaking = [p for p in out.unconnected if p in lit]
    print(f"verify: inputs={out.inputs} lit outputs={len(lit) - len(leaking)} unconnected ports={len(out.unconnected)}"
          f" ({len(leaking)} lit)")
    for s in out.summary:
        print(f"  {s.src} -> {s.dst}: {s.db_center:8.2f} dB at centre, peak {s.peak_db:.2f} dB @ {s.peak_wavelength_um} um, min {s.min_db:.2f} dB")
    for i in out.issues:
        print(f"ISSUE [{i.code}] {i.message}", file=sys.stderr)
    for w in out.warnings:
        print(f"warning: {w}", file=sys.stderr)


def _load_design(path: str) -> DesignerOutput:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DesignerOutput.model_validate(raw.get("design", raw))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="phido", description="PhIDO rebuild: interpreter, designer, layout, verify")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--no-store", action="store_true", help="do not write run artifacts")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("interpret", help="prompt -> PIC template (needs an LLM)")
    a.add_argument("prompt")

    sub.add_parser("templates", help="list library templates")

    b = sub.add_parser("design", help="PIC template YAML -> PIC DSL")
    b.add_argument("template", help="PICTemplate YAML (or an interpreter output.yaml)")
    b.add_argument("--llm", action="store_true", help="let an LLM resolve ambiguity and propose wiring")

    lay = sub.add_parser("layout", help="PIC DSL YAML (designer output) -> placement, routing, GDS")
    lay.add_argument("design")
    lay.add_argument("--out-dir", default="layouts")

    ver = sub.add_parser("verify", help="PIC DSL YAML (designer output) -> transfer functions")
    ver.add_argument("design")
    ver.add_argument("--csv", help="write the full spectra to this CSV")
    ver.add_argument("--wl", nargs=2, type=float, metavar=("START_UM", "STOP_UM"), default=(1.50, 1.60))
    ver.add_argument("--points", type=int, default=201)

    c = sub.add_parser("pipeline", help="prompt (or --adopt) -> template -> DSL [-> layout] [-> verify]")
    c.add_argument("prompt", nargs="?")
    c.add_argument("--adopt", metavar="NAME", help="use a library template instead of the LLM")
    c.add_argument("--llm-design", action="store_true")
    c.add_argument("--layout", action="store_true")
    c.add_argument("--verify", action="store_true")

    e = sub.add_parser("eval", help="evaluate one stage in isolation")
    e.add_argument("target", choices=["interpreter", "designer", "layout", "verify"])
    e.add_argument("--cases", help="path to a cases YAML (overrides --suite)")
    e.add_argument("--suite", choices=["basic", "hard", "holdout", "all"], default="basic", help="interpreter benchmark to use")
    e.add_argument("--trials", type=int, default=1)
    e.add_argument("--llm-design", action="store_true")

    m = sub.add_parser("compare", help="compare several models on the interpreter benchmark")
    m.add_argument("--models", required=True, help="comma-separated model names for the same endpoint")
    m.add_argument("--cases", help="path to a cases YAML (overrides --suite)")
    m.add_argument("--suite", choices=["basic", "hard", "holdout", "all"], default="basic")
    m.add_argument("--trials", type=int, default=3)

    args = p.parse_args(argv)
    try:
        if args.cmd == "interpret":
            _dump(build_interpreter(_llm())(InterpreterInput(prompt=args.prompt), _store(args)))
        elif args.cmd == "templates":
            for name, entry in sorted(build_library().entries.items()):
                print(f"{name}: {entry.description}")
        elif args.cmd == "design":
            raw = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
            tpl = PICTemplate.model_validate(raw.get("template", raw))
            designer = build_designer(_llm() if args.llm else None, use_llm=args.llm)
            out = designer(DesignerInput(template=tpl), _store(args))
            print(yaml.safe_dump(out.to_dsl(), sort_keys=False))
            _report_design(out)
        elif args.cmd == "layout":
            out = build_layout(out_dir=args.out_dir)(LayoutInput(design=_load_design(args.design)), _store(args))
            _print_layout(out)
        elif args.cmd == "verify":
            out = build_verifier()(
                VerifyInput(design=_load_design(args.design), wl_start_um=args.wl[0], wl_stop_um=args.wl[1],
                            points=args.points), _store(args))
            _print_verify(out)
            if args.csv:
                keys = list(out.transfer_db)
                with open(args.csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["wavelength_um", *keys])
                    for i, wl in enumerate(out.wavelengths_um):
                        w.writerow([wl, *(out.transfer_db[k][i] for k in keys)])
                print(f"spectra written to {args.csv}")
        elif args.cmd == "pipeline":
            store = _store(args)
            if args.adopt:
                interp = build_interpreter(None).adopt(args.adopt)  # type: ignore[arg-type]
                llm = _llm() if args.llm_design else None
            else:
                if not args.prompt:
                    p.error("give a prompt or --adopt NAME")
                llm = _llm()
                interp = build_interpreter(llm)(InterpreterInput(prompt=args.prompt), store)
            out_dir = (store.dir / "layout") if store else Path("layouts")
            res = run_full(
                interp, build_designer(llm, use_llm=args.llm_design),
                build_layout(out_dir=out_dir) if args.layout else None,
                build_verifier() if args.verify else None, store,
            )
            print(yaml.safe_dump(res.designer.to_dsl(), sort_keys=False))
            _report_design(res.designer)
            if res.layout:
                _print_layout(res.layout)
            if res.verify:
                _print_verify(res.verify)
        elif args.cmd == "eval":
            d = data_dir() / "testbench"
            if args.target == "interpreter":
                cases = load_interpreter_cases(args.cases) if args.cases else load_suite(args.suite)
                rep = eval_interpreter(build_interpreter(_llm()), cases, args.trials)
            else:
                cases = load_designer_cases(args.cases or d / "designer_gold.yaml")
                llm = _llm() if args.llm_design else None
                designer = build_designer(llm, use_llm=args.llm_design)
                if args.target == "designer":
                    rep = eval_designer(designer, cases, args.trials)
                elif args.target == "layout":
                    rep = eval_downstream(designer, build_layout(), cases)
                else:
                    rep = eval_downstream(designer, build_verifier(), cases)
            ks = tuple(sorted({1, args.trials} | {k for k in (1, 5) if k <= args.trials}))  # always include pass@trials
            print(json.dumps(rep.summary(ks), indent=2))
            for cid, why in rep.failures():
                print(f"FAIL {cid}: {why}", file=sys.stderr)
        elif args.cmd == "compare":
            cases = load_interpreter_cases(args.cases) if args.cases else load_suite(args.suite)
            def show_failures(model, rep):
                for cid, why in rep.failures():
                    print(f"FAIL [{model}] {cid}: {why}", file=sys.stderr)

            rows = compare_models(args.models.split(","), lambda name: build_interpreter(_llm(name)), cases,
                                  args.trials, on_report=show_failures)
            print(format_table(rows))
    except (StageError, RuntimeError, FileNotFoundError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
