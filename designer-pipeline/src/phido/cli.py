from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .core.errors import StageError
from .core.schemas import DesignerInput, InterpreterInput, PICTemplate
from .core.store import RunStore
from .eval.runner import eval_designer, eval_interpreter, load_designer_cases, load_interpreter_cases
from .llm.adapter import StructuredLLM
from .llm.openai_compat import OpenAICompatClient
from .paths import data_dir
from .pipeline import build_designer, build_interpreter, run_pipeline


def _llm() -> StructuredLLM:
    import os

    return StructuredLLM(OpenAICompatClient.from_env(), temperature=float(os.environ.get("PHIDO_TEMPERATURE", "0")))


def _dump(obj) -> None:
    print(yaml.safe_dump(obj.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


def _store(args) -> RunStore | None:
    return None if args.no_store else RunStore(args.runs_dir)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="phido", description="PhIDO rebuild: interpreter + designer")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--no-store", action="store_true", help="do not write run artifacts")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("interpret", help="prompt -> PIC template (needs an LLM)")
    a.add_argument("prompt")

    b = sub.add_parser("design", help="PIC template YAML -> PIC DSL")
    b.add_argument("template", help="path to a PICTemplate YAML (e.g. runs/<id>/interpreter/output.yaml)")
    b.add_argument("--llm", action="store_true", help="let an LLM resolve ambiguity and propose wiring")

    c = sub.add_parser("pipeline", help="prompt -> template -> DSL")
    c.add_argument("prompt")
    c.add_argument("--llm-design", action="store_true")

    e = sub.add_parser("eval", help="evaluate one stage in isolation")
    e.add_argument("target", choices=["interpreter", "designer"])
    e.add_argument("--cases")
    e.add_argument("--trials", type=int, default=1)
    e.add_argument("--llm-design", action="store_true")

    args = p.parse_args(argv)
    try:
        if args.cmd == "interpret":
            out = build_interpreter(_llm())(InterpreterInput(prompt=args.prompt), _store(args))
            _dump(out)
        elif args.cmd == "design":
            raw = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
            tpl = PICTemplate.model_validate(raw.get("template", raw))  # accepts interpreter output too
            designer = build_designer(_llm() if args.llm else None, use_llm=args.llm)
            out = designer(DesignerInput(template=tpl), _store(args))
            print(yaml.safe_dump(out.to_dsl(), sort_keys=False))
            for i in out.issues:
                print(f"ISSUE [{i.code}] {i.node or ''} {i.message}", file=sys.stderr)
        elif args.cmd == "pipeline":
            llm = _llm()
            interp, design = run_pipeline(
                args.prompt, build_interpreter(llm), build_designer(llm, use_llm=args.llm_design), _store(args)
            )
            print(yaml.safe_dump(design.to_dsl(), sort_keys=False))
        elif args.cmd == "eval":
            d = data_dir() / "testbench"
            if args.target == "interpreter":
                cases = load_interpreter_cases(args.cases or d / "interpreter_gold.yaml")
                rep = eval_interpreter(build_interpreter(_llm()), cases, args.trials)
            else:
                cases = load_designer_cases(args.cases or d / "designer_gold.yaml")
                llm = _llm() if args.llm_design else None
                rep = eval_designer(build_designer(llm, use_llm=args.llm_design), cases, args.trials)
            ks = tuple(k for k in (1, 5) if k <= args.trials) or (1,)
            print(json.dumps(rep.summary(ks), indent=2))
            for cid, why in rep.failures():
                print(f"FAIL {cid}: {why}", file=sys.stderr)
    except StageError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
