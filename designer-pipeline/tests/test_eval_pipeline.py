import json

import pytest

from phido.cli import main
from phido.core.schemas import InterpreterInput
from phido.core.store import RunStore
from phido.eval.metrics import conditional_rates, pass_at_k, summarize
from phido.eval.runner import (
    compare_templates, eval_designer, eval_interpreter, load_designer_cases, load_interpreter_cases,
)
from phido.pipeline import build_designer, build_interpreter, run_pipeline
from phido.core.schemas import PICTemplate
from conftest import j, make_llm


def test_pass_at_k_and_summary():
    assert pass_at_k(5, 0, 1) == 0.0 and pass_at_k(5, 5, 5) == 1.0
    assert pass_at_k(5, 1, 5) == 1.0 and pytest.approx(pass_at_k(5, 1, 1)) == 0.2
    s = summarize({"a": ["S", "EE", "EE", "EE", "EE"], "b": ["S"] * 5}, ks=(1, 5))
    assert s["pass@5"] == 1.0 and pytest.approx(s["pass@1"]) == 0.6
    c = conditional_rates({"a": ["EE", "CS", "S", "S"]})
    assert c["EE"] == 0.25 and pytest.approx(c["CS"]) == 1 / 3


def test_compare_templates_detects_mismatches():
    mk = lambda kinds, edges: PICTemplate(nodes=[{"id": f"n{i}", "kind": k} for i, k in enumerate(kinds)],
                                          edges=[{"src": f"n{a}", "dst": f"n{b}"} for a, b in edges])
    gold = mk(["mzi", "mzi", "mmi"], [(0, 1), (1, 2)])
    assert compare_templates(mk(["MZI", "mzi", "multimode interferometer"], [(0, 1), (1, 2)]), gold)[0] == "S"
    assert "component" in compare_templates(mk(["mzi", "mmi", "mmi"], [(0, 1), (1, 2)]), gold)[1]
    assert "connectivity" in compare_templates(mk(["mzi", "mzi", "mmi"], [(0, 1), (0, 2)]), gold)[1]


def test_designer_eval_is_green_on_gold_without_any_llm(data, pdk):
    from phido.stages.designer import DesignerStage

    rep = eval_designer(DesignerStage(pdk), load_designer_cases(data / "testbench" / "designer_gold.yaml"))
    assert rep.failures() == [] and rep.summary((1,))["pass@1"] == 1.0


def oracle_llm(cases):
    """Mock model that 'knows' the gold answer: turns gold nodes/edges into an extraction."""
    by_prompt = {c.prompt: c for c in cases}

    def respond(system, user):
        prompt = user.split("Request: ", 1)[1].split("\nAnswer:")[0]
        c = by_prompt[prompt]
        return j({"blocks": [{"name": n["id"], "kind": n["kind"]} for n in c.nodes], "links": c.edges})

    return respond


def test_interpreter_eval_with_oracle_mock(data, library):
    cases = load_interpreter_cases(data / "testbench" / "interpreter_gold.yaml")
    llm, _ = make_llm(oracle_llm(cases))
    from phido.stages.interpreter import InterpreterStage

    rep = eval_interpreter(InterpreterStage(llm, library), cases, trials=2)
    assert rep.failures() == []
    assert rep.summary((1, 2))["pass@2"] == 1.0


def test_interpreter_eval_counts_failures(data):
    cases = load_interpreter_cases(data / "testbench" / "interpreter_gold.yaml")[:2]
    llm, _ = make_llm(lambda s, u: j({"blocks": [{"name": "x", "kind": "straight"}]}))
    from phido.stages.interpreter import InterpreterStage

    rep = eval_interpreter(InterpreterStage(llm), cases)
    assert rep.summary((1,))["absolute"] == {"EE": 1.0} and len(rep.failures()) == 2


def test_pipeline_end_to_end_with_run_store(data, tmp_path):
    ans = {"motif": {"name": "tree", "args": {"unit": "mmi", "stages": 3, "io_kind": "edge_coupler"}}}
    llm, _ = make_llm([j(ans)])
    store = RunStore(tmp_path, "run1")
    interp, design = run_pipeline("1x8 splitter", build_interpreter(llm, data), build_designer(data=data), store)
    assert design.ok and len(design.nodes) == 16 and len(design.edges) == 15
    for rel in ["interpreter/input.yaml", "interpreter/output.yaml", "interpreter/llm_trace.json",
                "designer/input.yaml", "designer/output.yaml"]:
        assert (tmp_path / "run1" / rel).exists(), rel
    # stage isolation: the designer can be re-run from the stored interpreter artifact alone
    from phido.core.schemas import InterpreterOutput
    saved = store.load("interpreter", "output", InterpreterOutput)
    again = build_designer(data=data).run(__import__("phido.core.schemas", fromlist=["DesignerInput"]).DesignerInput(template=saved.template))
    assert again.edges == design.edges


def test_cli_design_command_offline(data, tmp_path, capsys):
    t = tmp_path / "t.yaml"
    t.write_text("nodes: [{id: A, kind: mmi, ports: '1x2'}, {id: B, kind: mmi, ports: '2x2'}]\nedges: [{src: A, dst: B}]\n")
    assert main(["--no-store", "design", str(t)]) == 0
    assert "A,o2:B,o2" in capsys.readouterr().out
    assert main(["--no-store", "eval", "designer"]) == 0
    assert json.loads(capsys.readouterr().out)["pass@1"] == 1.0
