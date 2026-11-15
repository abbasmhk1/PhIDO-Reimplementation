import json

import pytest

from phido.cli import main
from phido.core.errors import StageError
from phido.core.schemas import InterpreterInput, PICTemplate
from phido.eval.compare import compare_models, format_table
from phido.eval.runner import (
    InterpreterCase, compare_templates, eval_downstream, load_designer_cases, load_interpreter_cases,
)
from phido.llm.adapter import StructuredLLM
from phido.llm.mock import MockLLM
from phido.pipeline import build_designer, build_library, build_verifier, load_pdk
from phido.stages.interpreter import InterpreterStage
from conftest import j


def tpl(nodes, edges=()):
    return PICTemplate(nodes=nodes, edges=[{"src": a, "dst": b} for a, b in edges])


# ----- stricter interpreter scoring
def test_ports_and_params_are_checked_when_gold_states_them():
    gold = tpl([{"id": "a", "kind": "mzi", "ports": "2x2", "params": {"modulation_bandwidth_ghz": 10}}])
    ok = tpl([{"id": "x", "kind": "MZI", "ports": "2x2", "params": {"modulation_bandwidth_ghz": 10.0}}])
    assert compare_templates(ok, gold)[0] == "S"
    assert "port" in compare_templates(tpl([{"id": "x", "kind": "mzi", "ports": "1x2",
                                             "params": {"modulation_bandwidth_ghz": 10}}]), gold)[1]
    assert "parameter" in compare_templates(tpl([{"id": "x", "kind": "mzi", "ports": "2x2",
                                                  "params": {"modulation_bandwidth_ghz": 1}}]), gold)[1]
    assert "parameter" in compare_templates(tpl([{"id": "x", "kind": "mzi", "ports": "2x2"}]), gold)[1]


def test_unrecognised_kind_falls_back_to_the_description():
    gold = tpl([{"id": "a", "kind": "heater"}, {"id": "b", "kind": "mzi"}], [("a", "b")])
    pred = tpl([{"id": "x", "kind": "phase_shifter", "description": "thermo-optic heater"}, {"id": "y", "kind": "mzi"}], [("x", "y")])
    assert compare_templates(pred, gold)[0] == "S"
    bare = tpl([{"id": "x", "kind": "phase_shifter"}, {"id": "y", "kind": "mzi"}], [("x", "y")])
    assert compare_templates(bare, gold)[0] == "EE"  # no description to rescue it
    # a recognised kind is never overridden by its description
    assert compare_templates(tpl([{"id": "x", "kind": "mzi", "description": "with heater"}]), tpl([{"id": "a", "kind": "mzi"}]))[0] == "S"


def test_parameter_mismatch_message_says_what_was_predicted():
    gold = tpl([{"id": "a", "kind": "mzi", "params": {"delta_length_um": 30}}, {"id": "b", "kind": "mzi", "params": {"delta_length_um": 50}}], [("a", "b")])
    pred = tpl([{"id": "a", "kind": "mzi", "params": {"delta_length_um": 50}}, {"id": "b", "kind": "mzi", "params": {"delta_length_um": 30}}], [("a", "b")])
    code, msg = compare_templates(pred, gold)
    assert code == "EE" and "predicted" in msg and "'delta_length_um': 50" in msg and "gold" in msg


def test_prompt_teaches_hierarchy_expansion_and_listed_parts():
    from phido.stages.interpreter import build_system_prompt

    p = build_system_prompt()
    assert "Expand hierarchy" in p and "do not collapse them into one component" in p


def test_param_keys_match_through_unit_suffix_and_unstated_ports_are_not_required():
    gold = tpl([{"id": "a", "kind": "ring", "params": {"radius_um": 10}}])
    assert compare_templates(tpl([{"id": "x", "kind": "ring", "ports": "1x1", "params": {"radius": 10}}]), gold)[0] == "S"


def test_param_check_finds_the_right_assignment_among_symmetric_nodes():
    gold = tpl([{"id": "a", "kind": "mzi", "params": {"phase": 1}}, {"id": "b", "kind": "mzi"}, {"id": "c", "kind": "mmi"}],
               [("a", "c"), ("b", "c")])
    pred = tpl([{"id": "p", "kind": "mzi"}, {"id": "q", "kind": "mzi", "params": {"phase": 1}}, {"id": "r", "kind": "mmi"}],
               [("p", "r"), ("q", "r")])
    assert compare_templates(pred, gold)[0] == "S"


def test_output_only_coupler_prompt_matches_its_gold_case(data):
    from phido.core.schemas import MotifSpec
    from phido.knowledge.motifs import expand_motif

    cases = {c.id: c for c in load_interpreter_cases(data / "testbench" / "interpreter_gold.yaml")}
    pred = expand_motif(MotifSpec(name="cascade", args={"unit": "mzi", "n": 4, "out_kind": "grating_coupler"}))
    assert compare_templates(pred, cases["L3-four-mzi-grating"].gold())[0] == "S"
    both = expand_motif(MotifSpec(name="cascade", args={"unit": "mzi", "n": 4, "io_kind": "grating_coupler"}))
    assert compare_templates(both, cases["L3-four-mzi-grating"].gold())[0] == "EE"  # the failure seen with real models


def test_gold_given_as_motif(data):
    cases = {c.id: c for c in load_interpreter_cases(data / "testbench" / "interpreter_gold.yaml")}
    g = cases["L4-tree-1x16-io"].gold()
    assert len(g.nodes) == 15 + 1 + 16 and cases["L4-cascade-8"].gold().edges[-1].dst == "N8"


# ----- LLM stats and model comparison
class UsageClient(MockLLM):
    last_usage = {"prompt_tokens": 100, "completion_tokens": 20}


def test_llm_stats_accumulate_and_reset():
    llm = StructuredLLM(UsageClient(['{"blocks":[{"name":"a","kind":"mzi"}]}'] * 3))
    stage = InterpreterStage(llm)
    for _ in range(3):
        stage.run(InterpreterInput(prompt="an mzi"))
    s = llm.reset_stats()
    assert s["calls"] == 3 and s["prompt_tokens"] == 300 and s["completion_tokens"] == 60 and s["seconds"] >= 0
    assert llm.stats["calls"] == 0


def test_compare_models_table(data, library):
    cases = [c for c in load_interpreter_cases(data / "testbench" / "interpreter_gold.yaml") if c.id in ("L2-mmi-mmi", "L1-ring")]
    by_prompt = {c.prompt: c for c in cases}

    def respond_good(system, user):
        c = by_prompt[user.split("Request: ", 1)[1].split("\nAnswer:")[0]]
        blocks = [{"name": n["id"], "kind": n["kind"], **{k: n[k] for k in ("params",) if k in n}} for n in c.nodes]
        return j({"blocks": blocks, "links": c.edges})

    makers = {"good": respond_good, "bad": lambda s, u: j({"blocks": [{"name": "x", "kind": "straight"}]})}
    rows = compare_models(["good", "bad"], lambda m: InterpreterStage(StructuredLLM(UsageClient(makers[m])), library),
                          cases, trials=2, ks=(1, 2))
    assert {r["model"]: r["pass@1"] for r in rows} == {"good": 1.0, "bad": 0.0}
    assert rows[0]["calls_per_trial"] == 1.0 and rows[0]["prompt_tokens_per_trial"] == 100.0
    assert rows[0]["pass@2"] == 1.0 and rows[1]["pass@2"] == 0.0  # pass@k for k = trials is always shown
    seen = {}
    compare_models(["bad"], lambda m: InterpreterStage(StructuredLLM(UsageClient(makers[m])), library), cases,
                   trials=1, on_report=lambda m, rep: seen.update({m: rep.failures()}))
    assert len(seen["bad"]) == 2 and "component mismatch" in seen["bad"][0][1]
    table = format_table(rows)
    assert table.startswith("| model |") and "| good |" in table


# ----- adopting a library template
def test_adopt_library_template_without_llm(library):
    llm = StructuredLLM(MockLLM([]))
    out = InterpreterStage(llm, library).adopt("mzi_cascade_2")
    assert out.extraction is None and len(out.template.nodes) == 2 and out.template.source == "library"
    assert llm.client.calls == []
    with pytest.raises(StageError, match="unknown library template"):
        InterpreterStage(llm, library).adopt("nope")


def test_title_is_not_cut_mid_word(library):
    llm = StructuredLLM(MockLLM([j({"blocks": [{"name": "a", "kind": "mzi"}]})]))
    long = "Four MZIs in series, each with a heater, then a photodetector and some more words to be long enough"
    t = InterpreterStage(llm, library).run(InterpreterInput(prompt=long)).template.title
    assert len(t) <= 80 and not t.endswith("detecto")


# ----- downstream evals (designer gold -> layout / verify)
def test_verify_eval_is_green_on_gold(data):
    cases = load_designer_cases(data / "testbench" / "designer_gold.yaml")
    rep = eval_downstream(build_designer(), build_verifier(), cases)
    assert rep.failures() == []


def test_layout_eval_is_green_on_gold(data):
    pytest.importorskip("gdsfactory")
    from phido.pipeline import build_layout

    rep = eval_downstream(build_designer(), build_layout(), load_designer_cases(data / "testbench" / "designer_gold.yaml"))
    assert rep.failures() == []


# ----- CLI
def test_cli_pipeline_adopt_with_verify_and_csv(tmp_path, capsys):
    runs = tmp_path / "runs"
    assert main(["--runs-dir", str(runs), "pipeline", "--adopt", "edge_mzi_edge", "--verify"]) == 0
    cap = capsys.readouterr()
    assert "verify:" in cap.out and "leaves through" not in cap.err  # the designer now wires the MZI through its cross path
    out_yaml = next(runs.glob("*/designer/output.yaml"))
    csv_path = tmp_path / "s.csv"
    assert main(["--no-store", "verify", str(out_yaml), "--points", "5", "--csv", str(csv_path)]) == 0
    import csv

    header = next(csv.reader(csv_path.open()))
    assert header == ["wavelength_um", "IN1,fiber->OUT1,fiber"]  # names contain commas, so the writer quotes them


def test_cli_layout_command(tmp_path, capsys):
    pytest.importorskip("gdsfactory")
    runs = tmp_path / "runs"
    assert main(["--runs-dir", str(runs), "pipeline", "--adopt", "mmi_splitter_1x4_tree", "--layout"]) == 0
    capsys.readouterr()
    gds = next(runs.glob("*/layout/*.gds"))
    assert gds.stat().st_size > 500
    out_yaml = next(runs.glob("*/designer/output.yaml"))
    assert main(["--no-store", "layout", str(out_yaml), "--out-dir", str(tmp_path / "l")]) == 0
    assert "2/2 routes ok" in capsys.readouterr().out


def test_explicit_model_does_not_need_the_env_variable(monkeypatch):
    from phido.llm.openai_compat import OpenAICompatClient

    monkeypatch.delenv("PHIDO_LLM_MODEL", raising=False)
    monkeypatch.setenv("PHIDO_LLM_BASE_URL", "http://localhost:11435/v1")
    c = OpenAICompatClient.from_env("flint:40b")  # what `phido compare --models ...` does
    assert c.model == "flint:40b" and c.url == "http://localhost:11435/v1/chat/completions"
    with pytest.raises(RuntimeError, match="PHIDO_LLM_MODEL"):
        OpenAICompatClient.from_env()


def test_cli_reports_missing_llm_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("PHIDO_LLM_MODEL", raising=False)
    assert main(["--no-store", "interpret", "an mzi"]) == 2
    assert "PHIDO_LLM_MODEL" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["designer", "verify"])
def test_cli_offline_evals(target, capsys):
    assert main(["--no-store", "eval", target]) == 0
    assert json.loads(capsys.readouterr().out)["pass@1"] == 1.0
