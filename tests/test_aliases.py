from phido.core.schemas import PICTemplate
from phido.eval.runner import compare_templates, eval_interpreter, load_suite, param_aliases
from phido.llm.adapter import StructuredLLM
from phido.llm.mock import MockLLM
from phido.pipeline import load_pdk
from phido.stages.interpreter import InterpreterStage
from conftest import j


def one(params):
    return PICTemplate(nodes=[{"id": "a", "kind": "directional_coupler", "params": params}])


def test_pdk_aliases_are_accepted_when_given():
    aliases = param_aliases(load_pdk())
    assert aliases["coupling_length"] == "length" and aliases["ring_radius"] == "radius"
    gold, pred = one({"length_um": 30}), one({"coupling_length_um": 30})
    assert compare_templates(pred, gold, aliases)[0] == "S"
    assert compare_templates(pred, gold)[0] == "EE"  # without the alias table the names differ


def test_aliases_do_not_make_different_parameters_equal():
    aliases = param_aliases(load_pdk())
    assert compare_templates(one({"gap_um": 30}), one({"length_um": 30}), aliases)[0] == "EE"
    assert compare_templates(one({"coupling_length_um": 31}), one({"length_um": 30}), aliases)[0] == "EE"  # value still checked


def test_the_holdout_case_that_both_models_got_marked_wrong_now_passes():
    case = next(c for c in load_suite("holdout") if c.id == "HO-directional-coupler-length")
    llm = StructuredLLM(MockLLM(lambda s, u: j({"blocks": [
        {"name": "dc1", "kind": "directional_coupler", "params": {"coupling_length_um": 30}}]})))
    rep = eval_interpreter(InterpreterStage(llm), [case])
    assert rep.failures() == []
