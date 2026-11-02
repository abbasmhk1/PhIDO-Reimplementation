import pytest

from phido.core.errors import StageError
from phido.core.schemas import EntityExtraction, InterpreterInput
from phido.stages.interpreter import EXAMPLES, InterpreterStage
from conftest import j, make_llm

TWO_MZI = {
    "blocks": [
        {"name": "mzi1", "kind": "mzi", "ports": "2x2", "params": {"modulation_bandwidth_ghz": 10}},
        {"name": "mzi2", "kind": "mzi", "ports": "2x2", "params": {"modulation_bandwidth_ghz": 10}},
    ],
    "links": [{"src": "mzi1", "dst": "mzi2"}],
}


def run(stage, prompt="Two MZIs in series"):
    return stage.run(InterpreterInput(prompt=prompt))


def test_blocks_to_template_and_library_match(library):
    llm, _ = make_llm([j(TWO_MZI)])
    out = run(InterpreterStage(llm, library))
    assert [n.id for n in out.template.nodes] == ["mzi1", "mzi2"]
    assert [(e.src, e.dst) for e in out.template.edges] == [("mzi1", "mzi2")]
    assert out.template.nodes[0].params == {"modulation_bandwidth_ghz": 10}
    assert out.matches[0].name == "mzi_cascade_2"


def test_motif_is_expanded_by_code_not_llm():
    ans = {"motif": {"name": "tree", "args": {"unit": "mmi", "stages": 4}}}
    llm, mock = make_llm([j(ans)])
    out = run(InterpreterStage(llm), "1x16 splitter from 1x2 MMIs")
    assert len(out.template.nodes) == 15 and out.template.source == "motif"
    assert len(mock.calls) == 1


def test_unknown_motif_triggers_retry_with_message():
    bad = {"motif": {"name": "mesh", "args": {}}}
    llm, mock = make_llm([j(bad), j(TWO_MZI)])
    out = run(InterpreterStage(llm))
    assert out.template.source == "blocks"
    assert "unknown motif" in mock.calls[1][1]


def test_invalid_link_triggers_retry():
    bad = {"blocks": [{"name": "a", "kind": "mzi"}], "links": [{"src": "a", "dst": "b"}]}
    llm, mock = make_llm([j(bad), j(TWO_MZI)])
    run(InterpreterStage(llm))
    assert len(mock.calls) == 2


def test_gives_ee_when_llm_never_valid():
    llm, _ = make_llm(["nope"] * 5, max_retries=1)
    with pytest.raises(StageError) as e:
        run(InterpreterStage(llm))
    assert e.value.code == "EE"


def test_interpreter_is_independent_of_designer_and_pdk():
    from pathlib import Path

    import phido.stages.interpreter as m

    src = Path(m.__file__).read_text()
    assert "designer" not in src.lower() and "pdk" not in src.lower()


def test_prompt_examples_are_valid_schema_instances():
    for _, ans in EXAMPLES:
        EntityExtraction.model_validate(ans)
