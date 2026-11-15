import pytest
from pydantic import ValidationError

from phido.core.schemas import EntityExtraction, PICTemplate
from phido.core.store import RunStore
from phido.knowledge.text import canonical_kind, tokenize
from phido.stages.interpreter import EXAMPLES


def test_extraction_needs_content_and_valid_links():
    with pytest.raises(ValidationError):
        EntityExtraction()
    with pytest.raises(ValidationError):
        EntityExtraction(blocks=[{"name": "a", "kind": "mzi"}], links=[{"src": "a", "dst": "zzz"}])
    with pytest.raises(ValidationError):
        EntityExtraction(blocks=[{"name": "bad name", "kind": "mzi"}])


def test_null_values_from_the_llm_mean_not_given():
    ex = EntityExtraction.model_validate({
        "motif": {"name": "cascade", "args": {"unit": "mzi", "n": 8, "ports": None, "io_kind": None}},
        "blocks": [{"name": "a", "kind": "mzi", "params": {"radius_um": None, "phase": 1}}],
    })
    assert ex.motif.args == {"unit": "mzi", "n": 8} and ex.blocks[0].params == {"phase": 1}


def test_template_validation():
    with pytest.raises(ValidationError):
        PICTemplate(nodes=[{"id": "A", "kind": "x"}], edges=[{"src": "A", "dst": "A"}])
    with pytest.raises(ValidationError):
        PICTemplate(nodes=[{"id": "A", "kind": "x"}, {"id": "A", "kind": "y"}])


def test_prompt_examples_are_valid_schema_instances():
    for _, ans in EXAMPLES:
        EntityExtraction.model_validate(ans)


def test_store_roundtrip(tmp_path):
    store = RunStore(tmp_path, "r1")
    t = PICTemplate(nodes=[{"id": "A", "kind": "mzi"}])
    store.save("x", "t", t)
    assert store.load("x", "t", PICTemplate) == t


def test_text_helpers():
    assert tokenize("A 2 × 2 Mach-Zehnder interferometer") == ["2x2", "mzi"]
    assert canonical_kind("MZI with thermo-optic heater") == "mzi"
    assert canonical_kind("edge_coupler") == "edge_coupler"
    assert canonical_kind("thermo phase shifter") == "heater"
    assert canonical_kind("directional coupler") == "coupler"
    assert canonical_kind("multimode interferometer") == "mmi"
