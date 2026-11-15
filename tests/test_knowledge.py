import pytest

from phido.core.schemas import MotifSpec
from phido.knowledge.motifs import MotifError, expand_motif
from phido.knowledge.pdk import Cell


def test_pdk_loads_and_port_numbering(pdk):
    assert len(pdk.cells) >= 15
    c = pdk.cell("mmi_1x2")
    assert c.signature == "1x2"
    assert [(p.name, p.side) for p in c.ports] == [("o1", "left"), ("o2", "right"), ("o3", "right")]
    assert c.ports[1].y > c.ports[2].y  # o2 above o3
    assert pdk.cell("edge_coupler").signature == "1x1"
    assert pdk.cell("photodetector").signature == "1x0"


def test_cell_requires_ports():
    with pytest.raises(ValueError):
        Cell(name="x", ports=[])


def test_retrieval_ranks_expected_cell_first(pdk):
    top = pdk.search("mzi 2x2 thermo-optic heater", k=3, focus="mzi")
    assert top[0].id == "mzi_2x2_heater"
    assert pdk.search("", k=3) == []
    assert pdk.search("flux capacitor", k=3) == []


def test_library_search(library):
    hits = library.search("Two MZIs in series mzi mzi", k=3)
    assert hits[0].id == "mzi_cascade_2"
    assert library.get("single_mzi").source == "library"


def test_tree_motif_counts():
    t = expand_motif(MotifSpec(name="tree", args={"unit": "mmi", "stages": 4}))
    assert len(t.nodes) == 15 and len(t.edges) == 14
    assert {n.ports for n in t.nodes} == {"1x2"}
    out = [e.dst for e in t.edges if e.src == "N1"]
    assert out == ["N2", "N3"]


def test_tree_motif_with_io():
    t = expand_motif(MotifSpec(name="tree", args={"unit": "mmi", "stages": 3, "io_kind": "edge_coupler"}))
    assert len(t.nodes) == 7 + 1 + 8 and len(t.edges) == 6 + 1 + 8


def test_cascade_motif():
    t = expand_motif(MotifSpec(name="cascade", args={"unit": "mzi", "n": "4", "io_kind": "grating_coupler"}))
    assert [n.id for n in t.nodes] == ["IN1", "N1", "N2", "N3", "N4", "OUT1"]
    assert len(t.edges) == 5


def test_cascade_input_and_output_components_are_separate():
    kinds = lambda t: [n.kind for n in t.nodes]
    out_only = expand_motif(MotifSpec(name="cascade", args={"unit": "mzi", "n": 4, "out_kind": "grating_coupler"}))
    assert kinds(out_only).count("grating_coupler") == 1 and [n.id for n in out_only.nodes][-1] == "OUT1"
    assert len(out_only.nodes) == 5 and len(out_only.edges) == 4
    in_only = expand_motif(MotifSpec(name="cascade", args={"unit": "mzi", "n": 4, "in_kind": "edge_coupler"}))
    assert [n.id for n in in_only.nodes][0] == "IN1" and len(in_only.nodes) == 5
    both = expand_motif(MotifSpec(name="cascade", args={"unit": "mzi", "n": 4, "io_kind": "edge_coupler"}))
    assert len(both.nodes) == 6
    mixed = expand_motif(MotifSpec(name="cascade", args={"unit": "mzi", "n": 2, "io_kind": "edge_coupler", "out_kind": "photodetector"}))
    assert [n.kind for n in mixed.nodes] == ["edge_coupler", "mzi", "mzi", "photodetector"]


def test_tree_input_and_output_components_are_separate():
    in_only = expand_motif(MotifSpec(name="tree", args={"unit": "mmi", "stages": 3, "in_kind": "edge_coupler"}))
    assert len(in_only.nodes) == 8
    out_only = expand_motif(MotifSpec(name="tree", args={"unit": "mmi", "stages": 3, "out_kind": "edge_coupler"}))
    assert len(out_only.nodes) == 7 + 8 and not any(e.src == "IN1" for e in out_only.edges)


@pytest.mark.parametrize("spec", [
    MotifSpec(name="mesh"),
    MotifSpec(name="tree", args={"stages": 0}),
    MotifSpec(name="tree", args={"unit": "mmi", "stages": 9}),
    MotifSpec(name="cascade", args={"unit": "mzi"}),
    MotifSpec(name="cascade", args={"n": 3, "ports": "two"}),
])
def test_motif_errors(spec):
    with pytest.raises(MotifError):
        expand_motif(spec)
