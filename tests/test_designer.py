import pytest

from phido.core.errors import StageError
from phido.core.schemas import Connect, DesignerInput, PICTemplate, TemplateEdge
from phido.knowledge.pdk import Cell
from phido.stages.designer import CircuitGraph, DesignerStage, LLMProposer, WiringError
from conftest import j, make_llm


def tpl(nodes, edges=()):
    return PICTemplate(nodes=nodes, edges=[{"src": a, "dst": b} for a, b in edges])


def design(pdk, template, **kw):
    return DesignerStage(pdk, **kw).run(DesignerInput(template=template))


def test_tree_wiring_is_planar_and_ordered(pdk):
    out = design(pdk, tpl([{"id": f"N{i}", "kind": "mmi", "ports": "1x2"} for i in (1, 2, 3)],
                          [("N1", "N2"), ("N1", "N3")]))
    assert out.ok and not out.warnings
    assert [e.link for e in out.edges.values()] == ["N1,o2:N2,o1", "N1,o3:N3,o1"]  # top port -> first child


def test_requirement_selects_fast_cell(pdk):
    out = design(pdk, tpl([{"id": "M", "kind": "mzi", "ports": "2x2", "params": {"modulation_bandwidth_ghz": 10}}]))
    assert out.nodes["M"].component == "mzi_2x2_pndiode" and out.ok


def test_unmet_requirement_is_reported(pdk):
    out = design(pdk, tpl([{"id": "M", "kind": "mzi", "ports": "2x2", "params": {"modulation_bandwidth_ghz": 500}}]))
    assert [i.code for i in out.issues] == ["PC"]


def test_param_mapping_alias_and_range(pdk):
    t = tpl([{"id": "R", "kind": "ring", "ports": "2x2", "params": {"radius_um": 12, "ring_radius": 400, "nope": 1}}])
    out = design(pdk, t)
    assert out.nodes["R"].params == {"radius": 12.0}
    msgs = " ".join(i.message for i in out.issues)
    assert "outside allowed range" in msgs and "'nope'" in msgs


def test_no_matching_cell_is_cs_error(pdk):
    with pytest.raises(StageError) as e:
        design(pdk, tpl([{"id": "X", "kind": "flux capacitor"}]))
    assert e.value.code == "CS"


def test_ambiguity_flagged_and_llm_chooser_decides(pdk):
    t = tpl([{"id": "M", "kind": "mzi", "ports": "2x2"}])
    plain = design(pdk, t)
    assert plain.choices[0].ambiguous and plain.choices[0].chosen_by == "retrieval"
    llm, _ = make_llm([j({"cell": "mzi_2x2_pindiode"})])
    out = DesignerStage.with_llm(pdk, llm, wire=False).run(DesignerInput(template=t))
    assert out.nodes["M"].component == "mzi_2x2_pindiode" and out.choices[0].chosen_by == "llm"


def test_llm_chooser_rejects_non_candidate_then_recovers(pdk):
    t = tpl([{"id": "M", "kind": "mzi", "ports": "2x2"}])
    llm, mock = make_llm([j({"cell": "made_up"}), j({"cell": "mzi_2x2_pndiode"})])
    out = DesignerStage.with_llm(pdk, llm, wire=False).run(DesignerInput(template=t))
    assert out.nodes["M"].component == "mzi_2x2_pndiode" and len(mock.calls) == 2


# ----- circuit graph rules
def cells(pdk):
    return {"A": pdk.cell("mmi_1x2"), "B": pdk.cell("mmi_1x2")}


def graph(pdk):
    return CircuitGraph(cells(pdk), [TemplateEdge(src="A", dst="B")])


@pytest.mark.parametrize("op,msg", [
    (Connect(src_node="A", src_port="o1", dst_node="B", dst_port="o1"), "input-side"),
    (Connect(src_node="A", src_port="o2", dst_node="B", dst_port="o2"), "output-side"),
    (Connect(src_node="A", src_port="o9", dst_node="B", dst_port="o1"), "no port"),
    (Connect(src_node="A", src_port="o2", dst_node="A", dst_port="o1"), "itself"),
    (Connect(src_node="B", src_port="o2", dst_node="A", dst_port="o1"), "no pending"),
])
def test_invalid_edits_rejected(pdk, op, msg):
    with pytest.raises(WiringError, match=msg):
        graph(pdk).apply(op)


def test_port_reuse_rejected(pdk):
    g = CircuitGraph(cells(pdk), [TemplateEdge(src="A", dst="B")] * 2)
    g.apply(Connect(src_node="A", src_port="o2", dst_node="B", dst_port="o1"))
    with pytest.raises(WiringError, match="already connected"):
        g.apply(Connect(src_node="A", src_port="o3", dst_node="B", dst_port="o1"))


def test_nonplanar_topology_warns():
    ports = "3x3"
    cell = Cell(name="k33", ports=ports)
    a = {f"a{i}": cell for i in range(3)}
    b = {f"b{i}": cell for i in range(3)}
    edges = [TemplateEdge(src=x, dst=y) for x in a for y in b]
    g = CircuitGraph({**a, **b}, edges)
    from phido.stages.designer import GreedyProposer

    for op in GreedyProposer().propose(g, g.pending(edges)):
        g.apply(op)
    assert not g.pending(edges) and g.warnings()


# ----- wiring proposers
def test_llm_proposer_retries_invalid_edits(pdk):
    t = tpl([{"id": "N1", "kind": "mmi", "ports": "1x2"}, {"id": "N2", "kind": "mmi", "ports": "1x2"}], [("N1", "N2")])
    bad = {"edits": [{"src_node": "N1", "src_port": "o1", "dst_node": "N2", "dst_port": "o1"}]}
    good = {"edits": [{"src_node": "N1", "src_port": "o3", "dst_node": "N2", "dst_port": "o1"}]}
    llm, mock = make_llm([j(bad), j(good)])
    out = DesignerStage(pdk, proposer=LLMProposer(llm), llm=llm).run(DesignerInput(template=t))
    assert [e.link for e in out.edges.values()] == ["N1,o3:N2,o1"] and len(mock.calls) == 2


def test_llm_proposer_falls_back_to_greedy(pdk):
    t = tpl([{"id": "N1", "kind": "mmi", "ports": "1x2"}, {"id": "N2", "kind": "mmi", "ports": "1x2"}], [("N1", "N2")])
    llm, _ = make_llm(["junk"] * 6, max_retries=1)
    out = DesignerStage(pdk, proposer=LLMProposer(llm)).run(DesignerInput(template=t))
    assert out.ok and len(out.edges) == 1


def test_unwireable_edge_becomes_sg_issue(pdk):
    t = tpl([{"id": "P", "kind": "photodetector"}, {"id": "M", "kind": "mmi", "ports": "1x2"}], [("P", "M")])
    out = design(pdk, t)  # photodetector has no output port
    assert [i.code for i in out.issues] == ["SG"]


def test_kind_outranks_port_mismatch(pdk):
    # a model may say a photodetector has "1x1" ports; the PDK cell is 1x0. The right kind must still win.
    t = tpl([{"id": "M", "kind": "mzi", "ports": "2x2"}, {"id": "PD", "kind": "photodetector", "ports": "1x1"}],
            [("M", "PD")])
    out = design(pdk, t)
    assert out.nodes["PD"].component == "photodetector" and out.ok
    assert out.choices[1].match == "partial"
    assert any("PD: photodetector is a partial match" in w for w in out.warnings)
