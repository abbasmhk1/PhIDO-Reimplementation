import networkx as nx
import pytest

from phido.core.schemas import DesignerInput, LayoutInput, MotifSpec, VerifyInput
from phido.eval.runner import compare_templates, load_suite
from phido.geometry.placement import segment_hits_box
from phido.knowledge.motifs import MotifError, expand_motif
from phido.pipeline import build_designer, load_pdk
from phido.stages.layout import stage as layout_stage
from phido.stages.layout.stage import LayoutStage
from phido.stages.verify.stage import VerifyStage
from phido.core.schemas import DesignerOutput, DSLEdge, DSLNode


def mesh(n, topo="clements", **kw):
    return expand_motif(MotifSpec(name="mesh", args={"modes": n, "topology": topo, **kw}))


# ----- mesh motif
@pytest.mark.parametrize("topo", ["clements", "reck"])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_mesh_structure(topo, n):
    t = mesh(n, topo)
    assert len(t.nodes) == n * (n - 1) // 2 and len(t.edges) == n * (n - 2)
    g = nx.DiGraph([(e.src, e.dst) for e in t.edges]) if t.edges else nx.DiGraph()
    g.add_nodes_from(x.id for x in t.nodes)
    assert nx.is_directed_acyclic_graph(g)
    assert max(dict(g.in_degree()).values()) <= 2 and max(dict(g.out_degree()).values()) <= 2
    assert nx.check_planarity(g.to_undirected())[0]


def test_mesh_io_and_errors():
    t = mesh(4, io_kind="edge_coupler")
    assert sum(n.kind == "edge_coupler" for n in t.nodes) == 8
    outs = mesh(4, out_kind="grating_coupler").nodes
    assert sum(n.kind == "grating_coupler" for n in outs) == 4 and not any(n.id.startswith("IN") for n in outs)
    for bad in ({"modes": 1}, {"modes": 4, "topology": "spanke"}, {}):
        with pytest.raises(MotifError):
            expand_motif(MotifSpec(name="mesh", args=bad))


# ----- hard suite is sound: every gold circuit can be built by the designer and simulated
def test_hard_suite_gold_is_realizable_and_self_consistent():
    designer, verifier = build_designer(), VerifyStage(load_pdk())
    cases = load_suite("hard")
    assert len(cases) >= 18 and len({c.id for c in cases}) == len(cases)
    for c in cases:
        gold = c.gold()
        assert compare_templates(gold, gold)[0] == "S", c.id
        d = designer.run(DesignerInput(template=gold))
        assert d.ok, (c.id, d.issues)
        assert verifier.run(VerifyInput(design=d, points=3)).ok, c.id


def test_suites():
    assert len(load_suite("all")) == len(load_suite("basic")) + len(load_suite("hard"))
    with pytest.raises(KeyError):
        load_suite("nope")


# ----- layout regressions found with meshes
@pytest.fixture(scope="module")
def gf():
    return pytest.importorskip("gdsfactory")


@pytest.mark.parametrize("n", [3, 4, 6, 8])
def test_clements_meshes_lay_out_cleanly(gf, n):
    pdk = load_pdk()
    d = build_designer().run(DesignerInput(template=mesh(n, io_kind="edge_coupler")))
    assert d.ok and not any("crossing" in w for w in d.warnings)
    out = LayoutStage(pdk).run(LayoutInput(design=d))
    assert out.ok, out.issues
    assert all(r.ok for r in out.routes)


def test_segment_hits_box():
    box = (0, 0, 10, 10)
    assert segment_hits_box((-5, 5), (15, 5), box)  # runs through
    assert not segment_hits_box((-5, 12), (15, 12), box)  # passes above
    assert not segment_hits_box((-5, 10), (15, 10), box)  # grazing the edge is not inside


def test_route_through_another_component_is_reported(gf, monkeypatch):
    pdk = load_pdk()
    d = DesignerOutput(
        nodes={"A": DSLNode(component="edge_coupler"), "B": DSLNode(component="photodetector"),
               "C": DSLNode(component="mmi_2x2")},
        edges={"E1": DSLEdge(link="A,o1:B,o1")},
    )
    # force a blocker C onto the straight A -> B wire
    monkeypatch.setattr(layout_stage, "place", lambda nodes, pairs, sizes, gx, gy, port_dy=None: {
        "A": (0.0, 0.0), "B": (600.0, 0.0), "C": (300.0, 0.0)})
    out = LayoutStage(pdk).run(LayoutInput(design=d))
    assert any("passes through component C" in i.message for i in out.issues)


def test_layout_retries_with_more_spacing_only_when_needed(gf):
    pdk = load_pdk()
    clean = build_designer().run(DesignerInput(template=mesh(2, io_kind="edge_coupler")))
    assert not any("vertical gap" in w for w in LayoutStage(pdk).run(LayoutInput(design=clean)).warnings)
