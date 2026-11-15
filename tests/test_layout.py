import pytest

from phido.core.schemas import (
    DesignerInput, DesignerOutput, DSLEdge, DSLNode, LayoutInput, MotifSpec, PICTemplate,
)
from phido.knowledge.motifs import expand_motif
from phido.stages.designer import DesignerStage
from phido.stages.layout import stage as layout_stage
from phido.stages.layout.stage import LayoutStage


def design_for(pdk, template):
    return DesignerStage(pdk).run(DesignerInput(template=template))


def test_without_gdsfactory_falls_back_to_estimate(pdk, monkeypatch):
    monkeypatch.setattr(layout_stage, "gf_available", lambda: False)
    t = PICTemplate(nodes=[{"id": "A", "kind": "mmi", "ports": "1x2"}, {"id": "B", "kind": "mmi", "ports": "1x2"}],
                    edges=[{"src": "A", "dst": "B"}])
    out = LayoutStage(pdk).run(LayoutInput(design=design_for(pdk, t)))
    assert out.gds_path is None and out.ok and all(p.backend == "estimate" for p in out.placements)
    assert any("gdsfactory is not installed" in w for w in out.warnings)


@pytest.fixture(scope="module")
def gf():
    return pytest.importorskip("gdsfactory")


def test_tree_layout_routes_every_link_and_writes_gds(pdk, gf, tmp_path):
    t = expand_motif(MotifSpec(name="tree", args={"unit": "mmi", "stages": 3, "io_kind": "edge_coupler"}))
    design = design_for(pdk, t)
    out = LayoutStage(pdk, out_dir=tmp_path).run(LayoutInput(design=design))
    assert out.ok, out.issues
    assert len(out.routes) == 15 and all(r.ok and r.length_um > 0 for r in out.routes)
    assert out.gds_path and (tmp_path / out.gds_path.split("/")[-1]).stat().st_size > 1000
    rot = {p.node: p.rotation for p in out.placements}
    assert rot["IN1"] == 180 and rot["OUT1"] == 0  # input coupler is flipped so its waveguide faces right


def test_cell_without_gdsfactory_equivalent_becomes_black_box(pdk, gf):
    t = PICTemplate(nodes=[{"id": "Y", "kind": "y_branch"}, {"id": "R", "kind": "ring", "ports": "1x1"}],
                    edges=[{"src": "Y", "dst": "R"}])
    out = LayoutStage(pdk).run(LayoutInput(design=design_for(pdk, t)))
    assert {p.node: p.backend for p in out.placements} == {"Y": "blackbox", "R": "gdsfactory"}
    assert any("black box" in w for w in out.warnings) and out.ok


def test_crossing_routes_are_reported(pdk, gf):
    # IN1 sits above IN2 but is wired to the LOWER input port of the MMI -> the two routes must cross
    design = DesignerOutput(
        nodes={"IN1": DSLNode(component="edge_coupler"), "IN2": DSLNode(component="edge_coupler"),
               "M1": DSLNode(component="mmi_2x2")},
        edges={"E1": DSLEdge(link="IN1,o1:M1,o1"), "E2": DSLEdge(link="IN2,o1:M1,o2")},
    )
    out = LayoutStage(pdk).run(LayoutInput(design=design))
    assert any(("cross" in i.message or "overlap" in i.message) and "routes" in i.message for i in out.issues)


def test_user_params_reach_the_gdsfactory_component(pdk, gf):
    def width(params):
        d = DesignerOutput(nodes={"S": DSLNode(component="straight", params=params)})
        return LayoutStage(pdk).run(LayoutInput(design=d)).placements[0].width

    assert width({"length": 200.0}) > width({"length": 20.0}) + 100
