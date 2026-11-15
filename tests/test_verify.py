import numpy as np
import pytest

from phido.core.schemas import (
    DesignerInput, DesignerOutput, DSLEdge, DSLNode, MotifSpec, PICTemplate, VerifyInput,
)
from phido.knowledge.motifs import expand_motif
from phido.knowledge.pdk import PDK, Cell
from phido.stages.designer import DesignerStage
from phido.stages.verify import models
from phido.stages.verify.models import MODELS, m_mzi_2x2, m_ring_double, m_ring_single
from phido.stages.verify.solver import Model, solve
from phido.stages.verify.stage import VerifyStage

WL = np.linspace(1.5, 1.6, 2001)


def P(m, i, o):
    return np.abs(m.S[:, m.ports.index(o), m.ports.index(i)]) ** 2


# ----- solver
def test_solver_rejects_bad_connections():
    m = models.m_straight(WL[:3], {})
    with pytest.raises(ValueError, match="unknown port"):
        solve({"a": m}, [(("a", "o1"), ("b", "o1"))])
    with pytest.raises(ValueError, match="connected twice"):
        solve({"a": m, "b": m, "c": m}, [(("a", "o2"), ("b", "o1")), (("a", "o2"), ("c", "o1"))])


def test_solver_two_waveguides_equal_one_long_one():
    wl = WL[:5]
    a, b, long_ = models._wg(wl, 30.0), models._wg(wl, 70.0), models._wg(wl, 100.0)
    refs, S = solve({"a": a, "b": b}, [(("a", "o2"), ("b", "o1"))])
    assert refs == [("a", "o1"), ("b", "o2")]
    assert np.allclose(S[:, 1, 0], long_.S[:, 1, 0])


def test_solver_matches_sax_when_installed():
    sax = pytest.importorskip("sax")
    import jax.numpy as jnp

    wl = WL[:7]
    cpl, top, bot = models._coupler(wl, 0.5), models._wg(wl, 100.0), models._wg(wl, 140.0)

    grid = wl

    def mk(model):
        def f(wl=1.55):  # SAX passes the wavelength only to a parameter literally named "wl"
            i = int(np.argmin(np.abs(grid - float(wl))))
            return {(a, b): jnp.asarray(model.S[i, model.ports.index(b), model.ports.index(a)])
                    for a in model.ports for b in model.ports}
        return f

    net = {"instances": {"c1": "cpl", "c2": "cpl", "top": "top", "bot": "bot"},
           "connections": {"c1,o3": "top,o1", "top,o2": "c2,o2", "c1,o4": "bot,o1", "bot,o2": "c2,o1"},
           "ports": {"o1": "c1,o1", "o2": "c1,o2", "o3": "c2,o3", "o4": "c2,o4"}}
    circ, _ = sax.circuit(net, {"cpl": mk(cpl), "top": mk(top), "bot": mk(bot)})
    ours = m_mzi_2x2(wl, {"delta_length": 40.0, "arm_length": 100.0})
    for k, w in enumerate(wl):
        S = circ(wl=float(w))
        for a in ("o1", "o2"):
            for b in ("o3", "o4"):
                assert abs(complex(S[(a, b)]) - ours.S[k, ours.ports.index(b), ours.ports.index(a)]) < 1e-9


# ----- physics of the models
def test_mzi_states_and_fsr():
    m = m_mzi_2x2(WL, {})
    assert P(m, "o1", "o3").mean() > 0.99 and P(m, "o1", "o4").max() < 1e-3  # cross state at dL=0
    pi = m_mzi_2x2(WL, {"phase": np.pi})
    assert P(pi, "o1", "o4")[1000] > 0.99 and P(pi, "o1", "o3")[1000] < 1e-3  # pi shift -> bar
    y = P(m_mzi_2x2(WL, {"delta_length": 100.0}), "o1", "o3")
    pk = WL[1:-1][(y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])]
    assert np.diff(pk).mean() * 1e3 == pytest.approx(1.55**2 / (4.2 * 100) * 1e3, rel=0.02)  # lam^2/(ng dL)


def test_rings_are_passive_and_show_resonances():
    rd = m_ring_double(WL, {"radius": 10.0})
    assert (P(rd, "o1", "o4") + P(rd, "o1", "o3")).max() <= 1 + 1e-9
    rs = m_ring_single(WL, {})
    assert P(rs, "o1", "o2").min() < 0.95 and P(rs, "o1", "o2").max() > 0.999


def test_every_pdk_model_name_exists(pdk):
    assert all(c.model in MODELS for c in pdk.cells.values())


# ----- stage
def design_for(pdk, template):
    return DesignerStage(pdk).run(DesignerInput(template=template))


def test_splitter_tree_loss_budget(pdk):
    t = expand_motif(MotifSpec(name="tree", args={"unit": "mmi", "stages": 3, "io_kind": "edge_coupler"}))
    v = VerifyStage(pdk).run(VerifyInput(design=design_for(pdk, t), points=5))
    assert v.ok and v.inputs == ["IN1,fiber"] and len(v.outputs) == 8 and not v.warnings
    expect = -10 * np.log10(8) - 3 * 0.3 - 2 * 1.5  # 1/8 split + three MMIs + two fibre couplers
    assert all(s.db_center == pytest.approx(expect, abs=0.01) for s in v.summary) and len(v.summary) == 8


def edge_mzi_edge(params=None):
    return PICTemplate(nodes=[{"id": "IN1", "kind": "edge_coupler"},
                              {"id": "N1", "kind": "mzi", "ports": "2x2", "params": params or {}},
                              {"id": "OUT1", "kind": "edge_coupler"}],
                       edges=[{"src": "IN1", "dst": "N1"}, {"src": "N1", "dst": "OUT1"}])


def test_verify_warns_when_light_leaves_through_unconnected_port(pdk):
    # hand-built bad wiring: top input -> top output, but a 2x2 MZI at its default setting is in the cross state
    bad = DesignerOutput(
        nodes={"IN1": DSLNode(component="edge_coupler"), "N1": DSLNode(component="mzi_2x2_heater"),
               "OUT1": DSLNode(component="edge_coupler")},
        edges={"E1": DSLEdge(link="IN1,o1:N1,o2"), "E2": DSLEdge(link="N1,o3:OUT1,o1")},
    )
    v = VerifyStage(pdk).run(VerifyInput(design=bad, points=5))
    text = " ".join(v.warnings)
    assert "leaves through unconnected port N1,o4" in text and "OUT1,fiber receives no light" in text


def test_unconnected_ports_are_data_not_a_warning(pdk):
    v = VerifyStage(pdk).run(VerifyInput(design=design_for(pdk, edge_mzi_edge()), points=5))
    assert v.unconnected == ["N1,o1", "N1,o3"]  # the MZI's two unused ports (it is entered at o2 and left at o4)
    assert not any("unconnected component ports" in w for w in v.warnings) and v.lit_outputs == ["OUT1,fiber"]


def test_verify_is_clean_after_flipping_the_mzi_with_a_pi_phase(pdk):
    v = VerifyStage(pdk).run(VerifyInput(design=design_for(pdk, edge_mzi_edge({"phase": 3.1416})), points=5))
    assert not any("OUT1" in w or "leaves through" in w for w in v.warnings)
    assert v.summary[0].dst == "OUT1,fiber" and v.summary[0].db_center == pytest.approx(-3.0 - 0.6 * 0 - 0.03, abs=0.2)


def test_detector_output_and_missing_model(pdk):
    t = PICTemplate(nodes=[{"id": "IN1", "kind": "edge_coupler"}, {"id": "PD1", "kind": "photodetector"}],
                    edges=[{"src": "IN1", "dst": "PD1"}])
    v = VerifyStage(pdk).run(VerifyInput(design=design_for(pdk, t), points=3))
    assert v.summary[0].dst == "PD1,pd" and v.summary[0].db_center == pytest.approx(-1.5, abs=0.01)
    bare = PDK([Cell(name="x", ports="1x1")])
    d = DesignerOutput(nodes={"A": DSLNode(component="x")})
    out = VerifyStage(bare).run(VerifyInput(design=d))
    assert [i.code for i in out.issues] == ["V"] and "no simulation model" in out.issues[0].message


def test_gain_is_flagged_as_non_passive(monkeypatch):
    monkeypatch.setitem(MODELS, "amp", lambda wl, p: models._two_port(2.0 * np.ones(len(wl), dtype=complex)))
    pdk = PDK([Cell(name="amp", ports="1x1", model="amp")])
    out = VerifyStage(pdk).run(VerifyInput(design=DesignerOutput(nodes={"A": DSLNode(component="amp")}), points=3))
    assert any("not passive" in i.message for i in out.issues)
