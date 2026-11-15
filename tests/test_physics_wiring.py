import numpy as np
import pytest

from phido.core.schemas import DesignerInput, PICTemplate, VerifyInput
from phido.pipeline import build_designer, load_pdk
from phido.stages.designer.throughput import decisive_pair, directivity
from phido.stages.verify.stage import VerifyStage


def tpl(nodes, edges=()):
    return PICTemplate(nodes=nodes, edges=[{"src": a, "dst": b} for a, b in edges])


def chain(n=4, mzi=None):
    nodes = [{"id": "IN1", "kind": "edge_coupler"}] + [{"id": f"M{i}", "kind": "mzi", **(mzi or {})} for i in range(1, n + 1)] \
        + [{"id": "PD1", "kind": "photodetector"}]
    ids = [x["id"] for x in nodes]
    return tpl(nodes, list(zip(ids, ids[1:])))


def run(template):
    d = build_designer().run(DesignerInput(template=template))
    return d, VerifyStage(load_pdk()).run(VerifyInput(design=d, points=5))


def test_directivity_distinguishes_switches_from_splitters():
    pdk = load_pdk()
    assert directivity(pdk.cell("mzi_2x2_heater"), {}) > 0.99  # cross state: all light to one output
    assert directivity(pdk.cell("mzi_1x2_heater"), {}) == pytest.approx(0.5, abs=0.01)  # 50/50 at default
    assert directivity(pdk.cell("mmi_1x2"), {}) == pytest.approx(0.5, abs=0.01)
    assert directivity(pdk.cell("straight"), {}) == pytest.approx(1.0)


def test_through_pair_follows_the_default_state():
    pdk = load_pdk()
    mzi = pdk.cell("mzi_2x2_heater")
    assert decisive_pair(mzi, {}, ["o1", "o2"], ["o3", "o4"]) in {("o1", "o3"), ("o2", "o4")}  # cross
    assert decisive_pair(mzi, {"phase": np.pi}, ["o1", "o2"], ["o3", "o4"]) in {("o1", "o4"), ("o2", "o3")}  # bar
    assert decisive_pair(pdk.cell("mmi_2x2"), {}, ["o1", "o2"], ["o3", "o4"]) is None  # 50/50: no clear path


def test_unspecified_mzi_chain_is_wired_without_leaks():
    d, v = run(chain(4))
    assert d.ok and all(d.nodes[f"M{i}"].component.startswith("mzi_2x2") for i in range(1, 5))  # not a 50/50 1x2
    assert not any("leaves through" in w for w in v.warnings), v.warnings
    assert v.summary[-1].dst == "PD1,pd" and v.summary[-1].db_center == pytest.approx(-1.5, abs=0.2)  # one fibre coupler + tiny wg loss


def test_through_path_adapts_to_the_phase_setting():
    for phase in (0.0, 3.1416):  # cross, then bar
        d, v = run(chain(2, {"ports": "2x2", "params": {"phase": phase}}))
        assert not any("leaves through" in w for w in v.warnings), (phase, v.warnings)
        assert v.summary[-1].db_center > -2.0


def test_ambiguous_choice_is_reported_once():
    d, _ = run(chain(4))
    msgs = [w for w in d.warnings if "matched several cells" in w]
    assert len(msgs) == 1 and "Name a technology or port count" in msgs[0]  # four identical nodes -> one message


def test_explicit_description_still_overrides_the_tiebreak():
    # "tunable splitter" is a request for a 1x2 splitting MZI: intent beats the clean-pass preference
    t = tpl([{"id": "IN1", "kind": "edge_coupler"}, {"id": "M", "kind": "mzi", "ports": "1x2", "description": "tunable splitter"},
             {"id": "PD", "kind": "photodetector"}], [("IN1", "M"), ("M", "PD")])
    d, _ = run(t)
    assert d.nodes["M"].component.startswith("mzi_1x2")


class PositionOnly:
    """Stand-in for an LLM that only sees port positions: top input -> top output everywhere."""

    def propose(self, graph, pending, feedback=()):
        from phido.core.schemas import Connect

        return [Connect(src_node=e.src, src_port=graph.free_ports(e.src, "out")[0].name,
                        dst_node=e.dst, dst_port=graph.free_ports(e.dst, "in")[0].name) for _, e in pending]


def test_proposer_that_ignores_physics_is_corrected_for_chain_elements():
    from phido.stages.designer import DesignerStage

    template = chain(2, {"ports": "2x2"})
    d = DesignerStage(load_pdk(), proposer=PositionOnly()).run(DesignerInput(template=template))
    v = VerifyStage(load_pdk()).run(VerifyInput(design=d, points=3))
    assert d.ok and not any("leaves through" in w for w in v.warnings), v.warnings
    assert v.summary[-1].dst == "PD1,pd" and v.summary[-1].db_center > -2.5  # light actually arrives


def test_correction_never_breaks_a_valid_wiring():
    from phido.stages.designer import DesignerStage

    # a splitter feeding two chain MZIs: only the chain elements are touched, the tree stays valid
    t = tpl([{"id": "S", "kind": "mmi", "ports": "1x2"}, {"id": "A", "kind": "mzi", "ports": "2x2"},
             {"id": "B", "kind": "mzi", "ports": "2x2"}, {"id": "PA", "kind": "photodetector"},
             {"id": "PB", "kind": "photodetector"}], [("S", "A"), ("S", "B"), ("A", "PA"), ("B", "PB")])
    d = DesignerStage(load_pdk(), proposer=PositionOnly()).run(DesignerInput(template=t))
    assert d.ok and len(d.edges) == 4
    used = [p for e in d.edges.values() for p in (e.parse()[:2], e.parse()[2:])]
    assert len(used) == len(set(used))  # no port used twice
