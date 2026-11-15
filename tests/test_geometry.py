import pytest

from phido.core.schemas import Connect, TemplateEdge
from phido.geometry.placement import (
    assign_layers, boxes_overlap, count_crossings, order_layers, place, segments_cross, segments_overlap,
)
from phido.stages.designer import CircuitGraph
from phido.stages.designer.crossings import estimated_crossings, repair_crossings


def test_layers_follow_edges_and_survive_cycles():
    assert assign_layers(list("abcd"), [("a", "b"), ("b", "c"), ("a", "d")]) == {"a": 0, "b": 1, "c": 2, "d": 1}
    cyc = assign_layers(list("ab"), [("a", "b"), ("b", "a")])  # cycle must not hang
    assert cyc["a"] == 0 and cyc["b"] == 1


def test_ordering_untangles_crossing_layer():
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "d"), ("b", "c")]  # naive order crosses
    layers = order_layers(nodes, edges, assign_layers(nodes, edges))
    top = layers[0]
    assert top.index("a") < top.index("b") and layers[1].index("d") < layers[1].index("c") or \
        top.index("b") < top.index("a") and layers[1].index("c") < layers[1].index("d")


def test_place_is_left_to_right_and_non_overlapping():
    sizes = {n: (40.0, 10.0) for n in "abc"}
    c = place(list("abc"), [("a", "b"), ("a", "c")], sizes)
    assert c["a"][0] < c["b"][0] == c["c"][0]
    assert c["b"][1] > c["c"][1]  # first node of a layer is on top
    box = lambda n: (c[n][0] - 20, c[n][1] - 5, c[n][0] + 20, c[n][1] + 5)
    assert not boxes_overlap(box("b"), box("c"))


def test_segment_crossing_rules():
    assert segments_cross((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_cross((0, 0), (1, 1), (1, 1), (2, 0))  # sharing an endpoint is not a crossing
    assert count_crossings([((0, 0), (2, 2)), ((0, 2), (2, 0)), ((5, 5), (6, 6))]) == 1


def test_port_alignment_straightens_single_wires():
    sizes = {"a": (40.0, 10.0), "b": (40.0, 10.0)}
    c = place(["a", "b"], [("a", "b")], sizes, port_dy=[(0.6, -0.2)])
    assert c["b"][1] - c["a"][1] == pytest.approx(0.8)  # b's input lands exactly at a's output height
    assert place(["a", "b"], [("a", "b")], sizes)["b"][1] == pytest.approx(c["a"][1], abs=1e-9)  # default: same-height centres


def test_port_alignment_fanout_is_centred_and_separated():
    sizes = {n: (40.0, 10.0) for n in "abc"}
    c = place(list("abc"), [("a", "b"), ("a", "c")], sizes, gap_y=40.0, port_dy=[(0.6, 0.0), (-0.6, 0.0)])
    assert c["b"][1] > c["a"][1] > c["c"][1]  # order kept: b above, c below
    assert c["b"][1] - c["c"][1] == pytest.approx(50.0)  # height 10 + gap 40: minimum separation, no overlap
    assert (c["b"][1] + c["c"][1]) / 2 == pytest.approx(c["a"][1], abs=0.01)  # centred on the parent


def test_long_edge_gets_its_own_track():
    sizes = {n: (40.0, 10.0) for n in "abc"}
    edges = [("a", "b"), ("b", "c"), ("a", "c")]  # a -> c skips b's layer
    c = place(list("abc"), edges, sizes)
    assert set(c) == {"a", "b", "c"}  # the stand-in node is not returned
    assert abs(c["b"][1] - c["a"][1]) > sizes["b"][1] / 2  # b is pushed off the straight a->c wire


def test_collinear_overlap_is_detected_but_touching_is_not():
    assert segments_overlap((0, 0), (0, 10), (0, 5), (0, 20))  # shared stretch of the same vertical line
    assert not segments_overlap((0, 0), (0, 10), (0, 10), (0, 20))  # only touching
    assert not segments_overlap((0, 0), (0, 10), (1, 0), (1, 10))  # parallel but apart
    assert not segments_overlap((0, 0), (10, 0), (5, -5), (5, 5))  # perpendicular


def test_port_swap_repairs_a_crossing(pdk):
    cells = {"A": pdk.cell("mmi_1x2"), "B": pdk.cell("straight"), "C": pdk.cell("straight")}
    g = CircuitGraph(cells, [TemplateEdge(src="A", dst="B"), TemplateEdge(src="A", dst="C")])
    # B is placed above C, but A's TOP port (o2) is sent to C and its bottom port (o3) to B -> crossing
    g.apply(Connect(src_node="A", src_port="o2", dst_node="C", dst_port="o1"))
    g.apply(Connect(src_node="A", src_port="o3", dst_node="B", dst_port="o1"))
    assert estimated_crossings(g) == 1
    assert repair_crossings(g) == 0
    assert estimated_crossings(g) == 0
    assert sorted(c.src_port for c in g.connections) == ["o2", "o3"]  # same ports, just reassigned
