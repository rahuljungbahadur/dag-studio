"""Backend tests. The MCMC test is slow (~30 s); run `pytest -m "not slow"` to skip it."""

import polars as pl
import pytest

from dag_studio import backend as b


def test_edge_key_is_order_independent():
    assert b.edge_key("x", "a") == b.edge_key("a", "x") == "a~x"


def test_creates_cycle():
    edges = [b.Edge("a", "b"), b.Edge("b", "c")]
    assert b.creates_cycle(edges, "c", "a")
    assert not b.creates_cycle(edges, "a", "c")
    # Undirected edges never count towards a directed cycle.
    assert not b.creates_cycle([b.Edge("a", "b", False)], "b", "a")


def test_graph_problems():
    assert b.graph_problems(["a"], []) == ["The graph has no edges."]
    undirected = b.graph_problems(["a", "b"], [b.Edge("a", "b", False)])
    assert len(undirected) == 1 and "a–b" in undirected[0]
    cyclic = b.graph_problems(["a", "b"], [b.Edge("a", "b"), b.Edge("b", "a")])
    assert cyclic == ["The graph contains a directed cycle."]
    assert b.graph_problems(["a", "b"], [b.Edge("a", "b")]) == []


def test_to_dot_round_trip_through_pathmc():
    import pathmc

    dot = b.to_dot(["a", "b", "c", "lonely"], [b.Edge("a", "b"), b.Edge("b", "c")], ["node a"])
    assert '"lonely";' in dot and "// review: node a" in dot
    assert pathmc.dag_to_spec(dot).split("\n") == ["b ~ a", "c ~ b"]


def test_to_dot_carries_descriptions():
    import pathmc

    notes = {"a": 'Price in "USD"\nC:\\data -> b', "lonely": "unused", "b": "   "}
    dot = b.to_dot(["a", "b", "lonely"], [b.Edge("a", "b")], descriptions=notes)
    assert '"a" [tooltip="Price in \\"USD\\"\\nC:\\\\data -> b"];' in dot
    assert '"lonely" [tooltip="unused"];' in dot
    assert '"b" [' not in dot  # blank descriptions are dropped
    assert pathmc.dag_to_spec(dot) == "b ~ a"


def test_clean_table_handles_messy_input():
    raw = pl.DataFrame(
        {
            "unit price": [float(i) for i in range(30)],
            "2nd": [float(i % 7) for i in range(30)],
            "label": ["x"] * 30,
            "const": [1.0] * 30,
            "gappy": [None] + [float(i * i) for i in range(29)],
        }
    )
    df, notes = b.clean_table(raw)
    assert df.columns == ["unit_price", "v_2nd", "gappy"]
    assert df.height == 29
    assert len(notes) == 4


def test_bundled_test_file_is_current_and_exercises_cleaning():
    on_disk = pl.read_csv(b.TEST_FILE, infer_schema_length=10_000)
    assert on_disk.equals(b.test_file_table()), (
        "Regenerate with: b.test_file_table().write_csv(b.TEST_FILE)"
    )
    df, notes = b.load_table(b.TEST_FILE)
    assert "region" not in df.columns and "tax_rate" not in df.columns
    assert {"ad_spend", "v_2024_price_index", "web_visits"} <= set(df.columns)
    assert df.height == 585
    assert len(notes) == 4  # non-numeric, renamed, constant, missing rows


def test_clean_table_rejects_unusable_input():
    with pytest.raises(ValueError, match="two numeric"):
        b.clean_table(pl.DataFrame({"a": [1.0] * 30, "s": ["x"] * 30}))
    with pytest.raises(ValueError, match="20 complete rows"):
        b.clean_table(pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]}))


def test_layout_puts_causes_above_and_target_last():
    nodes = ["a", "b", "c", "y", "alone"]
    edges = [b.Edge("a", "b"), b.Edge("b", "y"), b.Edge("c", "y"), b.Edge("a", "c", False)]
    pos = b.layout(nodes, edges, target="y")
    assert set(pos) == set(nodes)
    assert pos["a"][1] < pos["b"][1] < pos["y"][1]
    assert pos["y"][1] == max(p[1] for p in pos.values())
    assert len({tuple(p) for p in pos.values()}) == len(nodes)  # no overlaps
    width, height = b.canvas_size(pos)
    assert all(0 < x < width and 0 < y < height for x, y in pos.values())


def test_edge_geometry():
    d, arrow = b.edge_geometry((300, 40), (300, 136), True)
    assert d.startswith("M ") and " Q " in d and len(arrow.split()) == 3
    # The path starts and ends outside the node boxes.
    start_y = float(d.split()[2])
    assert start_y > 40 + b.NODE_H / 2
    _, no_arrow = b.edge_geometry((300, 40), (300, 136), False)
    assert no_arrow == ""
    # With nothing in the way the edge is straight (control point on the chord).
    assert float(d.split()[4]) == 300.0
    # A node sitting on the chord forces a sideways bow around it.
    d_bowed, _ = b.edge_geometry((300, 40), (300, 232), True, obstacles=[(300, 136)])
    assert abs(float(d_bowed.split()[4]) - 300) > b.NODE_W / 2


def test_discover_recovers_demo_structure():
    df = b.demo_table()
    edges, evidence = b.discover(df, target="revenue")
    pairs = {e.key for e in edges}
    assert {"demand~revenue", "price~revenue", "demand~price", "demand~promo"} <= pairs
    assert not any("noise" in k for k in pairs)
    # Evidence exists for removed pairs too, and says why they were removed.
    removed = evidence[b.edge_key("noise", "revenue")]
    assert removed.independent and removed.bf10 < b.BF_THRESHOLD
    assert "BF₁₀" in removed.describe()
    # Restricting the node set drops the excluded column entirely.
    sub_edges, _ = b.discover(df, target="revenue", nodes=["price", "demand"])
    assert all("promo" not in e.key and "season" not in e.key for e in sub_edges)


@pytest.mark.slow
def test_fit_and_curve_recover_known_effect(monkeypatch):
    monkeypatch.setattr(b, "DRAWS", 200)
    monkeypatch.setattr(b, "TUNE", 200)
    df = b.demo_table()
    nodes = ["season", "price", "promo", "demand", "revenue"]
    edges = [
        b.Edge("season", "promo"),
        b.Edge("season", "demand"),
        b.Edge("promo", "demand"),
        b.Edge("price", "demand"),
        b.Edge("demand", "revenue"),
        b.Edge("price", "revenue"),
    ]
    model, spec = b.fit_model(df, nodes, edges)
    assert "revenue ~ demand + price" in spec

    curve = b.edge_curve(model, df, "price", "revenue")
    # Direct coefficient: true value 0.4.
    assert curve.coef_lower < 0.4 < curve.coef_upper
    # Total effect of price on revenue: 0.4 + (-0.8 * 0.9) = -0.32 per unit.
    slope = (curve.mean[-1] - curve.mean[0]) / (curve.x[-1] - curve.x[0])
    assert slope == pytest.approx(-0.32, abs=0.08)
    assert all(lo <= m <= hi for lo, m, hi in zip(curve.lower, curve.mean, curve.upper))

    fig = b.make_figure(curve, df)
    assert len(fig.data) == 3


def test_fit_refuses_invalid_graph():
    with pytest.raises(ValueError, match="Orient the dashed edges"):
        b.fit_model(b.demo_table(), ["price", "demand"], [b.Edge("price", "demand", False)])
