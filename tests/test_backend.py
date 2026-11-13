"""Backend tests. The MCMC test is slow (~30 s); run `pytest -m "not slow"` to skip it."""

import numpy as np
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


def test_model_equations_from_summary():
    import pandas as pd

    rows = {
        "beta_unit_sales[Intercept]": (0.5, 0.4, 0.6),
        "beta_unit_sales[v_2024_price]": (-0.7, -0.8, -0.6),
        "beta_unit_sales[promo]": (0.25, 0.1, 0.4),
        "sigma_unit_sales": (1.1, 1.0, 1.2),
        "mu_unit_sales[0]": (0.3, 0.2, 0.4),
        "beta_promo[Intercept]": (0.0, -0.1, 0.1),
    }
    summary = pd.DataFrame(
        list(rows.values()), index=list(rows), columns=["mean", "eti89_lb", "eti89_ub"]
    )

    class FakeModel:
        def summary(self):
            return summary

    equations = b.model_equations(FakeModel())
    assert set(equations) == {"unit_sales", "promo"}
    eq = equations["unit_sales"]
    assert [t.name for t in eq.terms] == ["v_2024_price", "promo"]
    assert eq.interval == "89%" and equations["promo"].sigma is None

    lines = eq.lines(highlight="promo")
    assert lines == [
        ("unit_sales = 0.500", "[0.400, 0.600]", False),
        ("− 0.700 · v_2024_price", "[-0.800, -0.600]", False),
        ("+ 0.250 · promo", "[0.100, 0.400]", True),
        ("+ ε,  ε ~ Normal(0, 1.100)", "[1.000, 1.200]", False),
    ]


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
    model, spec, code = b.fit_model(
        df, nodes, edges, "revenue", priors={"revenue": {"beta.sigma": "5"}}
    )
    assert code.startswith("builder = BuildModelFromDAG(") and 'target="revenue"' in code
    assert '"beta_revenue": Prior("Normal", mu=0, sigma=5),' in code
    assert "revenue ~ demand + price" in spec

    equations = b.model_equations(model)
    assert set(equations) == {"promo", "demand", "revenue"}
    direct = next(t for t in equations["revenue"].terms if t.name == "price")
    # Direct coefficient: true value 0.4.
    assert direct.lower < 0.4 < direct.upper
    assert equations["revenue"].sigma.mean == pytest.approx(0.8, abs=0.15)

    curve = b.edge_curve(model, df, "price", "revenue")
    # Total effect of price on revenue: 0.4 + (-0.8 * 0.9) = -0.32 per unit.
    slope = (curve.mean[-1] - curve.mean[0]) / (curve.x[-1] - curve.x[0])
    assert slope == pytest.approx(-0.32, abs=0.08)
    assert all(lo <= m <= hi for lo, m, hi in zip(curve.lower, curve.mean, curve.upper))

    fig = b.make_figure(curve, df)
    assert len(fig.data) == 3


def test_fit_refuses_invalid_graph():
    df = b.demo_table()
    with pytest.raises(ValueError, match="Orient the dashed edges"):
        b.fit_model(df, ["price", "demand"], [b.Edge("price", "demand", False)], "demand")
    with pytest.raises(ValueError, match="outcome revenue has no edges"):
        b.fit_model(df, ["price", "demand", "revenue"], [b.Edge("price", "demand")], "revenue")
    with pytest.raises(ValueError, match="only 0 and 1"):
        b.fit_model(
            df, ["price", "demand"], [b.Edge("price", "demand")], "demand",
            families={"demand": "bernoulli"},
        )


def test_prior_params_follow_the_family():
    keys = lambda family: [p.key for p in b.prior_params(family)]  # noqa: E731
    assert keys("gaussian") == ["beta.mu", "beta.sigma", "sigma.sigma"]
    assert keys("studentt") == ["beta.mu", "beta.sigma", "sigma.sigma", "nu.alpha", "nu.beta"]
    assert keys("bernoulli") == keys("poisson") == ["beta.mu", "beta.sigma"]
    assert keys("negbinomial") == ["beta.mu", "beta.sigma", "alpha_disp.sigma"]


def test_prior_specs_fill_defaults_and_skip_root_nodes():
    edges = [b.Edge("x", "m"), b.Edge("m", "y"), b.Edge("x", "y")]
    typed = {
        "y": {"beta.sigma": " 2 ", "sigma.sigma": ""},
        "x": {"beta.sigma": "3"},  # x has no parents, so pathmc would reject this
        "m": {"nu.alpha": "4"},  # m is gaussian, so it has no nu prior
    }
    assert b.prior_specs(edges, {}, typed) == {"beta_y": ("Normal", {"mu": 0.0, "sigma": 2.0})}
    specs = b.prior_specs(edges, {"m": "studentt"}, typed)
    assert specs["nu_m"] == ("Gamma", {"alpha": 4.0, "beta": 0.1})
    with pytest.raises(ValueError, match="must be a number"):
        b.prior_specs(edges, {}, {"y": {"beta.mu": "abc"}})
    with pytest.raises(ValueError, match="must be a positive number"):
        b.prior_specs(edges, {}, {"y": {"sigma.sigma": "-1"}})


def test_family_problems():
    df = pl.DataFrame({"flag": [0.0, 1.0, 1.0], "count": [0.0, 2.0, 5.0], "z": [0.5, -1.0, 2.0]})
    ok = {"flag": "bernoulli", "count": "poisson"}
    assert b.family_problems(df, ok) == []
    bad = b.family_problems(df, {"z": "bernoulli", "flag": "negbinomial", "count": "studentt"})
    assert len(bad) == 1 and "z must contain only 0 and 1" in bad[0]
    assert b.family_problems(df, {"z": "poisson"})[0].startswith("z must contain non-negative")


def test_builder_code():
    code = b.builder_code(
        "X->M, M->Y, X->Y", "Y", {"Y": "bernoulli"}, {"beta_Y": ("Normal", {"mu": 0.0, "sigma": 2.5})}
    )
    assert code == "\n".join([
        "builder = BuildModelFromDAG(",
        '    dag="X->M, M->Y, X->Y",',
        "    df=data,",
        '    target="Y",',
        '    style="native",',
        '    families={"Y": "bernoulli"},',
        "    priors={",
        '        "beta_Y": Prior("Normal", mu=0, sigma=2.5),',
        "    },",
        ")",
    ])


def test_equation_lines_for_other_families():
    t = lambda name, mean: b.Term(name, mean, mean - 0.1, mean + 0.1)  # noqa: E731
    logistic = b.Equation("y", t("Intercept", 0.1), [t("x", 0.6)], None, "89%", "bernoulli")
    assert [r[0] for r in logistic.lines("x")] == ["logit P(y = 1) = 0.100", "+ 0.600 · x"]
    robust = b.Equation(
        "y", t("Intercept", 0.0), [t("x", -0.5)], t("sigma", 1.0), "89%", "studentt",
        nu=t("nu", 4.0),
    )
    assert [r[0] for r in robust.lines()][2:] == [
        "+ ε,  ε ~ StudentT(ν, 0, σ)", "    σ = 1.000", "    ν = 4.000",
    ]
    counts = b.Equation(
        "n", t("Intercept", 1.0), [t("x", 0.2)], None, "89%", "negbinomial", alpha=t("a", 2.0)
    )
    assert counts.lines()[0][0] == "log E[n] = 1.000"
    assert counts.lines()[-1][0] == "n ~ NegBinomial(E[n], α = 2.000)"


@pytest.mark.slow
def test_bernoulli_fit_through_builder(monkeypatch):
    monkeypatch.setattr(b, "DRAWS", 200)
    monkeypatch.setattr(b, "TUNE", 200)
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    m = 0.8 * x + rng.normal(size=500)
    y = (rng.random(500) < 1 / (1 + np.exp(-(0.5 * m + 0.7 * x)))).astype(float)
    df = pl.DataFrame({"x": x, "m": m, "y": y})
    edges = [b.Edge("x", "m"), b.Edge("m", "y"), b.Edge("x", "y")]
    families = {"y": "bernoulli"}
    model, _, code = b.fit_model(
        df, ["x", "m", "y"], edges, "y", families, {"y": {"beta.sigma": "2"}}
    )
    assert 'families={"y": "bernoulli"},' in code

    equations = b.model_equations(model, families)
    assert equations["y"].sigma is None and equations["m"].sigma is not None
    assert equations["y"].lines("x")[0][0].startswith("logit P(y = 1) = ")
    slope = next(t for t in equations["y"].terms if t.name == "x")
    assert slope.lower < 0.7 < slope.upper

    curve = b.edge_curve(model, df, "x", "y")
    assert all(0 <= v <= 1 for v in curve.mean)  # a probability
    assert curve.mean[-1] > curve.mean[0]


def _test_file_graph():
    df, _ = b.load_table(b.TEST_FILE)
    edges, _ = b.discover(df, target="revenue")
    return df, [b.Edge(e.source, e.target) for e in edges]  # orient dashed edges as drawn


def test_validate_flags_the_missing_edge_and_clears_once_added():
    df, edges = _test_file_graph()
    result = b.validate_graph(df, df.columns, edges)
    assert result.tone == "good" and result.n_tests > 20
    strongest = result.violations[0]
    assert strongest.key == b.edge_key("revenue", "v_2024_price_index")
    assert strongest.given == ["orders"] and strongest.p_value < 1e-10
    assert [v.p_value for v in result.violations] == sorted(v.p_value for v in result.violations)

    fixed = b.validate_graph(df, df.columns, edges + [b.Edge("v_2024_price_index", "revenue")])
    assert strongest.key not in {v.key for v in fixed.violations}
    # What's left is no more than chance would produce.
    assert len(fixed.violations) <= 2 * fixed.expected_by_chance


def test_validate_tests_unconnected_nodes():
    df, edges = _test_file_graph()
    connected = {e.source for e in edges} | {e.target for e in edges}
    assert "noise" not in connected
    with_noise = b.validate_graph(df, df.columns, edges)
    without = b.validate_graph(df, [c for c in df.columns if c != "noise"], edges)
    # One statement per other node: noise ⊥ y | parents(y).
    assert with_noise.n_tests - without.n_tests == len(df.columns) - 1

    empty = b.validate_graph(df, ["orders", "revenue", "noise"], [])
    assert empty.tone == "neutral" and empty.n_tests == 3
    assert [v.key for v in empty.violations] == [b.edge_key("orders", "revenue")]


def test_validate_refuses_unfinished_graphs():
    df = b.demo_table()
    with pytest.raises(ValueError, match="dashed"):
        b.validate_graph(df, ["price", "demand"], [b.Edge("price", "demand", False)])
    with pytest.raises(ValueError, match="cycle"):
        b.validate_graph(
            df, ["price", "demand"], [b.Edge("price", "demand"), b.Edge("demand", "price")]
        )


def test_edge_effects_sign_and_uncertainty():
    t = lambda name, mean, lo, hi: b.Term(name, mean, lo, hi)  # noqa: E731
    equations = {
        "demand": b.Equation(
            "demand", t("Intercept", 0, -1, 1),
            [t("price", -0.7, -0.8, -0.6), t("promo", 0.05, -0.1, 0.2)], None, "89%",
        ),
        "revenue": b.Equation(
            "revenue", t("Intercept", 0, -1, 1),
            [t("demand", 0.85, 0.8, 0.9), t("price", -0.02, -0.1, 0.06)], None, "89%",
        ),
    }
    edges = [
        b.Edge("price", "demand"), b.Edge("promo", "demand"), b.Edge("demand", "revenue"),
        b.Edge("price", "revenue"), b.Edge("season", "promo"),  # no equation for promo
        b.Edge("demand", "x", directed=False),
    ]
    signs = {k: sign for k, (sign, _) in b.edge_effects(equations, edges).items()}
    assert signs == {
        b.edge_key("price", "demand"): "neg",
        b.edge_key("promo", "demand"): "pos_weak",
        b.edge_key("demand", "revenue"): "pos",
        b.edge_key("price", "revenue"): "neg_weak",
    }
    assert set(signs.values()) <= set(b.EFFECT_COLOURS)
    assert b.EFFECT_COLOURS["pos"] == "#008FD5" and b.EFFECT_COLOURS["neg"] == "#FC4F30"
