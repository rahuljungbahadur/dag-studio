"""Backend tests. The MCMC test is slow (~30 s); run `pytest -m "not slow"` to skip it."""

import re

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
    fit = b.fit_model(df, nodes, edges, "revenue", priors={"revenue": {"beta.sigma": "5"}})
    model, spec, code = fit.model, fit.spec, fit.code
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

    # A node with no edge into the outcome: E[revenue | do(season)] is the total effect through
    # promo and demand: (0.5 + 0.6 * 0.7) * 0.9 = 0.83 per unit.
    season = b.edge_curve(model, df, "season", "revenue")
    season_slope = (season.mean[-1] - season.mean[0]) / (season.x[-1] - season.x[0])
    assert season_slope == pytest.approx(0.83, abs=0.1)
    assert b.effect_unavailable(edges, "season", "revenue") == ""

    # Bayesian R² for a gaussian node matches ordinary least squares.
    quality = b.fit_quality(fit)
    r2 = {n.node: n for n in quality.nodes}
    assert set(r2) == {"promo", "demand", "revenue"}
    X = np.column_stack([np.ones(df.height), df["demand"].to_numpy(), df["price"].to_numpy()])
    y = df["revenue"].to_numpy()
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    assert r2["revenue"].r2 == pytest.approx(1 - resid.var() / y.var(), abs=0.02)
    assert r2["revenue"].lower < r2["revenue"].r2 < r2["revenue"].upper
    assert quality.divergences == 0 and quality.max_rhat < b.RHAT_BAD


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
    fit = b.fit_model(df, ["x", "m", "y"], edges, "y", families, {"y": {"beta.sigma": "2"}})
    model, code = fit.model, fit.code
    assert 'families={"y": "bernoulli"},' in code

    equations = b.model_equations(model, families)
    assert equations["y"].sigma is None and equations["m"].sigma is not None
    assert equations["y"].lines("x")[0][0].startswith("logit P(y = 1) = ")
    slope = next(t for t in equations["y"].terms if t.name == "x")
    assert slope.lower < 0.7 < slope.upper

    curve = b.edge_curve(model, df, "x", "y")
    assert all(0 <= v <= 1 for v in curve.mean)  # a probability
    assert curve.mean[-1] > curve.mean[0]

    quality = {n.node: n for n in b.fit_quality(fit).nodes}
    assert quality["y"].family == "bernoulli" and 0.05 < quality["y"].r2 < 0.6


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


def _fake_idata(**posterior):
    """Just enough of a DataTree for bayes_r2: one chain of each variable."""
    return {"posterior": {k: np.asarray(v)[None] for k, v in posterior.items()},
            "observed_data": {}}


def test_bayes_r2_by_family():
    mean = np.array([[0.0, 1.0, 2.0, 3.0]] * 2)  # 2 draws, 4 rows; var = 5/3
    gaussian = b.bayes_r2(_fake_idata(mu_y=mean, sigma_y=[1.0, 2.0]), "y")
    assert gaussian == pytest.approx([(5 / 3) / (5 / 3 + 1), (5 / 3) / (5 / 3 + 4)])

    p = np.array([[0.2, 0.8, 0.2, 0.8]])
    bern = b.bayes_r2(_fake_idata(pathmc_resp_y=p), "y", "bernoulli")
    assert bern == pytest.approx([0.12 / (0.12 + 0.16)])  # var(p) / (var(p) + mean p(1-p))

    lam = np.array([[1.0, 3.0]])
    pois = b.bayes_r2(_fake_idata(pathmc_resp_y=lam), "y", "poisson")
    assert pois == pytest.approx([2.0 / (2.0 + 2.0)])
    nb = b.bayes_r2(_fake_idata(pathmc_resp_y=lam, alpha_disp_y=[2.0]), "y", "negbinomial")
    assert nb == pytest.approx([2.0 / (2.0 + (2.0 + 5.0 / 2))])  # mean(mu + mu²/alpha)


def test_fit_quality_warning_levels():
    good = b.FitQuality([], max_rhat=1.004, min_ess=900, divergences=0)
    assert good.warnings() == []
    borderline = b.FitQuality([], max_rhat=1.02, min_ess=900, divergences=0).warnings()
    assert len(borderline) == 1 and borderline[0].startswith("Borderline")
    bad = b.FitQuality([], max_rhat=1.2, min_ess=50, divergences=3).warnings()
    assert len(bad) == 3 and bad[0].startswith("The chains disagree")


def test_next_selection():
    pick = b.next_selection
    assert pick([], "a", False) == ["a"]
    assert pick(["a"], "a", False) == []  # plain click on the selected node deselects
    assert pick(["a"], "b", False) == ["b"]  # plain click replaces the selection
    assert pick(["a", "b"], "c", False) == ["c"]
    # cmd-click builds up the pair in click order.
    assert pick(["a"], "b", True) == ["a", "b"]
    assert pick([], "a", True) == ["a"]
    # cmd-click on a selected node removes just that one.
    assert pick(["a", "b"], "a", True) == ["b"]
    # A third node pushes out the oldest, so the button never gets stuck.
    assert pick(["a", "b"], "c", True) == ["b", "c"]


def test_cycle_helpers():
    chain = [b.Edge("a", "b"), b.Edge("b", "c"), b.Edge("c", "d")]
    assert b.directed_path(chain, "a", "d") == ["a", "b", "c", "d"]
    assert b.directed_path(chain, "d", "a") == []
    assert b.directed_path(chain, "a", "a") == ["a"]
    # Undirected edges are not paths.
    assert b.directed_path([b.Edge("a", "b", False)], "a", "b") == []
    # The cycle a refused edge would close runs target -> ... -> source.
    assert b.cycle_closed_by(chain, "d", "a") == ["a", "b", "c", "d"]
    assert b.cycle_closed_by(chain, "a", "d") == []
    # A cycle that is already in the graph.
    assert b.find_cycle(chain) == []
    ring = b.find_cycle(chain + [b.Edge("d", "b")])
    assert sorted(ring) == ["b", "c", "d"]
    assert all(b.Edge(x, y) in chain + [b.Edge("d", "b")] for x, y in zip(ring, ring[1:]))


def test_effect_unavailable_explains_why_a_node_has_no_curve():
    edges = [b.Edge("a", "b"), b.Edge("b", "y"), b.Edge("y", "after"), b.Edge("a", "x", False)]
    assert b.effect_unavailable(edges, "a", "y") == ""  # through b
    assert b.effect_unavailable(edges, "b", "y") == ""
    assert "is the outcome" in b.effect_unavailable(edges, "y", "y")
    assert "no directed path to y" in b.effect_unavailable(edges, "after", "y")  # downstream
    assert "no directed path to y" in b.effect_unavailable(edges, "lonely", "y")
    assert "no directed path to y" in b.effect_unavailable(edges, "x", "y")  # only a dashed edge


class RecordingProgress(b.Progress):
    """A Progress that also keeps every update, to check what a task reported."""

    def __init__(self):
        super().__init__()
        self.log: list[tuple[str, float | None]] = []

    def set(self, label, fraction=None):
        super().set(label, fraction)
        self.log.append((label, self.snapshot()[1]))


def test_progress_snapshot_clamps_and_allows_unknown():
    progress = b.Progress()
    assert progress.snapshot() == ("", None)
    progress.set("Working", 0.4)
    assert progress.snapshot() == ("Working", 0.4)
    progress.set("Over", 1.7)
    assert progress.snapshot()[1] == 1.0
    progress.set("Under", -0.2)
    assert progress.snapshot()[1] == 0.0
    progress.set("Unknown")
    assert progress.snapshot() == ("Unknown", None)


def test_discover_reports_progress_without_changing_its_result():
    df = b.demo_table()
    progress = RecordingProgress()
    edges, evidence = b.discover(df, target="revenue", progress=progress)
    plain_edges, plain_evidence = b.discover(df, target="revenue")
    assert edges == plain_edges
    assert {k: (v.bf10, v.conditioning_set) for k, v in evidence.items()} == {
        k: (v.bf10, v.conditioning_set) for k, v in plain_evidence.items()
    }

    fractions = [f for _, f in progress.log if f is not None]
    assert fractions == sorted(fractions)  # never goes backwards
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    phases = []
    for label, _ in progress.log:
        name = label.split(" (")[0]
        if not phases or phases[-1] != name:
            phases.append(name)
    assert phases == [
        "Starting TBFPC",
        "Testing which variables drive the outcome",
        "Testing links between the other variables",
        "Reading the results",
    ]
    # 5 drivers -> 5 outcome passes then 10 driver-pair passes, each counted "n of total".
    assert any(label.endswith("(5 of 5)") for label, _ in progress.log)
    assert any(label.endswith("(10 of 10)") for label, _ in progress.log)


def test_fit_step_labels():
    assert b.fit_step(1) == "Step 1 of 4 · Compiling the model"
    assert b.fit_step(4) == "Step 4 of 4 · Computing equations and fit quality"


@pytest.mark.slow
def test_fit_reports_progress_and_the_hook_does_not_leak(monkeypatch):
    monkeypatch.setattr(b, "DRAWS", 200)
    monkeypatch.setattr(b, "TUNE", 200)
    df = b.demo_table(300)
    nodes, edges = ["price", "demand"], [b.Edge("price", "demand")]

    progress = RecordingProgress()
    b.fit_model(df, nodes, edges, "demand", progress=progress)
    labels = [label for label, _ in progress.log]
    assert labels[0] == b.fit_step(1) and progress.log[0][1] is None  # compiling: no count
    sampling = [f for label, f in progress.log if label == b.fit_step(2)]
    assert sampling and all(f is not None for f in sampling)
    assert sampling == sorted(sampling) and sampling[-1] < 1.0
    assert labels[-1] == b.fit_step(3)  # all draws done: finishing, no count
    assert b._sink.callback is None  # cleared once the fit is over

    # A later fit without progress must not write into the earlier fit's progress.
    n_before = len(progress.log)
    b.fit_model(df, nodes, edges, "demand")
    assert len(progress.log) == n_before

    # If PyMC's internals ever change, the fit still works and just reports step 1.
    monkeypatch.setattr(b, "_install_sampling_hook", lambda: False)
    fallback = RecordingProgress()
    fit = b.fit_model(df, nodes, edges, "demand", progress=fallback)
    assert [label for label, _ in fallback.log] == [b.fit_step(1)]
    assert fit.idata is not None


IMPORT_COLUMNS = ["ad_spend", "season", "web_visits", "orders", "revenue", "v_2024_price_index", "noise"]


def imported(text: str) -> b.ImportedDag:
    return b.parse_dag_text(text, IMPORT_COLUMNS)


def pairs(result: b.ImportedDag) -> set[tuple[str, str, bool]]:
    return {(e.source, e.target, e.directed) for e in result.edges}


def test_import_spec_syntax():
    result = imported("web_visits ~ ad_spend + season\norders ~ web_visits\nrevenue ~ orders")
    assert result.format == "spec" and result.notes == []
    assert pairs(result) == {
        ("ad_spend", "web_visits", True), ("season", "web_visits", True),
        ("web_visits", "orders", True), ("orders", "revenue", True),
    }


def test_import_spec_keeps_only_the_structure_and_says_so():
    result = imported(
        "revenue ~ adstock(orders, decay=t) + season:ad_spend + 2*noise\n"
        "web_visits ~ lag(web_visits) + lag(season)\n"
        "ad_spend ~~ season"
    )
    assert pairs(result) == {
        ("orders", "revenue", True), ("season", "revenue", True), ("ad_spend", "revenue", True),
        ("noise", "revenue", True), ("season", "web_visits", True),  # own past is not a parent
    }
    assert len(result.notes) == 2
    assert "Only the structure" in result.notes[0] and "residual covariance" in result.notes[1]


def test_import_dot_with_chains_comments_and_undirected_edges():
    result = imported(
        "// hand drawn\n/* v2 */\ndigraph G { rankdir=LR; ad_spend -> web_visits -> orders;\n"
        '  "v_2024_price_index" -> orders; season -- web_visits [style=dashed] }'
    )
    assert result.format == "DOT"
    assert pairs(result) == {
        ("ad_spend", "web_visits", True), ("web_visits", "orders", True),
        ("v_2024_price_index", "orders", True), ("season", "web_visits", False),
    }
    assert result.notes == ["1 undirected edge(s) were imported as dashed edges."]
    # A directed edge for the same pair wins over an undirected one.
    both = imported("digraph { orders -> revenue; orders -- revenue }")
    assert pairs(both) == {("orders", "revenue", True)}


def test_import_edge_list():
    result = imported("ad_spend->web_visits, web_visits->orders\norders->revenue;")
    assert result.format == "edge list"
    assert pairs(result) == {
        ("ad_spend", "web_visits", True), ("web_visits", "orders", True),
        ("orders", "revenue", True),
    }


def test_import_matches_names_the_way_an_upload_cleans_them():
    result = imported('digraph { "ad spend" -> "web visits"; "2024 price index" -> orders }')
    assert pairs(result) == {
        ("ad_spend", "web_visits", True), ("v_2024_price_index", "orders", True),
    }
    assert "Matched ad spend to the column ad_spend." in result.notes


def test_import_round_trips_the_apps_own_export():
    edges = [b.Edge("ad_spend", "web_visits"), b.Edge("web_visits", "orders"),
             b.Edge("season", "web_visits", False)]
    notes = {"orders": 'Orders "placed"\nincl. C:\\x -> y [see]', "noise": "unused", "season": "  "}
    dot = b.to_dot(IMPORT_COLUMNS, edges, ["node orders", "edge season -- web_visits"], notes)
    result = imported(dot)
    assert pairs(result) == {(e.source, e.target, e.directed) for e in edges}
    assert result.descriptions == {"orders": notes["orders"], "noise": "unused"}
    assert result.flagged_nodes == ["orders"]
    assert result.flagged_edges == ["season~web_visits"]


def test_import_ignores_tooltips_on_edges():
    result = imported('digraph { orders [tooltip="mine"]; ad_spend -> revenue [tooltip="not a node note"] }')
    assert result.descriptions == {"orders": "mine"}


@pytest.mark.parametrize(
    "text,message",
    [
        ("orders ~ web_visitz + price", "web_visitz (did you mean web_visits?), price. Nothing"),
        ("digraph { orders -> revenue; revenue -> web_visits; web_visits -> orders }",
         "cycle: revenue → web_visits → orders → revenue"),
        ("orders->revenue, revenue->orders", "contains a cycle"),
        ("orders ~ orders + season", "orders → orders is a self-loop"),
        ("digraph { orders }", "No edges found"),
        ("hello there", "Could not tell what format"),
        ("digraph { orders -> }", "Could not read the DOT"),
        ("orders ~ ", "Could not read the spec"),
        ("Q ~ 1", "Could not read the spec"),
    ],
)
def test_import_refuses_bad_input_with_a_reason(text, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        imported(text)


def test_import_refuses_oversized_text():
    with pytest.raises(ValueError, match="too large"):
        imported("a->b," * b.MAX_IMPORT_CHARS)
