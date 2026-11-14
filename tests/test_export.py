"""The exported script: what it contains, and that it reproduces the app's numbers."""

import ast
import builtins
import csv
import json
import re
import shutil
import subprocess
import sys

import polars as pl
import pytest

from dag_studio import backend as b
from dag_studio import export
from dag_studio.state import State

DEMO_NODES = ["season", "price", "promo", "demand", "revenue", "noise"]
DEMO_EDGES = [
    b.Edge("season", "promo"), b.Edge("season", "demand"), b.Edge("promo", "demand"),
    b.Edge("price", "demand"), b.Edge("demand", "revenue"), b.Edge("price", "revenue"),
]


def demo_script(**overrides) -> str:
    df = b.demo_table()
    plan = b.prepare_fit(df, DEMO_NODES, DEMO_EDGES, "revenue",
                         priors={"revenue": {"beta.sigma": "5"}})
    settings = dict(
        nodes=DEMO_NODES, outcome="revenue", plan=plan, data_kind="demo", data_file="",
        group_cols=[], time_col="", alpha=0.05, generated_at="2026-01-02 03:04",
    )
    return export.build_script(**{**settings, **overrides})


def undefined_names(source: str) -> set[str]:
    """Names the script uses but never defines. Not scope-aware: it catches a missing helper."""
    tree = ast.parse(source)
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loaded - bound


# --------------------------------------------------------------------------
# What the fit takes from the app's settings
# --------------------------------------------------------------------------


def test_prepare_fit_sorts_the_edges_so_a_graph_always_fits_the_same_way():
    df = b.demo_table()
    shuffled = [DEMO_EDGES[i] for i in (4, 1, 5, 0, 3, 2)]
    a = b.prepare_fit(df, DEMO_NODES, DEMO_EDGES, "revenue")
    c = b.prepare_fit(df, DEMO_NODES, shuffled, "revenue")
    assert a.dag == c.dag == (
        "demand->revenue, price->demand, price->revenue, promo->demand, season->demand, "
        "season->promo"
    )
    assert a.pairs == c.pairs and a.used == ["demand", "price", "promo", "revenue", "season"]


def test_prepare_fit_keeps_only_what_the_fit_will_use():
    df = b.demo_table().with_columns((pl.col("promo") > 0).cast(pl.Float64).alias("flag"))
    edges = DEMO_EDGES + [b.Edge("price", "flag")]
    plan = b.prepare_fit(
        df, DEMO_NODES + ["flag"], edges, "revenue",
        families={"flag": "bernoulli", "season": "poisson", "demand": "gaussian"},
        priors={"flag": {"beta.sigma": "2"}, "season": {"beta.sigma": "9"}},
    )
    # season has no parents, and gaussian is the default: neither is kept.
    assert plan.families == {"flag": "bernoulli"}
    assert plan.specs == {"beta_flag": ("Normal", {"mu": 0.0, "sigma": 2.0})}


def test_prepare_fit_refuses_what_cannot_be_fitted():
    df = b.demo_table()
    with pytest.raises(ValueError, match="Orient the dashed edges"):
        b.prepare_fit(df, DEMO_NODES, [b.Edge("price", "demand", False)], "demand")
    with pytest.raises(ValueError, match="once group means are removed"):
        b.prepare_fit(df, DEMO_NODES, DEMO_EDGES, "revenue", {"demand": "bernoulli"}, grouped=True)
    with pytest.raises(ValueError, match="must be a number"):
        b.prepare_fit(df, DEMO_NODES, DEMO_EDGES, "revenue", priors={"demand": {"beta.mu": "x"}})


# --------------------------------------------------------------------------
# The script's text
# --------------------------------------------------------------------------


def test_script_is_valid_python_that_defines_everything_it_uses():
    script = demo_script()
    ast.parse(script)
    assert undefined_names(script) == set()
    assert "@@" not in script


def test_script_is_standalone():
    script = demo_script()
    assert not re.search(r"^\s*(from|import)\s+dag_studio", script, re.MULTILINE)
    assert "from pathmc import BuildModelFromDAG" in script


def test_script_names_its_packages_and_the_dag_as_a_digraph():
    script = demo_script()
    pins = re.search(r"pip install (.+)", script).group(1).split()
    assert any(p.startswith("pathmc==") for p in pins) and any(p.startswith("pymc==") for p in pins)
    assert 'DAG = """digraph {\n  "demand" -> "revenue";' in script
    assert '  "season" -> "promo";\n}"""' in script  # sorted, one edge per line


def test_script_shows_the_fit_call_and_the_settings():
    script = demo_script()
    assert "DRAWS, TUNE, CHAINS, SEED = 500, 500, 4, 0" in script
    assert 'OUTCOME = "revenue"' in script and "ALPHA = 0.05" in script
    assert 'PRIORS = {\n    "beta_revenue": Prior("Normal", mu=0, sigma=5),\n}' in script
    assert "FAMILIES = {}" in script
    call = script[script.index("builder = BuildModelFromDAG("):]
    call = call[: call.index(")\n") + 2]
    for part in ("dag=graph", "target=OUTCOME", 'style="native"', "families=FAMILIES or None",
                 "priors=PRIORS or None"):
        assert part in call
    assert "model.fit(draws=DRAWS, tune=TUNE, chains=CHAINS, progressbar=False, random_seed=SEED)" in script


def test_script_data_section_depends_on_where_the_data_came_from():
    demo = demo_script()
    assert "def demo_table" in demo and "DATA_FILE" not in demo
    assert "the built-in demo data" in demo
    file_script = demo_script(data_kind="file", data_file="sales 2026.csv")
    assert 'DATA_FILE = "sales 2026.csv"' in file_script and "def demo_table" not in file_script
    assert "sales 2026.csv (put it in the same folder as this script)" in file_script
    assert undefined_names(file_script) == set()


def test_script_survives_awkward_file_names_and_grouping_settings():
    script = demo_script(
        data_kind="file", data_file='my "odd" file.csv', group_cols=["region", "store"],
        time_col="week", nodes=[f"variable_{i}" for i in range(12)],
    )
    tree = ast.parse(script)
    assigned = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert ast.literal_eval(assigned["DATA_FILE"]) == 'my "odd" file.csv'
    assert ast.literal_eval(assigned["GROUP_COLUMNS"]) == ["region", "store"]
    assert ast.literal_eval(assigned["TIME_COLUMN"]) == "week"
    assert ast.literal_eval(assigned["NODES"]) == [f"variable_{i}" for i in range(12)]


# --------------------------------------------------------------------------
# The export button, through the state
# --------------------------------------------------------------------------


def demo_state() -> State:
    st = State(_reflex_internal_init=True)
    State.load_demo.fn(st)
    st.edges = [{"source": e.source, "target": e.target, "directed": True} for e in DEMO_EDGES]
    st._refresh()
    return st


def test_export_code_in_the_state_uses_the_apps_current_settings():
    st = demo_state()
    st.families = {"demand": "studentt"}
    st.prior_text = {"revenue": {"beta.sigma": "3"}}
    st.alpha = "0.01"
    script = st._export_script()
    assert 'FAMILIES = {"demand": "studentt"}' in script and "ALPHA = 0.01" in script
    assert 'Prior("Normal", mu=0, sigma=3)' in script and 'OUTCOME = "revenue"' in script
    assert "demo data" in script
    assert State.export_code.fn(st) is not None and st.error == ""


def test_export_code_says_what_to_fix_instead_of_exporting_a_graph_that_cannot_be_fitted():
    st = demo_state()
    st.edges = st.edges + [{"source": "noise", "target": "price", "directed": False}]
    State.export_code.fn(st)
    assert st.error.startswith("Could not export the code: ") and "dashed" in st.error
    assert st.can_export_code is False

    st = demo_state()
    st.prior_text = {"revenue": {"beta.mu": "abc"}}
    State.export_code.fn(st)
    assert "must be a number" in st.error


def test_export_code_remembers_the_file_and_the_grouping():
    st = State(_reflex_internal_init=True)
    df, notes, extras = b.load_dataset(b.PANEL_TEST_FILE)
    st._set_table(df, notes, "dag_studio_panel_data.csv", target="revenue", extras=extras)
    State.add_group_col.fn(st, "region")
    State.add_group_col.fn(st, "store")
    State.set_time_col.fn(st, "week")
    st.edges = [{"source": s, "target": t, "directed": True} for s, t in
                [("price", "demand"), ("promo", "demand"), ("demand", "revenue")]]
    script = st._export_script()
    assert 'DATA_FILE = "dag_studio_panel_data.csv"' in script
    assert 'GROUP_COLUMNS = ["region", "store"]' in script and 'TIME_COLUMN = "week"' in script
    assert "footfall" in script  # an unconnected variable is still in NODES, for Validate DAG
    # Grouped data cannot take a non-Gaussian family, and the export says so too.
    st.families = {"demand": "bernoulli"}
    State.export_code.fn(st)
    assert "once group means are removed" in st.error


# --------------------------------------------------------------------------
# Running the script: it must reproduce the app
# --------------------------------------------------------------------------


def run_script(script: str, workdir, *files) -> tuple[str, dict]:
    for source in files:
        shutil.copy(source, workdir)
    (workdir / "dag_studio_analysis.py").write_text(script)
    done = subprocess.run(
        [sys.executable, "dag_studio_analysis.py"], cwd=workdir, capture_output=True,
        text=True, timeout=900,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout, json.loads((workdir / "dag_studio_output" / "results.json").read_text())


def close(a: float, b_: float, tol: float = 1e-6) -> None:
    assert a == pytest.approx(b_, rel=tol, abs=tol)


def assert_matches_the_app(results: dict, fit: b.FitResult, data: pl.DataFrame) -> None:
    equations = b.model_equations(fit.model, fit.families)
    quality = b.fit_quality(fit)
    assert set(results["equations"]) == set(equations)
    for node, eq in equations.items():
        got = results["equations"][node]
        for field in ("mean", "lower", "upper"):
            close(got["intercept"][field], getattr(eq.intercept, field))
            if eq.sigma is not None:
                close(got["sigma"][field], getattr(eq.sigma, field))
        assert [t["name"] for t in got["terms"]] == [t.name for t in eq.terms]
        for theirs, ours in zip(got["terms"], eq.terms):
            for field in ("mean", "lower", "upper"):
                close(theirs[field], getattr(ours, field))
    for n in quality.nodes:
        for field in ("r2", "lower", "upper"):
            close(results["r2"][n.node][field], getattr(n, field))
    close(results["diagnostics"]["max_rhat"], quality.max_rhat)
    assert results["diagnostics"]["divergences"] == quality.divergences


@pytest.mark.slow
def test_script_reproduces_the_app_on_ungrouped_data(tmp_path):
    df = b.demo_table()
    priors = {"revenue": {"beta.sigma": "5"}}
    fit = b.fit_model(df, DEMO_NODES, DEMO_EDGES, "revenue", priors=priors)  # what the app does
    stdout, results = run_script(demo_script(), tmp_path)
    assert_matches_the_app(results, fit, df)

    # The printed report has the app's sections, in order.
    sections = re.findall(r"^== (.+?) ==$", stdout, re.MULTILINE)
    assert sections == ["Data", "Validate DAG", "Fit quality", "Fitted equations",
                        "Direct effects (the colours the app draws on the edges)",
                        "Effect curves, E[child | do(parent)]"]
    assert "demo data · 800 rows · 6 variables" in stdout
    assert b.CONVERGED_NOTE in stdout or "Borderline" in stdout
    assert re.search(r"4 chains × 500 draws · largest r̂ \d\.\d{3} · smallest ESS \d+ · 0 divergences", stdout)
    assert "demand → revenue: direct coefficient" in stdout and "raises its child" in stdout

    # The validation result is the app's.
    check = b.validate_graph(df, DEMO_NODES, DEMO_EDGES, 0.05)
    assert results["validation"]["n_tests"] == check.n_tests
    assert [(v["x"], v["y"]) for v in results["validation"]["violations"]] == [
        (v.x, v.y) for v in check.violations
    ]

    # The effect curves are the app's: one per edge, plus each node's effect on the outcome.
    with open(tmp_path / "dag_studio_output" / "effect_curves.csv") as handle:
        rows = list(csv.DictReader(handle))
    pairs = {(r["source"], r["target"]) for r in rows}
    assert {(e.source, e.target) for e in DEMO_EDGES} <= pairs
    assert ("season", "revenue") in pairs  # no edge, but a path: the node-effect curve
    curve = b.edge_curve(fit.model, df, "season", "revenue")
    mine = [r for r in rows if (r["source"], r["target"]) == ("season", "revenue")]
    assert len(mine) == len(curve.x)
    for row, x, m, lo, hi in zip(mine, curve.x, curve.mean, curve.lower, curve.upper):
        close(float(row["x"]), x)
        close(float(row["mean"]), m, 1e-5)
        close(float(row["lower"]), lo, 1e-5)
        close(float(row["upper"]), hi, 1e-5)
    assert len(list((tmp_path / "dag_studio_output").glob("effect_*.html"))) == len(pairs)


@pytest.mark.slow
def test_script_reproduces_the_app_on_grouped_data(tmp_path):
    df, notes, extras = b.load_dataset(b.PANEL_TEST_FILE)
    grouped = b.apply_grouping(df.hstack(extras), ["region", "store"], "week", list(df.columns))
    nodes = grouped.df.columns
    edges = [b.Edge("price", "demand"), b.Edge("promo", "demand"), b.Edge("demand", "revenue"),
             b.Edge("price", "revenue")]
    fit = b.fit_model(grouped.df, nodes, edges, "revenue", grouped=True)
    plan = b.prepare_fit(grouped.df, nodes, edges, "revenue", grouped=True)
    script = export.build_script(
        nodes=nodes, outcome="revenue", plan=plan, data_kind="file",
        data_file=b.PANEL_TEST_FILE.name, group_cols=["region", "store"], time_col="week",
        alpha=0.05,
    )
    stdout, results = run_script(script, tmp_path, b.PANEL_TEST_FILE)
    assert_matches_the_app(results, fit, grouped.df)
    assert "dag_studio_panel_data.csv · 240 rows · 5 variables (group means removed)" in stdout
    assert "12 groups (region × store), 20 rows each, balanced, 20 periods" in stdout
    assert "within groups, after group means are removed" in stdout
    assert "Group means were removed first" in stdout
    close(results["equations"]["demand"]["terms"][0]["mean"], -0.67, 0.1)  # the within-group effect


@pytest.mark.slow
def test_script_reproduces_a_non_gaussian_fit_with_custom_priors(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * m + 0.7 * x)))).astype(float)
    pl.DataFrame({"x": x, "m": m, "y": y}).write_csv(tmp_path / "outcomes.csv")
    df, _ = b.load_table(tmp_path / "outcomes.csv")
    edges = [b.Edge("x", "m"), b.Edge("m", "y"), b.Edge("x", "y")]
    families, priors = {"y": "bernoulli"}, {"y": {"beta.sigma": "2"}}

    fit = b.fit_model(df, ["x", "m", "y"], edges, "y", families, priors)
    plan = b.prepare_fit(df, ["x", "m", "y"], edges, "y", families, priors)
    script = export.build_script(
        nodes=["x", "m", "y"], outcome="y", plan=plan, data_kind="file", data_file="outcomes.csv",
        group_cols=[], time_col="", alpha=0.05,
    )
    assert 'FAMILIES = {"y": "bernoulli"}' in script
    stdout, results = run_script(script, tmp_path)
    assert_matches_the_app(results, fit, df)
    assert "logit P(y = 1)" in stdout  # the bernoulli equation, worded as in the app
    assert results["equations"]["y"]["family"] == "bernoulli"
    assert results["r2"]["y"]["family"] == "bernoulli"
