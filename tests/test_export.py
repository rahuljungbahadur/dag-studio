"""Export code: a short script for the DAG the model was last fitted on."""

import ast
import asyncio
import builtins
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
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
PANEL_EDGES = [
    b.Edge("price", "demand"), b.Edge("promo", "demand"), b.Edge("demand", "revenue"),
    b.Edge("price", "revenue"),
]


def demo_snapshot(**changes) -> dict:
    plan = b.prepare_fit(
        b.demo_table(), DEMO_NODES, DEMO_EDGES, "revenue", priors={"revenue": {"beta.sigma": "5"}}
    )
    snapshot = b.fit_snapshot(plan, "revenue", [], "", "demo", "demo data")
    return {**snapshot, **changes}


def file_snapshot(file="sales 2026.csv", **changes) -> dict:
    return demo_snapshot(data_kind="file", data_file=file, **changes)


def panel_data():
    df, _, extras = b.load_dataset(b.PANEL_TEST_FILE)
    grouped = b.apply_grouping(df.hstack(extras), ["region", "store"], "week", list(df.columns))
    return grouped.df


def panel_snapshot() -> dict:
    data = panel_data()
    plan = b.prepare_fit(data, data.columns, PANEL_EDGES, "revenue", grouped=True)
    return b.fit_snapshot(
        plan, "revenue", ["region", "store"], "week", "file", b.PANEL_TEST_FILE.name
    )


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


def main_block(script: str) -> str:
    return script[script.index('if __name__ == "__main__":'):]


# --------------------------------------------------------------------------
# What the script is
# --------------------------------------------------------------------------


@pytest.mark.parametrize("snapshot", [demo_snapshot(), file_snapshot(), panel_snapshot()],
                         ids=["demo", "file", "grouped file"])
def test_script_is_valid_python_that_defines_everything_it_uses(snapshot):
    script = export.build_script(snapshot)
    ast.parse(script)
    assert undefined_names(script) == set()
    assert not re.search(r"^\s*(from|import)\s+dag_studio", script, re.MULTILINE)  # standalone


def test_script_loads_the_dag_as_a_digraph_and_names_its_packages():
    script = export.build_script(file_snapshot())
    assert 'DAG = """digraph {\n  "demand" -> "revenue";' in script
    assert '  "season" -> "promo";\n}"""' in script  # one edge per line, sorted
    pins = re.search(r"pip install (.+)", script).group(1).split()
    assert {"pathmc", "pymc", "nutpie", "polars", "networkx"} <= {p.split("==")[0] for p in pins}
    assert all("==" in p for p in pins)  # versions are pinned


def test_script_fits_the_model_inside_main_and_keeps_it_in_variables():
    main = main_block(export.build_script(file_snapshot()))
    for line in ("builder = BuildModelFromDAG(", "dag=graph,", "target=OUTCOME,",
                 'style="native",', "families=FAMILIES or None,", "priors=PRIORS or None,",
                 "model = builder.to_pathmodel()",
                 "idata = model.fit(draws=DRAWS, tune=TUNE, chains=CHAINS, progressbar=False, "
                 "random_seed=SEED)"):
        assert line in main
    assert "graph = load_dag(DAG)" in main


def test_script_makes_the_user_provide_the_data_path_in_main():
    script = export.build_script(file_snapshot(file="sales 2026.csv"))
    main = main_block(script)
    assert 'data_path = "path/to/sales 2026.csv"' in main  # a placeholder, not a real path
    assert "raise SystemExit" in main and "Set data_path in the __main__ section" in main
    assert "data_path" not in script.split('if __name__ == "__main__":')[0].split('"""')[2]  # not a global
    # The data is loaded with polars, and prepared as the app prepared it.
    assert "raw = pl.read_csv(data_path, infer_schema_length=10_000)" in main
    assert "table, _, extras = clean_table_with_extras(raw)" in main
    assert "pandas" not in "".join(main.split("to_pandas")[0:1])  # pandas is only for pathmc's input
    parquet = main_block(export.build_script(file_snapshot(file="sales.parquet")))
    assert "raw = pl.read_parquet(data_path)" in parquet


def test_script_for_the_demo_data_needs_no_path():
    script = export.build_script(demo_snapshot())
    main = main_block(script)
    assert "data_path" not in main and "table = demo_table()" in main
    assert "def demo_table" in script and "clean_table_with_extras" not in script


def test_script_carries_only_what_this_fit_needs():
    plain = export.build_script(file_snapshot())
    assert "apply_grouping" not in plain and "dataclass" not in plain
    assert "GROUP_COLUMNS" not in plain and "TIME_COLUMN" not in plain
    assert len(plain.splitlines()) < 150  # was ~1000 lines when it also printed a report
    assert "Prior(" in plain and "from pymc_extras.prior import Prior" in plain
    no_priors = export.build_script(file_snapshot(specs={}))
    assert "PRIORS = {}" in no_priors and "pymc_extras" not in no_priors
    assert "FAMILIES = {}" in no_priors

    timed = export.build_script(file_snapshot(time_column="week"))
    assert 'TIME_COLUMN = "week"' in timed
    assert "data = table.drop(TIME_COLUMN, strict=False)" in main_block(timed)

    grouped = export.build_script(panel_snapshot())
    assert 'GROUP_COLUMNS = ["region", "store"]' in grouped and 'TIME_COLUMN = "week"' in grouped
    assert "data = apply_grouping(frame, GROUP_COLUMNS, TIME_COLUMN, list(table.columns)).df" in grouped
    assert "SHORT_GROUP_ROWS = 5" in grouped


def test_script_writes_settings_that_round_trip():
    script = export.build_script(
        file_snapshot(file='my "odd" file.csv', families={"demand": "studentt"},
                      group_columns=["a b", "c"], time_column="t",
                      specs={"beta_demand": ["Normal", {"mu": 0.0, "sigma": 2.5}]})
    )
    assigned = {t.id: n.value for n in ast.parse(script).body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert ast.literal_eval(assigned["GROUP_COLUMNS"]) == ["a b", "c"]
    assert ast.literal_eval(assigned["FAMILIES"]) == {"demand": "studentt"}
    assert ast.literal_eval(assigned["TIME_COLUMN"]) == "t"
    assert 'Prior("Normal", mu=0, sigma=2.5)' in script
    assert '"path/to/my \\"odd\\" file.csv"' in script  # an awkward name stays valid Python


# --------------------------------------------------------------------------
# Which DAG is exported
# --------------------------------------------------------------------------


def demo_state() -> State:
    st = State(_reflex_internal_init=True)
    State.load_demo.fn(st)
    st.edges = [{"source": e.source, "target": e.target, "directed": True} for e in DEMO_EDGES]
    st._refresh()
    return st


def test_export_needs_a_fit_first():
    st = demo_state()
    assert st.can_export_code is False
    State.export_code.fn(st)
    assert st.error.startswith("Could not export the code: Fit the model first")


def test_export_is_the_dag_that_was_fitted_not_the_edits_since():
    st = demo_state()
    st.fit_snapshot = demo_snapshot()
    st.fitted, st.stale = True, False
    # The user then edits the graph: an edge removed, another added.
    st.edges = [e for e in st.edges if (e["source"], e["target"]) != ("season", "promo")]
    st.edges.append({"source": "noise", "target": "price", "directed": True})
    st.stale = True
    script = st._export_script()
    assert '"season" -> "promo";' in script  # still the fitted graph
    assert "noise" not in script.split("# What DAG Studio fitted it with.")[0].split("DAG = ")[1]
    assert st.can_export_code is True
    assert State.export_code.fn(st) is not None and st.error == ""


def test_the_fitted_dag_is_forgotten_when_the_data_or_graph_it_came_from_is_replaced():
    for forget in (
        lambda st: st._discard_results(),  # an import or a change of grouping
        lambda st: st._reset_graph(),  # new data
    ):
        st = demo_state()
        st.fit_snapshot = demo_snapshot()
        forget(st)
        assert st.fit_snapshot == {} and st.can_export_code is False

    st = State(_reflex_internal_init=True)
    df, notes, extras = b.load_dataset(b.PANEL_TEST_FILE)
    st._set_table(df, notes, "panel.csv", target="revenue", extras=extras)
    st.fit_snapshot = panel_snapshot()
    State.add_group_col.fn(st, "region")
    assert st.fit_snapshot == {}  # the fit was on different data


def test_state_records_what_a_fit_used(tmp_path):
    """Run the app's real fit handler and check it remembers the fitted graph and settings."""
    st = demo_state()
    st.families = {"demand": "studentt"}
    st.prior_text = {"revenue": {"beta.sigma": "5"}}

    class Proxy:  # stands in for the StateProxy that a background handler runs on
        def __init__(self, state):
            object.__setattr__(self, "_state", state)

        def __getattr__(self, name):
            return getattr(self._state, name)

        def __setattr__(self, name, value):
            setattr(self._state, name, value)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    asyncio.run(State.fit.fn(Proxy(st)))
    assert st.fitted and st.error == ""
    snap = st.fit_snapshot
    assert snap["outcome"] == "revenue" and snap["data_kind"] == "demo"
    assert snap["pairs"] == sorted([e.source, e.target] for e in DEMO_EDGES)
    assert snap["families"] == {"demand": "studentt"}
    assert snap["specs"] == {"beta_revenue": ["Normal", {"mu": 0.0, "sigma": 5.0}]}
    assert (snap["draws"], snap["chains"], snap["seed"]) == (b.DRAWS, b.CHAINS, b.RANDOM_SEED)
    assert st.can_export_code is True


# --------------------------------------------------------------------------
# Running the script: it must refit the app's model
# --------------------------------------------------------------------------


def run_script(script: str, tmp_path: Path) -> dict:
    """Run the script's __main__ section in this process and return what it defined."""
    import runpy

    path = tmp_path / "dag_studio_model.py"
    path.write_text(script)
    return runpy.run_path(str(path), run_name="__main__")


def parameters(model) -> pd.DataFrame:
    summary = model.summary()
    return summary[~summary.index.str.startswith(("mu_", "pathmc_resp_"))]


def assert_same_model(script_model, app_model) -> None:
    pd.testing.assert_frame_equal(parameters(script_model), parameters(app_model), rtol=1e-9)


@pytest.mark.slow
def test_script_refits_the_app_model_exactly_on_the_demo_data(tmp_path):
    fit = b.fit_model(b.demo_table(), DEMO_NODES, DEMO_EDGES, "revenue",
                      priors={"revenue": {"beta.sigma": "5"}})
    namespace = run_script(export.build_script(demo_snapshot()), tmp_path)
    assert {"builder", "model", "idata"} <= set(namespace)
    assert_same_model(namespace["model"], fit.model)


@pytest.mark.slow
def test_script_refits_the_app_model_exactly_on_grouped_panel_data(tmp_path):
    fit = b.fit_model(panel_data(), panel_data().columns, PANEL_EDGES, "revenue", grouped=True)
    script = export.build_script(panel_snapshot(), data_path=str(b.PANEL_TEST_FILE))
    assert_same_model(run_script(script, tmp_path)["model"], fit.model)


@pytest.mark.slow
def test_script_refits_messy_data_with_a_non_gaussian_family_and_priors(tmp_path):
    df, _ = b.load_table(b.TEST_FILE)  # renamed, text, constant and incomplete columns
    nodes = df.columns
    edges = [b.Edge("ad_spend", "web_visits"), b.Edge("web_visits", "orders"),
             b.Edge("v_2024_price_index", "orders"), b.Edge("orders", "revenue"),
             b.Edge("orders", "customer_satisfaction_score"),
             b.Edge("customer_satisfaction_score", "repeat_purchase")]
    families, priors = {"repeat_purchase": "bernoulli"}, {"revenue": {"beta.sigma": "3"}}
    fit = b.fit_model(df, nodes, edges, "revenue", families, priors)
    plan = b.prepare_fit(df, nodes, edges, "revenue", families, priors)
    snapshot = b.fit_snapshot(plan, "revenue", [], "", "file", b.TEST_FILE.name)
    script = export.build_script(snapshot, data_path=str(b.TEST_FILE))
    assert_same_model(run_script(script, tmp_path)["model"], fit.model)


@pytest.mark.slow
def test_script_runs_as_a_program_and_refuses_to_run_without_a_path(tmp_path):
    script = tmp_path / "dag_studio_model.py"
    script.write_text(export.build_script(demo_snapshot()))
    done = subprocess.run([sys.executable, script.name], cwd=tmp_path, capture_output=True,
                          text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "beta_revenue[demand]" in done.stdout and "sigma_revenue" in done.stdout
    assert "mu_revenue" not in done.stdout  # the per-row means are left out

    script.write_text(export.build_script(file_snapshot(file="missing.csv")))
    done = subprocess.run([sys.executable, script.name], cwd=tmp_path, capture_output=True,
                          text=True, timeout=600)
    assert done.returncode != 0
    assert "Set data_path in the __main__ section" in done.stderr
