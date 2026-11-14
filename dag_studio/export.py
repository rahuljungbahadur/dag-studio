"""Export the DAG the model was last fitted on as a short, standalone Python script.

The script loads the data with polars from a path the user must set, loads the DAG as
a digraph, and fits it with pathmc's ``BuildModelFromDAG`` with the settings and seed
the app used, assigning the model to variables in the ``__main__`` section.

The few functions that prepare the data (cleaning, and removing group means for panel
data) are copied into the script from this package's source at export time, so the
script prepares the data with the same code the app did and imports no DAG Studio code.
"""

from __future__ import annotations

import inspect
import json
import textwrap
from datetime import datetime
from importlib import metadata

from . import backend as b

PACKAGES = [
    "pathmc", "pymc", "pymc-extras", "nutpie", "polars", "pyarrow", "pandas", "numpy",
    "networkx",
]


def requirements() -> str:
    """``pip install`` arguments pinning the versions this was exported from."""
    pins = []
    for name in PACKAGES:
        try:
            pins.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pins.append(name)
    return " ".join(pins)


def _source(objects) -> str:
    return "\n\n\n".join(textwrap.dedent(inspect.getsource(o)).rstrip() for o in objects)


def _format_list(values: list[str]) -> str:
    """A Python list of strings, wrapped if it is long."""
    quoted = [json.dumps(v) for v in values]
    line = "[" + ", ".join(quoted) + "]"
    if len(line) <= 84:
        return line
    body = textwrap.fill(", ".join(quoted), width=80, initial_indent="    ", subsequent_indent="    ")
    return "[\n" + body + ",\n]"


def _format_priors(specs: dict[str, list]) -> str:
    if not specs:
        return "PRIORS = {}"
    lines = ["PRIORS = {"]
    for name, (dist, params) in sorted(specs.items()):
        args = ", ".join(f"{k}={b._fmt_number(v)}" for k, v in params.items())
        lines.append(f'    "{name}": Prior("{dist}", {args}),')
    lines.append("}")
    return "\n".join(lines)


_LOAD_DAG = '''def load_dag(text: str) -> nx.DiGraph:
    """The digraph above as a networkx graph.

    pathmc reads a digraph string in an order that changes from run to run, which can
    change the order of the equations and so the sampler's random numbers. Adding the
    edges in sorted order gives the same model, and the same fit, every time.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(sorted(re.findall(r'"([^"]+)"\\s*->\\s*"([^"]+)"', text)))
    return graph'''


def build_script(
    snapshot: dict, *, generated_at: str | None = None, data_path: str | None = None
) -> str:
    """The script for a fit, from :func:`backend.fit_snapshot`.

    ``data_path`` is for tests and notebooks. The app leaves it out, so the script
    carries a placeholder that its user must replace with the path to their file.
    """
    when = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    pairs = [tuple(pair) for pair in snapshot["pairs"]]
    used = sorted({node for pair in pairs for node in pair})
    dag = b.to_dot(used, [b.Edge(u, v) for u, v in pairs]).rstrip("\n")
    outcome, demo = snapshot["outcome"], snapshot["data_kind"] == "demo"
    group_cols, time_col = snapshot["group_columns"], snapshot["time_column"]
    data_file, specs = snapshot["data_file"], snapshot["specs"]
    grouped = bool(group_cols)
    parquet = data_file.lower().endswith((".parquet", ".pq"))

    # ---- imports: only what this script uses ---------------------------------------
    stdlib = ["import re"]
    if grouped:
        stdlib.append("from dataclasses import dataclass")
    if not demo:
        stdlib.append("from pathlib import Path")
    third_party = ["import networkx as nx"]
    if demo:
        third_party.append("import numpy as np")
    third_party += ["import polars as pl", "from pathmc import BuildModelFromDAG"]
    if specs:
        third_party.append("from pymc_extras.prior import Prior")

    # ---- settings ----------------------------------------------------------------------
    settings = [f"OUTCOME = {json.dumps(outcome)}"]
    if grouped:
        settings.append(
            f"GROUP_COLUMNS = {_format_list(group_cols)}  # together they name one group"
        )
    if time_col:
        settings.append(f"TIME_COLUMN = {json.dumps(time_col)}  # kept out of the graph")
    settings += [
        "FAMILIES = " + json.dumps(dict(sorted(snapshot["families"].items()))),
        _format_priors(specs),
        "DRAWS, TUNE, CHAINS, SEED = "
        f"{snapshot['draws']}, {snapshot['tune']}, {snapshot['chains']}, {snapshot['seed']}",
    ]

    # ---- the functions that prepare the data: copied from the app ------------------------
    helpers = [b.demo_table] if demo else [b._safe_name, b.clean_table_with_extras]
    helper_text = ""
    if grouped:
        helper_text = "SHORT_GROUP_ROWS = %r\n\n\n" % b.SHORT_GROUP_ROWS
        helpers += [b.GroupSummary, b.GroupedData, b.apply_grouping]
    helper_text += _source(helpers)

    # ---- __main__ ------------------------------------------------------------------------
    if demo:
        load = [
            "    # DAG Studio's built-in demo data, generated here: there is no file to load.",
            "    table = demo_table()",
        ]
        if grouped:
            load.append("    extras = pl.DataFrame()  # the demo data has no text columns")
        data_note = "DAG Studio's built-in demo data (generated in the script)"
    else:
        reader = "pl.read_parquet(data_path)" if parquet else (
            "pl.read_csv(data_path, infer_schema_length=10_000)"
        )
        placeholder = data_path if data_path is not None else f"path/to/{data_file}"
        load = [
            f"    # REQUIRED: the path to the data file you loaded in DAG Studio ({data_file}).",
            f"    data_path = {json.dumps(placeholder)}",
            "    if not Path(data_path).is_file():",
            '        raise SystemExit(f"Set data_path in the __main__ section: no file at {data_path!r}.")',
            "",
            "    # Load it with polars and prepare it as DAG Studio did: numeric columns, tidy names,",
            "    # no constant columns, complete rows.",
            f"    raw = {reader}",
            "    table, _, extras = clean_table_with_extras(raw)",
        ]
        data_note = f"{data_file} (set data_path in the __main__ section)"
    if grouped:
        prepare = [
            "    # Panel data: remove each group's mean, so only differences within a group remain.",
            "    frame = table.hstack(extras) if extras.width else table",
            "    data = apply_grouping(frame, GROUP_COLUMNS, TIME_COLUMN, list(table.columns)).df",
        ]
    elif time_col:
        prepare = ["    data = table.drop(TIME_COLUMN, strict=False)  # a time index is not a variable"]
    else:
        prepare = ["    data = table"]

    main = "\n".join(
        ["if __name__ == \"__main__\":", *load, "", *prepare, "",
         "    # The DAG the model was last fitted on, and the model fitted on it.",
         "    graph = load_dag(DAG)",
         "    builder = BuildModelFromDAG(",
         "        dag=graph,",
         "        df=data.select(sorted(graph.nodes)).to_pandas(),",
         "        target=OUTCOME,",
         '        style="native",',
         "        families=FAMILIES or None,",
         "        priors=PRIORS or None,",
         "    )",
         "    model = builder.to_pathmodel()",
         "    idata = model.fit(draws=DRAWS, tune=TUNE, chains=CHAINS, progressbar=False, "
         "random_seed=SEED)",
         "",
         "    # The fitted parameters, without the per-row fitted means.",
         "    summary = model.summary()",
         '    print(summary[~summary.index.str.startswith(("mu_", "pathmc_resp_"))])']
    )

    return f'''"""The model DAG Studio last fitted, as code (generated {when}).

The digraph below is the DAG the model was last fitted on. Run this file, or paste it into
a notebook: afterwards `model` is the fitted pathmc model and `idata` its posterior.

    Data     {data_note}
    Outcome  {outcome}

Install (the versions this was exported from):

    pip install {requirements()}
"""

from __future__ import annotations

{chr(10).join(stdlib)}

{chr(10).join(third_party)}

# The DAG the model was fitted on, as a digraph.
DAG = """{dag}"""

# What DAG Studio fitted it with.
{chr(10).join(settings)}


{_LOAD_DAG}


# The functions below prepare the data. They are copied from DAG Studio's own source, so the
# data is prepared exactly as it was in the app.
{helper_text}


{main}
'''
