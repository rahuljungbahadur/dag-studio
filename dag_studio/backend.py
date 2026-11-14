"""Pure-Python backend for DAG Studio.

Everything that touches data, pathmc, or graph geometry lives here so it can
be tested without a browser. The Reflex state in ``state.py`` only calls
these functions and stores their (serialisable) results.
"""

from __future__ import annotations

import difflib
import math
import re
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

BF_THRESHOLD = 1.0  # TBFPC: declare independence when BF10 < this
MAX_CONDITIONING_SET = 3  # TBFPC: largest conditioning set searched
DRAWS, TUNE, CHAINS = 500, 500, 4  # MCMC budget for one fit (4 chains for a reliable r̂)
INTERVAL = 0.89  # equal-tailed credible interval used everywhere
GRID_POINTS = 15  # x values on the fitted-relationship curve
MAX_SCATTER = 600  # observed points drawn behind the curve
RANDOM_SEED = 0

NODE_W, NODE_H = 104.0, 34.0
LAYER_GAP, ROW_GAP = 96.0, 136.0  # vertical gap between layers, horizontal gap within one
MARGIN_X, MARGIN_Y = 70.0, 36.0
MIN_CENTRE_X = 320.0  # keeps small graphs horizontally centred on the canvas


# --------------------------------------------------------------------------
# Graph representation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """One adjacency. ``directed=False`` means discovery left it unoriented."""

    source: str
    target: str
    directed: bool = True

    @property
    def key(self) -> str:
        return edge_key(self.source, self.target)


def edge_key(a: str, b: str) -> str:
    """Order-independent id: at most one edge may connect a pair of nodes."""
    lo, hi = sorted((a, b))
    return f"{lo}~{hi}"


def directed_path(edges: list[Edge], start: str, end: str) -> list[str]:
    """Nodes of a directed path ``start -> ... -> end`` (shortest), or [] if none."""
    children: dict[str, list[str]] = {}
    for e in edges:
        if e.directed:
            children.setdefault(e.source, []).append(e.target)
    came_from: dict[str, str | None] = {start: None}
    queue = [start]
    for node in queue:
        if node == end:
            path = [node]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])  # type: ignore[arg-type]
            return path[::-1]
        for child in children.get(node, []):
            if child not in came_from:
                came_from[child] = node
                queue.append(child)
    return []


def creates_cycle(edges: list[Edge], source: str, target: str) -> bool:
    """True if adding ``source -> target`` closes a directed cycle."""
    return bool(directed_path(edges, target, source))


def cycle_closed_by(edges: list[Edge], source: str, target: str) -> list[str]:
    """The cycle that adding ``source -> target`` would close, as nodes in order.

    The list runs ``target -> ... -> source`` along existing edges; the new edge
    would close it from the last node back to the first. Empty if it closes none.
    """
    return directed_path(edges, target, source)


def find_cycle(edges: list[Edge]) -> list[str]:
    """Nodes of one directed cycle already in the graph, in order, or [] if acyclic.

    The edge from the last node back to the first is the one that closes it.
    """
    directed = [e for e in edges if e.directed]
    for e in directed:
        path = directed_path([x for x in directed if x is not e], e.target, e.source)
        if path:
            return path
    return []


def graph_problems(nodes: list[str], edges: list[Edge], target: str = "") -> list[str]:
    """Reasons the current graph cannot be fitted yet (empty list = ready)."""
    problems = []
    if not edges:
        problems.append("The graph has no edges.")
    elif target and target not in {e.source for e in edges} | {e.target for e in edges}:
        problems.append(f"The outcome {target} has no edges.")
    undirected = [e for e in edges if not e.directed]
    if undirected:
        names = ", ".join(f"{e.source}–{e.target}" for e in undirected[:4])
        more = "…" if len(undirected) > 4 else ""
        problems.append(f"Orient the dashed edges first: {names}{more}")
    directed = [e for e in edges if e.directed]
    for e in directed:
        rest = [x for x in directed if x is not e]
        if creates_cycle(rest, e.source, e.target):
            problems.append("The graph contains a directed cycle.")
            break
    return problems


def next_selection(current: list[str], node: str, multi: bool) -> list[str]:
    """Node selection after a click.

    A plain click selects only ``node`` (or clears it if it was the only one
    selected). With ``multi`` (cmd or ctrl held) the node is toggled in the
    selection, which holds at most two nodes: a third drops the oldest. Order
    is kept, because the first node picked is the start of a new edge.
    """
    if not multi:
        return [] if current == [node] else [node]
    if node in current:
        return [n for n in current if n != node]
    return (current + [node])[-2:]


def _dot_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped.replace("\r\n", "\n").replace("\n", "\\n") + '"'


def to_dot(
    nodes: list[str],
    edges: list[Edge],
    flagged: list[str] | None = None,
    descriptions: dict[str, str] | None = None,
) -> str:
    """DOT text for the graph.

    Node descriptions become ``tooltip`` attributes; flagged items are listed
    as comments.
    """
    lines = ["digraph {"]
    connected = {e.source for e in edges} | {e.target for e in edges}
    descriptions = descriptions or {}
    for n in nodes:
        text = descriptions.get(n, "").strip()
        if text:
            lines.append(f'  "{n}" [tooltip={_dot_string(text)}];')
        elif n not in connected:
            lines.append(f'  "{n}";')
    for e in edges:
        if e.directed:
            lines.append(f'  "{e.source}" -> "{e.target}";')
        else:
            lines.append(f'  "{e.source}" -> "{e.target}" [dir=none, style=dashed];')
    lines.append("}")
    for item in flagged or []:
        lines.append(f"// review: {item}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Grouped (panel) data: remove each group's mean, so only differences within a
# group are compared. This is the fixed-effects estimate: it works because a
# regression on group-demeaned data gives the same slopes as one with a dummy
# for every group, and it keeps discovery, validation and the fit on one table.
# --------------------------------------------------------------------------

NON_GAUSSIAN = ("bernoulli", "poisson", "negbinomial")
SHORT_GROUP_ROWS = 5  # at or below this many rows per group, demeaned intervals run narrow


@dataclass
class GroupSummary:
    headline: str  # e.g. "12 groups (region × store), 20 rows each, balanced"
    warnings: list[str]
    n_groups: int


@dataclass
class GroupedData:
    df: pl.DataFrame  # the variables only, each minus its group's mean
    summary: GroupSummary


def apply_grouping(
    frame: pl.DataFrame, keys: list[str], time_col: str, variables: list[str]
) -> GroupedData:
    """Remove group means from ``variables``.

    ``keys`` are the group columns (say ``["region", "store"]``): a group is one
    combination of their values, so a store name that repeats in several regions
    still gives separate groups, and the order of ``keys`` makes no difference
    to the result. ``time_col`` is optional; it is kept out
    of the variables and used to check the panel. Rows with a missing group value,
    and groups of a single row (which have nothing to compare within), are
    dropped and reported, as are variables that never vary within a group.
    """
    if not keys:
        raise ValueError("Choose at least one group column.")
    if len(set(keys)) != len(keys):
        raise ValueError("Each group column can only be used once.")
    if time_col and time_col in keys:
        raise ValueError("The time column cannot also be a group column.")
    named = keys + ([time_col] if time_col else [])
    missing = [c for c in named if c not in frame.columns]
    if missing:
        raise ValueError(f"Not columns of the data: {', '.join(missing)}.")
    measured = [v for v in variables if v not in named]
    frame = frame.select(list(dict.fromkeys(named + measured)))
    warns: list[str] = []

    before = frame.height
    frame = frame.filter(pl.all_horizontal([pl.col(k).is_not_null() for k in keys]))
    if frame.height < before:
        warns.append(f"{before - frame.height} row(s) with a missing group value were dropped.")

    singles = frame.filter(pl.len().over(keys) == 1)
    if singles.height:
        frame = frame.filter(pl.len().over(keys) > 1)
        warns.append(
            f"{singles.height} group(s) with a single row were dropped: with one row there is "
            "nothing to compare within the group."
        )

    sizes = frame.group_by(keys).len()["len"]
    n_groups = sizes.len()
    if n_groups < 2:
        raise ValueError(
            f"These columns give {n_groups} group with two or more rows. Removing group means "
            "needs at least two."
        )
    lo, hi, median = int(sizes.min()), int(sizes.max()), float(sizes.median())

    demeaned = frame.select(
        [(pl.col(v) - pl.col(v).mean().over(keys)).alias(v) for v in measured]
    )
    flat = [
        v for v in measured
        if float(demeaned[v].abs().max()) <= 1e-9 * max(float(frame[v].abs().max()), 1.0)
    ]
    kept = [v for v in measured if v not in flat]
    if flat:
        warns.append(
            f"Removed because they never vary within a group: {', '.join(flat)}. Once group "
            "differences are taken out, a group-level variable has nothing left to say."
        )
    if len(kept) < 2:
        raise ValueError("Fewer than two variables vary within groups, so there is nothing to model.")
    if demeaned.height < 20:
        raise ValueError("Need at least 20 rows that sit in groups of two or more.")

    if time_col:
        repeated = int(frame.select(keys + [time_col]).is_duplicated().sum())
        if repeated:
            warns.append(
                f"{repeated} row(s) repeat a (group, {time_col}) pair. Either time repeats within a "
                "group, or different groups share a label (say store 1 in every region): add the "
                "column that tells them apart, such as the region."
            )
    if median <= SHORT_GROUP_ROWS:
        warns.append(
            f"Groups have few rows (median {median:g}). With group means removed, the intervals "
            "come out somewhat too narrow."
        )

    size_text = f"{lo} rows each, balanced" if lo == hi else f"{lo}–{hi} rows each (median {median:g})"
    periods = f", {frame[time_col].n_unique()} periods" if time_col else ""
    headline = f"{n_groups} groups ({' × '.join(keys)}), {size_text}{periods}"
    return GroupedData(demeaned.select(kept), GroupSummary(headline, warns, n_groups))


# --------------------------------------------------------------------------
# Importing a DAG: DOT, pathmc spec syntax, or an "A->B, B->C" edge list
# --------------------------------------------------------------------------

MAX_IMPORT_CHARS = 500_000


@dataclass
class ImportedDag:
    edges: list[Edge]
    descriptions: dict[str, str]  # from DOT ``tooltip`` attributes, which export_dot writes
    flagged_nodes: list[str]  # from the ``// review:`` comments export_dot writes
    flagged_edges: list[str]
    notes: list[str]
    format: str  # "DOT", "spec" or "edge list"


def _strip_leading_comments(text: str) -> str:
    """Drop comment lines before the graph header, which pathmc's DOT reader rejects."""
    text = text.lstrip()
    while True:
        if text.startswith(("//", "#")):
            newline = text.find("\n")
            text = text[newline + 1 :].lstrip() if newline != -1 else ""
        elif text.startswith("/*"):
            end = text.find("*/")
            text = text[end + 2 :].lstrip() if end != -1 else ""
        else:
            return text


def detect_dag_format(text: str) -> str:
    """"DOT", "spec" (``Y ~ X + M``) or "edge list" (``X->M, M->Y``); ValueError if none."""
    body = _strip_leading_comments(text)
    if re.match(r"(strict\s+)?(di)?graph\b", body, re.IGNORECASE):
        return "DOT"
    if "~" in body:
        return "spec"
    if "->" in body:
        return "edge list"
    raise ValueError(
        "Could not tell what format this is. Paste DOT (digraph { X -> M; M -> Y }), "
        "the spec syntax (M ~ X, then Y ~ M + X on the next line) or an edge list "
        "(X->M, M->Y)."
    )


def _read_dot(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Directed and undirected edges from DOT, via pathmc's own reader."""
    from pathmc._dot import parse_dot  # the reader pathmc.dag_to_spec uses

    _, directed, undirected = parse_dot(text)
    singles = [e for e in undirected if len(e) == 1]
    if singles:
        raise ValueError(f"{next(iter(singles[0]))} -- {next(iter(singles[0]))} is a self-loop.")
    return sorted(directed), sorted(tuple(sorted(e)) for e in undirected)  # type: ignore[misc]


def _read_spec(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Parent -> child edges from the spec syntax, via pathmc's parser, and notes."""
    from pathmc.parse import parse_spec

    spec = parse_spec(text)
    directed: list[tuple[str, str]] = []
    details = False
    for regression in spec.regressions:
        for term in regression.terms:
            if term.interaction_of:
                parents, details = list(term.interaction_of), True
            elif term.lag_of:
                details = True
                # A variable's own past is not another node; any other lag is a parent.
                parents = [] if term.lag_of == regression.lhs else [term.lag_of]
            else:
                parents = [term.variable]
                details = details or bool(
                    term.transform or term.categorical or term.basis
                    or term.fixed_value is not None
                )
            directed.extend((v, regression.lhs) for v in parents)
    notes = []
    if details:
        notes.append(
            "Only the structure was imported: transforms, lags, interactions, categorical "
            "terms and fixed coefficients are not carried over."
        )
    if spec.residual_covs:
        notes.append(
            f"{len(spec.residual_covs)} residual covariance(s) (~~) were ignored: the graph "
            "has no bidirected edges."
        )
    if spec.defined_params:
        notes.append(f"{len(spec.defined_params)} defined parameter(s) (:=) were ignored.")
    return directed, notes


def _unescape_dot(text: str) -> str:
    return re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), text)


_DOT_QUOTED = r'"(?:[^"\\]|\\.)*"'
_DOT_NODE_STATEMENT = re.compile(
    rf"({_DOT_QUOTED}|[A-Za-z_]\w*)\s*\[((?:[^\]\"]|{_DOT_QUOTED})*)\]"
)
_DOT_TOOLTIP = re.compile(rf"tooltip\s*=\s*({_DOT_QUOTED}|[^\s,\]]+)")


def _dot_id(token: str) -> str:
    return _unescape_dot(token[1:-1]) if token.startswith('"') else token


def _dot_descriptions(text: str) -> dict[str, str]:
    """``"node" [tooltip="..."]`` statements, which is how ``to_dot`` stores descriptions."""
    found: dict[str, str] = {}
    for match in _DOT_NODE_STATEMENT.finditer(text):
        if text[: match.start()].rstrip().endswith(("->", "--")):
            continue  # an edge's attributes, not a node's
        tooltip = _DOT_TOOLTIP.search(match.group(2))
        if tooltip:
            found[_dot_id(match.group(1))] = _dot_id(tooltip.group(1))
    return found


def _dot_review_flags(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """``// review: node a`` and ``// review: edge a -- b`` comments, as ``to_dot`` writes."""
    nodes = re.findall(r"^//\s*review:\s*node\s+(.+?)\s*$", text, re.MULTILINE)
    edges = re.findall(r"^//\s*review:\s*edge\s+(.+?)\s+--\s+(.+?)\s*$", text, re.MULTILINE)
    return nodes, edges


def parse_dag_text(text: str, columns: list[str]) -> ImportedDag:
    """Read a DAG from DOT, the pathmc spec syntax or an edge list.

    Parsing is pathmc's own: ``parse_dot`` for DOT and edge lists (an edge list
    is the body of a ``digraph``), ``parse_spec`` for ``Y ~ X + M`` lines. Names
    are matched to ``columns``, then to ``columns`` after the same cleaning an
    upload gets (``ad spend`` -> ``ad_spend``). Anything that cannot be imported
    whole raises a ValueError saying why, rather than importing part of a graph.
    """
    if len(text) > MAX_IMPORT_CHARS:
        raise ValueError("That text is too large to be a DAG.")
    kind = detect_dag_format(text)
    notes: list[str] = []
    undirected: list[tuple[str, str]] = []
    try:
        if kind == "DOT":
            directed, undirected = _read_dot(_strip_leading_comments(text))
        elif kind == "edge list":
            body = re.sub(r"[,\n]+", ";", text.strip())
            directed, undirected = _read_dot(f"digraph {{ {body} }}")
        else:
            directed, notes = _read_spec(text)
    except ImportError as exc:
        raise ValueError(f"This version of pathmc cannot read {kind}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - pathmc's ParseError, ValueError and friends
        raise ValueError(f"Could not read the {kind}: {exc}") from exc

    # Match every name an edge uses to a column.
    used = list(dict.fromkeys(n for pair in directed + undirected for n in pair))
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    for name in used:
        if name in columns:
            mapping[name] = name
        elif _safe_name(name) in columns:
            mapping[name] = _safe_name(name)
            notes.append(f"Matched {name} to the column {mapping[name]}.")
        else:
            unknown.append(name)
    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, columns, n=1, cutoff=0.6)
            hints.append(f"{name} (did you mean {close[0]}?)" if close else name)
        shown = ", ".join(columns[:15]) + ("…" if len(columns) > 15 else "")
        raise ValueError(
            f"These variables are not columns of the loaded data: {', '.join(hints)}. "
            f"Nothing was imported. The columns are: {shown}."
        )

    directed = [(mapping[u], mapping[v]) for u, v in directed]
    undirected = [(mapping[u], mapping[v]) for u, v in undirected]
    for u, v in directed + undirected:
        if u == v:
            raise ValueError(f"{u} → {u} is a self-loop. Nothing was imported.")
    if not directed and not undirected:
        raise ValueError("No edges found. Nothing was imported.")

    edges = [Edge(u, v, True) for u, v in dict.fromkeys(directed)]
    cycle = find_cycle(edges)
    if cycle:
        raise ValueError(
            f"The graph contains a cycle: {' → '.join(cycle + [cycle[0]])}. Nothing was imported."
        )
    seen = {e.key for e in edges}
    for u, v in undirected:
        key = edge_key(u, v)
        if key not in seen:  # a directed edge for the same pair wins
            seen.add(key)
            edges.append(Edge(*sorted((u, v)), False))

    descriptions: dict[str, str] = {}
    flagged_nodes: list[str] = []
    flagged_edges: list[str] = []
    if kind == "DOT":
        for name, note in _dot_descriptions(text).items():
            column = mapping.get(name) or (name if name in columns else None)
            if column and note.strip():
                descriptions[column] = note
        review_nodes, review_edges = _dot_review_flags(text)
        flagged_nodes = [mapping[n] for n in review_nodes if n in mapping]
        flagged_edges = [
            edge_key(mapping[a], mapping[c])
            for a, c in review_edges
            if a in mapping and c in mapping and edge_key(mapping[a], mapping[c]) in seen
        ]
    if undirected:
        notes.append(f"{len(undirected)} undirected edge(s) were imported as dashed edges.")
    return ImportedDag(edges, descriptions, flagged_nodes, flagged_edges, notes, kind)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    """Make a column name usable as a pathmc DSL identifier."""
    cleaned = re.sub(r"\W", "_", str(name).strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned


def load_dataset(path: str | Path) -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    """Read a CSV or Parquet file; return the usable numeric table, notes and the rest.

    The third value holds the columns that cannot be variables (text, dates),
    row for row with the numeric table, because they may still say which group
    a row belongs to.
    """
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pl.read_parquet(path)
    else:
        raw = pl.read_csv(path, infer_schema_length=10_000)
    return clean_table_with_extras(raw)


def load_table(path: str | Path) -> tuple[pl.DataFrame, list[str]]:
    """Read a CSV or Parquet file and keep the usable numeric columns.

    Returns the cleaned frame and human-readable notes about what changed.
    """
    df, notes, _ = load_dataset(path)
    return df, notes


def clean_table(raw: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    df, notes, _ = clean_table_with_extras(raw)
    return df, notes


def clean_table_with_extras(raw: pl.DataFrame) -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    notes: list[str] = []

    numeric = [c for c, dtype in raw.schema.items() if dtype.is_numeric()]
    dropped = [c for c in raw.columns if c not in numeric]
    if dropped:
        notes.append(f"Ignored non-numeric columns: {', '.join(dropped)}")
    df = raw.select(numeric).cast(pl.Float64)
    extras = raw.select(dropped)

    renames, used = {}, set()
    for col in df.columns:
        new = _safe_name(col)
        while new in used:
            new += "_"
        used.add(new)
        if new != col:
            renames[col] = new
    if renames:
        df = df.rename(renames)
        notes.append("Renamed columns: " + ", ".join(f"{a} → {b}" for a, b in renames.items()))

    constant = [c for c in df.columns if df[c].drop_nulls().n_unique() <= 1]
    if constant:
        df = df.drop(constant)
        notes.append(f"Ignored constant columns: {', '.join(constant)}")

    before = df.height
    complete = df.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
    df = df.filter(complete)
    if extras.width:  # keep the same rows in both; a file with no text columns has none to keep
        extras = extras.filter(complete)
    if df.height < before:
        notes.append(f"Dropped {before - df.height} rows with missing values")

    if df.width < 2:
        raise ValueError("Need at least two numeric, non-constant columns.")
    if df.height < 20:
        raise ValueError("Need at least 20 complete rows.")
    return df, notes, extras


# --------------------------------------------------------------------------
# Progress reporting for long tasks
# --------------------------------------------------------------------------


class Progress:
    """What a long task is doing: written by the worker thread, polled by the UI.

    ``fraction`` is 0 to 1 when the task can say how far along it is, and None
    when it cannot (the UI then shows an animated bar instead of a percentage).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._label, self._fraction = "", None

    def set(self, label: str, fraction: float | None = None) -> None:
        with self._lock:
            self._label = label
            self._fraction = None if fraction is None else min(max(fraction, 0.0), 1.0)

    def snapshot(self) -> tuple[str, float | None]:
        with self._lock:
            return self._label, self._fraction


FIT_STEPS = (
    "Compiling the model",
    "Drawing samples",
    "Finishing the posterior",
    "Computing equations and fit quality",
)


def fit_step(n: int) -> str:
    """Label for fit step ``n`` (1-based), e.g. "Step 2 of 4 · Drawing samples"."""
    return f"Step {n} of {len(FIT_STEPS)} · {FIT_STEPS[n - 1]}"


_sink = threading.local()  # the sampling-progress callback of the fit running on this thread
_hook_lock = threading.Lock()
_hook_installed = False


def _install_sampling_hook() -> bool:
    """Let a fit report nutpie's per-chain draw counts.

    PyMC already hands nutpie a progress callback (it feeds PyMC's own progress
    bar, which we keep switched off). This wraps that callback's ``update`` so a
    copy of the counts also reaches the sink of the fit that created it. Returns
    False, and the fit simply reports no sampling progress, if PyMC's internals
    are not as expected.
    """
    global _hook_installed
    with _hook_lock:
        if _hook_installed:
            return True
        try:
            from pymc.progress_bar.progress import NutpieProgressBarManager as Manager

            original_init, original_update = Manager.__init__, Manager.update
        except Exception:  # noqa: BLE001 - cosmetic feature, never break a fit
            return False

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._dag_studio_sink = getattr(_sink, "callback", None)

        def update(self, chain_progresses):
            callback = getattr(self, "_dag_studio_sink", None)
            if callback is not None:
                try:
                    callback(chain_progresses)
                except Exception:  # noqa: BLE001
                    pass
            return original_update(self, chain_progresses)

        Manager.__init__, Manager.update = init, update
        _hook_installed = True
        return True


@contextmanager
def _watch_sampling(progress: Progress | None):
    """Report MCMC draws to ``progress`` while the ``with`` body runs a fit."""
    if progress is None or not _install_sampling_hook():
        yield
        return

    def callback(chain_progresses) -> None:
        done = sum(int(c.finished_draws) for c in chain_progresses)
        total = sum(int(c.total_draws) for c in chain_progresses)
        if total <= 0:
            return
        if done >= total:
            progress.set(fit_step(3))
        else:
            progress.set(fit_step(2), done / total)

    _sink.callback = callback
    try:
        yield
    finally:
        _sink.callback = None


def _watch_tbfpc(model, drivers: list[str], progress: Progress) -> None:
    """Make ``model`` (a TBFPC) report how many variable pairs it has examined.

    TBFPC makes one pass per driver (its link to the outcome), then one per pair
    of drivers. Each pass runs many tests, so passes are the honest unit: the
    bar is "passes finished of passes to do". Counts are taken by watching which
    pair each conditional-independence test is for.
    """
    d = len(drivers)
    phases = [
        ("_test_target_edges", "Testing which variables drive the outcome", d),
        ("_test_driver_skeleton", "Testing links between the other variables", d * (d - 1) // 2),
    ]
    total = max(sum(n for _, _, n in phases), 1)
    ctx = {"offset": 0, "count": 0, "pair": None, "n": 1, "text": ""}

    original_test = model._ci_independent

    def counting_test(df, x, y, cond):
        pair = (x, y)
        if pair != ctx["pair"]:
            if ctx["pair"] is not None:
                ctx["count"] += 1
            ctx["pair"] = pair
            progress.set(
                f"{ctx['text']} ({ctx['count'] + 1} of {ctx['n']})",
                (ctx["offset"] + ctx["count"]) / total,
            )
        return original_test(df, x, y, cond)

    model._ci_independent = counting_test

    offset = 0
    for name, text, n in phases:
        original_phase = getattr(model, name)

        def phase(*args, _original=original_phase, _text=text, _n=n, _offset=offset, **kwargs):
            ctx.update(offset=_offset, count=0, pair=None, n=max(_n, 1), text=_text)
            progress.set(f"{_text} (1 of {max(_n, 1)})", _offset / total)
            result = _original(*args, **kwargs)
            progress.set(_text, (_offset + _n) / total)
            return result

        setattr(model, name, phase)
        offset += n


# --------------------------------------------------------------------------
# Discovery (pathmc TBFPC)
# --------------------------------------------------------------------------


@dataclass
class Evidence:
    """Weakest dependence evidence found for a pair during discovery."""

    bf10: float
    conditioning_set: list[str]
    independent: bool
    n_tests: int

    def describe(self) -> str:
        given = ", ".join(self.conditioning_set) if self.conditioning_set else "nothing"
        verdict = "judged independent" if self.independent else "dependence kept"
        return (
            f"Weakest of {self.n_tests} tests: BF₁₀ = {format_bf(self.bf10)} "
            f"given {given} ({verdict})."
        )


def format_bf(value: float) -> str:
    if not math.isfinite(value):
        return "∞"
    if value >= 1e4 or (0 < value < 1e-3):
        return f"{value:.1e}"
    return f"{value:.3g}"


def discover(
    df: pl.DataFrame,
    target: str,
    nodes: list[str] | None = None,
    progress: Progress | None = None,
) -> tuple[list[Edge], dict[str, Evidence]]:
    """Run TBFPC and return its CPDAG plus per-pair evidence.

    ``nodes`` restricts discovery to a subset of columns (the target is always
    included). Columns are standardised first, as pathmc recommends. If
    ``progress`` is given it is kept up to date while TBFPC runs.
    """
    from pathmc import TBFPC

    cols = [c for c in (nodes or df.columns) if c in df.columns]
    if target not in cols:
        cols.append(target)
    drivers = [c for c in cols if c != target]
    if not drivers:
        raise ValueError("Discovery needs at least one variable besides the outcome.")

    pdf = df.select(cols).to_pandas()
    z = (pdf - pdf.mean()) / pdf.std(ddof=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # TBFPC warns that it is experimental
        model = TBFPC(
            target=target,
            bf_thresh=BF_THRESHOLD,
            max_conditioning_set_size=MAX_CONDITIONING_SET,
        )
        if progress is not None:
            progress.set("Starting TBFPC")
            try:
                _watch_tbfpc(model, drivers, progress)
            except Exception:  # noqa: BLE001 - progress is cosmetic; fall back to no counts
                pass
        model.fit(z, drivers=drivers)
    if progress is not None:
        progress.set("Reading the results")

    edges = [Edge(u, v, True) for u, v in model.get_directed_edges()]
    edges += [Edge(u, v, False) for u, v in model.get_undirected_edges()]

    evidence: dict[str, Evidence] = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            results = model.get_test_results(a, b)
            if not results:
                continue
            weakest = min(results, key=lambda r: float(r["BF10"]))
            evidence[edge_key(a, b)] = Evidence(
                bf10=float(weakest["BF10"]),
                conditioning_set=list(weakest["conditioning_set"]),
                independent=bool(weakest["independent"]),
                n_tests=len(results),
            )
    return edges, evidence


# --------------------------------------------------------------------------
# Layout and edge geometry (plain SVG coordinates)
# --------------------------------------------------------------------------


def layout(nodes: list[str], edges: list[Edge], target: str = "") -> dict[str, tuple[float, float]]:
    """Top-down layered layout. Causes sit above their effects."""
    if not nodes:
        return {}

    # Orient undirected edges only for the purpose of layering.
    oriented: list[Edge] = [e for e in edges if e.directed]
    for e in edges:
        if not e.directed:
            a, b = sorted((e.source, e.target))
            if b == target or not creates_cycle(oriented, a, b):
                oriented.append(Edge(a, b))
            else:
                oriented.append(Edge(b, a))

    parents: dict[str, list[str]] = {n: [] for n in nodes}
    for e in oriented:
        if e.source in parents and e.target in parents:
            parents[e.target].append(e.source)

    layer: dict[str, int] = {}

    def depth(n: str, trail: frozenset[str] = frozenset()) -> int:
        if n in layer:
            return layer[n]
        if n in trail:  # defensive: a cycle slipped in
            return 0
        d = 1 + max((depth(p, trail | {n}) for p in parents[n]), default=-1)
        layer[n] = d
        return d

    for n in nodes:
        depth(n)
    if target in layer:
        layer[target] = max(layer.values())

    rows: dict[int, list[str]] = {}
    for n in sorted(nodes):
        rows.setdefault(layer[n], []).append(n)

    # One barycentre sweep to reduce crossings.
    slot: dict[str, float] = {}
    for depth_i in sorted(rows):
        members = rows[depth_i]
        if depth_i > 0:
            members.sort(
                key=lambda n: (
                    np.mean([slot[p] for p in parents[n] if p in slot]) if parents[n] else 0.0,
                    n,
                )
            )
        for i, n in enumerate(members):
            slot[n] = i - (len(members) - 1) / 2

    widest = max(len(m) for m in rows.values())
    centre_x = max(MARGIN_X + (widest - 1) * ROW_GAP / 2, MIN_CENTRE_X)
    return {
        n: (centre_x + slot[n] * ROW_GAP, MARGIN_Y + layer[n] * LAYER_GAP) for n in nodes
    }


def canvas_size(positions: dict[str, tuple[float, float]]) -> tuple[float, float]:
    if not positions:
        return 2 * MIN_CENTRE_X, 320.0
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    return max(max(xs) + MARGIN_X, 2 * MIN_CENTRE_X), max(ys) + MARGIN_Y


def _clip(cx: float, cy: float, tx: float, ty: float) -> tuple[float, float]:
    """Point where the ray from a node centre towards (tx, ty) leaves its box."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    hw, hh = NODE_W / 2 + 3, NODE_H / 2 + 3
    scale = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
    return cx + dx * scale, cy + dy * scale


def _clearance(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    obstacles: list[tuple[float, float]],
) -> float:
    """How far a quadratic curve stays from other node boxes (<= 0 means it hits one)."""
    if not obstacles:
        return math.inf
    worst = math.inf
    for i in range(1, 12):
        t = i / 12
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
        for ox, oy in obstacles:
            gap = max(abs(x - ox) - NODE_W / 2, abs(y - oy) - NODE_H / 2)
            worst = min(worst, gap)
    return worst


def edge_geometry(
    p_source: tuple[float, float],
    p_target: tuple[float, float],
    directed: bool,
    obstacles: list[tuple[float, float]] | None = None,
) -> tuple[str, str]:
    """SVG path ``d`` and arrowhead polygon ``points`` for one edge.

    The edge is straight when nothing is in the way. Otherwise it is bowed left
    or right, whichever keeps it clear of the other node boxes (``obstacles``
    are their centres).
    """
    (x1, y1), (x2, y2) = p_source, p_target
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    span = max(abs(y2 - y1) / LAYER_GAP, 1.0)
    step = 70.0 + 30.0 * (span - 1)

    best, best_gap = (mx, my), -math.inf
    for bow in (0.0, -step, step, -2 * step, 2 * step):
        control = (max(mx + bow, 2.0), my)
        start, end = _clip(x1, y1, *control), _clip(x2, y2, *control)
        gap = _clearance(start, control, end, obstacles or [])
        if gap > best_gap + 1e-9:
            best, best_gap = control, gap
        if gap >= 8.0:  # first candidate with comfortable room wins
            best = control
            break
    cx, cy = best

    sx, sy = _clip(x1, y1, cx, cy)
    ex, ey = _clip(x2, y2, cx, cy)
    d = f"M {sx:.1f} {sy:.1f} Q {cx:.1f} {cy:.1f} {ex:.1f} {ey:.1f}"

    if not directed:
        return d, ""
    ang = math.atan2(ey - cy, ex - cx)
    size, spread = 9.0, 0.42
    a = (ex - size * math.cos(ang - spread), ey - size * math.sin(ang - spread))
    b = (ex - size * math.cos(ang + spread), ey - size * math.sin(ang + spread))
    points = f"{ex:.1f},{ey:.1f} {a[0]:.1f},{a[1]:.1f} {b[0]:.1f},{b[1]:.1f}"
    return d, points


# --------------------------------------------------------------------------
# Validation: does the data contradict the graph? (no fit needed)
# --------------------------------------------------------------------------


@dataclass
class Violation:
    """An independence the graph implies but the data reject."""

    x: str
    y: str
    given: list[str]
    partial_corr: float
    p_value: float

    @property
    def key(self) -> str:
        return edge_key(self.x, self.y)


@dataclass
class Validation:
    verdict: str  # one-line whole-graph verdict
    detail: str
    tone: str  # "good", "bad" or "neutral"
    n_tests: int  # independence statements tested
    alpha: float
    violations: list[Violation]  # strongest (smallest p) first
    n_collapsed: int  # tests skipped because a conditioner was redundant

    @property
    def expected_by_chance(self) -> float:
        return self.alpha * self.n_tests


def validate_graph(
    df: pl.DataFrame, nodes: list[str], edges: list[Edge], alpha: float = 0.05
) -> Validation:
    """Test the graph against the data with pathmc, without fitting a model.

    1. Every conditional independence the DAG implies (Shipley's basis set,
       from ``PathModel.implied_independences``) is tested with pathmc's
       partial-correlation test. pathmc's spec cannot hold nodes without
       edges, so their statements are added here: an unconnected node ``n``
       implies ``n ⊥ y | parents(y)`` for every other node ``y``.
    2. ``PathModel.falsify`` grades the connected graph as a whole against
       randomly relabelled versions of itself (Eulig et al., 2023).
    """
    import narwhals as nw
    import pathmc
    from pathmc.identify import ConditionalIndependence, test_implications

    if any(not e.directed for e in edges):
        raise ValueError("Orient every dashed edge before validating.")
    for e in edges:
        if creates_cycle([x for x in edges if x is not e], e.source, e.target):
            raise ValueError("The graph contains a directed cycle.")
    if len(nodes) < 2:
        raise ValueError("Need at least two nodes to validate.")

    connected = sorted({e.source for e in edges} | {e.target for e in edges})
    statements: list = []
    model = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if edges:
            spec = dag_to_spec_from_edges(edges)
            model = pathmc.model(spec, data=df.select(connected).to_pandas())
            statements.extend(model.implied_independences())
        parents = {n: frozenset(e.source for e in edges if e.target == n) for n in nodes}
        isolated = [n for n in nodes if n not in connected]
        for i, n in enumerate(isolated):
            for other in nodes:
                if other == n or other in isolated[:i]:
                    continue
                statements.append(ConditionalIndependence(n, other, parents[other]))

        frame = pd.DataFrame()
        if statements:
            data = nw.from_native(df.select(nodes).to_pandas(), eager_only=True)
            frame = test_implications(statements, data, alpha=alpha).to_dataframe()

    violations = []
    if not frame.empty:
        for row in frame[frame["significant"]].sort_values("p_value").itertuples():
            given = [g.strip() for g in str(row.conditioning_set).split(",") if g.strip()]
            violations.append(
                Violation(str(row.x), str(row.y), given, float(row.partial_corr),
                          float(row.p_value))
            )
    n_collapsed = int(frame["rank_collapsed"].sum()) if not frame.empty else 0

    verdict, detail, tone = _falsify_verdict(model, connected, isolated)
    return Validation(verdict, detail, tone, len(frame), alpha, violations, n_collapsed)


def violation_given(given: list[str]) -> str:
    return f"given {', '.join(given)}" if given else "with nothing held fixed"


def violation_stats(partial_corr: float, p_value: float) -> str:
    return f"partial r = {partial_corr:.2f} · p = {p_value:.2g}"


def validation_summary(result: Validation) -> str:
    n = len(result.violations)
    text = (
        f"{n} of {result.n_tests} implied independences {'is' if n == 1 else 'are'} violated. "
        f"At α = {result.alpha:g}, about {result.expected_by_chance:.1f} would be flagged by "
        "chance alone."
    )
    if result.n_collapsed:
        text += f" {result.n_collapsed} test(s) skipped: a conditioning variable was redundant."
    return text


def dag_to_spec_from_edges(edges: list[Edge]) -> str:
    import pathmc

    return pathmc.dag_to_spec(", ".join(f"{e.source}->{e.target}" for e in edges))


def _falsify_verdict(model, connected: list[str], isolated: list[str]) -> tuple[str, str, str]:
    """Whole-graph verdict from ``PathModel.falsify``."""
    note = (
        " This whole-graph test covers connected nodes only; unconnected nodes are "
        "checked in the list below."
        if isolated and model is not None
        else ""
    )
    if model is None:
        return (
            "No edges to grade yet",
            "The graph says every variable is unrelated to every other. The list below "
            "tests exactly that.",
            "neutral",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.falsify(random_seed=RANDOM_SEED)
    except Exception as exc:  # noqa: BLE001 - small or saturated graphs can't be graded
        return "Whole-graph test not available", f"pathmc could not run it: {exc}", "neutral"

    status = str(result.status)
    p = getattr(result, "p_value_lmc", float("nan"))
    if status == "not_rejected":
        return (
            "Not contradicted by the data",
            f"It breaks fewer of its own independence claims than randomly relabelled "
            f"versions of the graph (p = {p:.3f}), and its structure is specific enough "
            f"for that to mean something." + note,
            "good",
        )
    if status == "rejected":
        return (
            "Contradicted by the data",
            f"Randomly relabelled versions of the graph fit the data about as well "
            f"(p = {p:.3f}). Look at the violations below." + note,
            "bad",
        )
    if status == "not_falsifiable":
        return (
            "Not testable from the data",
            "Too many rearrangements of this graph imply the same independences, so "
            "the data can neither support nor reject it. Treat it as untested." + note,
            "neutral",
        )
    return (
        "Whole-graph test not available",
        "The graph implies no testable independences between parents and children "
        "(for example, every pair is connected)." + note,
        "neutral",
    )


# --------------------------------------------------------------------------
# Fitting and fitted relationships (pathmc PathModel)
# --------------------------------------------------------------------------


FAMILIES = ("gaussian", "studentt", "bernoulli", "poisson", "negbinomial")


@dataclass(frozen=True)
class PriorParam:
    """One editable number in a node's priors (pathmc parameter ``<prefix>_<node>``)."""

    key: str  # id used in the UI, e.g. "beta.sigma"
    prefix: str
    label: str
    dist: str
    param: str
    default: float  # pathmc's default
    positive: bool


PRIOR_PARAMS = (
    PriorParam("beta.mu", "beta", "Coefficients", "Normal", "mu", 0.0, False),
    PriorParam("beta.sigma", "beta", "Coefficients", "Normal", "sigma", 10.0, True),
    PriorParam("sigma.sigma", "sigma", "Noise scale", "HalfNormal", "sigma", 1.0, True),
    PriorParam("nu.alpha", "nu", "Degrees of freedom", "Gamma", "alpha", 2.0, True),
    PriorParam("nu.beta", "nu", "Degrees of freedom", "Gamma", "beta", 0.1, True),
    PriorParam("alpha_disp.sigma", "alpha_disp", "Dispersion", "HalfNormal", "sigma", 1.0, True),
)


def prior_params(family: str) -> list[PriorParam]:
    """The priors a node with this family has. ``beta`` covers intercept and slopes."""
    keep = {"beta"}
    if family in ("gaussian", "studentt"):
        keep.add("sigma")
    if family == "studentt":
        keep.add("nu")
    if family == "negbinomial":
        keep.add("alpha_disp")
    return [p for p in PRIOR_PARAMS if p.prefix in keep]


def prior_specs(
    edges: list[Edge],
    families: dict[str, str],
    typed: dict[str, dict[str, str]],
) -> dict[str, tuple[str, dict[str, float]]]:
    """Turn the numbers typed per node into pathmc prior overrides.

    ``typed`` maps node -> {PriorParam.key -> text}; a blank box means the
    default. Only nodes with parents get priors (pathmc rejects the rest), and
    a prior is only overridden when at least one of its numbers was typed.
    Returns ``{"beta_y": ("Normal", {"mu": 0.0, "sigma": 2.0}), ...}``.
    """
    children = {e.target for e in edges if e.directed}
    specs: dict[str, tuple[str, dict[str, float]]] = {}
    for node in sorted(children):
        entered = typed.get(node, {})
        groups: dict[str, list[PriorParam]] = {}
        for p in prior_params(families.get(node, "gaussian")):
            groups.setdefault(p.prefix, []).append(p)
        for prefix, group in groups.items():
            if not any(entered.get(p.key, "").strip() for p in group):
                continue
            values: dict[str, float] = {}
            for p in group:
                text = entered.get(p.key, "").strip()
                try:
                    value = float(text) if text else p.default
                except ValueError:
                    raise ValueError(
                        f"{node}: the {p.label.lower()} prior's {p.param} must be a number, "
                        f"not '{text}'."
                    ) from None
                if not math.isfinite(value) or (p.positive and value <= 0):
                    need = "a positive number" if p.positive else "a finite number"
                    raise ValueError(
                        f"{node}: the {p.label.lower()} prior's {p.param} must be {need}."
                    )
                values[p.param] = value
            specs[f"{prefix}_{node}"] = (group[0].dist, values)
    return specs


def family_problems(
    df: pl.DataFrame, families: dict[str, str], grouped: bool = False
) -> list[str]:
    """Columns whose values the chosen family cannot model.

    With ``grouped`` the data have had group means removed, which turns a 0/1 or
    count outcome into continuous values, so those families are ruled out.
    """
    problems = []
    for node, family in sorted(families.items()):
        if node not in df.columns:
            continue
        if grouped and family in NON_GAUSSIAN:
            problems.append(
                f"{node}: the {family} family cannot be fitted once group means are removed "
                "(demeaning makes a 0/1 or count outcome continuous). Clear the grouping to use it."
            )
            continue
        values = df[node].to_numpy()
        if family == "bernoulli" and not np.isin(values, [0.0, 1.0]).all():
            problems.append(f"{node} must contain only 0 and 1 to use the bernoulli family.")
        if family in ("poisson", "negbinomial") and not (
            (values >= 0) & (values == np.round(values))
        ).all():
            problems.append(
                f"{node} must contain non-negative whole numbers to use the {family} family."
            )
    return problems


def _fmt_number(value: float) -> str:
    return f"{value:g}"


def builder_code(
    dag: str,
    target: str,
    families: dict[str, str],
    priors: dict[str, tuple[str, dict[str, float]]],
) -> str:
    """The BuildModelFromDAG call that reproduces a fit (``df`` is the cleaned data)."""
    lines = [
        "builder = BuildModelFromDAG(",
        f'    dag="{dag}",',
        "    df=data,",
        f'    target="{target}",',
        '    style="native",',
    ]
    if families:
        items = ", ".join(f'"{k}": "{v}"' for k, v in sorted(families.items()))
        lines.append(f"    families={{{items}}},")
    if priors:
        lines.append("    priors={")
        for name, (dist, params) in sorted(priors.items()):
            args = ", ".join(f"{k}={_fmt_number(v)}" for k, v in params.items())
            lines.append(f'        "{name}": Prior("{dist}", {args}),')
        lines.append("    },")
    lines.append(")")
    return "\n".join(lines)


@dataclass
class FitPlan:
    """Everything a fit takes from the app's settings, checked and in a fixed order."""

    used: list[str]  # the columns the model uses, sorted
    pairs: list[tuple[str, str]]  # (parent, child), sorted: the same graph always fits the same way
    families: dict[str, str]  # non-gaussian families, for nodes that have parents
    specs: dict[str, tuple[str, dict[str, float]]]  # prior overrides, see prior_specs

    @property
    def dag(self) -> str:
        return ", ".join(f"{u}->{v}" for u, v in self.pairs)


def prepare_fit(
    df: pl.DataFrame,
    nodes: list[str],
    edges: list[Edge],
    target: str,
    families: dict[str, str] | None = None,
    priors: dict[str, dict[str, str]] | None = None,
    grouped: bool = False,
) -> FitPlan:
    """Check the graph and settings, and say exactly what the fit will use.

    Raises a ValueError that says what to fix. The edges are put in sorted order,
    so pathmc builds its equations in the same order however the graph was
    edited, and so the exported script (which sorts the same way) fits identically.
    """
    problems = graph_problems(nodes, edges, target)
    if problems:
        raise ValueError(" ".join(problems))

    children = {e.target for e in edges}
    families = {
        n: f for n, f in (families or {}).items() if n in children and f != "gaussian"
    }
    unknown = sorted(f for f in families.values() if f not in FAMILIES)
    if unknown:
        raise ValueError(f"Unknown family: {', '.join(unknown)}.")
    problems = family_problems(df, families, grouped)
    if problems:
        raise ValueError(" ".join(problems))
    specs = prior_specs(edges, families, priors or {})
    used = sorted({e.source for e in edges} | {e.target for e in edges})
    return FitPlan(used, sorted((e.source, e.target) for e in edges), families, specs)


def fit_model(
    df: pl.DataFrame,
    nodes: list[str],
    edges: list[Edge],
    target: str,
    families: dict[str, str] | None = None,
    priors: dict[str, dict[str, str]] | None = None,
    progress: Progress | None = None,
    grouped: bool = False,
):
    """Build the model with pathmc's ``BuildModelFromDAG`` (native style) and run MCMC.

    ``families`` maps node -> family and ``priors`` maps node -> the numbers
    typed in the UI (see :func:`prior_specs`). Settings for nodes without
    parents, and "gaussian" (the default), are left out of the call.
    If ``progress`` is given it is kept up to date: compiling and finishing have
    no counts, but sampling reports the share of draws done.

    Returns a :class:`FitResult`.
    """
    import pathmc
    from pymc_extras.prior import Prior

    plan = prepare_fit(df, nodes, edges, target, families, priors, grouped)
    families, specs, dag = plan.families, plan.specs, plan.dag
    builder = pathmc.BuildModelFromDAG(
        dag=dag,
        df=df.select(plan.used).to_pandas(),
        target=target,
        style="native",
        families=families or None,
        priors={name: Prior(dist, **params) for name, (dist, params) in specs.items()} or None,
    )
    model = builder.to_pathmodel()
    spec = pathmc.dag_to_spec(dag)
    if progress is not None:
        progress.set(fit_step(1))
    with warnings.catch_warnings(), _watch_sampling(progress):
        warnings.simplefilter("ignore")
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            progressbar=False,
            random_seed=RANDOM_SEED,
        )
    return FitResult(
        model, idata, spec, builder_code(dag, target, families, specs), families, plan
    )


def fit_snapshot(
    plan: FitPlan,
    outcome: str,
    group_columns: list[str],
    time_column: str,
    data_kind: str,
    data_file: str,
) -> dict:
    """What a fit used, in plain types, so the app can remember it and export it later."""
    return {
        "pairs": [list(pair) for pair in plan.pairs],
        "families": dict(plan.families),
        "specs": {name: [dist, dict(params)] for name, (dist, params) in plan.specs.items()},
        "outcome": outcome,
        "group_columns": list(group_columns),
        "time_column": time_column,
        "data_kind": data_kind,  # "file", or "demo" for the built-in data
        "data_file": data_file,
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "seed": RANDOM_SEED,
    }


@dataclass
class FitResult:
    model: object  # pathmc.PathModel
    idata: object  # the posterior DataTree returned by PathModel.fit
    spec: str
    code: str  # the equivalent BuildModelFromDAG call
    families: dict[str, str]  # non-gaussian families actually used
    plan: FitPlan | None = None  # what the fit used: the graph, families and priors


# --------------------------------------------------------------------------
# Fit quality: Bayesian R² per node and sampler diagnostics
# --------------------------------------------------------------------------

# Vehtari et al. (2021): r̂ < 1.01 and bulk ESS >= 400. Between 1.01 and 1.05 is
# usually tail noise on a short run; above 1.05 the chains genuinely disagree.
RHAT_GOOD, RHAT_BAD, ESS_MIN = 1.01, 1.05, 400


@dataclass
class NodeFit:
    """Share of a node's variance explained by its parents (Bayesian R²)."""

    node: str
    family: str
    r2: float  # posterior median
    lower: float  # equal-tailed INTERVAL bounds
    upper: float


@dataclass
class FitQuality:
    nodes: list[NodeFit]
    max_rhat: float
    min_ess: float
    divergences: int

    def warnings(self) -> list[str]:
        """Sampler problems, worst first. Empty means the fit can be trusted."""
        out = []
        if not self.max_rhat < RHAT_BAD:
            out.append(
                f"The chains disagree (largest r̂ {self.max_rhat:.3f}), so the sampler has not "
                "converged. Don't trust this fit; try stronger priors or more draws."
            )
        elif not self.max_rhat < RHAT_GOOD:
            out.append(
                f"Borderline convergence (largest r̂ {self.max_rhat:.3f}, ideally below "
                f"{RHAT_GOOD}). Usually harmless on a short run; more draws would confirm it."
            )
        if not self.min_ess >= ESS_MIN:
            out.append(
                f"Too few effective draws (smallest ESS {self.min_ess:.0f}, should be "
                f"{ESS_MIN}+). Raise DRAWS in backend.py."
            )
        if self.divergences:
            out.append(
                f"{self.divergences} divergent transitions: the sampler struggled, so "
                "estimates may be biased. Stronger priors often help."
            )
        return out


CONVERGED_NOTE = "The sampler converged: these estimates can be trusted as fits."
R2_HEADING = "Variance explained by each node's parents (Bayesian R², 89% interval)"
R2_HEADING_GROUPED = (
    "Variance explained by each node's parents within groups, after group means are removed "
    "(Bayesian R², 89% interval)"
)


@dataclass
class QualityRow:
    """One node's R² as the Fit quality section shows it."""

    node: str
    family: str  # empty for gaussian
    r2: str
    interval: str
    bar: str  # width of the bar, e.g. "76%"
    is_target: bool


def r2_text(node: NodeFit) -> str:
    return f"{node.r2:.2f} [{node.lower:.2f}, {node.upper:.2f}]"


def quality_rows(quality: FitQuality, target: str) -> list[QualityRow]:
    """The outcome first, then the other nodes alphabetically."""
    nodes = sorted(quality.nodes, key=lambda n: (n.node != target, n.node))
    return [
        QualityRow(
            node=n.node,
            family="" if n.family == "gaussian" else n.family,
            r2=f"{n.r2:.2f}",
            interval=f"[{n.lower:.2f}, {n.upper:.2f}]",
            bar=f"{max(n.r2, 0) * 100:.0f}%",
            is_target=n.node == target,
        )
        for n in nodes
    ]


def quality_diagnostics(quality: FitQuality, chains: int, draws: int) -> str:
    return (
        f"{chains} chains × {draws} draws · largest r̂ {quality.max_rhat:.3f} · "
        f"smallest ESS {quality.min_ess:.0f} · {quality.divergences} divergences"
    )


def edge_tooltip(source: str, target: str, directed: bool, term: Term | None = None) -> str:
    """"a → b", with the fitted direct coefficient once there is one."""
    text = f"{source} {'→' if directed else '—'} {target}"
    if term is not None:
        text += f": direct coefficient {term.mean:.3f} [{term.lower:.3f}, {term.upper:.3f}]"
    return text


def _draws(posterior, name: str) -> np.ndarray:
    """Stack chains: (chain, draw, ...) -> (sample, ...)."""
    values = np.asarray(posterior[name])
    return values.reshape(-1, *values.shape[2:])


def bayes_r2(idata, node: str, family: str = "gaussian") -> np.ndarray:
    """Bayesian R² draws for one node (Gelman, Goodrich, Gabry & Vehtari, 2019).

    Per posterior draw: var(fitted means) / (var(fitted means) + residual
    variance). The residual variance comes from the family: sigma² (gaussian),
    p(1 - p) (bernoulli), lambda (poisson), mu + mu²/alpha (negbinomial). For
    studentt, whose sigma is not a variance, the observed residuals are used.
    """
    post = idata["posterior"]
    count_like = family in ("bernoulli", "poisson", "negbinomial")
    mean = _draws(post, f"pathmc_resp_{node}" if count_like else f"mu_{node}")
    var_fit = mean.var(axis=1, ddof=1)
    if family == "gaussian":
        var_res = _draws(post, f"sigma_{node}") ** 2
    elif family == "bernoulli":
        var_res = (mean * (1 - mean)).mean(axis=1)
    elif family == "poisson":
        var_res = mean.mean(axis=1)
    elif family == "negbinomial":
        alpha = _draws(post, f"alpha_disp_{node}")
        var_res = (mean + mean**2 / alpha[:, None]).mean(axis=1)
    else:
        observed = np.asarray(idata["observed_data"][node], dtype=float)
        var_res = (observed[None, :] - mean).var(axis=1, ddof=1)
    return var_fit / (var_fit + var_res)


def fit_quality(fit: FitResult) -> FitQuality:
    """R² for every node with parents, plus r̂, ESS and divergences."""
    post = fit.idata["posterior"]
    nodes = sorted(
        name[len("mu_"):] for name in post.data_vars if str(name).startswith("mu_")
    )
    lo_q, hi_q = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2
    node_fits = []
    for node in nodes:
        family = fit.families.get(node, "gaussian")
        draws = bayes_r2(fit.idata, node, family)
        node_fits.append(
            NodeFit(node, family, float(np.median(draws)),
                    float(np.quantile(draws, lo_q)), float(np.quantile(draws, hi_q)))
        )

    import arviz as az

    # Only the model's parameters: the per-row means are functions of them.
    ds = post.to_dataset()
    ds = ds[[v for v in ds.data_vars if not str(v).startswith(("mu_", "pathmc_resp_"))]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rhat, ess = az.rhat(ds), az.ess(ds, method="bulk")
    max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
    min_ess = min(float(ess[v].min()) for v in ess.data_vars)
    divergences = int(np.asarray(fit.idata["sample_stats"]["diverging"]).sum())
    return FitQuality(node_fits, max_rhat, min_ess, divergences)


@dataclass
class Curve:
    """E[target | do(source = x)] on a grid, with a credible band."""

    source: str
    target: str
    x: list[float]
    mean: list[float]
    lower: list[float]
    upper: list[float]


@dataclass
class Term:
    """One posterior summary: mean and equal-tailed interval."""

    name: str  # "Intercept" or the parent node
    mean: float
    lower: float
    upper: float


@dataclass
class Equation:
    """The fitted structural equation of one node.

    The linear predictor ``intercept + sum(coef * parent)`` is the mean for
    gaussian and studentt nodes, ``logit P(node = 1)`` for bernoulli and
    ``log E[node]`` for poisson and negbinomial.
    """

    node: str
    intercept: Term
    terms: list[Term]  # one per parent, in the order pathmc reports them
    sigma: Term | None
    interval: str  # e.g. "89%"
    family: str = "gaussian"
    nu: Term | None = None  # studentt degrees of freedom
    alpha: Term | None = None  # negbinomial dispersion

    def lines(self, highlight: str = "") -> list[tuple[str, str, bool]]:
        """Display rows: (expression, interval, is the ``highlight`` parent's term)."""

        def band(t: Term) -> str:
            return f"[{t.lower:.3f}, {t.upper:.3f}]"

        lhs = {
            "bernoulli": f"logit P({self.node} = 1)",
            "poisson": f"log E[{self.node}]",
            "negbinomial": f"log E[{self.node}]",
        }.get(self.family, self.node)
        rows = [(f"{lhs} = {self.intercept.mean:.3f}", band(self.intercept), False)]
        for t in self.terms:
            sign = "−" if t.mean < 0 else "+"
            rows.append((f"{sign} {abs(t.mean):.3f} · {t.name}", band(t), t.name == highlight))
        if self.family == "studentt" and self.sigma is not None and self.nu is not None:
            rows.append(("+ ε,  ε ~ StudentT(ν, 0, σ)", "", False))
            rows.append((f"    σ = {self.sigma.mean:.3f}", band(self.sigma), False))
            rows.append((f"    ν = {self.nu.mean:.3f}", band(self.nu), False))
        elif self.sigma is not None:
            rows.append(
                (f"+ ε,  ε ~ Normal(0, {self.sigma.mean:.3f})", band(self.sigma), False)
            )
        if self.family == "poisson":
            rows.append((f"{self.node} ~ Poisson(E[{self.node}])", "", False))
        if self.family == "negbinomial" and self.alpha is not None:
            rows.append((
                f"{self.node} ~ NegBinomial(E[{self.node}], α = {self.alpha.mean:.3f})",
                band(self.alpha), False,
            ))
        return rows


# FiveThirtyEight yellow, for cycles; the fill is 20% of it mixed with white.
CYCLE_COLOUR, CYCLE_FILL = "#E5AE38", "#FBF1DA"

# FiveThirtyEight blue and red; the pale tints are 45% of each mixed with white.
EFFECT_BLUE, EFFECT_RED = "#008FD5", "#FC4F30"
EFFECT_COLOURS = {
    "pos": EFFECT_BLUE,
    "neg": EFFECT_RED,
    "pos_weak": "#8CCCEC",
    "neg_weak": "#FDB0A2",
}


def edge_effects(equations: dict[str, "Equation"], edges: list[Edge]) -> dict[str, tuple[str, Term]]:
    """Sign of each fitted edge's direct coefficient, keyed by ``Edge.key``.

    The sign is "pos" or "neg" (on the link scale for non-gaussian children),
    with "_weak" appended when the credible interval includes zero.
    """
    effects: dict[str, tuple[str, Term]] = {}
    for e in edges:
        equation = equations.get(e.target) if e.directed else None
        term = next((t for t in equation.terms if t.name == e.source), None) if equation else None
        if term is None:
            continue
        sign = "pos" if term.mean >= 0 else "neg"
        if term.lower <= 0 <= term.upper:
            sign += "_weak"
        effects[e.key] = (sign, term)
    return effects


def _interval_columns(summary) -> tuple[str, str, str]:
    bounds = [c for c in summary.columns if c.startswith(("eti", "hdi"))]
    lb = next(c for c in bounds if c.endswith("lb"))
    ub = next(c for c in bounds if c.endswith("ub"))
    return lb, ub, f"{re.sub(r'[^0-9]', '', lb) or '?'}%"


def model_equations(model, families: dict[str, str] | None = None) -> dict[str, Equation]:
    """Fitted equation for every node that has parents, from pathmc's summary.

    Reads the ``beta_<node>[<parent or Intercept>]`` rows, plus ``sigma_<node>``,
    ``nu_<node>`` and ``alpha_disp_<node>`` where the node's family has them.
    """
    families = families or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = model.summary()
    lb, ub, level = _interval_columns(summary)

    def term(name: str, label: str) -> Term:
        row = summary.loc[name]
        return Term(label, float(row["mean"]), float(row[lb]), float(row[ub]))

    coefs: dict[str, list[Term]] = {}
    for name in summary.index:
        match = re.fullmatch(r"beta_(.+)\[(.+)\]", str(name))
        if match:
            coefs.setdefault(match[1], []).append(term(name, match[2]))

    equations = {}
    for node, terms in coefs.items():
        intercept = next((t for t in terms if t.name == "Intercept"), Term("Intercept", 0, 0, 0))
        def optional(name: str) -> Term | None:
            return term(name, name) if name in summary.index else None

        equations[node] = Equation(
            node=node,
            intercept=intercept,
            terms=[t for t in terms if t.name != "Intercept"],
            sigma=optional(f"sigma_{node}"),
            interval=level,
            family=families.get(node, "gaussian"),
            nu=optional(f"nu_{node}"),
            alpha=optional(f"alpha_disp_{node}"),
        )
    return equations


def effect_unavailable(edges: list[Edge], node: str, outcome: str) -> str:
    """Why E[outcome | do(node)] would say nothing, or "" when the node can move the outcome."""
    if node == outcome:
        return (
            f"{node} is the outcome. Select a node upstream of it to see how setting that "
            f"node changes {outcome}."
        )
    if not directed_path(edges, node, outcome):
        return (
            f"{node} has no directed path to {outcome}, so setting it does not change "
            f"{outcome} in this graph."
        )
    return ""


def edge_curve(model, df: pl.DataFrame, source: str, target: str) -> Curve:
    """Posterior interventional mean of ``target`` as ``source`` is set.

    Uses pathmc's ``do()`` (g-computation), so the curve is the *total* effect
    of ``source`` on ``target`` through every directed path in the fitted DAG.
    The direct effect is the ``source`` term in :func:`model_equations`.
    """
    lo_q, hi_q = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2
    values = df[source].to_numpy()
    grid = np.linspace(np.quantile(values, 0.02), np.quantile(values, 0.98), GRID_POINTS)

    mean, lower, upper = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x in grid:
            draws = np.asarray(model.do(set={source: float(x)}).draws(target)).ravel()
            mean.append(float(draws.mean()))
            lower.append(float(np.quantile(draws, lo_q)))
            upper.append(float(np.quantile(draws, hi_q)))
    return Curve(
        source=source,
        target=target,
        x=[float(v) for v in grid],
        mean=mean,
        lower=lower,
        upper=upper,
    )


def make_figure(curve: Curve, df: pl.DataFrame, grouped: bool = False):
    """Minimal Plotly figure: observed points, posterior mean, credible band.

    With ``grouped`` the values are deviations from each group's mean, and the axes say so.
    """
    import plotly.graph_objects as go

    ink, accent = "#1f2933", "#2563eb"
    pts = df.select(curve.source, curve.target)
    if pts.height > MAX_SCATTER:
        pts = pts.sample(MAX_SCATTER, seed=RANDOM_SEED)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pts[curve.source].to_list(),
            y=pts[curve.target].to_list(),
            mode="markers",
            marker=dict(size=4, color="#9aa5b1", opacity=0.45),
            name="observed",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve.x + curve.x[::-1],
            y=curve.upper + curve.lower[::-1],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.16)",
            line=dict(width=0),
            name=f"{int(INTERVAL * 100)}% interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve.x,
            y=curve.mean,
            mode="lines",
            line=dict(color=accent, width=2),
            name=f"E[{curve.target} | do({curve.source})]",
        )
    )
    axis = dict(showgrid=False, zeroline=False, linecolor="#cbd2d9", ticks="outside",
                tickcolor="#cbd2d9", tickfont=dict(size=11, color=ink))
    fig.update_layout(
        height=330,
        margin=dict(l=52, r=12, t=10, b=44),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=ink),
        xaxis=dict(title=curve.source + (" − group mean" if grouped else ""), **axis),
        yaxis=dict(title=curve.target + (" − group mean" if grouped else ""), **axis),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=11)),
        showlegend=True,
    )
    return fig


# --------------------------------------------------------------------------
# Demo data
# --------------------------------------------------------------------------


def demo_table(n: int = 800, seed: int = 7) -> pl.DataFrame:
    """Synthetic data with a known DAG, for trying the app without a file.

    price -> demand, promo -> demand, season -> promo, season -> demand,
    demand -> revenue, price -> revenue. ``noise`` is unrelated to everything.
    """
    rng = np.random.default_rng(seed)
    season = rng.normal(size=n)
    price = rng.normal(size=n)
    promo = 0.6 * season + rng.normal(size=n)
    demand = -0.8 * price + 0.7 * promo + 0.5 * season + rng.normal(size=n)
    revenue = 0.9 * demand + 0.4 * price + rng.normal(scale=0.8, size=n)
    noise = rng.normal(size=n)
    return pl.DataFrame(
        dict(season=season, price=price, promo=promo, demand=demand, revenue=revenue, noise=noise)
    )


TEST_FILE = Path(__file__).resolve().parent.parent / "assets" / "dag_studio_test_data.csv"


def test_file_table(n: int = 600, seed: int = 11) -> pl.DataFrame:
    """Raw, deliberately messy data for trying every part of the app by hand.

    It goes through the same cleaning as an upload. Known structure (four layers
    when drawn top-down):

        ad spend -> web visits, season -> web visits, season -> store visits,
        web visits -> orders, store visits -> orders, 2024 price index -> orders,
        orders -> revenue, 2024 price index -> revenue,
        orders -> customer_satisfaction_score,
        customer_satisfaction_score -> repeat_purchase (0/1, logistic)

    ``noise`` is unrelated to everything. ``repeat_purchase`` is for trying
    the bernoulli family. Cleaning should rename two columns,
    ignore ``region`` (text) and ``tax_rate`` (constant), and drop the rows
    where ``web visits`` is missing.
    """
    rng = np.random.default_rng(seed)
    ad_spend = rng.normal(size=n)
    season = rng.normal(size=n)
    price = rng.normal(size=n)
    web = 0.8 * ad_spend + 0.5 * season + rng.normal(size=n)
    store = 0.7 * season + rng.normal(size=n)
    orders = 0.6 * web + 0.5 * store - 0.7 * price + rng.normal(size=n)
    revenue = 0.9 * orders + 0.4 * price + rng.normal(scale=0.8, size=n)
    satisfaction = 0.6 * orders + rng.normal(size=n)
    noise = rng.normal(size=n)
    # Drawn last so the columns above match earlier versions of this file.
    repeat = rng.random(n) < 1 / (1 + np.exp(-(0.3 + 1.2 * satisfaction)))

    web_visits = [None if i % 40 == 7 else float(v) for i, v in enumerate(web)]
    regions = ["north", "south", "east", "west"]
    return pl.DataFrame(
        {
            "ad spend": ad_spend.round(4),
            "season": season.round(4),
            "2024 price index": price.round(4),
            "region": [regions[i % 4] for i in range(n)],
            "web visits": pl.Series(web_visits).round(4),
            "store visits": store.round(4),
            "orders": orders.round(4),
            "tax_rate": [0.2] * n,
            "customer_satisfaction_score": satisfaction.round(4),
            "noise": noise.round(4),
            "repeat_purchase": repeat.astype(np.int64),
            "revenue": revenue.round(4),
        }
    )


PANEL_TEST_FILE = Path(__file__).resolve().parent.parent / "assets" / "dag_studio_panel_data.csv"


def panel_test_table(seed: int = 5) -> pl.DataFrame:
    """Raw panel data with known structure, for trying the grouping controls by hand.

    3 regions x 4 stores x 20 weeks. Store names (s1 to s4) repeat in every
    region, so a store is only identified by region *and* store. Each group has a
    level, ``g``, that also drives ``price``, ``demand`` and ``footfall``.

        price -> demand (-0.8), promo -> demand (0.7), demand -> revenue (0.9),
        price -> revenue (0.4).

    ``footfall`` depends only on the group, so it is unrelated to everything once
    group differences are removed; ignoring groups makes it look connected to
    price and demand. ``store_size`` (constant within a store) and
    ``region_income`` (constant within a region) are group-level variables, which
    removing group means must drop.
    """
    rng = np.random.default_rng(seed)
    regions, stores, weeks = ["north", "south", "east"], ["s1", "s2", "s3", "s4"], 20
    rows = [(r, s_, w) for r in regions for s_ in stores for w in range(1, weeks + 1)]
    n = len(rows)
    region_effect = {r: rng.normal() for r in regions}
    store_effect = {(r, s_): rng.normal() for r in regions for s_ in stores}
    g = np.array([region_effect[r] + store_effect[(r, s_)] for r, s_, _ in rows])
    price = 0.9 * g + rng.normal(size=n)
    promo = rng.normal(size=n)
    demand = -0.8 * price + 0.7 * promo + 1.0 * g + rng.normal(size=n)
    revenue = 0.9 * demand + 0.4 * price + rng.normal(scale=0.8, size=n)
    footfall = 0.8 * g + rng.normal(size=n)
    return pl.DataFrame(
        {
            "region": [r for r, _, _ in rows],
            "store": [s_ for _, s_, _ in rows],
            "week": [w for _, _, w in rows],
            "price": price.round(4),
            "promo": promo.round(4),
            "demand": demand.round(4),
            "revenue": revenue.round(4),
            "footfall": footfall.round(4),
            "store_size": [round(50 + 10 * store_effect[(r, s_)], 1) for r, s_, _ in rows],
            "region_income": [round(60 + 8 * region_effect[r], 1) for r, _, _ in rows],
        }
    )
