"""Pure-Python backend for DAG Studio.

Everything that touches data, pathmc, or graph geometry lives here so it can
be tested without a browser. The Reflex state in ``state.py`` only calls
these functions and stores their (serialisable) results.
"""

from __future__ import annotations

import math
import re
import warnings
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
# Data loading
# --------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    """Make a column name usable as a pathmc DSL identifier."""
    cleaned = re.sub(r"\W", "_", str(name).strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned


def load_table(path: str | Path) -> tuple[pl.DataFrame, list[str]]:
    """Read a CSV or Parquet file and keep the usable numeric columns.

    Returns the cleaned frame and human-readable notes about what changed.
    """
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pl.read_parquet(path)
    else:
        raw = pl.read_csv(path, infer_schema_length=10_000)
    return clean_table(raw)


def clean_table(raw: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    notes: list[str] = []

    numeric = [c for c, dtype in raw.schema.items() if dtype.is_numeric()]
    dropped = [c for c in raw.columns if c not in numeric]
    if dropped:
        notes.append(f"Ignored non-numeric columns: {', '.join(dropped)}")
    df = raw.select(numeric).cast(pl.Float64)

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
    df = df.drop_nulls()
    if df.height < before:
        notes.append(f"Dropped {before - df.height} rows with missing values")

    if df.width < 2:
        raise ValueError("Need at least two numeric, non-constant columns.")
    if df.height < 20:
        raise ValueError("Need at least 20 complete rows.")
    return df, notes


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
    df: pl.DataFrame, target: str, nodes: list[str] | None = None
) -> tuple[list[Edge], dict[str, Evidence]]:
    """Run TBFPC and return its CPDAG plus per-pair evidence.

    ``nodes`` restricts discovery to a subset of columns (the target is always
    included). Columns are standardised first, as pathmc recommends.
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
        model.fit(z, drivers=drivers)

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


def family_problems(df: pl.DataFrame, families: dict[str, str]) -> list[str]:
    """Columns whose values the chosen family cannot model."""
    problems = []
    for node, family in sorted(families.items()):
        if node not in df.columns:
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


def fit_model(
    df: pl.DataFrame,
    nodes: list[str],
    edges: list[Edge],
    target: str,
    families: dict[str, str] | None = None,
    priors: dict[str, dict[str, str]] | None = None,
):
    """Build the model with pathmc's ``BuildModelFromDAG`` (native style) and run MCMC.

    ``families`` maps node -> family and ``priors`` maps node -> the numbers
    typed in the UI (see :func:`prior_specs`). Settings for nodes without
    parents, and "gaussian" (the default), are left out of the call.
    Returns a :class:`FitResult`.
    """
    import pathmc
    from pymc_extras.prior import Prior

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
    problems = family_problems(df, families)
    if problems:
        raise ValueError(" ".join(problems))
    specs = prior_specs(edges, families, priors or {})

    used = sorted({e.source for e in edges} | {e.target for e in edges})
    dag = ", ".join(f"{e.source}->{e.target}" for e in edges)
    builder = pathmc.BuildModelFromDAG(
        dag=dag,
        df=df.select(used).to_pandas(),
        target=target,
        style="native",
        families=families or None,
        priors={name: Prior(dist, **params) for name, (dist, params) in specs.items()} or None,
    )
    model = builder.to_pathmodel()
    spec = pathmc.dag_to_spec(dag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            progressbar=False,
            random_seed=RANDOM_SEED,
        )
    return FitResult(model, idata, spec, builder_code(dag, target, families, specs), families)


@dataclass
class FitResult:
    model: object  # pathmc.PathModel
    idata: object  # the posterior DataTree returned by PathModel.fit
    spec: str
    code: str  # the equivalent BuildModelFromDAG call
    families: dict[str, str]  # non-gaussian families actually used


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


def make_figure(curve: Curve, df: pl.DataFrame):
    """Minimal Plotly figure: observed points, posterior mean, credible band."""
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
        xaxis=dict(title=curve.source, **axis),
        yaxis=dict(title=curve.target, **axis),
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
