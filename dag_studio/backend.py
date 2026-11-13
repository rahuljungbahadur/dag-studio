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
DRAWS, TUNE, CHAINS = 500, 500, 2  # MCMC budget for one fit
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


def creates_cycle(edges: list[Edge], source: str, target: str) -> bool:
    """True if adding ``source -> target`` closes a directed cycle."""
    children: dict[str, list[str]] = {}
    for e in edges:
        if e.directed:
            children.setdefault(e.source, []).append(e.target)
    stack, seen = [target], set()
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children.get(node, []))
    return False


def graph_problems(nodes: list[str], edges: list[Edge]) -> list[str]:
    """Reasons the current graph cannot be fitted yet (empty list = ready)."""
    problems = []
    if not edges:
        problems.append("The graph has no edges.")
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
# Fitting and fitted relationships (pathmc PathModel)
# --------------------------------------------------------------------------


def fit_model(df: pl.DataFrame, nodes: list[str], edges: list[Edge]):
    """Translate the DAG to a pathmc spec, build the model and run MCMC."""
    import pathmc

    problems = graph_problems(nodes, edges)
    if problems:
        raise ValueError(" ".join(problems))

    used = sorted({e.source for e in edges} | {e.target for e in edges})
    dot = "digraph { " + " ".join(f"{e.source} -> {e.target};" for e in edges) + " }"
    spec = pathmc.dag_to_spec(dot)
    model = pathmc.model(spec, data=df.select(used).to_pandas())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            progressbar=False,
            random_seed=RANDOM_SEED,
        )
    return model, spec


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

    ``node = intercept + sum(coef * parent) + noise``, noise ~ Normal(0, sigma).
    """

    node: str
    intercept: Term
    terms: list[Term]  # one per parent, in the order pathmc reports them
    sigma: Term | None
    interval: str  # e.g. "89%"

    def lines(self, highlight: str = "") -> list[tuple[str, str, bool]]:
        """Display rows: (expression, interval, is the ``highlight`` parent's term)."""

        def band(t: Term) -> str:
            return f"[{t.lower:.3f}, {t.upper:.3f}]"

        rows = [(f"{self.node} = {self.intercept.mean:.3f}", band(self.intercept), False)]
        for t in self.terms:
            sign = "−" if t.mean < 0 else "+"
            rows.append((f"{sign} {abs(t.mean):.3f} · {t.name}", band(t), t.name == highlight))
        if self.sigma is not None:
            rows.append(
                (f"+ ε,  ε ~ Normal(0, {self.sigma.mean:.3f})", band(self.sigma), False)
            )
        return rows


def _interval_columns(summary) -> tuple[str, str, str]:
    bounds = [c for c in summary.columns if c.startswith(("eti", "hdi"))]
    lb = next(c for c in bounds if c.endswith("lb"))
    ub = next(c for c in bounds if c.endswith("ub"))
    return lb, ub, f"{re.sub(r'[^0-9]', '', lb) or '?'}%"


def model_equations(model) -> dict[str, Equation]:
    """Fitted equation for every node that has parents, from pathmc's summary.

    Reads the ``beta_<node>[<parent or Intercept>]`` and ``sigma_<node>`` rows.
    """
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
        sigma = f"sigma_{node}"
        equations[node] = Equation(
            node=node,
            intercept=intercept,
            terms=[t for t in terms if t.name != "Intercept"],
            sigma=term(sigma, "sigma") if sigma in summary.index else None,
            interval=level,
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
        orders -> customer_satisfaction_score

    ``noise`` is unrelated to everything. Cleaning should rename two columns,
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
            "revenue": revenue.round(4),
        }
    )
