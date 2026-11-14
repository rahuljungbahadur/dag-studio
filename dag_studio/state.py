"""Reflex state for DAG Studio.

The state holds only small, serialisable things (names, edges, positions,
messages). The uploaded table and the fitted pathmc model stay on the server
in ``_SESSIONS``, keyed by the browser tab's client token.
"""

from __future__ import annotations

import asyncio
import dataclasses
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import reflex as rx

from . import backend as b
from . import export

# token -> {"df": pl.DataFrame, "model": PathModel, "evidence": {...}, "curves": {...}}
_SESSIONS: dict[str, dict] = {}


@dataclasses.dataclass
class NodeView:
    """A node as the SVG needs it."""

    id: str
    label: str
    box_x: str  # top-left corner of the box
    box_y: str
    text_x: str
    text_y: str
    dot_x: str  # review marker, top-right corner
    dot_y: str
    note_x: str  # description marker, top-left corner
    flagged: bool
    picked: bool
    is_target: bool
    tooltip: str  # full name, plus the description when there is one
    described: bool
    hinted: bool  # part of the selected validation violation
    neighbour: bool  # connected to the picked node
    dimmed: bool  # a node is picked and this one is not connected to it
    in_cycle: bool  # part of the highlighted cycle


@dataclasses.dataclass
class EdgeView:
    """An edge as the SVG needs it."""

    id: str
    source: str
    target: str
    directed: bool
    flagged: bool
    selected: bool
    d: str
    arrow: str
    colour: str
    width: str
    opacity: str
    tooltip: str


@dataclasses.dataclass
class EquationRow:
    """One line of a fitted equation."""

    expr: str
    band: str
    highlight: bool  # the term for the selected edge


@dataclasses.dataclass
class R2Row:
    """One node's Bayesian R² in the Fit quality table."""

    node: str
    family: str
    r2: str
    interval: str
    bar: str  # CSS width of the bar, e.g. "76%"
    is_target: bool


@dataclasses.dataclass
class ViolationView:
    """One violated independence in the Validate step."""

    key: str
    x: str
    y: str
    given: str
    stats: str
    selected: bool
    forward: str  # button labels
    backward: str


@dataclasses.dataclass
class PriorRow:
    """One editable prior number in the node panel."""

    key: str  # backend.PriorParam.key
    caption: str  # e.g. "Coefficients ~ Normal"; empty for the prior's second number
    param: str  # e.g. "σ"
    value: str  # what the user typed; empty means pathmc's default
    default: str


_SYMBOLS = {"mu": "μ", "sigma": "σ", "alpha": "α", "beta": "β"}


# Graph colours (kept in step with dag_studio.py).
_ACCENT, _AMBER, _EDGE = "#2563eb", "#d97706", "#9aa5b1"


async def _follow(state, task: "asyncio.Task", progress: b.Progress) -> None:
    """Push a worker's progress to the page until ``task`` finishes.

    ``state`` is the background handler's ``self``. Only changes are sent, so a
    quiet task does not flood the connection.
    """
    last = None
    while True:
        done, _ = await asyncio.wait({task}, timeout=0.2)
        if done:
            return
        label, fraction = progress.snapshot()
        current = (label, None if fraction is None else round(fraction * 100))
        if current != last and label:
            last = current
            async with state:
                state._set_progress(label, fraction)


def _short(name: str, limit: int = 13) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


class State(rx.State):
    # --- data ---------------------------------------------------------
    filename: str = ""
    n_rows: int = 0
    columns: list[str] = []
    notes: list[str] = []
    target: str = ""

    # --- graph (source of truth) --------------------------------------
    nodes: list[str] = []
    removed: list[str] = []
    edges: list[dict] = []  # {"source", "target", "directed"}
    flagged_nodes: list[str] = []
    flagged_edges: list[str] = []
    positions: dict[str, list[float]] = {}
    descriptions: dict[str, str] = {}  # node -> free-text note written by the user
    families: dict[str, str] = {}  # node -> pathmc family; missing means gaussian
    prior_text: dict[str, dict[str, str]] = {}  # node -> {PriorParam.key -> typed text}

    # --- selection ----------------------------------------------------
    selected_nodes: list[str] = []  # in the order picked; cmd-click adds a second (max 2)
    selected_edge: str = ""

    # --- derived views for rendering ----------------------------------
    node_views: list[NodeView] = []
    edge_views: list[EdgeView] = []
    refused_cycle: list[str] = []  # cycle (target ... source) a refused edit would close
    cycle_note: str = ""
    cycle_line: list[str] = []  # x1, y1, x2, y2 of the refused edge
    canvas_w: int = 640
    canvas_h: int = 320

    # --- validation ---------------------------------------------------
    alpha: str = "0.05"
    validated_sig: str = ""  # graph signature at the last validation; "" = never
    verdict: str = ""
    verdict_detail: str = ""
    verdict_tone: str = "neutral"
    violation_summary: str = ""
    violation_views: list[ViolationView] = []
    selected_violation: str = ""
    violations_open: bool = False  # the violation list is collapsed by default
    n_found: int = 0  # violations the last check found, before any were fixed
    hint_line: list[str] = []  # x1, y1, x2, y2 of the selected violation's pair
    _violations: list[dict] = []

    # --- model --------------------------------------------------------
    fit_snapshot: dict = {}  # what the last fit used (see backend.fit_snapshot); {} if none
    fitted: bool = False
    stale: bool = False  # graph edited since the last fit
    r2_rows: list[R2Row] = []
    fit_diagnostics: str = ""
    fit_warnings: list[str] = []
    fit_trouble: bool = False  # r̂ above RHAT_BAD: the fit should not be trusted
    _r2_text: dict[str, str] = {}  # node -> "0.76 [0.73, 0.78]"
    fig: go.Figure = go.Figure()
    has_curve: bool = False
    curve_caption: str = ""
    effect_note: str = ""  # why a selected node has no E[outcome | do(node)] curve
    has_equation: bool = False
    equation_rows: list[EquationRow] = []
    equation_caption: str = ""
    evidence_caption: str = ""

    # --- feedback -----------------------------------------------------
    busy: str = ""  # non-empty while a long task runs; shown next to a spinner
    error: str = ""
    # Progress bar for discovery and fitting; empty label means no bar is shown.
    progress_label: str = ""
    progress_pct: int = 0
    progress_known: bool = False  # False: the task cannot say how far along it is

    # --- grouped (panel) data -----------------------------------------
    data_kind: str = "file"  # "file", or "demo" for the built-in data (which has no file)
    all_columns: list[str] = []  # every column a group or time column could be
    group_cols: list[str] = []  # together they name a group, e.g. region and store; order is free
    time_col: str = ""
    group_pick: str = ""  # always empty: the "add a group column" select shows its placeholder
    group_headline: str = ""
    group_warnings: list[str] = []
    group_error: str = ""
    grouping_open: bool = False

    # --- importing a DAG ----------------------------------------------
    paste_text: str = ""
    import_message: str = ""
    import_ok: bool = False
    import_notes: list[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session(self) -> dict:
        return _SESSIONS.setdefault(self.router.session.client_token, {})

    def _edge_objects(self) -> list[b.Edge]:
        return [b.Edge(e["source"], e["target"], e["directed"]) for e in self.edges]

    def _set_progress(self, label: str, fraction: float | None) -> None:
        self.progress_label = label
        self.progress_known = fraction is not None
        self.progress_pct = 0 if fraction is None else int(round(fraction * 100))

    def _graph_sig(self) -> str:
        edges = sorted(
            f"{e['source']}{'>' if e['directed'] else '-'}{e['target']}" for e in self.edges
        )
        return "|".join(sorted(self.nodes)) + "#" + ",".join(edges)

    def _refresh(self, relayout: bool = False) -> None:
        """Rebuild the SVG views from the source-of-truth graph."""
        edges = self._edge_objects()
        # A violation is fixed once its two nodes are connected (or one is removed).
        connected = {e.key for e in edges}
        live = [
            v for v in self._violations
            if v["key"] not in connected and v["x"] in self.nodes and v["y"] in self.nodes
        ]
        if self.selected_violation not in {v["key"] for v in live}:
            self.selected_violation = ""
        chosen = next((v for v in live if v["key"] == self.selected_violation), None)
        hinted = {chosen["x"], chosen["y"]} if chosen else set()
        if relayout or set(self.positions) != set(self.nodes):
            pos = b.layout(self.nodes, edges, self.target)
            self.positions = {n: [x, y] for n, (x, y) in pos.items()}
        pos = {n: (p[0], p[1]) for n, p in self.positions.items()}
        width, height = b.canvas_size(pos)
        self.canvas_w, self.canvas_h = int(width), int(height)

        chosen_nodes = set(self.selected_nodes)
        neighbours = {e.target for e in edges if e.source in chosen_nodes} | {
            e.source for e in edges if e.target in chosen_nodes
        }
        effects = (
            b.edge_effects(self._session().get("equations", {}), edges)
            if self.fitted and not self.stale
            else {}
        )

        # A cycle already in the graph, else the one a refused edit would have closed.
        ring = b.find_cycle(edges)
        closing = None  # the refused edge, drawn as a line, when the cycle is not in the graph
        if ring:
            ring_edges = {(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))}
            path = " → ".join(ring + [ring[0]])
            self.cycle_note = f"The graph contains a cycle: {path}"
        elif len(self.refused_cycle) > 1 and all(n in pos for n in self.refused_cycle):
            ring = self.refused_cycle
            ring_edges = set(zip(ring, ring[1:]))
            closing = (ring[-1], ring[0])
            self.cycle_note = f"Not added: it would close the cycle {' → '.join(ring + [ring[0]])}"
        else:
            ring, ring_edges, self.cycle_note = [], set(), ""
        in_ring = set(ring)
        if closing:
            (x1, y1), (x2, y2) = pos[closing[0]], pos[closing[1]]
            self.cycle_line = [f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"]
        else:
            self.cycle_line = []

        self.node_views = [
            NodeView(
                id=n,
                label=_short(n),
                box_x=f"{pos[n][0] - b.NODE_W / 2:.1f}",
                box_y=f"{pos[n][1] - b.NODE_H / 2:.1f}",
                text_x=f"{pos[n][0]:.1f}",
                text_y=f"{pos[n][1] + 4:.1f}",
                dot_x=f"{pos[n][0] + b.NODE_W / 2 - 2:.1f}",
                dot_y=f"{pos[n][1] - b.NODE_H / 2 + 2:.1f}",
                note_x=f"{pos[n][0] - b.NODE_W / 2 + 2:.1f}",
                flagged=n in self.flagged_nodes,
                picked=n in chosen_nodes,
                is_target=n == self.target,
                tooltip=f"{n}\n{self.descriptions[n]}" if self.descriptions.get(n) else n,
                described=bool(self.descriptions.get(n, "").strip()),
                hinted=n in hinted,
                neighbour=n in neighbours,
                dimmed=(
                    bool(chosen_nodes) and n not in chosen_nodes and n not in neighbours
                    and n not in in_ring
                ),
                in_cycle=n in in_ring,
            )
            for n in self.nodes
        ]
        if chosen and chosen["x"] in pos and chosen["y"] in pos:
            (x1, y1), (x2, y2) = pos[chosen["x"]], pos[chosen["y"]]
            self.hint_line = [f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"]
        else:
            self.hint_line = []
        self.violation_views = [
            ViolationView(
                key=v["key"],
                x=v["x"],
                y=v["y"],
                given=b.violation_given(v["given"]),
                stats=b.violation_stats(v["r"], v["p"]),
                selected=v["key"] == self.selected_violation,
                forward=f"Add {v['x']} → {v['y']}",
                backward=f"Add {v['y']} → {v['x']}",
            )
            for v in live
        ]
        views = []
        for e in edges:
            others = [p for n, p in pos.items() if n not in (e.source, e.target)]
            d, arrow = b.edge_geometry(pos[e.source], pos[e.target], e.directed, others)
            selected = e.key == self.selected_edge
            incident = e.source in chosen_nodes or e.target in chosen_nodes
            on_cycle = (e.source, e.target) in ring_edges
            sign, term = effects.get(e.key, ("", None))
            if on_cycle:
                colour = b.CYCLE_COLOUR  # a cycle is a problem: it wins over every other colour
            elif sign:
                colour = b.EFFECT_COLOURS[sign]  # the fitted sign wins over highlight colours
            elif selected or incident:
                colour = _ACCENT
            elif e.key in self.flagged_edges:
                colour = _AMBER
            else:
                colour = _EDGE
            tooltip = b.edge_tooltip(e.source, e.target, e.directed, term)
            views.append(
                EdgeView(
                    id=e.key,
                    source=e.source,
                    target=e.target,
                    directed=e.directed,
                    flagged=e.key in self.flagged_edges,
                    selected=selected,
                    d=d,
                    arrow=arrow,
                    colour=colour,
                    width=(
                        "3.4" if on_cycle
                        else "3" if selected or incident else ("2.2" if sign else "1.5")
                    ),
                    opacity="0.2" if chosen_nodes and not incident and not on_cycle else "1",
                    tooltip=tooltip,
                )
            )
        self.edge_views = views

    def _graph_changed(self) -> None:
        """Call after any structural edit: the fitted model no longer matches."""
        self.refused_cycle = []
        if self.fitted:
            self.stale = True
        self.has_curve = self.has_equation = False
        self.effect_note = ""
        self.error = ""

    def _reset_graph(self) -> None:
        self.nodes, self.removed, self.edges = list(self.columns), [], []
        self.flagged_nodes, self.flagged_edges = [], []
        self.positions, self.descriptions = {}, {}
        self.families, self.prior_text = {}, {}
        self.refused_cycle, self.violations_open, self.n_found = [], False, 0
        self.paste_text, self.import_message, self.import_notes = "", "", []
        self._violations, self.validated_sig, self.selected_violation = [], "", ""
        self.verdict = self.verdict_detail = self.violation_summary = ""
        self.selected_nodes, self.selected_edge = [], ""
        self.fitted = self.stale = self.has_curve = self.has_equation = False
        self.fit_snapshot = {}
        self.curve_caption = self.equation_caption = self.evidence_caption = ""
        session = self._session()
        session.pop("model", None)
        session["evidence"], session["curves"] = {}, {}
        self._refresh(relayout=True)

    def _set_table(
        self, df, notes: list[str], filename: str, target: str = "", extras=None,
        kind: str = "file",
    ) -> None:
        self.data_kind = kind
        session = self._session()
        session["raw"], session["df"] = df, df  # df is replaced by a group-demeaned copy if grouped
        session["extras"] = extras if extras is not None else pl.DataFrame()
        self.all_columns = list(session["extras"].columns) + list(df.columns)
        self.group_cols, self.time_col = [], ""
        self.group_headline, self.group_warnings, self.group_error = "", [], ""
        self.grouping_open = True  # open as soon as there is data, so it is easy to find
        self.filename, self.n_rows = filename, df.height
        self.columns, self.notes = df.columns, notes
        self.target = target or df.columns[-1]
        self.error = ""
        self._reset_graph()

    # ------------------------------------------------------------------
    # Computed vars
    # ------------------------------------------------------------------

    @rx.var
    def has_data(self) -> bool:
        return self.n_rows > 0

    @rx.var
    def problems(self) -> list[str]:
        if not self.nodes:
            return []
        return b.graph_problems(self.nodes, self._edge_objects(), self.target)

    @rx.var
    def can_fit(self) -> bool:
        return self.n_rows > 0 and not self.problems and self.busy == ""

    @rx.var
    def can_export_code(self) -> bool:
        return bool(self.fit_snapshot)

    @rx.var
    def is_busy(self) -> bool:
        return self.busy != ""

    @rx.var
    def show_progress(self) -> bool:
        return self.progress_label != ""

    @rx.var
    def selected_label(self) -> str:
        for e in self.edges:
            if b.edge_key(e["source"], e["target"]) == self.selected_edge:
                return f"{e['source']} {'→' if e['directed'] else '—'} {e['target']}"
        return ""

    @rx.var
    def selected_is_directed(self) -> bool:
        return any(
            e["directed"] and b.edge_key(e["source"], e["target"]) == self.selected_edge
            for e in self.edges
        )

    @rx.var
    def selected_edge_flagged(self) -> bool:
        return self.selected_edge in self.flagged_edges

    @rx.var
    def picked(self) -> str:
        """The selected node when exactly one is selected, else empty."""
        return self.selected_nodes[0] if len(self.selected_nodes) == 1 else ""

    @rx.var
    def two_selected(self) -> bool:
        return len(self.selected_nodes) == 2

    @rx.var
    def pair_label(self) -> str:
        return " → ".join(self.selected_nodes) if len(self.selected_nodes) == 2 else ""

    @rx.var
    def add_edge_label(self) -> str:
        if len(self.selected_nodes) != 2:
            return "Add edge"
        a, b_ = self.selected_nodes
        exists = any(
            b.edge_key(e["source"], e["target"]) == b.edge_key(a, b_) for e in self.edges
        )
        return f"{'Re-point to' if exists else 'Add edge'} {a} → {b_}"

    @rx.var
    def picked_flagged(self) -> bool:
        return self.picked in self.flagged_nodes

    @rx.var
    def shows_effects(self) -> bool:
        return self.fitted and not self.stale

    @rx.var
    def grouped(self) -> bool:
        return len(self.group_cols) > 0

    @rx.var
    def group_candidates(self) -> list[str]:
        return [c for c in self.all_columns if c not in self.group_cols and c != self.time_col]

    @rx.var
    def time_options(self) -> list[str]:
        return ["(none)"] + [c for c in self.all_columns if c not in self.group_cols]

    @rx.var
    def time_value(self) -> str:
        return self.time_col or "(none)"

    @rx.var
    def has_validation(self) -> bool:
        return self.validated_sig != ""

    @rx.var
    def violations_left(self) -> int:
        return len(self.violation_views)

    @rx.var
    def violations_header(self) -> str:
        left = len(self.violation_views)
        if left == self.n_found:
            return f"Violated pairs ({left})"
        return f"Violated pairs ({left} of {self.n_found} left)"

    @rx.var
    def validation_stale(self) -> bool:
        return self.validated_sig != "" and self.validated_sig != self._graph_sig()

    @rx.var
    def can_validate(self) -> bool:
        if not self.n_rows or self.busy or len(self.nodes) < 2:
            return False
        blocking = [p for p in b.graph_problems(self.nodes, self._edge_objects())
                    if p != "The graph has no edges."]
        return not blocking

    @rx.var
    def current_step(self) -> int:
        """Where the user is in the guided flow (6 = everything done)."""
        if not self.n_rows:
            return 1
        if not self.edges:
            return 2
        if self.problems:
            return 3
        if self.validated_sig != self._graph_sig():
            return 4
        if not self.fitted or self.stale:
            return 5
        return 6

    @rx.var
    def picked_family(self) -> str:
        return self.families.get(self.picked, "gaussian")

    @rx.var
    def picked_has_parents(self) -> bool:
        return any(e["directed"] and e["target"] == self.picked for e in self.edges)

    @rx.var
    def picked_prior_rows(self) -> list[PriorRow]:
        typed = self.prior_text.get(self.picked, {})
        rows, seen = [], set()
        for p in b.prior_params(self.families.get(self.picked, "gaussian")):
            rows.append(
                PriorRow(
                    key=p.key,
                    caption="" if p.prefix in seen else f"{p.label} ~ {p.dist}",
                    param=_SYMBOLS.get(p.param, p.param),
                    value=typed.get(p.key, ""),
                    default=f"{p.default:g}",
                )
            )
            seen.add(p.prefix)
        return rows

    @rx.var
    def effect_title(self) -> str:
        return f"E[{self.target} | do({self.picked})]" if self.picked else ""

    @rx.var
    def picked_r2(self) -> str:
        text = self._r2_text.get(self.picked, "")
        if not (text and self.fitted and not self.stale):
            return ""
        return f"R² = {text}" + (" within groups" if self.grouped else "")

    @rx.var
    def picked_description(self) -> str:
        return self.descriptions.get(self.picked, "")

    @rx.var
    def review_queue(self) -> list[str]:
        edges = [k.replace("~", " – ") for k in self.flagged_edges]
        return list(self.flagged_nodes) + edges

    @rx.var
    def edge_count(self) -> int:
        return len(self.edges)

    # ------------------------------------------------------------------
    # Data events
    # ------------------------------------------------------------------

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        upload = files[0]
        name = getattr(upload, "name", None) or getattr(upload, "filename", None) or "data.csv"
        suffix = Path(name).suffix.lower() or ".csv"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(await upload.read())
                tmp.flush()
                df, notes, extras = b.load_dataset(tmp.name)
        except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
            self.error = f"Could not read {name}: {exc}"
            return
        self._set_table(df, notes, Path(name).name, extras=extras)

    # ------------------------------------------------------------------
    # Importing a DAG
    # ------------------------------------------------------------------

    def _apply_import(self, result: "b.ImportedDag") -> None:
        """Replace the graph's edges with an imported DAG, as discovery would."""
        session = self._session()
        used = {n for e in result.edges for n in (e.source, e.target)}
        restored = sorted(n for n in used if n in self.removed)
        if restored:  # an edge needs its nodes, so bring back any the user had removed
            self.removed = [n for n in self.removed if n not in restored]
            self.nodes = self.nodes + restored
        self.edges = [dataclasses.asdict(e) for e in result.edges]
        self.descriptions = {**self.descriptions, **result.descriptions}
        keys = {e.key for e in result.edges}
        self.flagged_edges = list(
            dict.fromkeys([k for k in self.flagged_edges if k in keys] + result.flagged_edges)
        )
        self.flagged_nodes = list(
            dict.fromkeys(self.flagged_nodes + [n for n in result.flagged_nodes if n in self.nodes])
        )
        self._discard_results()  # the graph is new: nothing fitted or validated describes it
        self._refresh(relayout=True)

    def _discard_results(self) -> None:
        """Forget the fit, curves and validation: they described a different graph or data."""
        session = self._session()
        session.pop("model", None)
        session["curves"] = {}
        self.selected_nodes, self.selected_edge, self.refused_cycle = [], "", []
        self.fitted = self.stale = self.has_curve = self.has_equation = False
        self.fit_snapshot = {}
        self.evidence_caption = ""
        self._violations, self.validated_sig, self.selected_violation = [], "", ""
        self.verdict = self.verdict_detail = self.violation_summary = ""
        self.n_found = 0

    # ------------------------------------------------------------------
    # Grouped (panel) data
    # ------------------------------------------------------------------

    def _try_grouping(self, keys: list[str], time_col: str) -> bool:
        """Group the rows by the combination of ``keys``. On a problem, change nothing."""
        session = self._session()
        raw, extras = session["raw"], session["extras"]
        self.group_error = ""
        if keys:
            frame = raw.hstack(extras) if extras.width else raw
            try:
                result = b.apply_grouping(frame, keys, time_col, list(raw.columns))
            except ValueError as exc:
                self.group_error = str(exc)
                return False
            df, headline, warnings = result.df, result.summary.headline, result.summary.warnings
        else:
            df = raw.drop(time_col) if time_col in raw.columns else raw
            headline, warnings = "", []
            if df.width < 2:
                self.group_error = "Fewer than two variables would be left."
                return False
        session["df"] = df
        self.group_cols, self.time_col = list(keys), time_col
        self.group_headline, self.group_warnings = headline, warnings
        self.columns, self.n_rows = list(df.columns), df.height

        # Variables that appear or disappear change the graph's nodes; everything the
        # graph learned from the other data is forgotten.
        known = set(self.nodes) | set(self.removed)
        self.nodes = [n for n in self.nodes if n in self.columns] + [
            c for c in self.columns if c not in known
        ]
        self.removed = [n for n in self.removed if n in self.columns]
        self.edges = [
            e for e in self.edges if e["source"] in self.nodes and e["target"] in self.nodes
        ]
        edge_keys = {b.edge_key(e["source"], e["target"]) for e in self.edges}
        self.flagged_edges = [k for k in self.flagged_edges if k in edge_keys]
        self.flagged_nodes = [n for n in self.flagged_nodes if n in self.nodes]
        self.families = {n: f for n, f in self.families.items() if n in self.nodes}
        self.prior_text = {n: v for n, v in self.prior_text.items() if n in self.nodes}
        if self.target not in self.columns:
            self.target = self.columns[-1]
        session["evidence"] = {}
        self._discard_results()
        self.error = ""
        self._refresh(relayout=True)
        return True

    @rx.event
    def toggle_grouping(self):
        self.grouping_open = not self.grouping_open

    @rx.event
    def add_group_col(self, column: str):
        """Add a group column. A group is a combination of the chosen columns, in any order."""
        if column and column not in self.group_cols and column != self.time_col:
            self._try_grouping(self.group_cols + [column], self.time_col)

    @rx.event
    def remove_group_col(self, column: str):
        self._try_grouping([c for c in self.group_cols if c != column], self.time_col)

    @rx.event
    def set_time_col(self, value: str):
        self._try_grouping(self.group_cols, "" if value == "(none)" else value)

    @rx.event
    def clear_grouping(self):
        self._try_grouping([], "")

    def _import_text(self, text: str) -> None:
        self.import_ok, self.import_notes = False, []
        if not self.n_rows:
            self.import_message = "Load data first: the graph's nodes are the dataset's columns."
            return
        try:
            result = b.parse_dag_text(text, list(self.columns))
        except ValueError as exc:
            self.import_message = str(exc)
            return
        self._apply_import(result)
        n = len(result.edges)
        self.import_ok = True
        self.import_message = (
            f"Imported {n} edge{'s' if n != 1 else ''} from {result.format}. "
            "The previous edges were replaced."
        )
        self.import_notes = result.notes
        self.paste_text = ""
        self.error = ""

    @rx.event
    def set_paste_text(self, value: str):
        self.paste_text = value

    @rx.event
    def import_pasted(self):
        """Import the DAG typed or pasted into the box (DOT, spec syntax or edge list)."""
        if not self.paste_text.strip():
            self.import_ok, self.import_notes = False, []
            self.import_message = "Paste a DAG first."
            return
        self._import_text(self.paste_text)

    @rx.event
    async def import_file(self, files: list[rx.UploadFile]):
        """Import a DOT (.dot / .gv) or text file holding a DAG."""
        if not files:
            return
        upload = files[0]
        name = getattr(upload, "name", None) or getattr(upload, "filename", None) or "dag.dot"
        try:
            text = (await upload.read()).decode("utf-8-sig")
        except UnicodeDecodeError:
            self.import_ok, self.import_notes = False, []
            self.import_message = f"{Path(name).name} is not a text file."
            return
        self._import_text(text)

    @rx.event
    def load_demo(self):
        note = "Synthetic demo data with a known DAG."
        self._set_table(b.demo_table(), [note], "demo data", target="revenue", kind="demo")

    @rx.event
    def load_test_file(self):
        """Load the bundled messy CSV through the same cleaning as an upload."""
        try:
            df, notes, extras = b.load_dataset(b.TEST_FILE)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Could not read the test file: {exc}"
            return
        self._set_table(df, notes, b.TEST_FILE.name, target="revenue", extras=extras)

    @rx.event
    def load_panel_test_file(self):
        """Load the bundled panel CSV (region, store and week columns) for trying grouping."""
        try:
            df, notes, extras = b.load_dataset(b.PANEL_TEST_FILE)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Could not read the panel test file: {exc}"
            return
        self._set_table(df, notes, b.PANEL_TEST_FILE.name, target="revenue", extras=extras)

    @rx.event
    def set_target(self, value: str):
        self.target = value
        self._refresh(relayout=True)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @rx.event(background=True)
    async def discover(self):
        async with self:
            if self.busy or not self.n_rows:
                return
            if self.target not in self.nodes:
                self.error = f"The outcome '{self.target}' was removed. Restore it first."
                return
            self.busy, self.error = "Running TBFPC discovery…", ""
            self._set_progress("Starting TBFPC", None)
            session, target, nodes = self._session(), self.target, list(self.nodes)
        progress = b.Progress()
        task = asyncio.create_task(
            asyncio.to_thread(b.discover, session["df"], target, nodes, progress)
        )
        try:
            await _follow(self, task, progress)
            edges, evidence = task.result()
        except Exception as exc:  # noqa: BLE001
            async with self:
                self.busy, self.error = "", f"Discovery failed: {exc}"
                self.progress_label = ""
            return
        async with self:
            session["evidence"], session["curves"] = evidence, {}
            session.pop("model", None)
            self.edges = [dataclasses.asdict(e) for e in edges]
            self.flagged_edges, self.refused_cycle = [], []
            self.selected_nodes, self.selected_edge = [], ""
            self.fitted = self.stale = self.has_curve = self.has_equation = False
            self.fit_snapshot = {}
            self.evidence_caption = ""
            self._refresh(relayout=True)
            self.busy = self.progress_label = ""

    # ------------------------------------------------------------------
    # Graph editing
    # ------------------------------------------------------------------

    def _put_edge(self, source: str, target: str) -> None:
        """Set the edge between a pair to ``source -> target``."""
        key = b.edge_key(source, target)
        self.refused_cycle = []
        others = [e for e in self.edges if b.edge_key(e["source"], e["target"]) != key]
        other_objects = [b.Edge(e["source"], e["target"], e["directed"]) for e in others]
        cycle = b.cycle_closed_by(other_objects, source, target)
        if cycle:
            self.refused_cycle = cycle  # _refresh draws it in yellow until the next action
            self.error = f"{source} → {target} would create a cycle."
            return
        self.edges = others + [{"source": source, "target": target, "directed": True}]
        self.selected_edge = key
        self._graph_changed()

    @rx.event
    def click_node(self, node: str, info: dict):
        """Select a node; with cmd (or ctrl) held, add it to the selection instead.

        Selecting two nodes enables "Add edge" above the graph, drawn from the
        one picked first to the one picked second. Clicking never adds an edge
        by itself. A third cmd-click drops the oldest of the two.
        """
        self.error, self.refused_cycle = "", []
        multi = bool(info.get("meta_key") or info.get("ctrl_key"))
        self.selected_nodes = b.next_selection(list(self.selected_nodes), node, multi)
        self.selected_edge = ""
        self._refresh()
        return State.show_selection

    @rx.event
    def swap_selected(self):
        """Reverse the direction of the pending edge."""
        self.refused_cycle = []
        self.selected_nodes = list(reversed(self.selected_nodes))
        self._refresh()

    @rx.event
    def add_selected_edge(self):
        """Add (or re-point) the edge between the two selected nodes."""
        if len(self.selected_nodes) != 2:
            return
        source, target = self.selected_nodes
        self.error = ""
        self._put_edge(source, target)
        if not self.error:  # a cycle keeps the selection so the direction can be swapped
            self.selected_nodes = []
        self._refresh()
        return State.show_selection

    @rx.event
    def clear_selection(self):
        self.refused_cycle = []
        self.selected_nodes, self.selected_edge = [], ""
        self.has_curve = self.has_equation = False
        self._refresh()

    @rx.event
    def select_edge(self, key: str):
        self.selected_nodes, self.error, self.refused_cycle = [], "", []
        self.selected_edge = "" if self.selected_edge == key else key
        self._refresh()
        return State.show_selection

    @rx.event
    def delete_edge(self):
        key = self.selected_edge
        self.edges = [e for e in self.edges if b.edge_key(e["source"], e["target"]) != key]
        self.flagged_edges = [k for k in self.flagged_edges if k != key]
        self.selected_edge = ""
        self._graph_changed()
        self._refresh()

    @rx.event
    def reverse_edge(self):
        """Flip a directed edge, or orient an undirected one the other way."""
        for e in self.edges:
            if b.edge_key(e["source"], e["target"]) == self.selected_edge:
                self._put_edge(e["target"], e["source"])
                break
        self._refresh()
        return State.show_selection

    @rx.event
    def orient_edge(self):
        """Accept the drawn direction of an undirected edge."""
        for e in self.edges:
            if b.edge_key(e["source"], e["target"]) == self.selected_edge:
                self._put_edge(e["source"], e["target"])
                break
        self._refresh()
        return State.show_selection

    @rx.event
    def toggle_flag_edge(self):
        key = self.selected_edge
        if key in self.flagged_edges:
            self.flagged_edges = [k for k in self.flagged_edges if k != key]
        elif key:
            self.flagged_edges = self.flagged_edges + [key]
        self._refresh()

    @rx.event
    def toggle_flag_node(self):
        node = self.picked
        if node in self.flagged_nodes:
            self.flagged_nodes = [n for n in self.flagged_nodes if n != node]
        elif node:
            self.flagged_nodes = self.flagged_nodes + [node]
        self._refresh()

    @rx.event
    def set_description(self, text: str):
        """Store the note for the selected node. Kept if the node is removed and restored."""
        node = self.picked
        if not node:
            return
        descriptions = dict(self.descriptions)
        if text.strip():
            descriptions[node] = text
        else:
            descriptions.pop(node, None)
        self.descriptions = descriptions
        self._refresh()

    def _check_family(self, node: str, family: str) -> None:
        df = self._session().get("df")
        problems = b.family_problems(df, {node: family}, self.grouped) if df is not None else []
        self.error = problems[0] if problems else ""

    @rx.event
    def set_family(self, family: str):
        """Set the selected node's family. Applies at the next fit."""
        node = self.picked
        if not node or family not in b.FAMILIES:
            return
        families = dict(self.families)
        if family == "gaussian":
            families.pop(node, None)
        else:
            families[node] = family
        if families != self.families:
            self.families = families
            self._graph_changed()
            self._refresh()
            self._check_family(node, family)

    @rx.event
    def set_prior(self, key: str, text: str):
        """Store one typed prior number for the selected node (blank = default)."""
        node = self.picked
        if not node:
            return
        typed = dict(self.prior_text.get(node, {}))
        if text.strip():
            typed[key] = text
        else:
            typed.pop(key, None)
        prior_text = {n: v for n, v in self.prior_text.items() if n != node}
        if typed:
            prior_text[node] = typed
        if prior_text != self.prior_text:
            self.prior_text = prior_text
            self._graph_changed()
            self._refresh()

    @rx.event
    def reset_priors(self):
        if self.picked in self.prior_text:
            self.prior_text = {n: v for n, v in self.prior_text.items() if n != self.picked}
            self._graph_changed()
            self._refresh()

    @rx.event
    def remove_node(self):
        node = self.picked
        if not node:
            return
        self.nodes = [n for n in self.nodes if n != node]
        self.removed = sorted(self.removed + [node])
        dropped = {
            b.edge_key(e["source"], e["target"])
            for e in self.edges
            if node in (e["source"], e["target"])
        }
        self.edges = [e for e in self.edges if b.edge_key(e["source"], e["target"]) not in dropped]
        self.flagged_edges = [k for k in self.flagged_edges if k not in dropped]
        self.flagged_nodes = [n for n in self.flagged_nodes if n != node]
        self.selected_nodes, self.selected_edge = [], ""
        self._graph_changed()
        self._refresh(relayout=True)

    @rx.event
    def restore_node(self, node: str):
        if node in self.removed:
            self.removed = [n for n in self.removed if n != node]
            self.nodes = self.nodes + [node]
            self._refresh(relayout=True)

    @rx.event
    def set_alpha(self, value: str):
        self.alpha = value

    @rx.event(background=True)
    async def validate(self):
        """Test the graph's implied independences against the data (no fit needed)."""
        async with self:
            if self.busy or not self.can_validate:
                return
            self.busy, self.error = "Validating the graph against the data…", ""
            session, nodes, edges = self._session(), list(self.nodes), self._edge_objects()
            alpha, sig = float(self.alpha), self._graph_sig()
        try:
            result = await asyncio.to_thread(b.validate_graph, session["df"], nodes, edges, alpha)
        except Exception as exc:  # noqa: BLE001
            async with self:
                self.busy, self.error = "", f"Validation failed: {exc}"
            return
        async with self:
            self.validated_sig = sig
            self.verdict, self.verdict_detail = result.verdict, result.detail
            self.verdict_tone = result.tone
            self.violation_summary = b.validation_summary(result)
            self._violations = [
                {"key": v.key, "x": v.x, "y": v.y, "given": v.given,
                 "r": v.partial_corr, "p": v.p_value}
                for v in result.violations
            ]
            self.n_found = len(self._violations)
            self.selected_violation = ""
            self._refresh()
            self.busy = ""

    @rx.event
    def toggle_violations(self):
        """Expand or collapse the list of violated pairs."""
        self.violations_open = not self.violations_open
        if not self.violations_open:
            self.selected_violation = ""  # no stray highlight on the graph from a hidden row
        self._refresh()

    @rx.event
    def select_violation(self, key: str):
        self.selected_violation = "" if self.selected_violation == key else key
        self._refresh()

    @rx.event
    def add_violation_edge(self, source: str, target: str):
        """Add the edge a violation suggests (the user picks the direction)."""
        self.error, self.selected_nodes = "", []
        if source in self.nodes and target in self.nodes:
            self._put_edge(source, target)
        self.selected_violation = ""
        self._refresh()
        return State.show_selection

    @rx.event
    def tidy_layout(self):
        self._refresh(relayout=True)

    def _export_script(self) -> str:
        """The script for the DAG the model was last fitted on, with the settings it used.

        This is the fitted graph, not any edits made since. Raises a ValueError if
        nothing has been fitted yet.
        """
        if not self.fit_snapshot:
            raise ValueError(
                "Fit the model first: the script is for the DAG the model was last fitted on."
            )
        return export.build_script(self.fit_snapshot)

    @rx.event
    def export_code(self):
        """Download a Python script that refits the DAG the model was last fitted on."""
        try:
            script = self._export_script()
        except ValueError as exc:
            self.error = f"Could not export the code: {exc}"
            return
        self.error = ""
        return rx.download(data=script, filename="dag_studio_model.py")

    @rx.event
    def export_dot(self):
        flagged = [f"node {n}" for n in self.flagged_nodes]
        flagged += [f"edge {k.replace('~', ' -- ')}" for k in self.flagged_edges]
        dot = b.to_dot(self.nodes, self._edge_objects(), flagged, self.descriptions)
        return rx.download(data=dot, filename="dag.dot")

    # ------------------------------------------------------------------
    # Fitting and the fitted-relationship plot
    # ------------------------------------------------------------------

    @rx.event(background=True)
    async def fit(self):
        async with self:
            if self.busy or self.problems:
                return
            self.busy, self.error = "Fitting the model with pathmc (MCMC)…", ""
            self._set_progress(b.fit_step(1), None)
            session, nodes, edges = self._session(), list(self.nodes), self._edge_objects()
            target, families = self.target, dict(self.families)
            priors = {n: dict(v) for n, v in self.prior_text.items()}
            grouped = self.grouped
            where = (list(self.group_cols), self.time_col, self.data_kind, self.filename)
        progress = b.Progress()

        async def work():
            fit = await asyncio.to_thread(
                b.fit_model, session["df"], nodes, edges, target, families, priors, progress,
                grouped,
            )
            progress.set(b.fit_step(4))
            equations = await asyncio.to_thread(b.model_equations, fit.model, families)
            quality = await asyncio.to_thread(b.fit_quality, fit)
            return fit, equations, quality

        task = asyncio.create_task(work())
        try:
            await _follow(self, task, progress)
            fit, equations, quality = task.result()
        except Exception as exc:  # noqa: BLE001
            async with self:
                self.busy, self.error = "", f"Fit failed: {exc}"
                self.progress_label = ""
            return
        async with self:
            session["model"], session["curves"] = fit.model, {}
            session["equations"] = equations
            self.fitted, self.stale = True, False
            self.fit_snapshot = b.fit_snapshot(fit.plan, target, *where)
            self.busy = self.progress_label = ""
            self._set_quality(quality, target)
            self._refresh()
        return State.show_selection

    def _set_quality(self, quality: "b.FitQuality", target: str) -> None:
        self.r2_rows = [
            R2Row(node=r.node, family=r.family, r2=r.r2, interval=r.interval, bar=r.bar,
                  is_target=r.is_target)
            for r in b.quality_rows(quality, target)
        ]
        self._r2_text = {n.node: b.r2_text(n) for n in quality.nodes}
        self.fit_diagnostics = b.quality_diagnostics(quality, b.CHAINS, b.DRAWS)
        self.fit_warnings = quality.warnings()
        self.fit_trouble = not quality.max_rhat < b.RHAT_BAD or quality.divergences > 0

    @rx.event(background=True)
    async def show_selection(self):
        """Fill the right-hand panel for the selected node or edge.

        A node shows E[outcome | do(node)]: the total effect of setting it on the
        outcome. An edge shows the same curve for its end node, plus its equation.
        """
        async with self:
            node, key = self.picked, self.selected_edge
            session = self._session()
            evidence = session.get("evidence", {}).get(key)
            self.evidence_caption = (
                evidence.describe() if evidence else ("Not tested by discovery." if key else "")
            )
            edge = next(
                (e for e in self.edges if b.edge_key(e["source"], e["target"]) == key), None
            )
            self.has_curve = self.has_equation = False
            self.effect_note = ""
            ready = self.fitted and not self.stale
            if node and ready:
                self.effect_note = b.effect_unavailable(self._edge_objects(), node, self.target)
                if self.effect_note:
                    return
                source, target = node, self.target
            elif edge and edge["directed"] and ready:
                source, target = edge["source"], edge["target"]
                equation = session.get("equations", {}).get(target)
                if equation:
                    self.equation_rows = [
                        EquationRow(expr, band, hit) for expr, band, hit in equation.lines(source)
                    ]
                    self.equation_caption = (
                        f"Posterior means, with {equation.interval} intervals on the right. "
                        f"The highlighted term is the direct effect of {source}; the curve "
                        "above is the total effect through every path."
                    )
                    if target in self._r2_text:
                        self.equation_caption += (
                            f" Its parents explain R² = {self._r2_text[target]} of "
                            f"{target}'s variance{' within groups' if self.grouped else ''}."
                        )
                    if self.grouped:
                        self.equation_caption += (
                            " Group means were removed first, so the slopes are within-group "
                            "effects and the intercept is near 0."
                        )
                    self.has_equation = True
            else:
                return
            if self.busy:
                return
            cached = session["curves"].get((source, target))
            if not cached:
                self.busy = f"Computing E[{target} | do({source})]…"

        if not cached:
            try:
                cached = await asyncio.to_thread(
                    b.edge_curve, session["model"], session["df"], source, target
                )
            except Exception as exc:  # noqa: BLE001
                async with self:
                    self.busy, self.error = "", f"Could not compute the curve: {exc}"
                return
            session["curves"][(source, target)] = cached

        async with self:
            self.busy = ""
            if (self.picked, self.selected_edge) != (node, key) or self.stale:
                return  # the user moved on while we were computing
            self.fig = b.make_figure(cached, session["df"], self.grouped)
            level = int(b.INTERVAL * 100)
            if node:
                self.curve_caption = (
                    f"Total effect of {source} on {target}: posterior mean of {target} when "
                    f"{source} is set to x, through every directed path in your graph, with a "
                    f"{level}% credible band. Grey points are the observed data."
                )
            else:
                self.curve_caption = (
                    f"Total effect: posterior mean of {target} when {source} is set to x, "
                    f"with a {level}% credible band. Grey points are the observed data."
                )
            if self.grouped:
                self.curve_caption += (
                    " Values are deviations from each group's mean (group differences removed)."
                )
            self.has_curve = True
