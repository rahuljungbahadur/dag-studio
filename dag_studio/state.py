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
import reflex as rx

from . import backend as b

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
    picked: str = ""  # the selected node
    selected_edge: str = ""
    new_from: str = ""  # "add edge" dropdowns
    new_to: str = ""

    # --- derived views for rendering ----------------------------------
    node_views: list[NodeView] = []
    edge_views: list[EdgeView] = []
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
    hint_line: list[str] = []  # x1, y1, x2, y2 of the selected violation's pair
    _violations: list[dict] = []

    # --- model --------------------------------------------------------
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
    has_equation: bool = False
    equation_rows: list[EquationRow] = []
    equation_caption: str = ""
    evidence_caption: str = ""

    # --- feedback -----------------------------------------------------
    busy: str = ""  # non-empty while a long task runs; shown next to a spinner
    error: str = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session(self) -> dict:
        return _SESSIONS.setdefault(self.router.session.client_token, {})

    def _edge_objects(self) -> list[b.Edge]:
        return [b.Edge(e["source"], e["target"], e["directed"]) for e in self.edges]

    def _graph_sig(self) -> str:
        edges = sorted(
            f"{e['source']}{'>' if e['directed'] else '-'}{e['target']}" for e in self.edges
        )
        return "|".join(sorted(self.nodes)) + "#" + ",".join(edges)

    def _refresh(self, relayout: bool = False) -> None:
        """Rebuild the SVG views from the source-of-truth graph."""
        edges = self._edge_objects()
        chosen = next((v for v in self._violations if v["key"] == self.selected_violation), None)
        hinted = {chosen["x"], chosen["y"]} if chosen else set()
        if relayout or set(self.positions) != set(self.nodes):
            pos = b.layout(self.nodes, edges, self.target)
            self.positions = {n: [x, y] for n, (x, y) in pos.items()}
        pos = {n: (p[0], p[1]) for n, p in self.positions.items()}
        width, height = b.canvas_size(pos)
        self.canvas_w, self.canvas_h = int(width), int(height)

        picked = self.picked
        neighbours = {e.target for e in edges if e.source == picked} | {
            e.source for e in edges if e.target == picked
        }
        effects = (
            b.edge_effects(self._session().get("equations", {}), edges)
            if self.fitted and not self.stale
            else {}
        )

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
                picked=n == self.picked,
                is_target=n == self.target,
                tooltip=f"{n}\n{self.descriptions[n]}" if self.descriptions.get(n) else n,
                described=bool(self.descriptions.get(n, "").strip()),
                hinted=n in hinted,
                neighbour=n in neighbours,
                dimmed=bool(picked) and n != picked and n not in neighbours,
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
                given=f"given {', '.join(v['given'])}" if v["given"] else "with nothing held fixed",
                stats=f"partial r = {v['r']:.2f} · p = {v['p']:.2g}",
                selected=v["key"] == self.selected_violation,
                forward=f"Add {v['x']} → {v['y']}",
                backward=f"Add {v['y']} → {v['x']}",
            )
            for v in self._violations
        ]
        views = []
        for e in edges:
            others = [p for n, p in pos.items() if n not in (e.source, e.target)]
            d, arrow = b.edge_geometry(pos[e.source], pos[e.target], e.directed, others)
            selected = e.key == self.selected_edge
            incident = picked in (e.source, e.target)
            sign, term = effects.get(e.key, ("", None))
            if sign:
                colour = b.EFFECT_COLOURS[sign]  # the fitted sign wins over highlight colours
            elif selected or incident:
                colour = _ACCENT
            elif e.key in self.flagged_edges:
                colour = _AMBER
            else:
                colour = _EDGE
            arrow_char = "→" if e.directed else "—"
            tooltip = f"{e.source} {arrow_char} {e.target}"
            if term is not None:
                tooltip += (
                    f": direct coefficient {term.mean:.3f} "
                    f"[{term.lower:.3f}, {term.upper:.3f}]"
                )
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
                    width="3" if selected or incident else ("2.2" if sign else "1.5"),
                    opacity="0.2" if picked and not incident else "1",
                    tooltip=tooltip,
                )
            )
        self.edge_views = views

    def _graph_changed(self) -> None:
        """Call after any structural edit: the fitted model no longer matches."""
        if self.fitted:
            self.stale = True
        self.has_curve = self.has_equation = False
        self.error = ""

    def _reset_graph(self) -> None:
        self.nodes, self.removed, self.edges = list(self.columns), [], []
        self.flagged_nodes, self.flagged_edges = [], []
        self.positions, self.descriptions = {}, {}
        self.families, self.prior_text = {}, {}
        self._violations, self.validated_sig, self.selected_violation = [], "", ""
        self.verdict = self.verdict_detail = self.violation_summary = ""
        self.picked = self.selected_edge = ""
        self.fitted = self.stale = self.has_curve = self.has_equation = False
        self.curve_caption = self.equation_caption = self.evidence_caption = ""
        session = self._session()
        session.pop("model", None)
        session["evidence"], session["curves"] = {}, {}
        self._refresh(relayout=True)

    def _set_table(self, df, notes: list[str], filename: str, target: str = "") -> None:
        self._session()["df"] = df
        self.filename, self.n_rows = filename, df.height
        self.columns, self.notes = df.columns, notes
        self.target = target or df.columns[-1]
        self.new_from = self.new_to = ""
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
    def is_busy(self) -> bool:
        return self.busy != ""

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
    def picked_flagged(self) -> bool:
        return self.picked in self.flagged_nodes

    @rx.var
    def shows_effects(self) -> bool:
        return self.fitted and not self.stale

    @rx.var
    def has_validation(self) -> bool:
        return self.validated_sig != ""

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
    def picked_r2(self) -> str:
        text = self._r2_text.get(self.picked, "")
        return f"R² = {text}" if text and self.fitted and not self.stale else ""

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
                df, notes = b.load_table(tmp.name)
        except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
            self.error = f"Could not read {name}: {exc}"
            return
        self._set_table(df, notes, Path(name).name)

    @rx.event
    def load_demo(self):
        note = "Synthetic demo data with a known DAG."
        self._set_table(b.demo_table(), [note], "demo data", target="revenue")

    @rx.event
    def load_test_file(self):
        """Load the bundled messy CSV through the same cleaning as an upload."""
        try:
            df, notes = b.load_table(b.TEST_FILE)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Could not read the test file: {exc}"
            return
        self._set_table(df, notes, b.TEST_FILE.name, target="revenue")

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
            session, target, nodes = self._session(), self.target, list(self.nodes)
        try:
            edges, evidence = await asyncio.to_thread(b.discover, session["df"], target, nodes)
        except Exception as exc:  # noqa: BLE001
            async with self:
                self.busy, self.error = "", f"Discovery failed: {exc}"
            return
        async with self:
            session["evidence"], session["curves"] = evidence, {}
            session.pop("model", None)
            self.edges = [dataclasses.asdict(e) for e in edges]
            self.flagged_edges = []
            self.picked = self.selected_edge = ""
            self.fitted = self.stale = self.has_curve = self.has_equation = False
            self.evidence_caption = ""
            self._refresh(relayout=True)
            self.busy = ""

    # ------------------------------------------------------------------
    # Graph editing
    # ------------------------------------------------------------------

    def _put_edge(self, source: str, target: str) -> None:
        """Set the edge between a pair to ``source -> target``."""
        key = b.edge_key(source, target)
        others = [e for e in self.edges if b.edge_key(e["source"], e["target"]) != key]
        other_objects = [b.Edge(e["source"], e["target"], e["directed"]) for e in others]
        if b.creates_cycle(other_objects, source, target):
            self.error = f"{source} → {target} would create a cycle."
            return
        self.edges = others + [{"source": source, "target": target, "directed": True}]
        self.selected_edge = key
        self._graph_changed()

    @rx.event
    def click_node(self, node: str):
        """Select a node, or deselect it if it is already selected.

        Clicking nodes never adds edges; use the "Add edge" dropdowns for that.
        """
        self.error = ""
        self.picked = "" if self.picked == node else node
        self.selected_edge = ""
        self._refresh()
        return State.show_selected_edge

    @rx.event
    def set_new_from(self, value: str):
        self.new_from = value

    @rx.event
    def set_new_to(self, value: str):
        self.new_to = value

    @rx.event
    def add_edge(self):
        """Add an edge from the two dropdowns (same effect as clicking two nodes)."""
        self.error = ""
        if not self.new_from or not self.new_to:
            self.error = "Choose both a start and an end node."
        elif self.new_from == self.new_to:
            self.error = "Start and end must be different nodes."
        elif self.new_from not in self.nodes or self.new_to not in self.nodes:
            self.error = "Both nodes must be in the graph."
        else:
            self.picked = ""
            self._put_edge(self.new_from, self.new_to)
        self._refresh()
        return State.show_selected_edge

    @rx.event
    def clear_selection(self):
        self.picked = self.selected_edge = ""
        self.has_curve = self.has_equation = False
        self._refresh()

    @rx.event
    def select_edge(self, key: str):
        self.picked, self.error = "", ""
        self.selected_edge = "" if self.selected_edge == key else key
        self._refresh()
        return State.show_selected_edge

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
        return State.show_selected_edge

    @rx.event
    def orient_edge(self):
        """Accept the drawn direction of an undirected edge."""
        for e in self.edges:
            if b.edge_key(e["source"], e["target"]) == self.selected_edge:
                self._put_edge(e["source"], e["target"])
                break
        self._refresh()
        return State.show_selected_edge

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
        problems = b.family_problems(df, {node: family}) if df is not None else []
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
        self.picked = self.selected_edge = ""
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
            n = len(result.violations)
            self.violation_summary = (
                f"{n} of {result.n_tests} implied independences "
                f"{'is' if n == 1 else 'are'} violated. At α = {alpha:g}, about "
                f"{result.expected_by_chance:.1f} would be flagged by chance alone."
            )
            if result.n_collapsed:
                self.violation_summary += (
                    f" {result.n_collapsed} test(s) skipped: a conditioning variable was redundant."
                )
            self._violations = [
                {"key": v.key, "x": v.x, "y": v.y, "given": v.given,
                 "r": v.partial_corr, "p": v.p_value}
                for v in result.violations
            ]
            self.selected_violation = ""
            self._refresh()
            self.busy = ""

    @rx.event
    def select_violation(self, key: str):
        self.selected_violation = "" if self.selected_violation == key else key
        self._refresh()

    @rx.event
    def add_violation_edge(self, source: str, target: str):
        """Add the edge a violation suggests (the user picks the direction)."""
        self.error, self.picked = "", ""
        if source in self.nodes and target in self.nodes:
            self._put_edge(source, target)
        self.selected_violation = ""
        self._refresh()
        return State.show_selected_edge

    @rx.event
    def tidy_layout(self):
        self._refresh(relayout=True)

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
            session, nodes, edges = self._session(), list(self.nodes), self._edge_objects()
            target, families = self.target, dict(self.families)
            priors = {n: dict(v) for n, v in self.prior_text.items()}
        try:
            fit = await asyncio.to_thread(
                b.fit_model, session["df"], nodes, edges, target, families, priors
            )
            equations = await asyncio.to_thread(b.model_equations, fit.model, families)
            quality = await asyncio.to_thread(b.fit_quality, fit)
        except Exception as exc:  # noqa: BLE001
            async with self:
                self.busy, self.error = "", f"Fit failed: {exc}"
            return
        async with self:
            session["model"], session["curves"] = fit.model, {}
            session["equations"] = equations
            self.fitted, self.stale, self.busy = True, False, ""
            self._set_quality(quality, target)
            self._refresh()
        return State.show_selected_edge

    def _set_quality(self, quality: "b.FitQuality", target: str) -> None:
        nodes = sorted(quality.nodes, key=lambda n: (n.node != target, n.node))
        self.r2_rows = [
            R2Row(
                node=n.node,
                family="" if n.family == "gaussian" else n.family,
                r2=f"{n.r2:.2f}",
                interval=f"[{n.lower:.2f}, {n.upper:.2f}]",
                bar=f"{max(n.r2, 0) * 100:.0f}%",
                is_target=n.node == target,
            )
            for n in nodes
        ]
        self._r2_text = {n.node: f"{n.r2:.2f} [{n.lower:.2f}, {n.upper:.2f}]" for n in nodes}
        self.fit_diagnostics = (
            f"{b.CHAINS} chains × {b.DRAWS} draws · largest r̂ {quality.max_rhat:.3f} · "
            f"smallest ESS {quality.min_ess:.0f} · {quality.divergences} divergences"
        )
        self.fit_warnings = quality.warnings()
        self.fit_trouble = not quality.max_rhat < b.RHAT_BAD or quality.divergences > 0

    @rx.event(background=True)
    async def show_selected_edge(self):
        """Fill the right-hand panel for the selected edge."""
        async with self:
            key = self.selected_edge
            session = self._session()
            evidence = session.get("evidence", {}).get(key)
            self.evidence_caption = (
                evidence.describe() if evidence else ("Not tested by discovery." if key else "")
            )
            edge = next(
                (e for e in self.edges if b.edge_key(e["source"], e["target"]) == key), None
            )
            self.has_curve = self.has_equation = False
            ready = edge and edge["directed"] and self.fitted and not self.stale
            if not ready:
                return
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
                        f" Its parents explain R² = {self._r2_text[target]} of {target}'s variance."
                    )
                self.has_equation = True
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
            if self.selected_edge != key or self.stale:
                return  # the user moved on while we were computing
            self.fig = b.make_figure(cached, session["df"])
            level = int(b.INTERVAL * 100)
            self.curve_caption = (
                f"Total effect: posterior mean of {target} when {source} is set to x, "
                f"with a {level}% credible band. Grey points are the observed data."
            )
            self.has_curve = True
