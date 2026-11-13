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
    flagged: bool
    picked: bool
    is_target: bool


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

    # --- selection ----------------------------------------------------
    picked: str = ""  # node clicked first; the next node click adds an edge
    selected_edge: str = ""
    new_from: str = ""  # "add edge" dropdowns
    new_to: str = ""

    # --- derived views for rendering ----------------------------------
    node_views: list[NodeView] = []
    edge_views: list[EdgeView] = []
    canvas_w: int = 640
    canvas_h: int = 320

    # --- model --------------------------------------------------------
    fitted: bool = False
    stale: bool = False  # graph edited since the last fit
    spec: str = ""
    fig: go.Figure = go.Figure()
    has_curve: bool = False
    curve_caption: str = ""
    coef_caption: str = ""
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

    def _refresh(self, relayout: bool = False) -> None:
        """Rebuild the SVG views from the source-of-truth graph."""
        edges = self._edge_objects()
        if relayout or set(self.positions) != set(self.nodes):
            pos = b.layout(self.nodes, edges, self.target)
            self.positions = {n: [x, y] for n, (x, y) in pos.items()}
        pos = {n: (p[0], p[1]) for n, p in self.positions.items()}
        width, height = b.canvas_size(pos)
        self.canvas_w, self.canvas_h = int(width), int(height)

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
                flagged=n in self.flagged_nodes,
                picked=n == self.picked,
                is_target=n == self.target,
            )
            for n in self.nodes
        ]
        views = []
        for e in edges:
            others = [p for n, p in pos.items() if n not in (e.source, e.target)]
            d, arrow = b.edge_geometry(pos[e.source], pos[e.target], e.directed, others)
            views.append(
                EdgeView(
                    id=e.key,
                    source=e.source,
                    target=e.target,
                    directed=e.directed,
                    flagged=e.key in self.flagged_edges,
                    selected=e.key == self.selected_edge,
                    d=d,
                    arrow=arrow,
                )
            )
        self.edge_views = views

    def _graph_changed(self) -> None:
        """Call after any structural edit: the fitted model no longer matches."""
        if self.fitted:
            self.stale = True
        self.has_curve = False
        self.error = ""

    def _reset_graph(self) -> None:
        self.nodes, self.removed, self.edges = list(self.columns), [], []
        self.flagged_nodes, self.flagged_edges = [], []
        self.positions = {}
        self.picked = self.selected_edge = ""
        self.fitted = self.stale = self.has_curve = False
        self.spec = self.curve_caption = self.coef_caption = self.evidence_caption = ""
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
        return b.graph_problems(self.nodes, self._edge_objects()) if self.nodes else []

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
            self.fitted = self.stale = self.has_curve = False
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
        """First click picks a start node; clicking a second node adds the edge."""
        self.error = ""
        if not self.picked:
            self.picked, self.selected_edge = node, ""
        elif self.picked == node:
            self.picked = ""
        else:
            self._put_edge(self.picked, node)
            self.picked = ""
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
        self.has_curve = False
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
    def tidy_layout(self):
        self._refresh(relayout=True)

    @rx.event
    def export_dot(self):
        flagged = [f"node {n}" for n in self.flagged_nodes]
        flagged += [f"edge {k.replace('~', ' -- ')}" for k in self.flagged_edges]
        dot = b.to_dot(self.nodes, self._edge_objects(), flagged)
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
        try:
            model, spec = await asyncio.to_thread(b.fit_model, session["df"], nodes, edges)
        except Exception as exc:  # noqa: BLE001
            async with self:
                self.busy, self.error = "", f"Fit failed: {exc}"
            return
        async with self:
            session["model"], session["curves"] = model, {}
            self.fitted, self.stale, self.spec, self.busy = True, False, spec, ""
        return State.show_selected_edge

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
            self.has_curve = False
            ready = edge and edge["directed"] and self.fitted and not self.stale
            if not ready or self.busy:
                return
            source, target = edge["source"], edge["target"]
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
            self.coef_caption = (
                f"Direct coefficient {source} → {target}: {cached.coef_mean:.3f} "
                f"({cached.coef_interval} interval {cached.coef_lower:.3f} to "
                f"{cached.coef_upper:.3f})"
            )
            self.has_curve = True
