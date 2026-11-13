"""DAG Studio: a minimal Reflex front end for pathmc discovery and fitting."""

from __future__ import annotations

import reflex as rx

from . import backend as b
from .state import EdgeView, EquationRow, NodeView, PriorRow, R2Row, State, ViolationView

INK = "#1f2933"
MUTED = "#7b8794"
LINE = "#e4e7eb"
EDGE = "#9aa5b1"
ACCENT = "#2563eb"
AMBER = "#d97706"
VIOLET = "#7c3aed"  # validation hints
GREEN = "#15803d"
GREEN_SOFT = "#dcfce7"
RED = "#b42318"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"


# --------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------


def label(text: str) -> rx.Component:
    return rx.text(
        text, size="1", weight="medium", color=MUTED, text_transform="uppercase",
        letter_spacing="0.06em",
    )


def hint(*children, **props) -> rx.Component:
    props.setdefault("color", MUTED)
    return rx.text(*children, size="1", line_height="1.5", **props)


# --------------------------------------------------------------------------
# Left panel: data, discovery, edits, fit
# --------------------------------------------------------------------------


def step(n: int, title: str, instruction: str, *body, needs_data: bool = True) -> rx.Component:
    """A numbered step: ✓ when done, highlighted when it is the current one."""
    done = State.current_step > n
    current = State.current_step == n
    badge = rx.center(
        rx.cond(done, "✓", str(n)),
        width="20px", height="20px", border_radius="50%", flex_shrink="0",
        font_size="11px", font_weight="600",
        background=rx.cond(current, ACCENT, rx.cond(done, GREEN_SOFT, "transparent")),
        color=rx.cond(current, "white", rx.cond(done, GREEN, MUTED)),
        border=rx.cond(current | done, "1px solid transparent", f"1px solid {EDGE}"),
    )
    content = rx.vstack(*body, spacing="2", width="100%", align="stretch", padding_top="4px")
    return rx.vstack(
        rx.hstack(
            badge,
            rx.text(title, size="2", weight="medium", color=rx.cond(current | done, INK, MUTED)),
            spacing="2", align="center",
        ),
        hint(instruction, color=rx.cond(current, INK, MUTED)),
        rx.cond(State.has_data, content) if needs_data else content,
        spacing="1", width="100%", align="stretch", padding="10px", border_radius="8px",
        background=rx.cond(current, "#f5f8ff", "transparent"),
        border=rx.cond(current, "1px solid #dbe4ff", "1px solid transparent"),
    )


def data_step() -> rx.Component:
    return step(
        1, "Load data", "Upload a CSV or Parquet file. Numeric columns become nodes.",
        rx.upload(
            rx.text("Drop a file here, or click to browse", size="1", color=MUTED),
            id="data",
            accept={
                "text/csv": [".csv"],
                "application/vnd.apache.parquet": [".parquet"],
                "application/octet-stream": [".parquet", ".pq"],
            },
            max_files=1,
            multiple=False,
            on_drop=State.handle_upload(rx.upload_files(upload_id="data")),
            border=f"1px dashed {EDGE}",
            border_radius="6px",
            padding="12px 10px",
            text_align="center",
            cursor="pointer",
            background="white",
        ),
        rx.cond(
            State.has_data,
            hint(State.filename, " · ", State.n_rows, " rows · ", State.columns.length(), " variables"),
            hint(
                "No file yet? ",
                rx.link("Load demo data", on_click=State.load_demo, size="1", underline="always"),
            ),
        ),
        rx.foreach(State.notes, lambda note: hint(note)),
        hint(
            "Manual testing: ",
            rx.link("Load test file", on_click=State.load_test_file, size="1",
                    underline="always"),
            " · ",
            rx.link("Download test file", size="1", underline="always",
                    on_click=rx.download(url=f"/{b.TEST_FILE.name}")),
        ),
        needs_data=False,
    )


def discover_step() -> rx.Component:
    return step(
        2, "Discover",
        "Pick the outcome you care about, then let TBFPC propose a graph. This replaces "
        "the current edges.",
        rx.select(State.columns, value=State.target, on_change=State.set_target, size="1"),
        rx.button(
            "Discover DAG", on_click=State.discover, disabled=State.is_busy,
            size="1", variant="solid", color_scheme="gray", high_contrast=True,
        ),
    )


def edit_step() -> rx.Component:
    return step(
        3, "Review and edit",
        "Use what you know about the domain: give every dashed edge a direction, and add, "
        "reverse or delete edges. Click a node to describe it or set its model.",
        rx.foreach(State.problems, lambda p: hint("• ", p, color=AMBER)),
        rx.grid(
            rx.select(State.nodes, value=State.new_from, on_change=State.set_new_from,
                      placeholder="from", size="1"),
            rx.select(State.nodes, value=State.new_to, on_change=State.set_new_to,
                      placeholder="to", size="1"),
            columns="2", gap="8px", width="100%",
        ),
        rx.button("Add edge", on_click=State.add_edge, size="1", variant="soft",
                  color_scheme="gray"),
        hint("Adding an edge between two connected nodes re-points it."),
        rx.cond(
            State.removed.length() > 0,
            rx.vstack(
                label("Removed nodes"),
                rx.flex(
                    rx.foreach(
                        State.removed,
                        lambda n: rx.badge(
                            n, " +", on_click=State.restore_node(n), cursor="pointer",
                            variant="outline", color_scheme="gray", size="1",
                        ),
                    ),
                    wrap="wrap", spacing="1",
                ),
                hint("Click to put a node back."),
                spacing="1", width="100%",
            ),
        ),
        rx.cond(
            State.review_queue.length() > 0,
            rx.vstack(
                label("Flagged for review"),
                rx.flex(
                    rx.foreach(
                        State.review_queue,
                        lambda item: rx.badge(item, variant="soft", color_scheme="amber",
                                              size="1"),
                    ),
                    wrap="wrap", spacing="1",
                ),
                spacing="1", width="100%",
            ),
        ),
    )


def violation_row(v: ViolationView) -> rx.Component:
    return rx.vstack(
        rx.text(v.x, " — ", v.y, size="1", weight="medium", color=INK),
        hint(v.given),
        hint(v.stats),
        rx.cond(
            v.selected,
            rx.flex(
                rx.button(v.forward, size="1", variant="soft", color_scheme="violet",
                          on_click=State.add_violation_edge(v.x, v.y).stop_propagation),
                rx.button(v.backward, size="1", variant="soft", color_scheme="violet",
                          on_click=State.add_violation_edge(v.y, v.x).stop_propagation),
                wrap="wrap", gap="6px", padding_top="4px",
            ),
        ),
        spacing="0", width="100%", align="stretch", padding="6px 8px", border_radius="6px",
        cursor="pointer", on_click=State.select_violation(v.key),
        background=rx.cond(v.selected, "#f3efff", "white"),
        border=rx.cond(v.selected, f"1px solid {VIOLET}", f"1px solid {LINE}"),
    )


def validate_step() -> rx.Component:
    tone = rx.match(State.verdict_tone, ("good", GREEN), ("bad", RED), INK)
    return step(
        4, "Validate",
        "Check the graph against the data. A violation is a pair the graph says are "
        "unrelated, but the data link. No model fit is needed.",
        rx.hstack(
            rx.button("Validate DAG", on_click=State.validate, disabled=~State.can_validate,
                      size="1", variant="solid", color_scheme="gray", high_contrast=True),
            rx.spacer(),
            rx.text("α", size="1", color=MUTED),
            rx.select(["0.01", "0.05", "0.10"], value=State.alpha, on_change=State.set_alpha,
                      size="1"),
            width="100%", align="center", spacing="2",
        ),
        rx.cond(
            State.has_validation,
            rx.vstack(
                rx.cond(
                    State.validation_stale,
                    hint("The graph changed since this check. Validate again.", color=AMBER),
                ),
                rx.text(State.verdict, size="2", weight="medium", color=tone),
                hint(State.verdict_detail),
                rx.text(State.violation_summary, size="1", color=INK, line_height="1.5"),
                rx.foreach(State.violation_views, violation_row),
                rx.cond(
                    State.violation_views.length() > 0,
                    hint(
                        "Click a violation to see the pair on the graph. Add an edge only if it "
                        "makes sense in your domain: some violations are chance, and a link can "
                        "also come from a missing common cause."
                    ),
                ),
                spacing="2", width="100%", align="stretch",
            ),
        ),
    )


def fit_step() -> rx.Component:
    return step(
        5, "Fit and inspect",
        "Fit the model, then click an edge to see its effect and fitted equation.",
        rx.button(
            rx.cond(State.stale, "Refit model", "Fit model"),
            on_click=State.fit, disabled=~State.can_fit,
            size="1", variant="solid", color_scheme="blue",
        ),
        rx.cond(
            State.problems.length() > 0,
            hint("Finish step 3 first.", color=AMBER),
        ),
        rx.cond(State.stale, hint("The model changed since the last fit.", color=AMBER)),
        rx.button(
            "Export DOT", on_click=State.export_dot, disabled=~State.has_data,
            size="1", variant="ghost", color_scheme="gray",
        ),
    )


def left_panel() -> rx.Component:
    return rx.vstack(
        data_step(), discover_step(), edit_step(), validate_step(), fit_step(),
        spacing="2", width="100%", align="stretch",
    )


# --------------------------------------------------------------------------
# Centre: the graph
# --------------------------------------------------------------------------


def edge_el(e: EdgeView) -> rx.Component:
    # Colour, width and opacity are worked out in State._refresh.
    return rx.el.svg.g(
        rx.el.svg.path(
            d=e.d, fill="none", stroke=e.colour, stroke_width=e.width,
            stroke_dasharray=rx.cond(e.directed, "none", "5 4"),
        ),
        rx.cond(e.directed, rx.el.svg.polygon(points=e.arrow, fill=e.colour)),
        # Wide invisible stroke so the edge is easy to click.
        rx.el.svg.path(
            d=e.d, fill="none", stroke="transparent", stroke_width="16",
            pointer_events="stroke", cursor="pointer", on_click=State.select_edge(e.id),
        ),
        rx.el.svg.title(e.tooltip),
        opacity=e.opacity,
    )


def node_el(n: NodeView) -> rx.Component:
    return rx.el.svg.g(
        rx.el.svg.rect(
            x=n.box_x, y=n.box_y, width=str(int(b.NODE_W)), height=str(int(b.NODE_H)), rx="6",
            fill=rx.cond(n.picked, "#eff6ff", "white"),
            stroke=rx.cond(
                n.picked | n.neighbour, ACCENT,
                rx.cond(n.flagged, AMBER, rx.cond(n.is_target, INK, EDGE)),
            ),
            stroke_width=rx.cond(
                n.picked, "2.4", rx.cond(n.neighbour | n.flagged | n.is_target, "1.8", "1")
            ),
        ),
        rx.el.svg.text(
            n.label, x=n.text_x, y=n.text_y, text_anchor="middle",
            font_weight=rx.cond(n.is_target, "600", "400"), fill=INK,
            style={"userSelect": "none", "fontSize": "13px"},
        ),
        rx.cond(
            n.flagged,
            rx.el.svg.circle(cx=n.dot_x, cy=n.dot_y, r="4.5", fill=AMBER),
        ),
        rx.cond(
            n.described,
            rx.el.svg.circle(cx=n.note_x, cy=n.dot_y, r="3.5", fill=ACCENT),
        ),
        rx.cond(
            n.hinted,
            rx.el.svg.rect(
                x=n.box_x, y=n.box_y, width=str(int(b.NODE_W)), height=str(int(b.NODE_H)),
                rx="6", fill="none", stroke=VIOLET, stroke_width="2.2",
                stroke_dasharray="4 3", pointer_events="none",
            ),
        ),
        rx.el.svg.title(n.tooltip),
        on_click=State.click_node(n.id), cursor="pointer",
        opacity=rx.cond(n.dimmed, "0.3", "1"),
    )


def swatch(colour: str) -> rx.Component:
    return rx.box(width="18px", height="3px", border_radius="2px", background=colour)


def effect_legend() -> rx.Component:
    return rx.cond(
        State.shows_effects,
        rx.hstack(
            hint("Edge colour = sign of the fitted direct effect:"),
            swatch(b.EFFECT_BLUE), hint("raises its child"),
            swatch(b.EFFECT_RED), hint("lowers it"),
            swatch(b.EFFECT_COLOURS["pos_weak"]), swatch(b.EFFECT_COLOURS["neg_weak"]),
            hint("pale: 89% interval includes 0"),
            spacing="2", align="center", wrap="wrap", width="100%",
        ),
    )


def graph_canvas() -> rx.Component:
    return rx.el.svg(
        rx.el.svg.rect(
            x="0", y="0", width=State.canvas_w, height=State.canvas_h,
            fill="transparent", on_click=State.clear_selection,
        ),
        rx.foreach(State.edge_views, edge_el),
        rx.cond(
            State.hint_line.length() == 4,
            rx.el.svg.line(
                x1=State.hint_line[0], y1=State.hint_line[1],
                x2=State.hint_line[2], y2=State.hint_line[3],
                stroke=VIOLET, stroke_width="1.6", stroke_dasharray="2 4",
                pointer_events="none",
            ),
        ),
        rx.foreach(State.node_views, node_el),
        view_box=f"0 0 {State.canvas_w} {State.canvas_h}",
        width=State.canvas_w,
        # Drawn 1:1; shrinks to fit a narrow panel but never scales up.
        style={"maxWidth": "100%", "height": "auto", "display": "block", "margin": "0 auto",
               "overflow": "visible"},
    )


def centre_panel() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            hint(
                "Click a node to highlight its connections and describe it, or an edge to "
                "inspect it. Add edges from the left panel. Dashed edges have no direction yet."
            ),
            rx.spacer(),
            rx.button("Tidy layout", on_click=State.tidy_layout, size="1", variant="ghost",
                      color_scheme="gray"),
            width="100%", align="center",
        ),
        effect_legend(),
        rx.cond(
            State.has_data,
            rx.box(graph_canvas(), width="100%", border=f"1px solid {LINE}", border_radius="8px",
                   padding="8px", background="white"),
            rx.center(
                hint("Upload a dataset to begin."),
                width="100%", height="320px", border=f"1px solid {LINE}", border_radius="8px",
            ),
        ),
        spacing="3", width="100%", align="stretch",
    )


# --------------------------------------------------------------------------
# Right panel: selection details and the fitted relationship
# --------------------------------------------------------------------------


def small_button(text, handler, **props) -> rx.Component:
    props.setdefault("variant", "soft")
    props.setdefault("color_scheme", "gray")
    return rx.button(text, on_click=handler, size="1", **props)


def prior_row(row: PriorRow) -> rx.Component:
    return rx.hstack(
        rx.text(row.caption, size="1", color=INK),
        rx.spacer(),
        rx.text(row.param, size="1", color=MUTED, style={"fontFamily": MONO}),
        rx.input(
            value=row.value, placeholder=row.default,
            on_change=lambda text: State.set_prior(row.key, text),
            size="1", width="84px",
        ),
        width="100%", align="center", spacing="2",
    )


def node_model_section() -> rx.Component:
    return rx.vstack(
        label("Model for this node"),
        rx.cond(
            State.picked_has_parents,
            rx.vstack(
                rx.hstack(
                    rx.text("Family", size="1", color=INK),
                    rx.spacer(),
                    rx.select(list(b.FAMILIES), value=State.picked_family,
                              on_change=State.set_family, size="1"),
                    width="100%", align="center",
                ),
                rx.foreach(State.picked_prior_rows, prior_row),
                hint(
                    "Empty boxes use pathmc's default (shown in grey). The coefficient prior "
                    "covers the intercept and every parent's slope. Changes apply at the next fit."
                ),
                small_button("Reset priors to defaults", State.reset_priors, variant="ghost"),
                spacing="2", width="100%", align="stretch",
            ),
            hint(
                "This node has no incoming edges, so it has no equation of its own: family "
                "and priors only apply to nodes with parents."
            ),
        ),
        spacing="2", width="100%", align="stretch", padding_top="8px",
    )


def node_details() -> rx.Component:
    return rx.vstack(
        label("Node"),
        rx.text(State.picked, size="3", weight="medium", color=INK),
        rx.cond(
            State.picked_r2 != "",
            hint(State.picked_r2, " of this node's variance is explained by its parents."),
        ),
        rx.text_area(
            value=State.picked_description,
            on_change=State.set_description,
            placeholder="Describe this variable: what it measures, units, how it was collected…",
            size="1", rows="4", resize="vertical", width="100%",
        ),
        hint("Saved as you type. Shown when you hover the node and included in the DOT export."),
        node_model_section(),
        rx.hstack(
            small_button(rx.cond(State.picked_flagged, "Unflag", "Flag for review"),
                         State.toggle_flag_node),
            small_button("Remove node", State.remove_node, color_scheme="red"),
            small_button("Cancel", State.clear_selection, variant="ghost"),
            spacing="2", wrap="wrap",
        ),
        spacing="2", width="100%", align="stretch",
    )


def edge_plot() -> rx.Component:
    return rx.cond(
        State.has_curve,
        rx.vstack(
            rx.plotly(data=State.fig, width="100%"),
            hint(State.curve_caption),
            spacing="2", width="100%", align="stretch",
        ),
        rx.cond(
            State.is_busy,
            rx.fragment(),
            hint(
                rx.cond(
                    ~State.selected_is_directed,
                    "Give this edge a direction to see its fitted relationship.",
                    rx.cond(
                        State.fitted & ~State.stale,
                        "",
                        "Fit the model to see the fitted relationship for this edge.",
                    ),
                )
            ),
        ),
    )



def equation_row(row: EquationRow) -> rx.Component:
    return rx.hstack(
        rx.text(
            row.expr, size="1", color=rx.cond(row.highlight, ACCENT, INK),
            weight=rx.cond(row.highlight, "bold", "regular"),
            style={"fontFamily": MONO, "whiteSpace": "pre"},
        ),
        rx.spacer(),
        rx.text(row.band, size="1", color=MUTED, style={"fontFamily": MONO}),
        width="100%", align="center", padding="3px 6px", border_radius="4px",
        background=rx.cond(row.highlight, "#eff6ff", "transparent"),
    )


def equation_block() -> rx.Component:
    return rx.cond(
        State.has_equation,
        rx.vstack(
            label("Fitted equation"),
            rx.vstack(
                rx.foreach(State.equation_rows, equation_row),
                spacing="0", width="100%", padding="4px",
                border=f"1px solid {LINE}", border_radius="6px",
            ),
            hint(State.equation_caption),
            spacing="2", width="100%", align="stretch", padding_top="6px",
        ),
    )


def edge_details() -> rx.Component:
    return rx.vstack(
        label("Edge"),
        rx.text(State.selected_label, size="3", weight="medium", color=INK),
        hint(State.evidence_caption),
        rx.hstack(
            rx.cond(
                State.selected_is_directed,
                small_button("Reverse", State.reverse_edge),
                rx.fragment(
                    small_button("Keep direction", State.orient_edge),
                    small_button("Reverse", State.reverse_edge),
                ),
            ),
            small_button(rx.cond(State.selected_edge_flagged, "Unflag", "Flag for review"),
                         State.toggle_flag_edge),
            small_button("Delete edge", State.delete_edge, color_scheme="red"),
            spacing="2", wrap="wrap",
        ),
        edge_plot(),
        equation_block(),
        spacing="2", width="100%", align="stretch",
    )


def r2_row(row: R2Row) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(row.node, size="1", color=INK,
                    weight=rx.cond(row.is_target, "bold", "regular")),
            rx.cond(row.family != "", rx.text(row.family, size="1", color=MUTED)),
            rx.spacer(),
            rx.text(row.r2, size="1", color=INK, style={"fontFamily": MONO}),
            rx.text(row.interval, size="1", color=MUTED, style={"fontFamily": MONO}),
            width="100%", align="center", spacing="2",
        ),
        rx.box(
            rx.box(height="100%", width=row.bar, background=ACCENT, border_radius="2px"),
            height="4px", width="100%", background=LINE, border_radius="2px",
        ),
        spacing="1", width="100%",
    )


def fit_quality_section() -> rx.Component:
    return rx.cond(
        State.shows_effects,
        rx.vstack(
            label("Fit quality"),
            hint(State.fit_diagnostics),
            rx.cond(
                State.fit_warnings.length() > 0,
                rx.foreach(
                    State.fit_warnings,
                    lambda w: rx.text(w, size="1", line_height="1.5",
                                      color=rx.cond(State.fit_trouble, RED, AMBER)),
                ),
                hint("The sampler converged: these estimates can be trusted as fits.",
                     color=GREEN),
            ),
            rx.text("Variance explained by each node's parents (Bayesian R², 89% interval)",
                    size="1", color=INK, padding_top="4px"),
            rx.foreach(State.r2_rows, r2_row),
            hint(
                "R² says how well the parents predict a node, not whether the graph is causally "
                "right: a well-identified effect can sit in a noisy outcome, and a wrong graph "
                "can predict well. Use Validate DAG for the graph."
            ),
            spacing="2", width="100%", align="stretch",
        ),
    )


def right_panel() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.picked != "",
            node_details(),
            rx.cond(
                State.selected_edge != "",
                edge_details(),
                rx.vstack(
                    label("Selection"),
                    hint("Select a node or an edge in the graph."),
                    spacing="2",
                ),
            ),
        ),
        fit_quality_section(),
        rx.cond(
            State.fitted,
            rx.vstack(
                label("Fitted with"),
                rx.el.pre(
                    State.fit_code,
                    style={"fontSize": "12px", "margin": "0", "color": INK,
                           "whiteSpace": "pre-wrap", "fontFamily": MONO},
                ),
                hint("data is the cleaned table. Prior comes from pymc_extras.prior."),
                spacing="2", width="100%",
            ),
        ),
        rx.cond(
            State.fitted,
            rx.vstack(
                label("Fitted pathmc spec"),
                rx.el.pre(
                    State.spec,
                    style={"fontSize": "12px", "margin": "0", "color": INK,
                           "whiteSpace": "pre-wrap", "fontFamily": "ui-monospace, monospace"},
                ),
                spacing="2", width="100%",
            ),
        ),
        spacing="5", width="100%", align="stretch",
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def status_bar() -> rx.Component:
    return rx.hstack(
        rx.text("DAG Studio", size="3", weight="medium", color=INK),
        hint("pathmc discovery, review and fit"),
        rx.spacer(),
        rx.cond(
            State.is_busy,
            rx.hstack(rx.spinner(size="1"), hint(State.busy), spacing="2", align="center"),
        ),
        rx.cond(State.error != "", rx.text(State.error, size="1", color=RED)),
        width="100%", align="center", spacing="3",
        padding_bottom="12px", border_bottom=f"1px solid {LINE}",
    )


def index() -> rx.Component:
    return rx.box(
        status_bar(),
        rx.grid(
            left_panel(), centre_panel(), right_panel(),
            columns="250px minmax(0, 1fr) 390px", gap="28px", width="100%", padding_top="18px",
            align_items="start",
        ),
        max_width="1500px", margin="0 auto", padding="18px 24px",
    )


app = rx.App()
app.add_page(index, title="DAG Studio")
