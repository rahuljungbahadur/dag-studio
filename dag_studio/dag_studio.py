"""DAG Studio: a minimal Reflex front end for pathmc discovery and fitting."""

from __future__ import annotations

import reflex as rx

from . import backend as b
from .state import EdgeView, EquationRow, NodeView, State

INK = "#1f2933"
MUTED = "#7b8794"
LINE = "#e4e7eb"
EDGE = "#9aa5b1"
ACCENT = "#2563eb"
AMBER = "#d97706"


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


def section(title: str, *children) -> rx.Component:
    return rx.vstack(label(title), *children, spacing="2", width="100%", align="stretch")


# --------------------------------------------------------------------------
# Left panel: data, discovery, edits, fit
# --------------------------------------------------------------------------


def data_section() -> rx.Component:
    return section(
        "Data",
        rx.upload(
            rx.text("Drop a CSV or Parquet file, or click to browse", size="1", color=MUTED),
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
            padding="14px 10px",
            text_align="center",
            cursor="pointer",
        ),
        rx.cond(
            State.has_data,
            hint(State.filename, " · ", State.n_rows, " rows · ", State.columns.length(), " variables"),
            hint(
                "No file yet. ",
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
    )


def discover_section() -> rx.Component:
    return section(
        "Outcome",
        rx.select(State.columns, value=State.target, on_change=State.set_target, size="1"),
        rx.button(
            "Discover DAG", on_click=State.discover, disabled=State.is_busy,
            size="1", variant="solid", color_scheme="gray", high_contrast=True,
        ),
        hint("TBFPC searches for the drivers of the outcome. It replaces the current edges."),
    )


def add_edge_section() -> rx.Component:
    return section(
        "Add edge",
        rx.grid(
            rx.select(State.nodes, value=State.new_from, on_change=State.set_new_from,
                      placeholder="from", size="1"),
            rx.select(State.nodes, value=State.new_to, on_change=State.set_new_to,
                      placeholder="to", size="1"),
            columns="2", gap="8px", width="100%",
        ),
        rx.button("Add edge", on_click=State.add_edge, size="1", variant="soft",
                  color_scheme="gray"),
        hint("Adding an edge between two nodes that are already connected re-points it."),
    )


def removed_section() -> rx.Component:
    return rx.cond(
        State.removed.length() > 0,
        section(
            "Removed nodes",
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
        ),
    )


def review_section() -> rx.Component:
    return rx.cond(
        State.review_queue.length() > 0,
        section(
            "Flagged for review",
            rx.flex(
                rx.foreach(
                    State.review_queue,
                    lambda item: rx.badge(item, variant="soft", color_scheme="amber", size="1"),
                ),
                wrap="wrap", spacing="1",
            ),
        ),
    )


def fit_section() -> rx.Component:
    return section(
        "Model",
        rx.button(
            rx.cond(State.stale, "Refit model", "Fit model"),
            on_click=State.fit, disabled=~State.can_fit,
            size="1", variant="solid", color_scheme="blue",
        ),
        rx.cond(
            State.has_data,
            rx.foreach(State.problems, lambda p: hint(p)),
        ),
        rx.cond(State.stale, hint("The graph changed since the last fit.", color=AMBER)),
        rx.button(
            "Export DOT", on_click=State.export_dot, disabled=~State.has_data,
            size="1", variant="ghost", color_scheme="gray",
        ),
    )


def left_panel() -> rx.Component:
    return rx.vstack(
        data_section(),
        rx.cond(
            State.has_data,
            rx.vstack(
                discover_section(), add_edge_section(), removed_section(), review_section(),
                fit_section(), spacing="5", width="100%", align="stretch",
            ),
        ),
        spacing="5", width="100%", align="stretch",
    )


# --------------------------------------------------------------------------
# Centre: the graph
# --------------------------------------------------------------------------


def edge_el(e: EdgeView) -> rx.Component:
    colour = rx.cond(e.selected, ACCENT, rx.cond(e.flagged, AMBER, EDGE))
    return rx.el.svg.g(
        rx.el.svg.path(
            d=e.d, fill="none", stroke=colour,
            stroke_width=rx.cond(e.selected, "2.4", "1.5"),
            stroke_dasharray=rx.cond(e.directed, "none", "5 4"),
        ),
        rx.cond(e.directed, rx.el.svg.polygon(points=e.arrow, fill=colour)),
        # Wide invisible stroke so the edge is easy to click.
        rx.el.svg.path(
            d=e.d, fill="none", stroke="transparent", stroke_width="16",
            pointer_events="stroke", cursor="pointer", on_click=State.select_edge(e.id),
        ),
    )


def node_el(n: NodeView) -> rx.Component:
    return rx.el.svg.g(
        rx.el.svg.rect(
            x=n.box_x, y=n.box_y, width=str(int(b.NODE_W)), height=str(int(b.NODE_H)), rx="6",
            fill=rx.cond(n.picked, "#eff6ff", "white"),
            stroke=rx.cond(n.picked, ACCENT, rx.cond(n.flagged, AMBER, rx.cond(n.is_target, INK, EDGE))),
            stroke_width=rx.cond(n.picked | n.flagged | n.is_target, "1.8", "1"),
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
        rx.el.svg.title(n.tooltip),
        on_click=State.click_node(n.id), cursor="pointer",
    )


def graph_canvas() -> rx.Component:
    return rx.el.svg(
        rx.el.svg.rect(
            x="0", y="0", width=State.canvas_w, height=State.canvas_h,
            fill="transparent", on_click=State.clear_selection,
        ),
        rx.foreach(State.edge_views, edge_el),
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
                "Click a node to describe it, or an edge to inspect it. Add edges "
                "from the left panel. Dashed edges have no direction yet."
            ),
            rx.spacer(),
            rx.button("Tidy layout", on_click=State.tidy_layout, size="1", variant="ghost",
                      color_scheme="gray"),
            width="100%", align="center",
        ),
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


def node_details() -> rx.Component:
    return rx.vstack(
        label("Node"),
        rx.text(State.picked, size="3", weight="medium", color=INK),
        rx.text_area(
            value=State.picked_description,
            on_change=State.set_description,
            placeholder="Describe this variable: what it measures, units, how it was collected…",
            size="1", rows="4", resize="vertical", width="100%",
        ),
        hint("Saved as you type. Shown when you hover the node and included in the DOT export."),
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


MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"


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
        rx.cond(State.error != "", rx.text(State.error, size="1", color="#b42318")),
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
