"""DAG Studio: a minimal Reflex front end for pathmc discovery and fitting."""

from __future__ import annotations

import reflex as rx

from . import backend as b
from .state import EdgeView, EquationRow, NodeView, PriorRow, R2Row, State, ViolationView

INK = "#1f2933"
MUTED = "#5f6b7a"  # 4.7:1 or better on white and on every tint it sits on
LINE = "#e4e7eb"
EDGE = "#9aa5b1"
ACCENT = "#2563eb"
AMBER = "#a84f09"  # warnings: 4.9:1 or better
VIOLET = "#7c3aed"  # validation hints
GREEN = "#157a39"
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


def step(
    n: int, title: str, instruction: str, *body, needs_data: bool = True, more: tuple = ()
) -> rx.Component:
    """A numbered step: ✓ when done, highlighted when it is the current one.

    ``instruction`` is one short line; ``more`` is the longer explanation, shown on hover.
    """
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
            info(*more) if more else rx.fragment(),
            spacing="2", align="center",
        ),
        hint(instruction, color=rx.cond(current, INK, MUTED)),
        rx.cond(State.has_data, content) if needs_data else content,
        spacing="1", width="100%", align="stretch", padding="10px", border_radius="8px",
        background=rx.cond(current, "#f5f8ff", "transparent"),
        border=rx.cond(current, "1px solid #dbe4ff", "1px solid transparent"),
    )


def info(*paragraphs: str, side: str = "right") -> rx.Component:
    """An (i) that explains something on hover, in a card that wraps at a readable width.

    (Radix's plain tooltip has no maximum width, so long text would run on in one line.)
    """
    return rx.hover_card.root(
        rx.hover_card.trigger(rx.icon("info", size=15, color=MUTED, cursor="help")),
        rx.hover_card.content(
            rx.vstack(
                *[rx.text(text, size="1", line_height="1.5", color=INK) for text in paragraphs],
                spacing="2", align="stretch",
            ),
            side=side, size="1", max_width="300px",
        ),
    )


GROUPING_INFO = (
    "For panel data: rows that are repeated observations of the same thing, like a store, a "
    "country or a person. Pick the column that names it. Skip this if your rows are "
    "independent.",
    "If one column is not enough, pick several: a store number used in every region needs the "
    "region too. A group is one combination of the columns you pick, so the order makes no "
    "difference.",
    "Each group's mean is then removed from every variable, so only differences within a group "
    "are compared. That stops group differences from creating false links, in discovery, "
    "validation and the fit.",
)


def group_chip(column: str) -> rx.Component:
    return rx.badge(
        column, " ×", on_click=State.remove_group_col(column), cursor="pointer",
        variant="soft", color_scheme="blue", size="1",
    )


def grouping_controls() -> rx.Component:
    """A card for naming the columns that identify a group (a store, a country…), for panel data."""
    header = rx.hstack(
        rx.hstack(
            rx.text(rx.cond(State.grouping_open, "▾", "▸"), size="1", color=MUTED, width="10px"),
            rx.text("Group the rows", size="2", weight="medium", color=INK),
            rx.cond(State.grouped, rx.badge("on", color_scheme="blue", variant="soft", size="1")),
            spacing="2", align="center", cursor="pointer", on_click=State.toggle_grouping,
        ),
        info(*GROUPING_INFO),
        spacing="2", align="center", width="100%",
    )
    controls = rx.vstack(
        hint("Do the rows belong to a store, a country or a person? Pick the column that "
             "names it."),
        rx.cond(
            State.grouped,
            rx.hstack(
                rx.foreach(State.group_cols, group_chip),
                wrap="wrap", spacing="1", align="center",
            ),
        ),
        rx.select(
            State.group_candidates, value=State.group_pick, placeholder="Add a group column…",
            on_change=State.add_group_col, size="1",
        ),
        rx.hstack(
            rx.hstack(
                rx.text("Time column", size="1", color=MUTED),
                info("Optional. Kept out of the graph, and used to check the panel."),
                spacing="2", align="center",
            ),
            rx.spacer(),
            rx.select(State.time_options, value=State.time_value, on_change=State.set_time_col,
                      size="1"),
            width="100%", align="center",
        ),
        rx.cond(
            State.group_error != "",
            rx.text(State.group_error, size="1", color=RED, line_height="1.5"),
        ),
        rx.cond(
            State.grouped,
            rx.button("Clear grouping", on_click=State.clear_grouping, size="1", variant="ghost",
                      color_scheme="gray"),
        ),
        spacing="2", width="100%", align="stretch",
    )
    return rx.cond(
        State.has_data,
        rx.vstack(
            header,
            # Always visible once on: it changes what every later number means.
            rx.cond(
                State.grouped,
                rx.vstack(
                    rx.text(State.group_headline, size="1", color=INK, line_height="1.5"),
                    rx.foreach(
                        State.group_warnings,
                        lambda w: rx.text(w, size="1", color=AMBER, line_height="1.5"),
                    ),
                    spacing="1", width="100%", align="stretch",
                ),
            ),
            rx.cond(State.grouping_open, controls),
            spacing="2", width="100%", align="stretch", padding="10px", background="white",
            border=f"1px solid {LINE}", border_radius="8px",
        ),
    )


DEMO_IID_INFO = (
    "Synthetic data, 800 independent rows, with a known DAG: price → demand, promo → demand, "
    "season → promo and demand, demand → revenue, price → revenue. noise is unrelated to "
    "everything."
)
DEMO_PANEL_INFO = (
    "Synthetic panel data: 3 regions × 4 stores × 20 weeks. Store names repeat across regions, "
    "so group by region and store, with week as the time column. Known DAG: price → demand, "
    "promo → demand, demand → revenue, price → revenue. footfall depends only on the group."
)


def demo_button(text: str, handler, description: str) -> rx.Component:
    """A demo loader, with an (i) that explains the data on hover."""
    info_icon = info(description)
    return rx.hstack(
        rx.button(text, on_click=handler, size="1", variant="soft", color_scheme="gray",
                  flex="1"),
        info_icon,
        spacing="2", align="center", width="100%",
    )


def demo_section() -> rx.Component:
    return rx.vstack(
        label("Demo"),
        demo_button("Load data (iid)", State.load_demo, DEMO_IID_INFO),
        demo_button("Load panel data", State.load_panel_demo, DEMO_PANEL_INFO),
        spacing="2", width="100%", align="stretch", padding_top="6px",
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
            hint(State.filename, " · ", State.n_rows, " rows · ", State.columns.length(),
                 " variables", rx.cond(State.grouped, " (group means removed)", "")),
        ),
        rx.foreach(State.notes, lambda note: hint(note)),
        demo_section(),
        grouping_controls(),
        needs_data=False,
    )


def import_controls() -> rx.Component:
    """Import a DAG from a DOT file, or from pasted DOT, spec syntax or an edge list."""
    return rx.vstack(
        label("Or import a DAG"),
        rx.upload(
            rx.text("Drop a .dot file, or click to browse", size="1", color=MUTED),
            id="dagfile",
            accept={
                "text/vnd.graphviz": [".dot", ".gv"],
                "text/plain": [".dot", ".gv", ".txt"],
                "application/octet-stream": [".dot", ".gv"],
            },
            max_files=1,
            multiple=False,
            on_drop=State.import_file(rx.upload_files(upload_id="dagfile")),
            border=f"1px dashed {EDGE}", border_radius="6px", padding="10px 8px",
            text_align="center", cursor="pointer", background="white",
        ),
        rx.text_area(
            value=State.paste_text,
            on_change=State.set_paste_text,
            placeholder="…or paste one, in the spec syntax:\nM ~ X\nY ~ M + X\n\nor as DOT / an edge list:\ndigraph { X -> M; M -> Y }\nX->M, M->Y",
            size="1", rows="5", resize="vertical", width="100%",
            style={"fontFamily": MONO, "fontSize": "12px"},
        ),
        rx.button(
            "Import pasted DAG", on_click=State.import_pasted, size="1", variant="soft",
            color_scheme="gray", disabled=State.paste_text == "",
        ),
        rx.cond(
            State.import_message != "",
            rx.text(
                State.import_message, size="1", line_height="1.5",
                color=rx.cond(State.import_ok, GREEN, RED),
            ),
        ),
        rx.foreach(State.import_notes, lambda note: hint(note)),
        spacing="2", width="100%", align="stretch", padding_top="6px",
    )


def discover_step() -> rx.Component:
    return step(
        2, "Discover or import",
        "Pick the outcome, then discover a graph or import one.",
        rx.select(State.columns, value=State.target, on_change=State.set_target, size="1"),
        rx.cond(
            State.locked,
            hint(LOCKED_NOTE),
            rx.vstack(
                rx.button(
                    "Discover DAG", on_click=State.discover, disabled=State.is_busy,
                    size="1", variant="solid", color_scheme="gray", high_contrast=True,
                ),
                import_controls(),
                spacing="2", width="100%", align="stretch",
            ),
        ),
        more=(
            "TBFPC proposes a graph from the data, searching for what drives the outcome you "
            "picked. Or import a graph you already have.",
            "Either one replaces the current edges.",
        ),
    )


def edit_step() -> rx.Component:
    return step(
        3, "Review and edit",
        "Orient dashed edges, then add, reverse or delete edges.",
        rx.foreach(State.problems, lambda p: hint("• ", p, color=AMBER)),
        rx.cond(
            (State.removed.length() > 0) & ~State.locked,
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
        more=(
            "Use what you know about the domain: give every dashed edge a direction, and add, "
            "reverse or delete edges.",
            "To add an edge, ⌘-click two nodes in the graph (Ctrl-click on Windows), then press "
            "Add edge above the graph.",
            "Click a single node to describe it or set its model.",
        ),
    )


def violation_row(v: ViolationView) -> rx.Component:
    return rx.vstack(
        rx.text(v.x, " — ", v.y, size="1", weight="medium", color=INK),
        hint(v.given),
        hint(v.stats),
        rx.cond(
            v.selected & ~State.locked,
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


def violation_list() -> rx.Component:
    """The violated pairs, collapsed until the header is clicked."""
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.text(rx.cond(State.violations_open, "▾", "▸"), size="1", color=MUTED,
                        width="10px"),
                rx.text(State.violations_header, size="1", weight="medium", color=INK),
                spacing="2", align="center", cursor="pointer", on_click=State.toggle_violations,
            ),
            info(
                "Click a pair to see it on the graph.",
                "Add an edge only if it makes sense in your domain: some violations are chance, "
                "and a link can also come from a missing common cause.",
                "A pair leaves this list once you connect it.",
            ),
            spacing="2", align="center", width="100%",
        ),
        rx.cond(
            State.violations_open,
            rx.vstack(
                rx.foreach(State.violation_views, violation_row),
                spacing="2", width="100%", align="stretch",
            ),
        ),
        spacing="2", width="100%", align="stretch",
    )


def validate_step() -> rx.Component:
    tone = rx.match(State.verdict_tone, ("good", GREEN), ("bad", RED), INK)
    return step(
        4, "Validate",
        "Check the graph against the data.",
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
                rx.cond(
                    State.n_found == 0,
                    hint("No violations: the data show no link that the graph rules out."),
                    rx.cond(
                        State.violations_left == 0,
                        hint("Every flagged pair is now connected. Validate again to re-check.",
                             color=GREEN),
                        violation_list(),
                    ),
                ),
                spacing="2", width="100%", align="stretch",
            ),
        ),
        more=(
            "A violation is a pair the graph says is unrelated, but the data link.",
            "No model fit is needed.",
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
        rx.hstack(
            rx.button(
                "Export DOT", on_click=State.export_dot, disabled=~State.has_data,
                size="1", variant="ghost", color_scheme="gray",
            ),
            rx.button(
                "Export code", on_click=State.export_code, disabled=~State.can_export_code,
                size="1", variant="ghost", color_scheme="gray",
            ),
            info(
                "Export code downloads a Python script for the DAG the model was last fitted "
                "on. It loads your data from a path you set, loads the DAG as a digraph, and "
                "fits it with the same settings and seed.",
            ),
            spacing="3", align="center",
        ),
        rx.cond(
            State.can_export_code & State.stale,
            hint("Edited since the fit: Export code uses the fitted graph.", color=AMBER),
        ),
        rx.cond(
            ~State.can_export_code,
            hint("Fit the model first, to have a fitted DAG to export."),
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
            fill=rx.cond(n.in_cycle, b.CYCLE_FILL, rx.cond(n.picked, "#eff6ff", "white")),
            stroke=rx.cond(
                n.in_cycle, b.CYCLE_COLOUR,
                rx.cond(
                    n.picked | n.neighbour, ACCENT,
                    rx.cond(n.flagged, AMBER, rx.cond(n.is_target, INK, EDGE)),
                ),
            ),
            stroke_width=rx.cond(
                n.in_cycle, "3",
                rx.cond(
                    n.picked, "2.4", rx.cond(n.neighbour | n.flagged | n.is_target, "1.8", "1")
                ),
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
        on_click=lambda e: State.click_node(n.id, e), cursor="pointer",
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
        rx.cond(
            State.cycle_line.length() == 4,
            rx.el.svg.line(
                x1=State.cycle_line[0], y1=State.cycle_line[1],
                x2=State.cycle_line[2], y2=State.cycle_line[3],
                stroke=b.CYCLE_COLOUR, stroke_width="3", stroke_dasharray="6 5",
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


LOCK_INFO = (
    "After fitting, the DAG and the priors are locked, so what you see keeps matching the fitted "
    "model.",
    "Edit DAG unlocks them. As soon as you change something the fit is marked out of date, and "
    "Refit model fits the new graph. Done editing locks them again.",
    "Descriptions and review flags are never locked.",
)
LOCKED_NOTE = "Locked after fitting. Click Edit DAG above the graph to change this."


def lock_bar() -> rx.Component:
    """Once a model is fitted: say the DAG is locked, and offer the one way to unlock it."""
    return rx.cond(
        State.fitted,
        rx.hstack(
            rx.text(
                rx.cond(State.locked, "🔒 DAG and priors locked", "✎ Editing the DAG"),
                size="2", weight="medium", color=INK,
            ),
            rx.spacer(),
            rx.cond(
                State.locked,
                rx.button("Edit DAG", on_click=State.start_editing, size="2", variant="solid",
                          color_scheme="blue"),
                rx.button("Done editing", on_click=State.stop_editing, size="2", variant="soft",
                          color_scheme="gray"),
            ),
            info(*LOCK_INFO, side="bottom"),
            width="100%", align="center", spacing="3", padding="8px 12px", border_radius="8px",
            border=rx.cond(State.locked, "1px solid #dbe4ff", "1px solid #f3d9a8"),
            background=rx.cond(State.locked, "#f5f8ff", "#fff8eb"),
        ),
    )


def edge_toolbar() -> rx.Component:
    """Add an edge between the two nodes selected with ⌘-click."""
    return rx.hstack(
        rx.button(
            State.add_edge_label, on_click=State.add_selected_edge,
            disabled=~State.two_selected, size="1", variant="soft", color_scheme="blue",
        ),
        rx.button(
            "⇄", on_click=State.swap_selected, disabled=~State.two_selected,
            size="1", variant="ghost", color_scheme="gray", title="Swap direction",
        ),
        rx.cond(
            State.two_selected,
            hint("The edge runs from the node you picked first to the second."),
            hint("⌘-click (Ctrl-click on Windows) two nodes to enable."),
        ),
        spacing="2", align="center", width="100%", wrap="wrap",
    )


def cycle_note() -> rx.Component:
    return rx.cond(
        State.cycle_note != "",
        rx.hstack(
            swatch(b.CYCLE_COLOUR),
            rx.text(State.cycle_note, size="1", color=INK, line_height="1.5"),
            spacing="2", align="center", width="100%",
        ),
    )


def centre_panel() -> rx.Component:
    return rx.vstack(
        lock_bar(),
        rx.hstack(
            hint("Click a node or an edge to inspect it."),
            info(
                "Click a node to highlight its connections and describe it, or an edge to "
                "inspect it.",
                "Dashed edges have no direction yet.",
                side="bottom",
            ),
            rx.spacer(),
            rx.button("Tidy layout", on_click=State.tidy_layout, size="1", variant="ghost",
                      color_scheme="gray"),
            width="100%", align="center",
        ),
        rx.cond(State.has_data & ~State.locked, edge_toolbar()),
        cycle_note(),
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
        rx.hstack(
            label("Model for this node"),
            info(
                "Empty boxes use pathmc's default (shown in grey).",
                "The coefficient prior covers the intercept and every parent's slope.",
                "Changes apply at the next fit.",
            ),
            spacing="2", align="center",
        ),
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
                small_button("Reset priors to defaults", State.reset_priors, variant="ghost"),
                spacing="2", width="100%", align="stretch",
            ),
            hint("No incoming edges, so no equation: family and priors need a parent."),
        ),
        spacing="2", width="100%", align="stretch", padding_top="8px",
    )


def node_effect() -> rx.Component:
    """E[outcome | do(node)] for the selected node, once the model is fitted."""
    return rx.cond(
        State.shows_effects,
        rx.cond(
            State.has_curve,
            rx.vstack(
                rx.text(State.effect_title, size="2", weight="medium", color=INK,
                        style={"fontFamily": MONO}),
                rx.plotly(data=State.fig, width="100%"),
                hint(State.curve_caption),
                spacing="2", width="100%", align="stretch", padding_y="4px",
            ),
            rx.cond(State.effect_note != "", hint(State.effect_note)),
        ),
        rx.cond(
            State.fitted,
            hint("The model changed since the last fit. Refit to see how setting this node "
                 "changes the outcome.", color=AMBER),
            hint("Fit the model to see how setting this node changes the outcome."),
        ),
    )


def node_details() -> rx.Component:
    return rx.vstack(
        label("Node"),
        rx.text(State.picked, size="3", weight="medium", color=INK),
        rx.cond(
            State.picked_r2 != "",
            hint(State.picked_r2, " of this node's variance is explained by its parents."),
        ),
        node_effect(),
        rx.text_area(
            value=State.picked_description,
            on_change=State.set_description,
            placeholder="Describe this variable: what it measures, units, how it was collected…",
            size="1", rows="4", resize="vertical", width="100%",
        ),
        hint("Saved as you type. Shown when you hover the node and included in the DOT export."),
        rx.cond(State.locked, hint(LOCKED_NOTE), node_model_section()),
        rx.hstack(
            small_button(rx.cond(State.picked_flagged, "Unflag", "Flag for review"),
                         State.toggle_flag_node),
            rx.cond(~State.locked, small_button("Remove node", State.remove_node,
                                                color_scheme="red")),
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
                ~State.locked,
                rx.cond(
                    State.selected_is_directed,
                    small_button("Reverse", State.reverse_edge),
                    rx.fragment(
                        small_button("Keep direction", State.orient_edge),
                        small_button("Reverse", State.reverse_edge),
                    ),
                ),
            ),
            small_button(rx.cond(State.selected_edge_flagged, "Unflag", "Flag for review"),
                         State.toggle_flag_edge),
            rx.cond(~State.locked, small_button("Delete edge", State.delete_edge,
                                                color_scheme="red")),
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
            rx.hstack(
                label("Fit quality"),
                info(
                    "R² says how well the parents predict a node, not whether the graph is "
                    "causally right.",
                    "A well-identified effect can sit in a noisy outcome, and a wrong graph can "
                    "predict well. Use Validate DAG for the graph.",
                    side="left",
                ),
                spacing="2", align="center",
            ),
            hint(State.fit_diagnostics),
            rx.cond(
                State.fit_warnings.length() > 0,
                rx.foreach(
                    State.fit_warnings,
                    lambda w: rx.text(w, size="1", line_height="1.5",
                                      color=rx.cond(State.fit_trouble, RED, AMBER)),
                ),
                hint(b.CONVERGED_NOTE, color=GREEN),
            ),
            rx.text(
                rx.cond(State.grouped, b.R2_HEADING_GROUPED, b.R2_HEADING),
                size="1", color=INK, padding_top="4px",
            ),
            rx.foreach(State.r2_rows, r2_row),
            spacing="2", width="100%", align="stretch",
        ),
    )


def right_panel() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.picked != "",
            node_details(),
            rx.cond(
                State.two_selected,
                rx.vstack(
                    label("Two nodes selected"),
                    rx.text(State.pair_label, size="3", weight="medium", color=INK),
                    hint(
                        "Press Add edge above the graph to connect them, or ⇄ to swap the "
                        "direction. Click a single node to describe it or set its model."
                    ),
                    spacing="2", width="100%", align="stretch",
                ),
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
        ),
        fit_quality_section(),
        spacing="5", width="100%", align="stretch",
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def progress_bar() -> rx.Component:
    """Progress of discovery or a fit: a percentage when known, else an animated bar."""
    return rx.vstack(
        rx.hstack(
            rx.text(State.progress_label, size="1", color=INK, no_of_lines=1),
            rx.spacer(),
            rx.cond(State.progress_known, hint(State.progress_pct, "%")),
            width="100%", align="center", spacing="2",
        ),
        rx.cond(
            State.progress_known,
            rx.progress(value=State.progress_pct, max=100, size="2", color_scheme="blue"),
            # No value: Radix animates the bar instead. It grows over `duration`, then pulses.
            rx.progress(size="2", color_scheme="blue", duration="12s"),
        ),
        spacing="1", width="300px", align="stretch",
    )


def status_bar() -> rx.Component:
    return rx.hstack(
        rx.text("DAG Studio", size="3", weight="medium", color=INK),
        hint("pathmc discovery, review and fit"),
        rx.spacer(),
        rx.cond(
            State.show_progress,
            progress_bar(),
            rx.cond(
                State.is_busy,
                rx.hstack(rx.spinner(size="1"), hint(State.busy), spacing="2", align="center"),
            ),
        ),
        rx.cond(
            State.error != "",
            # Wraps inside half the header, so a long message grows the header instead of
            # running off the screen.
            rx.text(State.error, size="1", color=RED, line_height="1.4", max_width="50%",
                    style={"overflowWrap": "anywhere"}),
        ),
        width="100%", align="center", spacing="3", flex_shrink="0",
        padding_bottom="12px", border_bottom=f"1px solid {LINE}",
    )


def pane(*children) -> rx.Component:
    """One of the three columns. It scrolls on its own, so the header and the other two stay put."""
    return rx.box(*children, height="100%", min_height="0", overflow_y="auto", padding_right="6px")


def index() -> rx.Component:
    # The page is exactly one screen tall: the header (progress, errors) is always in view, and
    # the left panel, the graph and the right panel each scroll without moving the others.
    return rx.box(
        status_bar(),
        rx.grid(
            pane(left_panel()), pane(centre_panel()), pane(right_panel()),
            columns="250px minmax(0, 1fr) 390px", rows="minmax(0, 1fr)", gap="28px",
            width="100%", flex="1", min_height="0", padding_top="18px",
        ),
        max_width="1500px", margin="0 auto", padding="18px 24px", height="100vh",
        display="flex", flex_direction="column", overflow="hidden",
    )


app = rx.App()
app.add_page(index, title="DAG Studio")
