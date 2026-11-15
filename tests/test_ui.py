"""Properties of the page that are easy to break by accident: readability and layout."""

import ast
from pathlib import Path

from dag_studio import dag_studio as page

# Everything text sits on: white, and the tints behind the current step, a selected pair, a
# picked node and a chip.
BACKGROUNDS = ["#ffffff", "#f5f8ff", "#f3efff", "#eff6ff", "#eef0f2"]


def luminance(colour: str) -> float:
    h = colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_colours_meet_wcag_aa_on_every_background_they_sit_on():
    for name in ("INK", "MUTED", "AMBER", "RED", "GREEN", "ACCENT"):
        colour = getattr(page, name)
        for background in BACKGROUNDS:
            assert contrast(colour, background) >= 4.5, (name, colour, background)


def test_the_page_is_one_screen_with_a_fixed_header_and_three_scrolling_columns():
    rendered = str(page.index().render())
    assert '["height"] : "100vh"' in rendered and '["overflow"] : "hidden"' in rendered
    assert rendered.count('["overflowY"] : "auto"') == 3  # left panel, graph, right panel
    assert 'rows:"minmax(0, 1fr)"' in rendered  # the grid's one row is the space left under the header
    header = rendered[rendered.index("DAG Studio"):][:900]
    assert '["flexShrink"] : "0"' in rendered.split("DAG Studio")[0][-400:] or "flexShrink" in header


def test_a_long_error_wraps_instead_of_running_off_the_screen():
    rendered = str(page.status_bar().render())
    assert '["overflowWrap"] : "anywhere"' in rendered and '["maxWidth"] : "50%"' in rendered


def test_long_explanations_live_in_hover_cards_not_inline():
    """Any text over 30 words must be an argument to info(), or a constant for one."""
    tree = ast.parse(Path(page.__file__).read_text())
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def inside(node, names):
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in names:
                return True
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "").endswith("_INFO") for t in node.targets
            ):
                return True
            if isinstance(node, ast.keyword) and node.arg in {"more", "placeholder"}:
                return True  # a step's hover text, or the paste box's placeholder
        return False

    docstrings = {
        n.body[0].value for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.Module)) and n.body
        and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
    }
    offenders = [
        node.value[:60] for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node not in docstrings and len(node.value.split()) > 30
        and not inside(node, {"info", "demo_button"})
    ]
    assert offenders == [], f"move these into an info() hover: {offenders}"


def test_the_explanations_are_there_to_hover():
    rendered = str(page.index().render())
    assert rendered.count("HoverCard") >= 8  # steps 2-4, grouping, time, demos, export, ...
