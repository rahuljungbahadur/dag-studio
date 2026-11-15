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
                getattr(t, "id", "").endswith("_INFO") or getattr(t, "id", "") == "GITHUB_MARK"
                for t in node.targets  # hover text; or the logo's path data, which is not prose
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


def test_the_lock_bar_offers_edit_dag_and_explains_itself():
    rendered = str(page.index().render())
    assert "Edit DAG" in rendered and "Done editing" in rendered
    assert "DAG and priors locked" in rendered and "Editing the DAG" in rendered
    assert "After fitting, the DAG and the priors are locked" in rendered  # in the (i) hover
    assert page.LOCKED_NOTE in rendered  # why the controls are missing


def test_the_header_links_to_this_project_and_to_pathmc_on_github():
    from importlib import metadata

    rendered = str(page.status_bar().render())
    for url in (page.REPO_URL, page.PATHMC_URL):
        assert f'to:"{url}"' in rendered
    assert rendered.count('"_blank"') == 2  # both open in a new tab...
    assert rendered.count("noopener noreferrer") == 2  # ...safely
    assert page.ACCENT in rendered and "accent-8" not in rendered  # a readable link colour
    # pathmc's link is the repository pathmc itself declares.
    urls = " ".join(metadata.metadata("pathmc").get_all("Project-URL") or [])
    assert page.PATHMC_URL in urls
    assert page.REPO_URL.startswith("https://github.com/") and not page.REPO_URL.endswith(".git")


def test_the_links_show_the_github_logo_and_a_name_not_the_words_on_github():
    import hashlib

    rendered = str(page.status_bar().render())
    assert rendered.count(page.GITHUB_MARK) == 2  # one logo in each link
    assert 'viewBox:"0 0 16 16"' in rendered and '"aria-hidden":"true"' in rendered
    # "on GitHub" is only in the tooltip and the accessible label of each link, never visible.
    assert rendered.count("on GitHub") == 4
    for label in ("DAG Studio on GitHub", "pathmc on GitHub"):
        assert f'title:"{label}"' in rendered and f'"aria-label":"{label}"' in rendered
    # The logo path was rendered and checked by eye; this pins it against accidental edits.
    assert hashlib.sha256(page.GITHUB_MARK.encode()).hexdigest() == (
        "20201911fe31ee5dd9f0603c976d75ec881aa09d31041706385b59256548f1bb"
    )
