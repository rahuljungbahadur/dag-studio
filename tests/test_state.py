"""State logic that drives the graph view: cycle highlighting and the violation list.

These call State methods directly, so they need no browser or running app.
"""

import pytest

from dag_studio.state import State


def make_state(edges, nodes=("a", "b", "c", "d", "x"), target="c") -> State:
    st = State(_reflex_internal_init=True)
    st.nodes = list(nodes)
    st.target = target
    st.edges = [{"source": s, "target": t, "directed": True} for s, t in edges]
    st._refresh(relayout=True)
    return st


def click(st: State, node: str) -> None:
    State.click_node.fn(st, node, {})


def cycle_nodes(st: State) -> set[str]:
    return {n.id for n in st.node_views if n.in_cycle}


def yellow_edges(st: State) -> set[str]:
    return {e.id for e in st.edge_views if e.colour == "#E5AE38"}


def test_refused_edge_highlights_the_cycle_it_would_close():
    st = make_state([("a", "b"), ("b", "c"), ("x", "a")])
    st._put_edge("c", "a")
    assert "cycle" in st.error and st.refused_cycle == ["a", "b", "c"]
    assert [e["source"] + e["target"] for e in st.edges] == ["ab", "bc", "xa"]  # not added
    st._refresh()
    assert cycle_nodes(st) == {"a", "b", "c"}  # x feeds the cycle but is not on it
    assert yellow_edges(st) == {"a~b", "b~c"}
    assert st.cycle_note == "Not added: it would close the cycle a → b → c → a"
    assert len(st.cycle_line) == 4  # the refused edge, c to a, is drawn as a dashed line


def test_refusal_clears_on_the_next_action():
    for action in (
        lambda st: click(st, "d"),
        lambda st: State.clear_selection.fn(st),
        lambda st: State.swap_selected.fn(st),
        lambda st: State.select_edge.fn(st, "a~b"),
    ):
        st = make_state([("a", "b"), ("b", "c")])
        st._put_edge("c", "a")
        st._refresh()
        assert cycle_nodes(st)
        action(st)
        assert cycle_nodes(st) == set() and st.cycle_note == "" and st.cycle_line == []

    # A successful edit also clears it.
    st = make_state([("a", "b"), ("b", "c")])
    st._put_edge("c", "a")
    st._put_edge("d", "a")
    st._refresh()
    assert cycle_nodes(st) == set()


def test_a_cycle_already_in_the_graph_is_highlighted():
    st = make_state([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])  # not reachable via the UI
    assert cycle_nodes(st) == {"a", "b", "c"}
    assert yellow_edges(st) == {"a~b", "b~c", "a~c"}  # all three, including the closing edge
    assert st.cycle_note.startswith("The graph contains a cycle: ")
    assert st.cycle_line == []  # nothing refused, so no extra line
    others = {e.id: e.colour for e in st.edge_views if e.id == "c~d"}
    assert others["c~d"] != "#E5AE38"


def test_no_cycle_no_highlight():
    st = make_state([("a", "b"), ("b", "c")])
    assert cycle_nodes(st) == set() and yellow_edges(st) == set() and st.cycle_note == ""


VIOLATIONS = [
    {"key": "a~c", "x": "a", "y": "c", "given": ["b"], "r": 0.3, "p": 1e-6},
    {"key": "b~x", "x": "b", "y": "x", "given": [], "r": 0.1, "p": 0.01},
    {"key": "a~d", "x": "a", "y": "d", "given": [], "r": 0.1, "p": 0.04},
]


def validated_state() -> State:
    st = make_state([("a", "b"), ("b", "c")])
    st._violations, st.n_found = list(VIOLATIONS), len(VIOLATIONS)
    st.validated_sig = st._graph_sig()
    st._refresh()
    return st


def test_violation_list_is_collapsed_by_default_and_toggles():
    st = validated_state()
    assert st.violations_open is False
    assert [v.key for v in st.violation_views] == ["a~c", "b~x", "a~d"]
    assert st.violations_header == "Violated pairs (3)"
    State.toggle_violations.fn(st)
    assert st.violations_open is True
    State.toggle_violations.fn(st)
    assert st.violations_open is False


def test_collapsing_clears_the_highlight_from_a_hidden_row():
    st = validated_state()
    State.toggle_violations.fn(st)
    State.select_violation.fn(st, "a~c")
    assert st.selected_violation == "a~c" and len(st.hint_line) == 4
    State.toggle_violations.fn(st)
    assert st.selected_violation == "" and st.hint_line == []


def test_connecting_a_pair_removes_its_violation():
    st = validated_state()
    State.add_violation_edge.fn(st, "a", "c")  # the suggestion for a~c
    assert [v.key for v in st.violation_views] == ["b~x", "a~d"]
    assert st.violations_left == 2 and st.violations_header == "Violated pairs (2 of 3 left)"
    assert st.validation_stale  # the check itself is now out of date


def test_adding_the_edge_by_hand_also_removes_it():
    st = validated_state()
    st._put_edge("x", "b")  # same pair as b~x, opposite direction
    st._refresh()
    assert [v.key for v in st.violation_views] == ["a~c", "a~d"]


def test_selected_violation_is_dropped_when_its_pair_is_connected():
    st = validated_state()
    State.select_violation.fn(st, "a~c")
    assert st.hint_line
    st._put_edge("a", "c")
    st._refresh()
    assert st.selected_violation == "" and st.hint_line == []


def test_removing_a_node_removes_its_violations():
    st = validated_state()
    st.nodes = [n for n in st.nodes if n != "d"]
    st._refresh()
    assert [v.key for v in st.violation_views] == ["a~c", "b~x"]


def test_violation_removal_is_not_reverted_by_unrelated_refreshes():
    st = validated_state()
    st._put_edge("a", "c")
    for _ in range(3):
        st._refresh()
    assert st.violations_left == 2


@pytest.mark.parametrize("found,left", [(0, 0), (3, 0)])
def test_empty_states(found, left):
    st = make_state([("a", "b"), ("b", "c")])
    st.n_found = found
    st._violations = [] if not found else list(VIOLATIONS)
    st.edges = st.edges + [
        {"source": v["x"], "target": v["y"], "directed": True} for v in st._violations
    ]
    st._refresh()
    assert st.violations_left == left


class FakeState:
    """Stands in for a background handler's `self`: records what would be pushed."""

    def __init__(self):
        self.pushed: list[tuple[str, int | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def _set_progress(self, label, fraction):
        self.pushed.append((label, None if fraction is None else round(fraction * 100)))


def test_follow_pushes_only_changes_and_stops_when_the_task_ends():
    import asyncio

    from dag_studio import backend as b
    from dag_studio.state import _follow

    async def run():
        progress, state = b.Progress(), FakeState()

        async def worker():
            for label, fraction in [("one", 0.1), ("one", 0.1), ("two", 0.5), ("three", None)]:
                progress.set(label, fraction)
                await asyncio.sleep(0.5)
            return "done"

        task = asyncio.create_task(worker())
        await _follow(state, task, progress)
        return state.pushed, task.result()

    pushed, result = asyncio.run(run())
    assert result == "done"
    assert pushed == [("one", 10), ("two", 50), ("three", None)]  # the repeat was not re-sent


def test_follow_leaves_a_failed_task_for_the_caller_to_raise():
    import asyncio

    from dag_studio import backend as b
    from dag_studio.state import _follow

    async def run():
        async def worker():
            raise ValueError("boom")

        task = asyncio.create_task(worker())
        await _follow(FakeState(), task, b.Progress())
        return task

    task = asyncio.run(run())
    with pytest.raises(ValueError, match="boom"):
        task.result()


def test_set_progress_and_show_progress():
    st = State(_reflex_internal_init=True)
    assert st.show_progress is False
    st._set_progress("Step 2 of 4 · Drawing samples", 0.456)
    assert (st.progress_label, st.progress_pct, st.progress_known) == (
        "Step 2 of 4 · Drawing samples", 46, True,
    )
    assert st.show_progress is True
    st._set_progress("Compiling", None)
    assert st.progress_known is False and st.progress_pct == 0 and st.show_progress is True
    st.progress_label = ""
    assert st.show_progress is False
