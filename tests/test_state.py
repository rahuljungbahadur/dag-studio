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


COLUMNS = ["a", "b", "c", "d", "x"]


def data_state() -> State:
    st = State(_reflex_internal_init=True)
    st.columns, st.n_rows, st.nodes, st.target = list(COLUMNS), 100, list(COLUMNS), "c"
    st._session().clear()
    st._refresh(relayout=True)
    return st


def edge_pairs(st: State) -> set[tuple[str, str, bool]]:
    return {(e["source"], e["target"], e["directed"]) for e in st.edges}


def test_importing_replaces_the_edges_and_resets_what_depended_on_them():
    st = data_state()
    st.edges = [{"source": "a", "target": "b", "directed": True}]
    st.fitted, st.validated_sig, st.n_found = True, "old", 2
    st._violations = [{"key": "a~c", "x": "a", "y": "c", "given": [], "r": 0.3, "p": 0.01}]
    st._session()["model"] = object()
    st.paste_text = "c ~ a + b"
    State.import_pasted.fn(st)
    assert edge_pairs(st) == {("a", "c", True), ("b", "c", True)}
    assert st.import_ok and st.import_message.startswith("Imported 2 edges from spec")
    assert st.paste_text == ""  # a successful import clears the box
    assert st.fitted is False and "model" not in st._session()
    assert st.has_validation is False and st.violation_views == []
    assert [e.id for e in st.edge_views] == ["a~c", "b~c"]  # the graph was redrawn


def test_import_brings_back_removed_nodes_and_merges_descriptions_and_flags():
    st = data_state()
    st.nodes, st.removed = ["a", "b", "c", "d"], ["x"]
    st.descriptions = {"d": "kept"}
    st.flagged_nodes, st.flagged_edges = ["d"], ["a~b", "gone~edge"]
    st.paste_text = (
        'digraph { x -> c; a -> b; "d" [tooltip="note"]; "x" [tooltip="restored"] }\n'
        "// review: node a\n// review: edge a -- b"
    )
    State.import_pasted.fn(st)
    assert st.removed == [] and "x" in st.nodes  # x is in the graph again
    assert st.descriptions == {"d": "note", "x": "restored"}  # the import wins; the rest is kept
    assert st.flagged_nodes == ["d", "a"]
    assert st.flagged_edges == ["a~b"]  # the flag on an edge that no longer exists is dropped


def test_a_failed_import_changes_nothing_and_explains():
    st = data_state()
    st.edges = [{"source": "a", "target": "b", "directed": True}]
    st.paste_text = "c ~ zzz"
    State.import_pasted.fn(st)
    assert st.import_ok is False and "zzz" in st.import_message
    assert edge_pairs(st) == {("a", "b", True)}
    assert st.paste_text == "c ~ zzz"  # kept, so it can be corrected

    State.set_paste_text.fn(st, "   ")
    State.import_pasted.fn(st)
    assert st.import_message == "Paste a DAG first."


def test_import_needs_data_first():
    st = State(_reflex_internal_init=True)
    st.paste_text = "b ~ a"
    State.import_pasted.fn(st)
    assert st.import_ok is False and "Load data first" in st.import_message


def test_import_file_reads_text_and_rejects_binary():
    import asyncio

    class Upload:
        def __init__(self, name, data):
            self.name, self._data = name, data

        async def read(self):
            return self._data

    st = data_state()
    asyncio.run(State.import_file.fn(st, [Upload("g.dot", b"\xef\xbb\xbfdigraph { a -> b }")]))
    assert st.import_ok and edge_pairs(st) == {("a", "b", True)}  # a UTF-8 BOM is fine
    asyncio.run(State.import_file.fn(st, [Upload("pic.png", b"\x89PNG\r\n\x1a\n\xff\xfe")]))
    assert st.import_ok is False and "not a text file" in st.import_message
    assert edge_pairs(st) == {("a", "b", True)}


def panel_state() -> State:
    """A State with the panel test file loaded the way the app loads it."""
    from dag_studio import backend as b

    st = State(_reflex_internal_init=True)
    df, notes, extras = b.load_dataset(b.PANEL_TEST_FILE)
    st._set_table(df, notes, "panel", target="revenue", extras=extras)
    return st


def test_loading_keeps_the_text_columns_available_for_grouping():
    st = panel_state()
    assert st.grouped is False and st.group_candidates[:2] == ["region", "store"]
    assert "week" in st.group_candidates and st.time_options[0] == "(none)"
    assert "week" in st.columns and "region" not in st.columns  # ungrouped: week is a variable


def test_choosing_group_columns_swaps_the_data_for_group_demeaned_variables():
    st = panel_state()
    State.add_group_col.fn(st, "region")
    State.add_group_col.fn(st, "store")
    assert st.group_cols == ["region", "store"]
    assert st.group_headline.startswith("12 groups (region × store), 20 rows each")
    assert "region" not in st.group_candidates and "store" not in st.group_candidates
    State.set_time_col.fn(st, "week")
    assert st.time_col == "week" and "week" not in st.columns and "week" not in st.nodes
    assert st.columns == ["price", "promo", "demand", "revenue", "footfall"]
    assert st.nodes == st.columns and st.n_rows == 240 and st.target == "revenue"
    # The analysed data is the demeaned table, which the rest of the app reads.
    df = st._session()["df"]
    assert df.columns == st.columns and abs(df["price"].mean()) < 1e-9
    assert st._session()["raw"].columns != df.columns  # the original is kept for undoing it
    # Group-level variables were removed and named in the warnings.
    assert any("store_size" in w for w in st.group_warnings)


def test_changing_the_grouping_discards_what_the_old_data_produced():
    st = panel_state()
    st.edges = [
        {"source": "price", "target": "demand", "directed": True},
        {"source": "week", "target": "demand", "directed": True},
    ]
    st.flagged_edges, st.fitted, st.validated_sig = ["week~demand"], True, "old"
    st.families = {"demand": "studentt"}
    st._session()["model"], st._session()["evidence"] = object(), {"price~demand": object()}
    State.add_group_col.fn(st, "region")
    State.add_group_col.fn(st, "store")
    assert len(st.edges) == 2  # week is still a variable until it is made the time column
    State.set_time_col.fn(st, "week")
    assert [(e["source"], e["target"]) for e in st.edges] == [("price", "demand")]  # week is gone
    assert st.flagged_edges == [] and st.families == {"demand": "studentt"}
    assert st.fitted is False and st.has_validation is False
    assert "model" not in st._session() and st._session()["evidence"] == {}


def test_a_grouping_that_cannot_work_changes_nothing_and_says_why():
    st = panel_state()
    State.add_group_col.fn(st, "region")
    before = (list(st.group_cols), list(st.columns), st.n_rows, st.group_headline)
    State.add_group_col.fn(st, "price")  # every price is its own group: all singletons
    assert st.group_error != ""
    assert (list(st.group_cols), list(st.columns), st.n_rows, st.group_headline) == before
    State.add_group_col.fn(st, "store")  # a good choice clears the message
    assert st.group_error == ""


def test_removing_and_clearing_groups_restores_the_variables():
    st = panel_state()
    for column in ("region", "store"):
        State.add_group_col.fn(st, column)
    State.remove_group_col.fn(st, "store")
    assert st.group_cols == ["region"] and "store_size" in st.columns
    assert "store_size" in st.nodes  # a variable that reappears becomes a node again
    State.clear_grouping.fn(st)
    assert st.grouped is False and st.group_headline == "" and st.group_warnings == []
    assert st._session()["df"] is st._session()["raw"] and "week" in st.columns


def test_grouped_wording_reaches_the_numbers():
    st = panel_state()
    st._r2_text, st.fitted, st.stale = {"demand": "0.62 [0.55, 0.69]"}, True, False
    st.selected_nodes = ["demand"]
    assert st.picked_r2 == "R² = 0.62 [0.55, 0.69]"
    State.add_group_col.fn(st, "region")
    st._r2_text, st.fitted = {"demand": "0.62 [0.55, 0.69]"}, True
    st.selected_nodes = ["demand"]
    assert st.picked_r2.endswith("within groups")
    # A non-Gaussian family on grouped data is refused straight away.
    st.selected_nodes = ["demand"]
    State.set_family.fn(st, "bernoulli")
    assert "once group means are removed" in st.error


def test_loading_new_data_clears_the_grouping():
    st = panel_state()
    State.add_group_col.fn(st, "region")
    st.grouping_open = False
    st._set_table(st._session()["raw"], [], "other")
    assert st.group_cols == [] and st.time_col == "" and st.grouping_open is True
    assert st.group_headline == "" and st.group_candidates == st.all_columns


def test_importing_a_dag_uses_the_variables_not_the_group_columns():
    st = panel_state()
    State.add_group_col.fn(st, "region")
    State.add_group_col.fn(st, "store")
    st.paste_text = "demand ~ price + promo\nrevenue ~ demand"
    State.import_pasted.fn(st)
    assert st.import_ok and len(st.edges) == 3
    st.paste_text = "demand ~ week"  # week was the time column, so it is not a variable now
    State.set_time_col.fn(st, "week")
    State.import_pasted.fn(st)
    assert st.import_ok is False and "week" in st.import_message


def test_the_grouping_section_is_open_once_data_is_loaded():
    assert panel_state().grouping_open is True  # easy to find, not tucked away
    State.toggle_grouping.fn(st := panel_state())
    assert st.grouping_open is False and len(st.all_columns) > 0
