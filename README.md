# DAG Studio

A minimal [Reflex](https://reflex.dev) front end for
[pathmc](https://pathmc.pymc-labs.com). Upload a dataset, let pathmc's `TBFPC`
propose a causal graph, edit it by hand, flag what needs a second look, and see
each fitted relationship as a plot.

![layout: controls on the left, graph in the centre, fitted relationship on the right](docs/screenshot.png)

## What it computes

**Discovery.** For each pair, TBFPC tests `Y ⊥ X | S` by comparing two linear
models, `M0: Y ~ S` and `M1: Y ~ S + X`:

    BIC(M)     = n · log(RSS / n) + k · log(n)
    log BF10   ≈ −½ · [BIC(M1) − BIC(M0)]

- `n`: rows. `RSS`: unexplained variation in `Y`. `k`: number of coefficients.
- `BF10`: how much more the data support "X matters" than "X adds nothing given S".
- The edge is dropped when `BF10 < 1`.

The edge panel shows the *weakest* `BF10` found for that pair and the
conditioning set that produced it.

**Fitted relationship.** After a fit, selecting an edge `X → Y` plots

    E[ Y | do(X = x) ]   with an 89% credible band

- `do(X = x)`: X is set to x by intervention, not merely observed at x.
- The curve is the **total** effect of X on Y through every directed path in
  your graph.
- Grey points are the observed data.

Below the plot is Y's **fitted equation**, with posterior means and 89%
intervals:

    Y = b0 + b1 · X + b2 · Z + ε,   ε ~ Normal(0, σ)

The X term is highlighted: it is the **direct** effect of the edge, so you can
see when it differs from the total effect in the curve.

## Run it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run reflex run
```

Open http://localhost:3000. The first start takes a minute while Reflex builds
the front end. Use **Load demo data** to try it without a file.

## Using it

The left panel walks through these steps in order. Each step is numbered,
shows ✓ once done, and the current one is highlighted with a one-line
instruction.

1. **Upload** a CSV or Parquet file. Numeric columns become nodes; non-numeric
   and constant columns are ignored, and rows with missing values are dropped.
   The left panel lists exactly what was changed.

   **Panel data?** Open *Panel data: group the rows* and pick the columns that
   identify a group, from the broadest to the finest: `region`, then `store`.
   Text columns and ID columns are both offered; a group is one *combination*,
   so a store name used in several regions still gives separate groups. An
   optional time column is kept out of the graph and used to check the panel.
   Each group's mean is then removed from every variable (the fixed-effects
   estimate), and discovery, validation, the fit, the plots and R² all use that
   data, so group differences cannot create false links. On simulated data
   where the true effect was 0.5, ignoring groups gave 0.9 to 1.0 and removing
   group means gave 0.5 to 0.6. The panel summarises the result ("12 groups
   (region × store), 20 rows each, balanced") and warns about: rows with a
   missing group value (dropped), groups of one row (dropped), a repeated
   (group, time) pair (usually a label shared by different groups), short
   groups (intervals come out somewhat too narrow), and variables that never
   vary within a group, such as a region-level income, which are removed because
   nothing is left of them once group differences are taken out. Changing the
   grouping discards the fit, validation and discovery evidence, which used the
   old data. See the limits below.
2. **Pick the outcome** and press **Discover DAG**. A progress bar in the top
   right shows how many of TBFPC's passes (one per variable's link to the
   outcome, then one per pair of other variables) are done. Dashed edges are
   ones the data cannot orient.

   **Or import a DAG** you already have, instead of discovering one: drop a
   `.dot` / `.gv` file, or paste text into the box and press **Import pasted
   DAG**. Three formats are read, all by pathmc's own parsers:
   - pathmc's **spec syntax**, one equation per line: `M ~ X`, then
     `Y ~ M + X`. Only the structure is taken; transforms, lags,
     interactions, categorical terms and fixed coefficients are ignored (the
     app says so), as are `~~` and `:=` lines.
   - **DOT**: `digraph { X -> M; M -> Y }`, including chains, quoted names and
     comments. An undirected edge (`--`, or `dir=none`) becomes a dashed edge.
     Node `tooltip` attributes become descriptions and `// review:` comments
     become flags, so a file saved with **Export DOT** imports back whole.
   - An **edge list**: `X->M, M->Y`.

   Names must be columns of the loaded data; a name is also matched after the
   cleaning an upload gets (`ad spend` finds `ad_spend`). If anything cannot be
   imported (an unknown name, with a "did you mean", a cycle, which is named,
   a self-loop, unreadable syntax) nothing is imported and the message says
   why. A successful import replaces the current edges, brings back any removed
   nodes it uses, and clears the fit and validation results, which described
   the old graph.
3. **Edit the graph.**
   - Add an edge: **⌘-click** (Ctrl-click on Windows and Linux) two nodes,
     then press **Add edge** above the graph. The edge runs from the node you
     picked first to the second; **⇄** swaps it. Doing this on a connected
     pair re-points the edge. Clicking nodes only selects them; it never adds
     an edge by itself. A third ⌘-click drops the oldest of the two.
   - Select an edge by clicking it: reverse it, give a dashed edge a
     direction, flag it, or delete it.
   - Select a node by clicking it: describe it, flag it for review or remove
     it. Removed nodes appear as chips in the left panel and can be put back.
   - For a node with parents, the same panel sets its **family** (`gaussian`,
     `studentt`, `bernoulli`, `poisson`, `negbinomial`) and its **priors**:
     the coefficient prior Normal(μ, σ), shared by the intercept and every
     slope, plus the noise, degrees-of-freedom or dispersion prior the family
     uses. Empty boxes keep pathmc's defaults.
   - Edits that would create a cycle are refused, and the cycle the edit would
     have closed is drawn in yellow (FiveThirtyEight's #E5AE38): its nodes and
     edges, plus a dashed line for the refused edge, with the path named above
     the graph. It clears on your next action. A cycle already present in the
     graph is highlighted the same way.
4. **Validate DAG** checks the graph against the data, with no fit needed.
   Every independence the graph implies ("A and B are unrelated once you know
   Z") is tested with pathmc's `test_implications`, including the ones implied
   by unconnected nodes. `PathModel.falsify` then grades the whole graph
   against randomly relabelled versions of itself. The violated pairs sit in a
   list that is collapsed by default (click its header to open it), strongest
   first, with how many to expect by chance at the chosen α. Click a pair to
   highlight it on the graph and add an edge in either direction. A pair
   leaves the list as soon as you connect it, by any route, and the header
   counts what is left ("1 of 3 left"). The check is marked out of date when
   the graph changes.
5. **Fit model** once every edge has a direction and the outcome has at least
   one edge. The graph, outcome, families and priors go to pathmc's
   `BuildModelFromDAG(style="native")`, then MCMC runs (roughly 10 to 60
   seconds). The progress bar in the top right names the step: compiling the
   model, drawing samples (the only step that reports a percentage), finishing
   the posterior, then computing equations and fit quality. Steps that cannot
   say how far along they are show an animated bar instead of a number.

   **Fit quality** appears in the right panel: sampler diagnostics (4 chains; largest
   r̂, smallest effective sample size, divergences, with a warning when the
   fit should not be trusted) and each node's **Bayesian R²** (Gelman et al.,
   2019): the share of its variance its parents explain, with an 89%
   interval. R² measures prediction, not causal correctness; use Validate DAG
   for the graph.
6. **Click any node** to see E[outcome | do(node)]: the outcome's posterior mean
   (with an 89% band) when that node is set to x, summed over every directed
   path from the node to the outcome, so a node with no edge into the outcome
   still gets its indirect effect. A node with no path to the outcome, or the
   outcome itself, says so instead of drawing a flat line.
   **Click any edge** to see its fitted relationship and the fitted equation
   of its end node on the right.
   After a fit, each edge is coloured by the sign of its direct coefficient:
   blue (#008FD5) when it raises its child, red (#FC4F30) when it lowers it,
   and a pale tint when the 89% interval includes zero. Hover an edge for the
   number. Clicking a node highlights its edges and the nodes they connect
   to, and dims the rest.
7. **Export DOT** saves the graph, with node descriptions as `tooltip`
   attributes and flagged items listed as comments. The file can be passed
   straight to `pathmc.dag_to_spec`, or imported back (step 2).

After any structural edit, or a change of family or prior, the plot is hidden
and the button reads **Refit model**, so a plot never describes a model that is
no longer on screen.

## Manual testing

The left panel has **Load test file** and **Download test file** links for
`assets/dag_studio_test_data.csv`, a messy dataset with a known structure.
Loading it uses the same cleaning as an upload; downloading it lets you try the
drop zone too. Things to check:

- **Cleaning notes** list: `region` ignored (text), `tax_rate` ignored
  (constant), four columns renamed (`2024 price index` → `v_2024_price_index`),
  15 rows dropped for missing `web_visits`.
- **Discover DAG** (outcome `revenue`) draws four layers top-down, from
  `ad_spend`/`season`/`v_2024_price_index` down to `revenue`. `noise` stays
  unconnected and the long `customer_satisfaction_score` label is shortened.
- `season — store_visits` comes back dashed: orient it with **Keep direction**.
- Orient the dashed edge, then **Validate DAG**: the top violation is
  `revenue — v_2024_price_index` given `orders` (discovery missed that real
  effect). Click it and choose **Add v_2024_price_index → revenue**; the check
  is marked out of date. Validate again: what's left (about 2) is no more
  than chance would flag. Try ⌘-clicking `revenue`, then `ad_spend`, and
  pressing Add edge: the refused edit shows the cycle it would close in yellow.
- Click a node, type a description, hover the node to see it, then
  **Export DOT** and look for the `tooltip` attribute.
- **Fit model**, then click `orders → revenue`: the equation appears at once,
  with the `orders` term highlighted and near 0.9. Edit any edge and it hides
  until you refit.
- Click `repeat_purchase` (0/1), set its family to `bernoulli` and its
  coefficient σ to 2, then refit: its equation reads `logit P(repeat_purchase
  = 1) = …` with no noise term and its curve is a probability. Setting
  `bernoulli` on a continuous node such as `orders` shows an error straight
  away.

- **Panel data**: press **Load panel test file** (a `region`, `store`, `week`
  panel: 3 regions x 4 stores x 20 weeks, store names repeating across
  regions). First **Discover DAG** with no grouping: the graph is wrong (an
  edge reversed, `footfall` linked to `price`, `price → revenue` missing).
  Open *Panel data*, add `region` then `store`, and set the time column to
  `week`. The summary reads "12 groups (region × store), 20 rows each,
  balanced, 20 periods"; `store_size` and `region_income` are removed with an
  explanation. Discover again: exactly `price → demand`, `promo → demand`,
  `demand → revenue`, `price → revenue`, with `footfall` left alone. Try
  adding only `store` to see the repeated-label warning, and set `demand` to
  `bernoulli` to see it refused. After a fit the price → demand slope is
  about -0.67 (true -0.8; ungrouped it is about -0.22).
- **Import**: with the test file loaded, paste this into the box in step 2 and
  press **Import pasted DAG** (`2024 price index` is matched to
  `v_2024_price_index`):

      web_visits ~ ad_spend + season
      orders ~ web_visits + store_visits + 2024 price index
      revenue ~ orders

  It imports (after a note about the renamed column). Then paste
  `orders ~ revenue` plus `revenue ~ orders` to see the cycle named, and
  `revenue ~ ordrs` to see "did you mean orders?". **Export DOT**, load the
  file back with the drop zone, and the edges, descriptions and flags return.

The file is generated by `backend.test_file_table()`; a test fails if the two
drift apart.

## Layout of the code

| File | Role |
|---|---|
| `dag_studio/backend.py` | Data loading, TBFPC, layout and edge routing, fitting, curves. No Reflex imports. |
| `dag_studio/state.py` | Reflex state and event handlers. |
| `dag_studio/dag_studio.py` | The page: three panels and the SVG graph. |
| `tests/test_backend.py` | Backend tests, including recovery of known effects from synthetic data. |

Settings such as the MCMC budget, the Bayes-factor threshold and the interval
width are constants at the top of `backend.py`.

```bash
uv run pytest                 # everything (includes two short MCMC fits)
uv run pytest -m "not slow"   # skip the MCMC tests
uv build                      # sdist and wheel (the wheel holds the dag_studio package)
```

## Limits to know about

- **Linear, Gaussian, numeric only.** Both discovery and the fitted model are
  linear. A purely nonlinear dependence can be missed, and the plotted curve
  will always be a straight line.
- **Grouped data is a fixed-effects analysis only.** Removing group means
  compares each group with itself, so effects of group-level variables cannot
  be estimated, and `bernoulli`, `poisson` and `negbinomial` outcomes are
  refused (a demeaned 0/1 or count is no longer a 0/1 or count). Time effects,
  random intercepts and slopes, and lags are not done: pathmc supports them
  (`panel=` and `pooling=` in `pathmc.model`), but `BuildModelFromDAG` does not
  pass them on. Plots and R² are in deviations from group means, and say so.
- **Discovery is target-first.** TBFPC searches for drivers of the chosen
  outcome. It is not a symmetric whole-graph search, and pathmc marks it
  experimental.
- **No edge-stability estimate.** The Bayes factor is from one pass over the
  data; there is no bootstrap.
- **Single user, in memory.** The dataset and fitted model live in the server
  process, keyed by browser tab, and are lost on restart. This is a local
  tool, not a hosted service.
- **pathmc is pre-1.0.** The dependency is pinned to `0.6.x`.
- Node positions are computed, not draggable. **Tidy layout** recomputes them.
