"""
Default plots for explainability study.

Public surface of the package. Import everything you need from here:

>>> from mypackage.plot import *          # doctest: +SKIP
>>> result = plot_allocations(            # doctest: +SKIP
...     df,
...     Encoding(x="technology", y="value", color="effect", compare_by="scenario"),
...     facets=FacetSpec(col="region", col_wrap=3),
... )

The heavy lifting lives in ``mypackage.plot``:

* ``utils.specs``   -- the configuration dataclasses imported below.
* ``utils.prepare`` -- ordering, aggregation, style maps, render plan.
* ``utils.render``  -- the Matplotlib / Plotly renderers and figure export.

To add more public plots later, define them in this file (or another category
file such as ``analysis.py``) and append their names to ``__all__``.
"""

from __future__ import annotations  # String annotations.

import warnings  # Non-fatal configuration notices (e.g. show_total with kind="box").

import pandas as pd  # The input data type.
from typing import Any, Mapping  # highlight={column: value(s)} typing.

from .prepare import build_box_plan, build_fingerprint_plan, build_grid_plan, build_joint_plan, build_render_plan  # Spec + data -> plan.
from .render import (  # The renderers.
    render_box_matplotlib,
    render_scatter_explain,
    render_box_plotly,
    render_fingerprint_matplotlib,
    render_matplotlib,
    render_multi_jointplot,
    render_multiplot_matplotlib,
    render_multiplot_plotly,
    render_plotly,
)
from .specs import (  # Unified public config (A).
    Layout,
    Style,
    Regression,
    FingerprintStyle,
)
from .specs import Encoding, SaveSpec, PlotResult  # Public role container / save / result.
from .encodings import check_encoding  # Relevance guard for directly-built Encodings.
from ._context import (  # Internal contexts + small helper structs.
    BoxContext,
    DistSpec,
    FingerprintContext,
    FacetSpec,
    GridContext,
    JointContext,
    LabelSpec,
    MarginalSpec,
    OrderSpec,
    PlotContext,
    ScatterSpec,
)

# Names exported by ``from mypackage.plot import *`` (and, transitively, by
# ``from mypackage import *``). Only the entry point(s) and the public config /
# result objects are exposed; the internals stay in ``utils``.
__all__ = [
    "plot_allocations",  # Entry point 1: stacked/grouped allocation plot.
    "multiplot_grid",  # Entry point 2: N x K diagnostic grid.
    "multi_jointplot_subfig",  # Entry point 3: multi-panel jointplots.
    "grouped_boxplot",  # Entry point 4: categorical box/violin/strip comparison.
    "fingerprint_diagram",  # Entry point 5: radial model-fingerprint dashboard.
    "scatter_explain",  # Entry point 6: joint panel + two stacked allocation panels.
    # Unified public config (everything factorable lives here; the rest are
    # explicit function kwargs). SaveSpec and FingerprintStyle are kept dedicated.
    "Encoding", "Layout", "Style", "Regression", "SaveSpec", "FingerprintStyle",
    # shared result type:
    "PlotResult",
]

# Frozen default spec instances. Safe to share as default arguments precisely
# because the specs are immutable (no mutable-default pitfall).

# --- unified public config (A: adapter to the internal per-chart specs) ---
_DEF_LAYOUT = Layout()  # Generic figure furniture.
_DEF_STYLE = Style()  # Generic cosmetics.
_DEF_REGRESSION = Regression()  # No regression by default.
_DEF_FP_STYLE = FingerprintStyle()  # Fingerprint geometry (kept dedicated).


_VALID_GRAPH_TYPES = {"bar", "line", "bubble"}  # Accepted graph types.
_VALID_BUBBLE_SCOPES = {"figure", "row", "col", "panel"}  # Bubble normalisation scopes.
_VALID_BACKENDS = {"matplotlib", "plotly"}  # Accepted backends.
_VALID_LINE_MODES = {"lines", "lines+markers"}  # Accepted line modes.


def _emit(result: PlotResult, show: bool) -> PlotResult:
    """Handle inline display of a finished figure, then return it unchanged.

    Unifies the two backends' notebook behaviour:

    * **matplotlib** -- the inline backend auto-displays any *open* figure at the
      end of a cell, even when the call is only assigned to a variable. So to
      *suppress* display (run + save only) the figure must be closed. When
      ``show=True`` in a notebook we display it once explicitly and then close it
      (avoids a duplicate auto-display and frees memory, which matters in loops).
      Outside IPython the figure is left open for the caller (e.g. ``plt.show()``).
    * **plotly** -- nothing auto-displays, so ``show=True`` calls ``fig.show()``
      and ``show=False`` does nothing.

    The returned :class:`PlotResult` is unchanged either way; even a closed
    matplotlib figure can still be shown later via ``display(result.figure)``.

    Parameters
    ----------
    result:
        The finished plot result.
    show:
        Whether to render the figure inline now.

    Returns
    -------
    PlotResult
        The same result, for chaining / assignment.
    """
    if result.backend == "matplotlib":  # Matplotlib path.
        import matplotlib.pyplot as plt  # Local import: only needed for display control.
        if show:  # Display once, then close to avoid the duplicate auto-display + leaks.
            try:
                from IPython.display import display  # Available inside Jupyter/IPython.
                display(result.figure)  # Render exactly once.
                plt.close(result.figure)  # Prevent the inline backend's end-of-cell duplicate.
            except Exception:  # Not in IPython (plain script): leave it open for the caller.
                pass
        else:  # Suppress display entirely: closing is required, else inline auto-shows it.
            plt.close(result.figure)  # Run + save without rendering; also frees memory.
    else:  # Plotly path.
        if show:  # Plotly never auto-displays, so trigger it explicitly.
            result.figure.show()
    return result  # Always hand back the (still usable) result.


def _validate(data: pd.DataFrame, encoding: Encoding, graph_type: str, backend: str,
              facets: FacetSpec, style: Style, scatter: ScatterSpec | None,
              save: SaveSpec | None) -> None:
    """Validate the request up front with clear, actionable errors.

    Raises
    ------
    ValueError
        On any inconsistent or unsupported option.
    KeyError
        If a referenced column is absent from ``data``.
    """
    if graph_type not in _VALID_GRAPH_TYPES:  # Graph type.
        raise ValueError(f"graph_type must be one of {_VALID_GRAPH_TYPES}, got {graph_type!r}.")
    if backend not in _VALID_BACKENDS:  # Backend.
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}.")
    if style.line_mode not in _VALID_LINE_MODES:  # Line mode.
        raise ValueError(f"style.line_mode must be one of {_VALID_LINE_MODES}, got {style.line_mode!r}.")
    if facets.col_wrap is not None and facets.row_wrap is not None:  # Wraps are exclusive.
        raise ValueError("Use only one of facets.col_wrap or facets.row_wrap, not both.")
    if (facets.col_wrap is not None or facets.row_wrap is not None) \
            and facets.col is not None and facets.row is not None:  # Wrap needs a single dim.
        raise ValueError("col_wrap / row_wrap is intended for a single facet dimension; "
                         "do not combine a wrap with both facets.col and facets.row.")
    if scatter is not None and (not isinstance(scatter.category, dict) or len(scatter.category) != 1):
        raise ValueError("scatter.category must be a dict with exactly one item, "
                         "e.g. {'effect_type': 'Total'}.")  # Reference selector shape.
    if save is not None and (not save.path or not save.name):  # Saving needs path + name.
        raise ValueError("To save, SaveSpec needs both a non-empty path and name.")

    referenced = [encoding.x, encoding.y]  # Always-required columns.
    if not (graph_type == "bubble" and encoding.color is None):  # Bubble: colour is optional.
        referenced.append(encoding.color)
    for col in (encoding.compare_by, encoding.x_hierarchy, facets.col, facets.row):  # Optional.
        if col is not None:
            referenced.append(col)
    if scatter is not None:  # Reference selector column.
        referenced.append(scatter.column)
    if graph_type == "bubble" and encoding.size is not None:  # Bubble heatmap extra role.
        referenced.append(encoding.size)
    if graph_type == "bubble" and encoding.background is not None:
        referenced.append(encoding.background)
    missing = [col for col in dict.fromkeys(referenced) if col not in data.columns]  # De-duped check.
    if missing:  # Report all at once.
        raise KeyError(f"Columns not found in data: {missing}.")
    if graph_type == "bubble":  # Bubble heatmap: colour/size numeric, no bar-only roles.
        for role, col in (("color", encoding.color), ("size", encoding.size)):
            if col is not None and not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"graph_type='bubble' needs a numeric {role} column, "
                                 f"got {col!r} of dtype {data[col].dtype}.")
        if encoding.compare_by is not None or encoding.x_hierarchy is not None:
            raise ValueError("graph_type='bubble' does not support compare_by / x_hierarchy.")


def plot_allocations(
    data: pd.DataFrame,
    encoding: Encoding,
    *,
    graph_type: str = "bar",
    bubble_sizes: tuple[float, float] = (30.0, 500.0),
    bubble_cmap: str = "viridis",
    bubble_color: str = "steelblue",
    bubble_background_palette: Mapping[Any, str] | None = None,
    bubble_background_alpha: float = 0.15,
    bubble_scope: str = "figure",
    bubble_axes_scope: str = "figure",
    bubble_agg: str = "mean",
    backend: str = "matplotlib",
    layout: Layout = _DEF_LAYOUT,
    style: Style = _DEF_STYLE,
    save: SaveSpec | None = None,
    scatter_category=None,
    scatter_size: float = 40.0,
    scatter_facecolor: str = "black",
    scatter_edgecolor: str = "white",
    scatter_linewidth: float = 0.5,
    show: bool = True,
) -> PlotResult:
    """Draw a faceted, stacked/grouped allocation plot.

    The single public entry point. The original ~45 keyword arguments are grouped
    into immutable spec objects (see ``mypackage.plot.specs``).

    Parameters
    ----------
    data:
        Long-format input frame.
    encoding:
        Column-to-role mapping (x, y, colour, optional compare / hierarchy).
    graph_type:
        ``"bar"`` (stacked, grouped), ``"line"``, or ``"bubble"`` for a bubble
        heatmap: both axes categorical, one bubble per ``(x, y)`` cell, colour
        and area encoding two numeric columns with figure-wide scales (build
        the encoding with :func:`mypackage.plot.encodings.heatmap`).
    bubble_sizes:
        Bubble area range in Matplotlib points^2, mapped linearly onto the
        ``size`` column's aggregated range; Plotly derives its marker
        diameters from the same scale. Pass ``(max, min)`` -- e.g.
        ``(500, 30)`` -- to *invert* the mapping: the smallest values get the
        biggest bubbles (useful for ranks). (``graph_type="bubble"`` only.)
    bubble_cmap:
        Continuous colormap name for the ``color`` column; append ``_r`` to
        any name (``"viridis_r"``, ...) to invert the colour mapping
        (``graph_type="bubble"`` only).
    bubble_color:
        Single bubble colour used when ``encoding.color`` is ``None``
        (``graph_type="bubble"`` only; no colorbar in that case).
    bubble_background_palette:
        ``{class: colour}`` mapping for the background bands driven by
        ``encoding.background``; classes missing from the mapping fall back to
        a Matplotlib ``tab10`` cycle. ``None`` -> full ``tab10`` cycle.
        (``graph_type="bubble"`` only.)
    bubble_background_alpha:
        Opacity of those background bands, in ``[0, 1]``
        (``graph_type="bubble"`` only).
    bubble_scope:
        Scope over which the colour and size scales are normalised:
        ``"figure"`` (default -- every panel comparable), ``"row"`` or
        ``"col"`` (one scale per facet row / column, each with its own
        colorbar and size key) or ``"panel"``. Falls back to ``"figure"``
        with a warning when the requested facet dimension is absent
        (``graph_type="bubble"`` only).
    bubble_axes_scope:
        Same scopes, applied to the *categorical* axes: which ``x`` / ``y``
        values appear in a panel. ``"row"`` gives each facet row its own
        categories (e.g. its own top-N features), ``"figure"`` (default)
        keeps one shared grid (``graph_type="bubble"`` only).
    bubble_agg:
        Aggregation applied when several rows share an ``(x, y)`` cell:
        ``"mean"`` (default), ``"sum"`` or ``"median"``
        (``graph_type="bubble"`` only).
    backend:
        ``"matplotlib"`` (vector publication output) or ``"plotly"`` (interactive).
        ``encoding.x_hierarchy`` is honoured only by the Matplotlib backend.
    layout, style, save:
        The corresponding spec objects (``save`` defaults to ``None``: no file
        written).
    scatter_category:
        Reference overlay (e.g. a "Total") drawn as markers on top of the bars:
        a one-item mapping ``{column: value}`` selecting the reference rows.
        ``None`` -> no overlay. NB: a mapping, not a bare string.
    scatter_size, scatter_facecolor, scatter_edgecolor, scatter_linewidth:
        Marker styling of that reference overlay (area, face / edge colours,
        edge width).
    show:
        When ``True`` (default) the figure is displayed inline in a notebook;
        when ``False`` it is computed (and saved, if ``save`` is set) without any
        display. The :class:`PlotResult` is returned either way.

    Returns
    -------
    PlotResult
        ``figure``, ``axes`` (Matplotlib only), ``files`` and ``backend``.

    Raises
    ------
    ValueError, KeyError
        On invalid configuration.
    """
    graph_type = graph_type.lower().strip()  # Normalise.
    if graph_type == "bubble" and bubble_agg not in ("mean", "sum", "median"):
        raise ValueError(f"bubble_agg must be 'mean', 'sum' or 'median', got {bubble_agg!r}.")
    if graph_type == "bubble" and not 0.0 <= bubble_background_alpha <= 1.0:
        raise ValueError(f"bubble_background_alpha must be in [0, 1], got {bubble_background_alpha!r}.")
    if graph_type == "bubble":  # Normalisation scopes must name an available facet dimension.
        for name, scope in (("bubble_scope", bubble_scope),
                            ("bubble_axes_scope", bubble_axes_scope)):
            if scope not in _VALID_BUBBLE_SCOPES:
                raise ValueError(f"{name} must be one of {sorted(_VALID_BUBBLE_SCOPES)}, "
                                 f"got {scope!r}.")
            need = {"row": encoding.facet_row, "col": encoding.facet_col,
                    "panel": encoding.facet_row or encoding.facet_col}.get(scope)
            if scope != "figure" and need is None:
                warnings.warn(f"{name}={scope!r} needs the matching facet dimension; "
                              f"falling back to 'figure'.", UserWarning, stacklevel=2)
    backend = backend.lower().strip()  # Normalise.

    col_wrap, row_wrap = layout.col_wrap, layout.row_wrap  # Wraps shape a *single* facet dimension.
    if encoding.facet_col is not None and encoding.facet_row is not None \
            and (col_wrap is not None or row_wrap is not None):
        warnings.warn("col_wrap / row_wrap is ignored when both facet_col and facet_row are set: "
                      "the two facet dimensions already determine the grid.",
                      UserWarning, stacklevel=2)
        col_wrap = row_wrap = None  # Selective read: irrelevant Layout fields are ignored.
    facets = FacetSpec(col=encoding.facet_col, row=encoding.facet_row, col_wrap=col_wrap,
                      row_wrap=row_wrap, share_y=layout.share_y, hide_empty=layout.hide_empty)  # Internal helper.
    order = OrderSpec(category_orders=encoding.category_orders, color_order=encoding.color_order,
                      compare_order=encoding.compare_order, sort_categories=encoding.sort_categories)
    labels = LabelSpec(mapping=encoding.labels or {})
    scatter = (ScatterSpec(category=scatter_category, size=scatter_size, facecolor=scatter_facecolor,
                           edgecolor=scatter_edgecolor, linewidth=scatter_linewidth)
               if scatter_category is not None else None)

    check_encoding(encoding, "plot_allocations_bubble" if graph_type == "bubble"
                   else "plot_allocations")  # Warn on set-but-ignored roles.
    _validate(data, encoding, graph_type, backend, facets, style, scatter, save)  # Fail fast.

    ctx = PlotContext(  # Unified: Encoding + Style + Layout + internal helper structs.
        data=data, encoding=encoding, graph_type=graph_type, backend=backend,
        facets=facets, order=order, style=style, layout=layout,
        labels=labels, scatter=scatter,
        bubble_sizes=bubble_sizes, bubble_cmap=bubble_cmap, bubble_agg=bubble_agg,
        bubble_color=bubble_color,
        bubble_background_palette=bubble_background_palette,
        bubble_background_alpha=bubble_background_alpha,
        bubble_scope=bubble_scope, bubble_axes_scope=bubble_axes_scope, save=save,
    )
    plan = build_render_plan(ctx)  # Precompute orders, layout, maps, aggregators.

    result = render_matplotlib(ctx, plan) if backend == "matplotlib" else render_plotly(ctx, plan)
    return _emit(result, show)  # Handle inline display (or suppression), then return.


# Frozen default spec instances for multiplot_grid (safe to share: immutable).

_VALID_VARIABLES_BY = {"row", "col"}  # Accepted grid orientations.
_VALID_GRID_COLUMNS = {"series", "dist", "scatter"}  # Accepted grid cells.
_VALID_DIST_KINDS = {"kde", "hist", "box"}  # Accepted distribution kinds.
_VALID_DIST_NORMS = {"density", "frequency", "probability"}  # Accepted density-axis units.


def _validate_grid(data: pd.DataFrame, benchmark: pd.DataFrame | None, encoding: Encoding,
                   backend: str, variables_by: str, columns, dist: DistSpec, highlight,
                   highlight_lines, cell_ratios, save: SaveSpec | None) -> None:
    """Validate a :func:`multiplot_grid` request with clear, actionable errors.

    Raises
    ------
    ValueError
        On any inconsistent or unsupported option.
    KeyError
        If a referenced column is absent from ``data`` (or ``benchmark``).
    """
    if backend not in _VALID_BACKENDS:  # Backend.
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}.")
    if variables_by not in _VALID_VARIABLES_BY:  # Orientation.
        raise ValueError(f"variables_by must be one of {_VALID_VARIABLES_BY}, "
                         f"got {variables_by!r}.")
    cols = list(columns)  # Selected diagnostic cells (ordered).
    if not cols:  # At least one cell.
        raise ValueError("columns must select at least one cell.")
    bad = [c for c in cols if c not in _VALID_GRID_COLUMNS]  # Unknown cell names.
    if bad:
        raise ValueError(f"columns entries must be in {_VALID_GRID_COLUMNS}, got {bad}.")
    if len(set(cols)) != len(cols):  # Each cell at most once.
        raise ValueError("columns must not repeat a cell.")
    if "scatter" in cols and encoding.scatter_by is None:  # Scatter needs its x column.
        raise ValueError('encoding.scatter_by is required when the scatter cell is shown '
                         '("scatter" in columns).')
    if "dist" in cols and not dist.show_data and not dist.show_total \
            and not (dist.show_benchmark and benchmark is not None):  # Empty distribution cell.
        raise ValueError('The distribution cell would be empty: enable dist.show_data or '
                         'dist.show_total, provide a benchmark with dist.show_benchmark=True, '
                         'or drop "dist" from columns.')
    if dist.kind not in _VALID_DIST_KINDS:  # Distribution kind.
        raise ValueError(f"dist.kind must be one of {_VALID_DIST_KINDS}, got {dist.kind!r}.")
    if dist.norm not in _VALID_DIST_NORMS:  # Density-axis unit.
        raise ValueError(f"dist.norm must be one of {_VALID_DIST_NORMS}, got {dist.norm!r}.")
    if not 0.0 <= dist.group_alpha <= 1.0:  # Fill opacity of the per-group layer.
        raise ValueError(f"dist.group_alpha must be in [0, 1], got {dist.group_alpha!r}.")
    if dist.show_total and dist.kind == "box":  # No pooled overlay for boxes (would clutter).
        warnings.warn('dist.show_total is ignored for dist.kind="box" (a pooled box would '
                      'clutter the cell); use "kde" or "hist" for the pooled overlay.',
                      UserWarning, stacklevel=3)
    if dist.log_x and dist.kind == "box":  # Box x-axis is categorical (slot positions).
        warnings.warn('dist.log_x is ignored for dist.kind="box": the cell x-axis is '
                      'categorical, not a density axis.', UserWarning, stacklevel=3)
    if highlight is not None:  # {column: value(s)} spotlight selection (scatter cell).
        if not isinstance(highlight, Mapping) or len(highlight) != 1:
            raise ValueError(f"highlight must be a one-item {{column: value(s)}} mapping, got {highlight!r}.")
        h_col = next(iter(highlight))
        if h_col not in data.columns:
            raise KeyError(f"highlight column {h_col!r} is not in data.")
    if cell_ratios is not None:  # One relative size per drawn cell.
        if len(cell_ratios) != len(columns):
            raise ValueError(f"cell_ratios must have one value per cell: got {len(cell_ratios)} "
                             f"for columns={tuple(columns)}.")
        if any(r <= 0 for r in cell_ratios):
            raise ValueError(f"cell_ratios must be strictly positive, got {tuple(cell_ratios)}.")
    if highlight_lines is not None:  # {column: value(s)} trajectory spotlight (series cell).
        if not isinstance(highlight_lines, Mapping) or len(highlight_lines) != 1:
            raise ValueError(f"highlight_lines must be a one-item {{column: value(s)}} mapping, "
                             f"got {highlight_lines!r}.")
        hl_col = next(iter(highlight_lines))
        if hl_col not in data.columns:
            raise KeyError(f"highlight_lines column {hl_col!r} is not in data.")
    if dist.highlight_on is not None:  # Pin the highlight points on a density curve.
        if dist.highlight_on not in ("group", "total"):
            raise ValueError(f'dist_highlight must be None, "group" or "total", got {dist.highlight_on!r}.')
        if highlight is None:
            raise ValueError("dist_highlight requires the highlight={column: value(s)} selection.")
        if dist.kind == "box":  # No curve to pin on.
            warnings.warn('dist_highlight is ignored for dist.kind="box" (no density curve).',
                          UserWarning, stacklevel=3)
        elif dist.highlight_on == "total" and not dist.show_total:
            warnings.warn("dist_highlight='total' with dist_show_total=False: the markers are "
                          "pinned on a curve that is not drawn.", UserWarning, stacklevel=3)
    if not list(encoding.variables):  # At least one variable.
        raise ValueError("encoding.variables must list at least one variable column.")
    if save is not None and (not save.path or not save.name):  # Saving needs path + name.
        raise ValueError("To save, SaveSpec needs both a non-empty path and name.")

    # Every referenced column must exist in ``data``.
    referenced = list(encoding.variables) + [encoding.line_by]  # Core columns.
    if encoding.scatter_by is not None:  # Scatter x (optional unless the scatter cell is shown).
        referenced.append(encoding.scatter_by)
    for col in (encoding.by, encoding.line_id):  # Optional data columns.
        if col is not None:
            referenced.append(col)
    missing = [c for c in dict.fromkeys(referenced) if c not in data.columns]  # De-duped check.
    if missing:
        raise KeyError(f"Columns not found in data: {missing}.")

    # Benchmark axis / trajectory-id columns must exist when a benchmark is given.
    # Individual variables are NOT required: a variable absent from the benchmark
    # is plotted without its benchmark overlay (the renderers warn and skip it).
    if benchmark is not None:
        bench_ids = ([] if encoding.benchmark_line_id is None
                     else [encoding.benchmark_line_id] if isinstance(encoding.benchmark_line_id, str)
                     else list(encoding.benchmark_line_id))
        bench_ref = [encoding.line_by] + bench_ids  # Core benchmark columns.
        if encoding.scatter_by is not None:  # Scatter x, when used.
            bench_ref.append(encoding.scatter_by)
        bench_missing = [c for c in dict.fromkeys(bench_ref) if c not in benchmark.columns]
        if bench_missing:
            raise KeyError(f"Columns not found in benchmark: {bench_missing}.")


def multiplot_grid(
    data: pd.DataFrame,
    encoding: Encoding,
    *,
    benchmark: pd.DataFrame | None = None,
    year=None,
    backend: str = "matplotlib",
    layout: Layout = _DEF_LAYOUT,
    style: Style = _DEF_STYLE,
    regression: Regression = _DEF_REGRESSION,
    save: SaveSpec | None = None,
    variables_by: str = "row",
    columns=("series", "dist", "scatter"),
    year_range: tuple = (2020.0, 2050.0),
    var_window=None,
    cell_ratios=None,
    dist_kind: str = "kde",
    dist_norm: str = "density",
    dist_points: int = 200,
    dist_show_data: bool = True,
    dist_show_benchmark: bool = True,
    dist_group_alpha: float = 0.5,
    dist_show_total: bool = False,
    dist_total_color: str = "black",
    dist_total_label: str = "total",
    dist_log_x: bool = False,
    dist_highlight: str | None = None,
    highlight: Mapping[str, Any] | None = None,
    highlight_lines: Mapping[str, Any] | None = None,
    show: bool = True,
) -> PlotResult:
    """Draw an N x 3 (or 3 x N) diagnostic grid comparing data against a benchmark.

    Each variable in ``encoding.variables`` occupies one line of the grid (when
    ``variables_by == "row"``; transposed when ``"col"``):

    * **column 1** -- time series vs ``line_by``, coloured by ``by``, with the
      benchmark trajectories greyed in the background;
    * **column 2** -- the marginal distribution at the final ``year`` (kde / hist
      / box), drawn horizontally (variable on the y-axis), benchmark greyed
      behind;
    * **column 3** -- a scatter vs ``scatter_by`` at the final ``year``, coloured
      by ``by``, with optional LOESS regressions for the data and the benchmark.

    The set and order of cells are configurable via ``columns`` (any
    ordered subset of ``("series", "dist", "scatter")``), so the grid is really
    N x ``len(columns)``; dropping ``"scatter"`` makes ``scatter_by``
    optional. The distribution cell can show the benchmark only, the data only,
    or both, via ``dist.show_benchmark`` / ``dist.show_data``.

    Distribution-cell layering (back to front): grey benchmark
    (``style.benchmark_color`` / ``style.benchmark_alpha``), then the per-``by``
    distribution curves, *unfilled*, line opacity ``dist_group_alpha`` (box
    faces are filled at that alpha instead; colours from the same
    ``plan.color_map`` as the per-group scatter regression, so both agree by
    construction), then -- when ``dist_show_total=True`` -- the pooled
    distribution of *all* groups as one unfilled full-opacity curve in
    ``dist_total_color`` (kde / hist only; ignored with a warning for
    ``dist_kind="box"``).

    Parameters
    ----------
    data:
        The user's long-format data.
    benchmark:
        Keyword-only. The comparison/reference data, or ``None`` (default) for
        no benchmark overlay.
    encoding:
        Column-to-role mapping (variables, line_by, scatter_by, by, line ids).
    year:
        Value of ``line_by`` selecting the cross-section for columns 2 and 3.
        ``None`` uses all rows.
    backend:
        ``"matplotlib"`` or ``"plotly"``.
    layout, style, regression, save:
        The corresponding spec objects (all optional).
    variables_by:
        ``"row"`` (one variable per row, default) or ``"col"`` (transposed).
    columns:
        Ordered subset of ``("series", "dist", "scatter")`` -- which cells to
        draw, in which order.
    year_range:
        ``(min, max)`` window on ``line_by`` for the series cell.
    var_window:
        Per-variable value window ``{variable: (min, max)}`` clipping the
        drawn range. Without an entry, the axis is automatic -- its bottom is
        clamped at 0 only for variables whose values are all non-negative.
    cell_ratios:
        Relative size of each cell along the cell axis, one number per entry of
        ``columns`` (e.g. ``(2.0, 1.0)`` makes the series cell twice as wide as
        the distribution cell). Widths when ``variables_by="row"`` (the
        default), heights when ``"col"``. ``None`` (default) -> equal cells.
    dist_kind:
        Distribution-cell mark: ``"kde"``, ``"hist"`` or ``"box"``.
    dist_norm:
        Density-axis unit: ``"density"``, ``"frequency"`` or ``"probability"``.
    dist_points:
        Plotly KDE sampling resolution.
    dist_show_data:
        Draw the data's own distribution(s); ``False`` for a benchmark-only
        (or total-only) cell.
    dist_show_benchmark:
        Draw the grey benchmark distribution behind the data (when a
        benchmark is given).
    dist_group_alpha:
        Opacity of the per-``by`` distribution curves (line opacity for
        kde / hist, box face opacity for ``dist_kind="box"``). See the
        layering paragraph above.
    dist_show_total:
        Also draw the pooled all-groups distribution as one unfilled
        full-opacity curve on top (kde / hist only).
    dist_total_color:
        Colour of that pooled curve.
    dist_total_label:
        Legend label of that pooled curve (default ``"total"``; also passed
        through ``encoding.labels``).
    dist_log_x:
        Log-scale the density axis of the distribution cell (kde / hist only;
        ignored with a warning for ``dist_kind="box"``). Floored 3 decades
        below the cell's density peak so KDE tails don't stretch the axis.
    highlight:
        Points to spotlight in the scatter cell: a one-item
        ``{column: value}`` or ``{column: [values]}`` mapping selecting rows of
        ``data``. The matching points are redrawn *on top* of the data scatter,
        larger (2.5x ``style.scatter_size``), lightly hatched with a black edge
        (Matplotlib; Plotly markers cannot hatch, so the Plotly backend uses
        the larger size + black outline only). Legend entry ``"Highlight"``,
        renamable via ``encoding.labels``. ``None`` (default) does nothing.
    dist_highlight:
        Also pin the ``highlight`` points inside the distribution cell, at
        ``(density(value), value)`` on a reference curve: ``"group"`` = each
        point's own per-``by`` curve, ``"total"`` = the pooled curve (the
        marker keeps its group colour), ``None`` (default) = no markers.
        Requires ``highlight``; kde / hist only (ignored with a warning for
        ``dist_kind="box"``).
    highlight_lines:
        Trajectories to spotlight in the series cell, independent of
        ``highlight``: a one-item ``{column: value(s)}`` mapping selecting rows
        of ``data``; every trajectory containing a matching row is redrawn on
        top with a black outline (path effects on Matplotlib, an under-laid
        black trace on Plotly), keeping its group colour. ``None`` (default)
        does nothing.
    show:
        When ``True`` (default) the figure is displayed inline in a notebook;
        when ``False`` it is computed (and saved, if ``save`` is set) without any
        display. The :class:`PlotResult` is returned either way.

    Returns
    -------
    PlotResult
        ``figure``, ``axes`` (Matplotlib only), ``files`` and ``backend``.

    Raises
    ------
    ValueError, KeyError
        On invalid configuration.

    Notes
    -----
    Regression requires ``statsmodels`` and the Plotly KDE requires ``scipy``;
    both are imported lazily, only when actually used.
    """
    backend = backend.lower().strip()  # Normalise.

    order = OrderSpec(category_orders=encoding.category_orders, color_order=encoding.color_order,
                      compare_order=encoding.compare_order, sort_categories=encoding.sort_categories)  # Internal helper struct.
    labels = LabelSpec(mapping=encoding.labels or {})
    dist = DistSpec(kind=dist_kind, norm=dist_norm, n_points=dist_points,
                    show_data=dist_show_data, show_benchmark=dist_show_benchmark,
                    group_alpha=dist_group_alpha, show_total=dist_show_total,
                    total_color=dist_total_color, total_label=dist_total_label,
                    highlight_on=dist_highlight, log_x=dist_log_x)

    check_encoding(encoding, "multiplot_grid")  # Warn on set-but-ignored roles.
    _validate_grid(data, benchmark, encoding, backend, variables_by, columns, dist, highlight, highlight_lines, cell_ratios, save)  # Fail fast.

    ctx = GridContext(  # Unified: Encoding + Style + Regression + flat grid geometry/legend.
        data=data, benchmark=benchmark, encoding=encoding, style=style, dist=dist, highlight=highlight, highlight_lines=highlight_lines,
        panel_letters=layout.panel_letters,
        regression=regression, order=order, labels=labels, year=year, backend=backend, save=save,
        variables_by=variables_by, columns=columns, year_range=year_range, var_window=(var_window or {}),
        cell_ratios=(tuple(cell_ratios) if cell_ratios is not None else None),
        shared_xaxes=layout.share_x, shared_yaxes=layout.share_y,
        figsize_per_col=(layout.figsize_per_panel or (5.0, 3.6)),
        plotly_size_per_col=(layout.plotly_size_per_panel or (360, 300)),
        legend_y=layout.legend_y, legend_ncol=layout.legend_ncol,
        legend_max_per_row=layout.legend_max_per_row, legend_gap=layout.legend_gap,
        legend_row_height=layout.legend_row_height, top_margin=layout.top_margin,
        bottom_margin=layout.bottom_margin, left_margin=layout.left_margin,
    )
    plan = build_grid_plan(ctx)  # Resolve variables, categories, colour map.

    result = (render_multiplot_matplotlib(ctx, plan) if backend == "matplotlib"
              else render_multiplot_plotly(ctx, plan))
    return _emit(result, show)  # Handle inline display (or suppression), then return.


# Frozen default spec instances for multi_jointplot_subfig (safe to share: immutable).

_VALID_ORIENTATIONS = {None, "horizontal", "vertical"}  # Accepted joint orientations.
_VALID_MARGINAL_KINDS = {"kde", "hist"}  # Accepted marginal kinds.
_VALID_LIM_MODES = {"data", "benchmark", "both"}  # Accepted axis-limit modes.


def _validate_joint(data: pd.DataFrame, benchmark: pd.DataFrame | None, encoding: Encoding,
                    orientation, xlim_mode, ylim_mode, marginals: MarginalSpec, highlight,
                    save: SaveSpec | None) -> None:
    """Validate a :func:`multi_jointplot_subfig` request with clear errors.

    Raises
    ------
    ValueError
        On any inconsistent or unsupported option.
    KeyError
        If a referenced column is absent from ``data`` (or ``benchmark``).
    """
    pairs = [tuple(p) for p in encoding.var_pairs]  # Normalise.
    if not pairs:  # At least one pair.
        raise ValueError("encoding.var_pairs must contain at least one (x, y) pair.")
    if any(len(p) != 2 for p in pairs):  # Each must be a 2-tuple.
        raise ValueError("each entry of encoding.var_pairs must be an (x, y) pair.")
    degenerate = [p for p in pairs if p[0] == p[1]]  # X vs X is the diagonal: no information.
    if degenerate:
        raise ValueError(f"var_pairs entries must use two different variables, got {degenerate}.")
    if orientation not in _VALID_ORIENTATIONS:  # Orientation.
        raise ValueError(f"orientation must be one of {_VALID_ORIENTATIONS}, "
                         f"got {orientation!r}.")
    if marginals.kind not in _VALID_MARGINAL_KINDS:  # Marginal kind.
        raise ValueError(f"marginals.kind must be one of {_VALID_MARGINAL_KINDS}, got {marginals.kind!r}.")
    if not 0.0 <= marginals.group_alpha <= 1.0:  # Fill opacity of the per-group layer.
        raise ValueError(f"marginal_group_alpha must be in [0, 1], got {marginals.group_alpha!r}.")
    if highlight is not None:  # {column: value(s)} spotlight selection.
        if not isinstance(highlight, Mapping) or len(highlight) != 1:
            raise ValueError(f"highlight must be a one-item {{column: value(s)}} mapping, got {highlight!r}.")
        h_col = next(iter(highlight))
        if h_col not in data.columns:
            raise KeyError(f"highlight column {h_col!r} is not in data.")
    if (xlim_mode is not None and xlim_mode not in _VALID_LIM_MODES) or \
       (ylim_mode is not None and ylim_mode not in _VALID_LIM_MODES):  # Lim modes (None = default).
        raise ValueError(f"xlim_mode / ylim_mode must be one of {_VALID_LIM_MODES}.")
    if save is not None and (not save.path or not save.name):  # Saving needs path + name.
        raise ValueError("To save, SaveSpec needs both a non-empty path and name.")

    # Every variable in the pairs (+ the optional ``by``) must exist in ``data``.
    referenced = [v for pair in pairs for v in pair]  # Flatten the pairs.
    if encoding.by is not None:
        referenced.append(encoding.by)
    missing = [c for c in dict.fromkeys(referenced) if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}.")

    # The benchmark grouping column must exist when a benchmark + benchmark_by are given.
    # Individual pair variables are NOT required in the benchmark: a panel whose
    # variable is missing is drawn without its benchmark overlay (with a warning).
    if benchmark is not None and encoding.benchmark_by is not None \
            and encoding.benchmark_by not in benchmark.columns:
        raise KeyError(f"benchmark_by column {encoding.benchmark_by!r} not found in benchmark.")


def multi_jointplot_subfig(
    data: pd.DataFrame,
    encoding: Encoding,
    *,
    benchmark: pd.DataFrame | None = None,
    layout: Layout = _DEF_LAYOUT,
    style: Style = _DEF_STYLE,
    regression: Regression = _DEF_REGRESSION,
    save: SaveSpec | None = None,
    orientation=None,
    ncols=None,
    figsize_per_plot=None,
    main_height_ratio=None,
    marginal_height_ratio=None,
    main_width_ratio=None,
    marginal_width_ratio=None,
    xlim_mode=None,
    ylim_mode=None,
    axis_margin=None,
    diag_line=None,
    diag_margin=None,
    symmetric_xy=None,
    diag_quartiles=None,
    highlight: Mapping[str, Any] | None = None,
    marginal_show=None,
    marginal_kind=None,
    marginal_group_alpha=None,
    marginal_show_total=None,
    marginal_total_color=None,
    marginal_total_label=None,
    show: bool = True,
) -> PlotResult:
    """Draw a row / column / grid of jointplot-like panels (scatter + marginals).

    Each ``(x, y)`` pair in ``encoding.var_pairs`` becomes one panel: a scatter of
    ``x`` against ``y`` (coloured by ``by``), with optional marginal distributions
    (top = ``x``, right = ``y``), an optional dashed bold ``x = y`` diagonal, and
    optional LOESS regressions for the data and the benchmark. The benchmark can be
    coloured per category (``benchmark_by`` + a forced colour map, e.g. the IPCC
    temperature classes); otherwise it is a single grey overlay (black-line / grey-
    fill marginals).

    Parameters
    ----------
    data:
        The user's data.
    benchmark:
        The comparison/reference data, or ``None``.
    encoding:
        Column-to-role mapping (var_pairs, by, benchmark_by).
    layout:
        Titles, legend and margins (shared :class:`Layout`).
    orientation:
        ``"horizontal"`` (one row of panels), ``"vertical"`` (one column) or
        ``None`` -> grid controlled by ``ncols``.
    ncols:
        Panel columns of the grid arrangement (with ``orientation=None``).
    figsize_per_plot:
        Per-panel size (inches), marginals included.
    main_height_ratio, marginal_height_ratio, main_width_ratio, marginal_width_ratio:
        Relative sizes of the main scatter vs the top / right marginal axes.
    xlim_mode, ylim_mode:
        Which frames set the axis limits: ``"data"``, ``"benchmark"`` or
        ``"both"``.
    axis_margin:
        Relative padding added around the computed limits.
    diag_line:
        Draw the y = x diagonal.
    diag_margin:
        Relative extension of the diagonal beyond the limits.
    symmetric_xy:
        Give each panel identical x and y limits (the union of the two, after
        ``xlim_mode`` / ``ylim_mode`` / ``axis_margin``), so the y = x diagonal
        splits the plane symmetrically. Strictly per panel -- no cross-panel
        synchronisation.
    diag_quartiles:
        Also draw the y = 0.75x, y = 0.5x and y = 0.25x guide lines (dashed,
        thinner and lighter grey than the y = x diagonal), to read off which
        quartile of the x value the points sit in, with 25 / 50 / 75 tags
        just above each line's upper-right end (and 100 below y = x when
        ``diag_line`` is on). Independent of ``diag_line``; clearest combined
        with ``symmetric_xy=True``.
    highlight:
        Points to spotlight: a one-item ``{column: value}`` or
        ``{column: [values]}`` mapping selecting rows of ``data``. The matching
        points are redrawn *on top* of the data scatter, larger (2.5x
        ``style.scatter_size``), lightly hatched (colour stays visible), with a
        black edge -- same per-``by`` colours as the ordinary points. ``None``
        (default) does nothing. The data-point opacity itself is
        ``style.alpha``.
    style:
        Cosmetics, including the forced per-category colour maps (``palette`` /
        ``benchmark_cmap``).
    marginal_show:
        Whether to draw the top (x) and right (y) marginal distributions.
        Layering (back to front, on both marginals): grey / per-category
        benchmark (``style.benchmark_*``), then the per-``by``
        marginal curves, *unfilled*, line opacity ``marginal_group_alpha``
        (colours from the same ``plan.color_map`` as the scatter and its
        per-group regression), then --
        when ``marginal_show_total=True`` -- the pooled marginal of *all* groups
        as one unfilled full-opacity curve in ``marginal_total_color``.
    marginal_kind:
        ``"kde"`` or ``"hist"``.
    marginal_group_alpha:
        Line opacity of the per-``by`` marginal curves (see layering above).
    marginal_show_total:
        Also draw the pooled all-groups marginal on both marginals.
    marginal_total_color:
        Colour of that pooled curve.
    marginal_total_label:
        Legend label of that pooled curve (default ``"total"``; also passed
        through ``encoding.labels``).
    regression:
        LOESS regressions (``data`` and/or ``benchmark``). Both get a legend
        entry (``LOESS (data)`` / ``LOESS (benchmark)``, renamable via
        ``encoding.labels``).
    order, labels, save:
        Category ordering, display names, and save configuration.
    show:
        When ``True`` (default) the figure is displayed inline; when ``False`` it
        is computed (and saved, if ``save`` is set) without display.

    Returns
    -------
    PlotResult
        ``figure``, ``axes`` (a list of per-panel ``{"ax_main", "ax_top",
        "ax_right"}`` dicts), ``files`` and ``backend`` (always ``"matplotlib"``).

    Raises
    ------
    ValueError, KeyError
        On invalid configuration.

    Notes
    -----
    Matplotlib-only (the marginal layout uses SubFigures). Regression requires
    ``statsmodels`` (imported lazily, only when used).
    """
    from dataclasses import replace as _replace  # Local: apply only provided joint kwargs.

    order = OrderSpec(category_orders=encoding.category_orders, color_order=encoding.color_order,
                      compare_order=encoding.compare_order, sort_categories=encoding.sort_categories)  # Internal helper struct.
    labels = LabelSpec(mapping=encoding.labels or {})

    lay_over = {k: v for k, v in dict(  # Joint-specific geometry kwargs (None = keep default).
        orientation=orientation, ncols=ncols, figsize_per_plot=figsize_per_plot,
        main_height_ratio=main_height_ratio, marginal_height_ratio=marginal_height_ratio,
        main_width_ratio=main_width_ratio, marginal_width_ratio=marginal_width_ratio,
        xlim_mode=xlim_mode, ylim_mode=ylim_mode, axis_margin=axis_margin,
        diag_line=diag_line, diag_margin=diag_margin,
        symmetric_xy=symmetric_xy, diag_quartiles=diag_quartiles,
        highlight=highlight).items() if v is not None}
    if layout.legend_y is not None:  # Generic legend anchor, if set.
        lay_over["legend_y"] = layout.legend_y
    lay_over["legend_ncol"] = layout.legend_ncol  # Generic legend balancing knobs.
    lay_over["legend_max_per_row"] = layout.legend_max_per_row
    lay_over["panel_letters"] = layout.panel_letters  # Generic panel tagging.
    lay_over["x_min"] = layout.x_min  # Hard axis limits (override the computed ranges).
    lay_over["x_max"] = layout.x_max
    lay_over["y_min"] = layout.y_min
    lay_over["y_max"] = layout.y_max
    marg_over = {k: v for k, v in dict(show=marginal_show, kind=marginal_kind,
                                       group_alpha=marginal_group_alpha,
                                       show_total=marginal_show_total,
                                       total_color=marginal_total_color,
                                       total_label=marginal_total_label).items() if v is not None}
    marginals = _replace(MarginalSpec(), **marg_over)

    check_encoding(encoding, "multi_jointplot_subfig")  # Warn on set-but-ignored roles.
    _validate_joint(data, benchmark, encoding, orientation, xlim_mode, ylim_mode, marginals,
                    highlight, save)  # Fail fast.

    ctx = JointContext(  # Unified: Encoding + Style + Regression + flat geometry.
        data=data, benchmark=benchmark, encoding=encoding, style=style, marginals=marginals,
        regression=regression, order=order, labels=labels, save=save, **lay_over,
    )
    plan = build_joint_plan(ctx)  # Resolve pairs, grid, categories, colour maps.
    result = render_multi_jointplot(ctx, plan)  # Draw (matplotlib only).
    return _emit(result, show)  # Handle inline display (or suppression), then return.


def scatter_explain(
    data: pd.DataFrame,
    encoding: Encoding,
    alloc_data: pd.DataFrame,
    alloc_encoding: Encoding,
    *,
    benchmark: pd.DataFrame | None = None,
    alloc_panels: tuple[Any, Any] | None = None,
    hide_repeated_panels: bool = False,
    graph_type: str = "bar",
    scatter_category=None,
    scatter_size: float = 40.0,
    scatter_facecolor: str = "black",
    scatter_edgecolor: str = "white",
    scatter_linewidth: float = 0.5,
    layout: Layout = _DEF_LAYOUT,
    style: Style = _DEF_STYLE,
    regression: Regression = _DEF_REGRESSION,
    save: SaveSpec | None = None,
    figsize: tuple[float, float] = (16.0, 6.5),
    width_ratios: tuple[float, float] = (1.0, 1.6),
    xlim_mode=None,
    ylim_mode=None,
    axis_margin=None,
    diag_line=None,
    diag_margin=None,
    symmetric_xy=None,
    diag_quartiles=None,
    highlight: Mapping[str, Any] | None = None,
    marginal_show=None,
    marginal_kind=None,
    marginal_group_alpha=None,
    marginal_show_total=None,
    marginal_total_color=None,
    marginal_total_label=None,
    show: bool = True,
) -> PlotResult:
    """Composite figure: one jointplot panel (left) + two allocation panels side by side (right).

    The left column is a full :func:`multi_jointplot_subfig` panel for the single
    variable pair in ``encoding.var_pairs`` -- scatter, marginals, diagonal and
    quartile guides, highlight, LOESS regressions (incl. ``per_group``), all its
    options. The right block places two :func:`plot_allocations`-style panels
    side by side, *explaining* the two indicators: left = the x indicator's
    decomposition, right = the y indicator's. The allocation frame is **long**:
    ``alloc_encoding`` is a plain allocations encoding whose ``facet_col`` names
    the indicator column, and the two panels are its two values. Two independent
    legends are laid out over their half of the figure. Matplotlib only.

    Parameters
    ----------
    data:
        Scatter-side long data (the joint panel's frame).
    encoding:
        Joint-side roles (build with :func:`mypackage.plot.encodings.joint`);
        each ``(x, y)`` pair in ``var_pairs`` produces one **row** of the
        figure (joint panel + its two allocation panels).
    alloc_data:
        Allocation-side long frame (structure independent of ``data``): one
        value column (``alloc_encoding.y``) and one indicator column
        (``alloc_encoding.facet_col``) whose values select the two panels.
    alloc_encoding:
        Allocation-side roles -- the exact :func:`plot_allocations` interface,
        built with :func:`mypackage.plot.encodings.allocations`: ``x``, ``y``,
        ``color``, ``compare``, ``x_hierarchy``, and ``facet_col`` = the
        indicator column (required here). ``facet_row`` is ignored.
    benchmark:
        Keyword-only. Scatter-side reference frame, or ``None`` (default).
    alloc_panels:
        Which ``facet_col`` values feed the right-hand panels, left to right
        -- the tuple *is* the panel order. Accepts one ``(x, y)`` tuple
        (used for every row), a list of such tuples (one per ``var_pairs``
        row), or ``None`` (default) to auto-select per row: the pair's own
        variable names when both are values of ``facet_col`` (extra values in
        the column are simply ignored), else the column's two values when it
        has exactly two. A ``None`` *inside* a tuple hides that panel: its
        slot is left empty and the other panels keep their column alignment.
    hide_repeated_panels:
        Draw each indicator's decomposition only once: scanning the rows
        left to right, top to bottom, any panel repeating an indicator
        already drawn is hidden (empty slot). Useful when successive pairs
        share a variable -- e.g. ``[("A", "B"), ("A", "C")]`` draws ``A``
        once. Hidden panels don't consume a ``panel_letters`` letter.
    graph_type:
        Right-block mark: ``"bar"`` or ``"line"``.
    scatter_category:
        Reference overlay on the *right* panels (e.g. a "Total"), exactly as
        in :func:`plot_allocations`: a one-item ``{column: value}`` mapping
        selecting the reference rows of ``alloc_data``, drawn as markers on
        top of the bars / lines of each panel. ``None`` -> no overlay. (The
        left panel's point cloud is unrelated -- see ``highlight`` for it.)
    scatter_size, scatter_facecolor, scatter_edgecolor, scatter_linewidth:
        Marker styling of that reference overlay (area, face / edge colours,
        edge width).
    layout:
        Shared :class:`Layout` (title, legend balancing, ``share_y`` across the
        two right panels, ``panel_letters``: row-major a, b, c on the first
        row, d, e, f on the second, ...).
    style:
        Shared :class:`Style` (both columns).
    regression:
        LOESS configuration for the joint panel, including ``per_group=True``
        (one curve per ``by`` category, same ``color_map`` as the scatter).
    save:
        Save configuration for the composite figure, or ``None``.
    figsize:
        Figure size in inches as ``(width, height per row)``; the total height
        scales with the number of ``var_pairs`` rows.
    width_ratios:
        Left column vs right block width ratio (the right block splits evenly).
    xlim_mode, ylim_mode:
        Joint axis-limit sources (``"data"`` / ``"benchmark"`` / ``"both"``).
    axis_margin:
        Joint relative axis padding.
    diag_line:
        Draw the y = x diagonal on the joint panel.
    diag_margin:
        Relative extension of that diagonal.
    symmetric_xy:
        Same x and y limits on the joint panel (union), so y = x is the
        symmetry axis.
    diag_quartiles:
        Also draw the y = 0.75x / 0.5x / 0.25x guides with 25/50/75/100 tags.
    highlight:
        ``{column: value(s)}`` -- spotlighted points on the joint scatter
        (bigger, hatched, black edge; ``Highlight`` legend entry).
    marginal_show, marginal_kind, marginal_group_alpha:
        Joint marginals: toggle, ``"kde"``/``"hist"``, per-group line opacity.
    marginal_show_total, marginal_total_color, marginal_total_label:
        Pooled all-groups marginal curve: toggle, colour, legend label.
    show:
        Display inline when ``True`` (default); always returns the result.

    Returns
    -------
    PlotResult
        ``figure``, ``axes`` (a **list** of per-row dicts
        ``{"joint": {...}, "alloc_x": ax, "alloc_y": ax}``, one per
        ``var_pairs`` pair), ``files`` and ``backend`` (always
        ``"matplotlib"``).

    Raises
    ------
    ValueError, KeyError
        On invalid configuration (pair count, missing facet column / values, ...).
    """
    from dataclasses import replace as _replace  # Local: apply only provided kwargs.

    # ---- joint side: identical construction to multi_jointplot_subfig ----
    if not encoding.var_pairs:
        raise ValueError("scatter_explain needs at least one var pair in encoding.var_pairs.")
    order = OrderSpec(category_orders=encoding.category_orders, color_order=encoding.color_order,
                      compare_order=encoding.compare_order, sort_categories=encoding.sort_categories)
    labels = LabelSpec(mapping=encoding.labels or {})
    lay_over = {k: v for k, v in dict(
        xlim_mode=xlim_mode, ylim_mode=ylim_mode, axis_margin=axis_margin,
        diag_line=diag_line, diag_margin=diag_margin,
        symmetric_xy=symmetric_xy, diag_quartiles=diag_quartiles,
        highlight=highlight).items() if v is not None}
    lay_over["legend_ncol"] = layout.legend_ncol
    lay_over["legend_max_per_row"] = layout.legend_max_per_row
    lay_over["panel_letters"] = False  # Lettering is handled by the composite renderer.
    lay_over["x_min"] = layout.x_min  # Hard axis limits (override the computed ranges).
    lay_over["x_max"] = layout.x_max
    lay_over["y_min"] = layout.y_min
    lay_over["y_max"] = layout.y_max
    marg_over = {k: v for k, v in dict(show=marginal_show, kind=marginal_kind,
                                       group_alpha=marginal_group_alpha,
                                       show_total=marginal_show_total,
                                       total_color=marginal_total_color,
                                       total_label=marginal_total_label).items() if v is not None}
    marginals = _replace(MarginalSpec(), **marg_over)
    check_encoding(encoding, "multi_jointplot_subfig")  # The joint's relevance set applies.
    _validate_joint(data, benchmark, encoding, None, lay_over.get("xlim_mode", "both"),
                    lay_over.get("ylim_mode", "both"), marginals, highlight, save)
    jctx = JointContext(
        data=data, benchmark=benchmark, encoding=encoding, style=style, marginals=marginals,
        regression=regression, order=order, labels=labels, save=None, **lay_over,
    )
    jplan = build_joint_plan(jctx)

    # ---- allocation side: the plot_allocations interface, one panel per facet value ----
    facet_col = alloc_encoding.facet_col
    if facet_col is None:
        raise ValueError("alloc_encoding.facet_col must name the indicator column "
                         "(its two values select the right-hand panels).")
    if facet_col not in alloc_data.columns:
        raise KeyError(f"alloc_encoding.facet_col {facet_col!r} is not in alloc_data.")
    if alloc_encoding.y is None:
        raise ValueError("alloc_encoding.y must name the allocation value column.")
    facet_vals = list(dict.fromkeys(alloc_data[facet_col].dropna().tolist()))
    n_rows = len(encoding.var_pairs)
    if alloc_panels is None:  # One (x, y) selection per row.
        rows_panels = [None] * n_rows
    elif isinstance(alloc_panels[0], (tuple, list)):  # Explicit list, one tuple per row.
        rows_panels = [tuple(p) for p in alloc_panels]
        if len(rows_panels) != n_rows:
            raise ValueError(f"alloc_panels has {len(rows_panels)} entries for "
                             f"{n_rows} var pair(s).")
    else:  # A single tuple: same two panels on every row.
        rows_panels = [tuple(alloc_panels)] * n_rows
    resolved: list[tuple[Any, Any]] = []
    for (xvar_j, yvar_j), given in zip(encoding.var_pairs, rows_panels):
        if given is None:  # Auto: prefer the pair's own names, else the only two values.
            if xvar_j in facet_vals and yvar_j in facet_vals:
                given = (xvar_j, yvar_j)
            elif len(facet_vals) == 2:
                given = tuple(facet_vals)
            else:
                raise ValueError(
                    f"alloc_panels is required for pair ({xvar_j!r}, {yvar_j!r}): "
                    f"{facet_col!r} has {len(facet_vals)} distinct values and neither "
                    f"variable name is one of them.")
        missing = [v for v in given if v is not None and v not in facet_vals]
        if len(given) != 2 or missing:
            raise ValueError(f"each alloc_panels entry must be two values of {facet_col!r} "
                             f"(or None to hide that panel); missing from alloc_data: {missing}.")
        resolved.append(tuple(given))
    if hide_repeated_panels:  # Draw each indicator once, row-major; later repeats are hidden.
        seen: set = set()
        deduped = []
        for pair in resolved:
            keep = []
            for v in pair:
                keep.append(None if (v is None or v in seen) else v)
                seen.add(v)
            deduped.append(tuple(keep))
        resolved = deduped
    a_order = OrderSpec(category_orders=alloc_encoding.category_orders,
                        color_order=alloc_encoding.color_order,
                        compare_order=alloc_encoding.compare_order,
                        sort_categories=alloc_encoding.sort_categories)
    a_labels = LabelSpec(mapping=alloc_encoding.labels or {})
    a_enc = _replace(alloc_encoding, facet_col=None, facet_row=None)  # Panels ARE the layout.
    a_scatter = (ScatterSpec(category=scatter_category, size=scatter_size,
                             facecolor=scatter_facecolor, edgecolor=scatter_edgecolor,
                             linewidth=scatter_linewidth)
                 if scatter_category is not None else None)
    rows_ctx: list[list] = []  # Per row: two slots, each (ctx, plan, title) or None (hidden).
    for panels in resolved:
        row = []
        for pval in panels:
            if pval is None:  # Hidden slot: nothing to build.
                row.append(None)
                continue
            pdf = alloc_data.loc[alloc_data[facet_col] == pval]
            _validate(pdf, a_enc, graph_type, "matplotlib", FacetSpec(), style, a_scatter, None)
            pctx = PlotContext(data=pdf, encoding=a_enc, graph_type=graph_type,
                               backend="matplotlib", facets=FacetSpec(), order=a_order,
                               style=style, layout=layout, labels=a_labels,
                               scatter=a_scatter, save=None)
            row.append((pctx, build_render_plan(pctx), str(a_labels.lookup(pval, facet_col))))
        rows_ctx.append(row)
    if not any(slot is not None for row in rows_ctx for slot in row):
        raise ValueError("every allocation panel is hidden; nothing to explain.")

    result = render_scatter_explain(jctx, jplan, rows_ctx, layout, style,
                                    figsize, width_ratios, save)
    return _emit(result, show)


# Frozen default spec instances for grouped_boxplot (safe to share: immutable).

_VALID_BOX_KINDS = {"box", "violin", "strip"}  # Accepted mark types.


def _validate_box(data: pd.DataFrame, benchmark: pd.DataFrame | None, encoding: Encoding,
                  kind: str, backend: str, show_total: bool, save: SaveSpec | None) -> None:
    """Validate a :func:`grouped_boxplot` request with clear, actionable errors.

    Raises
    ------
    ValueError
        On any inconsistent or unsupported option.
    KeyError
        If a referenced column is absent from ``data`` (or ``benchmark``).
    """
    if backend not in _VALID_BACKENDS:  # Backend.
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}.")
    if kind not in _VALID_BOX_KINDS:  # Mark type.
        raise ValueError(f"kind must be one of {_VALID_BOX_KINDS}, got {kind!r}.")
    if show_total and encoding.by is None:  # Pool of a single group = a duplicate mark.
        warnings.warn("show_total is ignored when encoding.by is None: the pooled mark would "
                      "duplicate the single data mark in the next slot.",
                      UserWarning, stacklevel=3)
    if save is not None and (not save.path or not save.name):  # Saving needs path + name.
        raise ValueError("To save, SaveSpec needs both a non-empty path and name.")

    # Required + optional data columns must exist.
    referenced = [encoding.x, encoding.y]  # Always required.
    for col in (encoding.by, encoding.x_hierarchy,  # Optional data columns.
                encoding.facet_col, encoding.facet_row):
        if col is not None:
            referenced.append(col)
    missing = [c for c in dict.fromkeys(referenced) if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}.")

    # The benchmark must carry x + y (and benchmark_by when grouping it).
    if benchmark is not None:
        bench_ref = [encoding.x, encoding.y]
        if encoding.benchmark_by is not None:
            bench_ref.append(encoding.benchmark_by)
        for col in (encoding.facet_col, encoding.facet_row):  # Facets slice the benchmark too.
            if col is not None:
                bench_ref.append(col)
        bench_missing = [c for c in dict.fromkeys(bench_ref) if c not in benchmark.columns]
        if bench_missing:
            raise KeyError(f"Columns not found in benchmark: {bench_missing}.")


def grouped_boxplot(
    data: pd.DataFrame,
    encoding: Encoding,
    *,
    benchmark: pd.DataFrame | None = None,
    kind: str = "box",
    backend: str = "matplotlib",
    style: Style = _DEF_STYLE,
    layout: Layout = _DEF_LAYOUT,
    save: SaveSpec | None = None,
    show_total: bool = False,
    total_color: str = "black",
    total_label: str = "total",
    show: bool = True,
) -> PlotResult:
    """Draw a categorical box / violin / strip comparison of ``y`` by ``x``.

    For each category of ``encoding.x`` the distribution of ``encoding.y`` is
    drawn as one mark per ``encoding.by`` group, side by side. When a
    ``benchmark`` frame is given, its groups (optionally split by
    ``encoding.benchmark_by``) are drawn beside the data within the same ``x``
    slot, in grey or a forced ``style.benchmark_cmap``. An optional
    ``encoding.x_hierarchy`` adds bracket annotations beneath the axis grouping
    the ``x`` categories (Matplotlib backend only, exactly as in
    :func:`plot_allocations`).

    Parameters
    ----------
    data:
        The user's long-format data.
    benchmark:
        Keyword-only. The comparison/reference data, or ``None`` (default) for
        no benchmark.
    encoding:
        Column-to-role mapping (``x``, ``y``, optional ``by`` / ``x_hierarchy`` /
        ``benchmark_by``).
    kind:
        Mark type: ``"box"`` (default), ``"violin"`` or ``"strip"`` (points only).
        For ``"box"`` / ``"violin"``, ``style.show_points`` overlays the points.
    backend:
        ``"matplotlib"`` (vector publication output) or ``"plotly"`` (interactive).
        ``encoding.x_hierarchy`` is honoured only by the Matplotlib backend.
    style:
        Cosmetics, including the forced palettes (``palette`` / ``benchmark_cmap``),
        box geometry, point styling and the bracket offsets.
    layout:
        Shared :class:`Layout` -- titles, hard y-limits, legend anchor.
    show_total:
        Draw one extra *pooled* mark per ``x`` slot (all ``encoding.by`` groups
        together), unfilled, lines in ``total_color``, placed between the data
        groups and the benchmark -- for comparison against the benchmark, the
        same role as ``dist_show_total`` in :func:`multiplot_grid`. Requires
        ``encoding.by`` (ignored with a warning otherwise: the pool would
        duplicate the single data mark). The per-group fill opacity is
        ``style.data_alpha`` (the box analogue of the grid's
        ``dist_group_alpha``); group colours come from the same
        ``plan.color_map``. The legend entry is "total", renamable via
        ``encoding.labels`` (e.g. ``labels={"total": "All groups"}``).
    total_color:
        Colour of the pooled mark.
    total_label:
        Legend label of the pooled mark (default ``"total"``; also passed
        through ``encoding.labels``).

        Setting ``encoding.facet_col`` / ``facet_row`` repeats the whole slot
        layout in a facet grid (shared category order and slot positions, so
        panels stay comparable); ``Layout.share_y``, ``hide_empty``,
        ``col_wrap`` / ``row_wrap``, ``wspace`` / ``hspace`` and
        ``panel_letters`` then apply as they do for ``plot_allocations``.
    order, labels, save:
        Category ordering, display names, and save configuration.
    show:
        When ``True`` (default) the figure is displayed inline in a notebook;
        when ``False`` it is computed (and saved, if ``save`` is set) without any
        display. The :class:`PlotResult` is returned either way.

    Returns
    -------
    PlotResult
        ``figure``, ``axes`` (Matplotlib only), ``files`` and ``backend``.

    Raises
    ------
    ValueError, KeyError
        On invalid configuration.
    """
    kind = kind.lower().strip()  # Normalise.
    backend = backend.lower().strip()  # Normalise.

    order = OrderSpec(category_orders=encoding.category_orders, color_order=encoding.color_order,
                      compare_order=encoding.compare_order, sort_categories=encoding.sort_categories)
    if not isinstance(encoding.y, str) and encoding.y is not None:  # Wide input: one panel per column.
        ycols = list(encoding.y)
        if len(ycols) < 1:
            raise ValueError("encoding.y must name at least one column.")
        missing = [c for c in ycols if c not in data.columns]
        if missing:
            raise KeyError(f"Columns not found in data: {missing}.")
        if encoding.facet_col is None:  # Put the indicator on the free facet dimension.
            facet_role = "facet_col"
        elif encoding.facet_row is None:
            facet_role = "facet_row"
        else:
            raise ValueError("a sequence of y columns needs a free facet dimension, but both "
                             "facet_col and facet_row are set.")
        taken = set(data.columns) | (set(benchmark.columns) if benchmark is not None else set())
        var_name = next(n for n in ("indicator", "_indicator", "__indicator")  # Collision-free.
                        if n not in taken)
        value_name = next(n for n in ("value", "_value", "__value") if n not in taken)
        id_vars = [c for c in dict.fromkeys(
            (encoding.x, encoding.by, encoding.x_hierarchy, encoding.benchmark_by,
             encoding.facet_col, encoding.facet_row)) if c is not None]

        def _melt(df):  # Wide -> long, keeping the roles that address the marks.
            keep = [c for c in id_vars if c in df.columns]
            return df.melt(id_vars=keep, value_vars=[c for c in ycols if c in df.columns],
                           var_name=var_name, value_name=value_name)

        data = _melt(data)
        if benchmark is not None:
            benchmark = _melt(benchmark)
        from dataclasses import replace as _replace  # Local: rebuild the encoding.
        new_labels = {value_name: "", **dict(encoding.labels or {})}  # Panels name the indicator,
        encoding = _replace(encoding, y=value_name, labels=new_labels,  # so the y title stays empty
                            **{facet_role: var_name})  # (Layout.y_title still wins).
    labels = LabelSpec(mapping=encoding.labels or {})

    check_encoding(encoding, "grouped_boxplot")  # Warn on set-but-ignored roles.
    _validate_box(data, benchmark, encoding, kind, backend, show_total, save)  # Fail fast.

    ctx = BoxContext(  # Unified: Encoding + Style + Layout (+ flat box-specific fields).
        data=data, benchmark=benchmark, encoding=encoding, kind=kind, backend=backend,
        style=style, layout=layout, order=order, labels=labels, save=save,
        figsize=(layout.figsize_per_panel or (8.0, 6.0)),
        plotly_size=(layout.plotly_size_per_panel or (900, 650)),
        show_total=(show_total and encoding.by is not None), total_color=total_color,
        total_label=total_label,
        facets=FacetSpec(col=encoding.facet_col, row=encoding.facet_row,
                         col_wrap=layout.col_wrap if encoding.facet_row is None else None,
                         row_wrap=layout.row_wrap if encoding.facet_col is None else None,
                         share_y=layout.share_y, hide_empty=layout.hide_empty),
    )
    plan = build_box_plan(ctx)  # Resolve categories, colour maps, slot geometry.

    result = (render_box_matplotlib(ctx, plan) if backend == "matplotlib"
              else render_box_plotly(ctx, plan))
    return _emit(result, show)  # Handle inline display (or suppression), then return.


# Frozen default spec instances for fingerprint_diagram (safe to share: immutable).


def _validate_fingerprint(data: pd.DataFrame, benchmark: pd.DataFrame | None,
                          encoding: Encoding, save: SaveSpec | None) -> None:
    """Validate a :func:`fingerprint_diagram` request with clear errors.

    Raises
    ------
    ValueError
        On an empty / inconsistent structure.
    KeyError
        If a referenced indicator column is absent from ``data`` (or ``benchmark``).
    """
    if not encoding.categories:  # At least one category.
        raise ValueError("encoding.categories must contain at least one category.")
    if any(len(cols) == 0 for cols in encoding.categories.values()):  # Each non-empty.
        raise ValueError("Each category in encoding.categories must list at least one indicator.")
    if save is not None and (not save.path or not save.name):  # Saving needs path + name.
        raise ValueError("To save, SaveSpec needs both a non-empty path and name.")

    if encoding.category_order is not None:  # Order override must be a permutation of the keys.
        if set(encoding.category_order) != set(encoding.categories):
            raise ValueError("encoding.category_order must be a permutation of "
                             "encoding.categories' keys.")

    indicators = [col for cols in encoding.categories.values() for col in cols]  # All spokes.
    dup = [c for c in set(indicators) if indicators.count(c) > 1]  # Indicators are unique spokes.
    if dup:
        raise ValueError(f"Indicators must not repeat across categories; duplicates: {dup}.")

    missing = [c for c in dict.fromkeys(indicators) if c not in data.columns]  # Data columns.
    if missing:
        raise KeyError(f"Indicator columns not found in data: {missing}.")
    if benchmark is not None:  # Benchmark must carry the same indicators.
        bmissing = [c for c in dict.fromkeys(indicators) if c not in benchmark.columns]
        if bmissing:
            raise KeyError(f"Indicator columns not found in benchmark: {bmissing}.")


def fingerprint_diagram(
    data: pd.DataFrame,
    encoding: Encoding,
    *,
    benchmark: pd.DataFrame | None = None,
    style: FingerprintStyle = _DEF_FP_STYLE,
    layout: Layout = _DEF_LAYOUT,
    save: SaveSpec | None = None,
    title=None,
    footnotes=None,
    quantile_guide=None,
    quantile_labels=None,
    legend=None,
    legend_loc=None,
    legend_fontsize=None,
    xlim=None,
    ylim=None,
    iqr_mult=None,
    band_quantiles=None,
    na_sentinels=None,
    mark=None,
    band_alpha=None,
    band_lw=None,
    tick_lw=None,
    tick_scale=None,
    show: bool = True,
) -> PlotResult:
    """Draw a radial "model fingerprint" dashboard (Matplotlib only).

    Categories occupy equal angular sectors; each category's indicators are
    radial spokes spread evenly inside its sector, so the layout adapts to any
    number of categories and any per-category indicator count. Concentric rings
    mark the reference quantiles (q25 / median / q75) and the outer border; the
    short code of each indicator is drawn at its spoke tip and a legend maps
    codes to display names, grouped and coloured by category.

    This renders the fully-labelled diagram. When a ``benchmark`` is given, each
    indicator's radial scale is set by the benchmark's Tukey fence
    (``Q25 - k*IQR`` -> centre, ``Q75 + k*IQR`` -> outer ring) and the focal
    ``data`` distribution is drawn on each spoke as a min..max band with quantile
    cross-ticks. With ``benchmark=None`` only the labelled scaffold is drawn.

    Parameters
    ----------
    data:
        Focal frame, indicators as columns (drawn as a band per spoke).
    benchmark:
        Keyword-only. Reference ensemble, same indicator columns; sets the
        radial scale. ``None`` (default) draws the scaffold only.
    encoding:
        Structure: ``categories`` (``{category: [indicator columns]}``, ordered),
        the ``names`` legend mapping and optional ``codes`` / ``category_order``.
    style:
        Radial geometry + cosmetics (:class:`FingerprintStyle`).
    layout:
        Shared :class:`Layout`; the fingerprint reads only ``panel_letters``
        (the single diagram is tagged ``a``).
    title:
        Figure title.
    footnotes:
        Free-text footnote lines printed under the diagram.
    quantile_guide:
        Draw the ring-legend guide explaining the radial scale.
    quantile_labels:
        The five guide labels, from the fence rings inward (LaTeX allowed).
    legend:
        Draw the indicator legend.
    legend_loc:
        Legend placement (e.g. ``"right"``).
    legend_fontsize:
        Legend font size.
    xlim, ylim:
        Axes window around the unit-radius diagram (data coordinates).
    iqr_mult:
        Tukey-fence multiplier: the outer ring sits at Q25/Q75 -/+
        ``iqr_mult`` x IQR of the benchmark.
    band_quantiles:
        Focal-frame quantiles drawn on each spoke (``mark="band"``).
    na_sentinels:
        Treat IIASA-style sentinel values (0 and >= 1e20) as missing when
        cleaning the focal and benchmark series.
    mark:
        Focal mark type: ``"band"`` (quantile band) or ``"tick"``.
    band_alpha:
        Focal band opacity.
    band_lw:
        Focal band edge width.
    tick_lw:
        Tick mark width (``mark="tick"``).
    tick_scale:
        Tick mark length, as a share of the spoke.
    save:
        Save configuration, or ``None``.
    show:
        Display inline when ``True`` (default); always returns the result.

    Returns
    -------
    PlotResult
        ``figure``, ``axes`` and any written ``files`` (``backend="matplotlib"``).

    Raises
    ------
    ValueError, KeyError
        On invalid configuration.
    """
    from dataclasses import replace as _replace  # Apply only provided fingerprint kwargs.

    check_encoding(encoding, "fingerprint_diagram")  # Warn on set-but-ignored roles.
    _validate_fingerprint(data, benchmark, encoding, save)  # Fail fast.

    over = {k: v for k, v in dict(  # Only the fingerprint kwargs the caller set (None = default).
        title=title, footnotes=footnotes, quantile_guide=quantile_guide,
        quantile_labels=quantile_labels, legend=legend, legend_loc=legend_loc,
        legend_fontsize=legend_fontsize, xlim=xlim, ylim=ylim,
        iqr_mult=iqr_mult, band_quantiles=band_quantiles, na_sentinels=na_sentinels, mark=mark,
        band_alpha=band_alpha, band_lw=band_lw, tick_lw=tick_lw, tick_scale=tick_scale).items()
        if v is not None}

    ctx = FingerprintContext(  # Unified: Encoding + FingerprintStyle + flat kwargs.
        panel_letters=layout.panel_letters,
        data=data, benchmark=benchmark, encoding=encoding, style=style, save=save, **over,
    )
    plan = build_fingerprint_plan(ctx)  # Order, colours, codes, angular geometry.
    result = render_fingerprint_matplotlib(ctx, plan)  # Matplotlib only.
    return _emit(result, show)  # Inline display (or suppression), then return.
