"""Per-function :class:`Encoding` builders + a relevance guard.

The unified :class:`~mypackage.plot.specs.Encoding` is a superset that serves all
five chart functions, so a bare ``Encoding(...)`` gives the caller no signal about
which fields matter for a given chart -- and silently ignores the rest. These thin
builders solve that at the call site: each exposes *only* the roles relevant to one
function, with a homogenized name for the colour role (always ``color``), and a
docstring describing what every argument does *in that chart*. They construct the
same single ``Encoding`` under the hood -- there is still exactly one class to
maintain, and the renderers are untouched.

``check_encoding`` is the safety net for callers who build ``Encoding`` directly:
it warns when a set field is irrelevant to the target function (the mistake the
builders make structurally impossible).

Naming note -- the colour role is called ``color`` everywhere here, even though it
maps to ``Encoding.color`` for allocations (stacking colour) and to ``Encoding.by``
for grid / joint / box (colour grouping): the visible result is the same (a colour
code), so the surface name is the same.
"""

from __future__ import annotations

import dataclasses as _dc
import warnings
from typing import Any, Mapping, Sequence

from .specs import Encoding

__all__ = [
    "allocations", "grid", "joint", "box", "fingerprint", "check_encoding",
]


# ---------------------------------------------------------------------------
# Relevance sets: the Encoding fields each function actually reads. Single
# source of truth for both the guard below and the builders' surface.
# ---------------------------------------------------------------------------
_ORDERING = {"category_orders", "color_order", "compare_order", "sort_categories", "labels"}

RELEVANT_FIELDS: Mapping[str, frozenset] = {
    "plot_allocations": frozenset(
        {"x", "y", "color", "compare_by", "x_hierarchy", "facet_col", "facet_row"} | _ORDERING),
    "plot_allocations_bubble": frozenset(
        {"x", "y", "color", "size", "background", "facet_col", "facet_row"} | _ORDERING),
    "multiplot_grid": frozenset(
        {"variables", "line_by", "by", "line_id", "scatter_by", "benchmark_line_id",
         "aggregate"} | _ORDERING),
    "multi_jointplot_subfig": frozenset(
        {"var_pairs", "by", "benchmark_by"} | _ORDERING),
    "grouped_boxplot": frozenset(
        {"x", "y", "by", "x_hierarchy", "benchmark_by", "facet_col", "facet_row"} | _ORDERING),
    "fingerprint_diagram": frozenset(
        {"categories", "names", "codes", "category_order", "explanations", "footnotes",
         "labels"}),
    "scatter_explain": frozenset(  # Joint side: same roles as the jointplot.
        {"var_pairs", "by", "benchmark_by"} | _ORDERING),
}

# Map homogenized builder names -> the underlying Encoding field, per function,
# so the guard can report the *builder* name a caller should have used.
_ALIAS_BACK: Mapping[str, Mapping[str, str]] = {
    "multiplot_grid": {"by": "color"},
    "multi_jointplot_subfig": {"by": "color"},
    "grouped_boxplot": {"by": "color"},
}

_DEFAULTS = {f.name: f for f in _dc.fields(Encoding)}


def _is_set(enc: Encoding, name: str) -> bool:
    """True if field ``name`` on ``enc`` differs from its dataclass default."""
    field = _DEFAULTS[name]
    current = getattr(enc, name)
    if field.default is not _dc.MISSING:
        return current != field.default
    if field.default_factory is not _dc.MISSING:  # e.g. dict-valued fields
        return bool(current)
    return current is not None


def check_encoding(encoding: Encoding, func_name: str) -> None:
    """Warn if ``encoding`` carries fields ``func_name`` will ignore.

    Called at the top of every entry point. Silent when the encoding is clean
    (e.g. always the case for encodings built via this module's builders).

    Parameters
    ----------
    encoding:
        The encoding handed to the function.
    func_name:
        The public function name, e.g. ``"multiplot_grid"``.
    """
    relevant = RELEVANT_FIELDS.get(func_name)
    if relevant is None:  # Unknown function -> nothing to check against.
        return
    stray = sorted(n for n in _DEFAULTS if n not in relevant and _is_set(encoding, n))
    if not stray:
        return
    pretty = ", ".join(f"Encoding.{n}" for n in stray)
    warnings.warn(
        f"{func_name} ignores these Encoding fields: {pretty}. "
        f"They belong to a different chart; use plot.encodings.{_short(func_name)}(...) "
        f"to see only the relevant roles.",
        stacklevel=3,
    )


def _short(func_name: str) -> str:
    """Public function name -> builder name in this module."""
    return {
        "plot_allocations": "allocations", "plot_allocations_bubble": "heatmap",
        "multiplot_grid": "grid",
        "multi_jointplot_subfig": "joint", "grouped_boxplot": "box",
        "fingerprint_diagram": "fingerprint",
    }.get(func_name, func_name)


# ===========================================================================
# Builders. Keyword-only, so there is never a positional-order surprise.
# ===========================================================================
def allocations(
    *,
    x: str,
    y: str,
    color: str | None = None,
    compare: str | None = None,
    x_hierarchy: str | None = None,
    facet_col: str | None = None,
    facet_row: str | None = None,
    labels: Mapping[Any, Any] | None = None,
    category_orders: Mapping[str, Sequence[Any]] | None = None,
    color_order: Sequence[Any] | None = None,
    compare_order: Sequence[Any] | None = None,
    sort_categories: bool = False,
) -> Encoding:
    """Encoding for :func:`plot_allocations` (faceted bar / line breakdowns).

    Parameters
    ----------
    x:
        Column on the horizontal axis (e.g. the year). One group of bars, or one
        line's x, per distinct value.
    y:
        Value column: bar height (stacked) or line value.
    color:
        Column whose values become the stacked colour segments within each ``x``
        (bars) or the coloured lines. ``None`` -> a single series, one colour.
    compare:
        Second categorical split drawn side by side (bars) or as line styles,
        *within* each ``color`` -- e.g. two scenarios compared at each ``x``.
    x_hierarchy:
        Coarser grouping of ``x`` values, drawn as labelled brackets under the axis.
    facet_col, facet_row:
        Columns to facet on: one subplot per value across columns / rows.
    labels:
        Display names. Flat ``{value: label}`` and/or per-column
        ``{column: {value: label}}``; applied to axis ticks, legend and titles.
    category_orders:
        ``{column: [ordered values]}`` to fix the order of any categorical axis / split.
    color_order:
        Explicit order (or subset) of the ``color`` categories.
    compare_order:
        Explicit order of the ``compare`` categories.
    sort_categories:
        Alphabetically sort any categories not pinned by an explicit order.
    """
    return Encoding(
        x=x, y=y, color=color, compare_by=compare, x_hierarchy=x_hierarchy,
        facet_col=facet_col, facet_row=facet_row, labels=dict(labels or {}),
        category_orders=dict(category_orders or {}), color_order=color_order,
        compare_order=compare_order, sort_categories=sort_categories,
    )


def grid(
    *,
    variables: Sequence[str],
    line_by: str,
    color: str | None = None,
    line_id: str | None = None,
    scatter_by: str | None = None,
    benchmark_line_id: str | Sequence[str] | None = None,
    aggregate: str | None = "mean",
    labels: Mapping[Any, Any] | None = None,
    category_orders: Mapping[str, Sequence[Any]] | None = None,
    color_order: Sequence[Any] | None = None,
    sort_categories: bool = False,
) -> Encoding:
    """Encoding for :func:`multiplot_grid` (per-variable series / distribution / scatter grid).

    Each variable gets a row (or column); the three cells are a time series, a
    distribution, and a scatter.

    Parameters
    ----------
    variables:
        The variable columns, one grid line each. Their display names come from
        ``labels`` (``{column: name}``).
    line_by:
        Column on the series x-axis (e.g. the period/year). Also the column the
        frame is *sliced* on to build the distribution and scatter cells at the
        cross-section ``year=`` argument of the function.
    color:
        Column whose values colour the series / distributions / scatter groups
        (maps to ``Encoding.by``). ``None`` -> one ungrouped series.
    line_id:
        Per-trajectory id in the data: one line is drawn per ``line_id`` value
        (use with ``aggregate=None`` to keep individual trajectories rather than
        collapsing them).
    scatter_by:
        Column on the scatter cell's x-axis. Required if the ``"scatter"`` cell is shown.
    benchmark_line_id:
        Trajectory id column(s) in the benchmark frame (single name or list).
    aggregate:
        How to reduce multiple rows sharing a ``(color, line_by)`` key:
        ``"mean"`` / ``"median"`` / ``None`` (no reduction -> raw trajectories).
    labels:
        Display names, notably ``{variable_column: plain-English name}`` used for
        cell titles and axes.
    category_orders, color_order, sort_categories:
        Ordering controls for the ``color`` groups (see :func:`allocations`).
    """
    return Encoding(
        variables=list(variables), line_by=line_by, by=color, line_id=line_id,
        scatter_by=scatter_by, benchmark_line_id=benchmark_line_id, aggregate=aggregate,
        labels=dict(labels or {}), category_orders=dict(category_orders or {}),
        color_order=color_order, sort_categories=sort_categories,
    )


def joint(
    *,
    var_pairs: Sequence[tuple[str, str]],
    color: str | None = None,
    benchmark_by: str | None = None,
    labels: Mapping[Any, Any] | None = None,
    category_orders: Mapping[str, Sequence[Any]] | None = None,
    color_order: Sequence[Any] | None = None,
    sort_categories: bool = False,
) -> Encoding:
    """Encoding for :func:`multi_jointplot_subfig` (one jointplot per variable pair).

    Parameters
    ----------
    var_pairs:
        List of ``(x, y)`` column pairs; one jointplot (scatter + marginals) each.
    color:
        Column whose values colour the points and the marginal distributions
        (maps to ``Encoding.by``). ``None`` -> a single colour.
    benchmark_by:
        Colour-grouping column for the benchmark overlay, when a benchmark is given.
    labels:
        Display names for axes / legend.
    category_orders, color_order, sort_categories:
        Ordering controls for the ``color`` groups (see :func:`allocations`).
    """
    return Encoding(
        var_pairs=list(var_pairs), by=color, benchmark_by=benchmark_by,
        labels=dict(labels or {}), category_orders=dict(category_orders or {}),
        color_order=color_order, sort_categories=sort_categories,
    )


def box(
    *,
    x: str,
    y: str | Sequence[str],
    color: str | None = None,
    x_hierarchy: str | None = None,
    benchmark_by: str | None = None,
    facet_col: str | None = None,
    facet_row: str | None = None,
    labels: Mapping[Any, Any] | None = None,
    category_orders: Mapping[str, Sequence[Any]] | None = None,
    color_order: Sequence[Any] | None = None,
    sort_categories: bool = False,
) -> Encoding:
    """Encoding for :func:`grouped_boxplot` (grouped box / violin / strip).

    Parameters
    ----------
    x:
        Categorical column on the x-axis: one box group per value.
    y:
        Value column whose distribution each box / violin summarises. A
        *sequence* of column names instead draws one facet panel per column
        (wide input, melted internally), each panel titled with the column
        name -- renamable through ``labels``. Indicators usually carry
        different units, so pair it with ``Layout(share_y=False)``; the
        shared y title is dropped (each panel names its indicator) unless
        ``Layout.y_title`` is set.
    color:
        Column that splits each ``x`` group into coloured sub-boxes
        (maps to ``Encoding.by``). ``None`` -> one box per ``x``.
    x_hierarchy:
        Coarser grouping of the ``x`` values, drawn as labelled brackets under
        the axis (matplotlib only).
    benchmark_by:
        Colour-grouping column for the benchmark overlay, when a benchmark is given.
    facet_col:
        Column whose values create one panel per value, laid out in columns.
    facet_row:
        Column whose values create one panel per value, laid out in rows.
    labels:
        Display names for axis ticks / legend.
    category_orders, color_order, sort_categories:
        Ordering controls (see :func:`allocations`).
    """
    return Encoding(
        x=x, y=y, by=color, x_hierarchy=x_hierarchy, benchmark_by=benchmark_by,
        facet_col=facet_col, facet_row=facet_row,
        labels=dict(labels or {}),
        category_orders=dict(category_orders or {}), color_order=color_order,
        sort_categories=sort_categories,
    )


def heatmap(
    *,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    background: str | None = None,
    facet_col: str | None = None,
    facet_row: str | None = None,
    labels: Mapping[Any, Any] | None = None,
    category_orders: Mapping[str, Sequence[Any]] | None = None,
    sort_categories: bool = False,
) -> Encoding:
    """Encoding for a bubble heatmap (``plot_allocations`` with ``graph_type="bubble"``).

    Both axes are categorical; each ``(x, y)`` cell is one bubble whose colour
    encodes ``color`` (continuous, shared colormap + colorbar) and whose area
    encodes ``size`` (continuous, shared scale + size legend).

    Parameters
    ----------
    x:
        Categorical column on the x-axis (one column of bubbles per value).
    y:
        Categorical column on the y-axis (one row of bubbles per value).
    color:
        Numeric column mapped to the bubble colour (continuous colormap; the
        scale is shared across all facets); ``None`` -> single colour
        (``plot_allocations``'s ``bubble_color``), no colorbar.
    size:
        Numeric column mapped to the bubble area (shared scale across facets);
        ``None`` -> uniform bubbles. ``color`` and ``size`` are each optional
        and independent.
    background:
        Categorical column colouring the background bands (one colour per
        class via ``plot_allocations``'s ``bubble_background_palette``, alpha
        via ``bubble_background_alpha``). Must be constant within each x
        category (vertical bands) or each y category (horizontal bands) --
        the axis is inferred.
    facet_col:
        Column whose values create one panel per value, laid out in columns.
    facet_row:
        Column whose values create one panel per value, laid out in rows.
    labels:
        Display names for ticks / axis and legend titles.
    category_orders:
        ``{column: ordered values}`` for the x / y / facet categories.
    sort_categories:
        Sort unordered categories alphabetically instead of by appearance.
    """
    return Encoding(
        x=x, y=y, color=color, size=size, background=background,
        facet_col=facet_col, facet_row=facet_row,
        labels=dict(labels or {}), category_orders=dict(category_orders or {}),
        sort_categories=sort_categories,
    )


def fingerprint(
    *,
    categories: Mapping[str, Sequence[str]],
    names: Mapping[str, str] | None = None,
    codes: Mapping[str, str] | None = None,
    category_order: Sequence[str] | None = None,
    explanations: Mapping[str, str] | None = None,
    footnotes: Mapping[str, str] | None = None,
    labels: Mapping[Any, Any] | None = None,
) -> Encoding:
    """Encoding for :func:`fingerprint_diagram` (radial per-indicator dashboard).

    Parameters
    ----------
    categories:
        ``{category: [indicator columns]}`` -- the spokes, grouped into arcs.
    names:
        ``{indicator: display name}`` shown in the legend.
    codes:
        ``{indicator: short code}`` printed on each spoke.
    category_order:
        Explicit order of the category arcs around the circle.
    explanations, footnotes:
        Optional ``{indicator: text}`` notes attached to indicators.
    labels:
        Additional flat display-name overrides.
    """
    return Encoding(
        categories=dict(categories),
        names=dict(names) if names is not None else None,
        codes=dict(codes) if codes is not None else None,
        category_order=list(category_order) if category_order is not None else None,
        explanations=dict(explanations) if explanations is not None else None,
        footnotes=dict(footnotes) if footnotes is not None else None,
        labels=dict(labels or {}),
    )
