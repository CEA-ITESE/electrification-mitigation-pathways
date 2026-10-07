"""
Computation layer for the plotting API (explainability study).

Turns the user-facing specs + the input DataFrame into a backend-agnostic
``RenderPlan``: cleaned data, category orders, the facet-grid shape, colour and
compare-style maps, bar x-positions and *precomputed aggregators*.

Performance note
----------------
The original ``_aggregate`` re-filtered the panel frame inside the innermost
drawing loop (panels x |x| x |colors| x |compare|). Here a single
``groupby(...).sum()`` is computed once and every height lookup is an O(1) index
access (:class:`Aggregator`). This matters on large factorial ensembles.

The colour map (:func:`build_color_map`) and category ordering
(:func:`ordered_categories`) preserve the original ``auxil`` semantics so
existing figures render identically.
"""

from __future__ import annotations  # String annotations: no runtime cost, easy forward refs.

import math  # ``ceil`` sizes a wrapped facet grid.
from dataclasses import dataclass, field  # Small internal result structs.
from typing import Any, Iterable, Mapping, Sequence  # Documentation aliases.

import matplotlib.pyplot as plt  # Colormap fallback in the colour map.
import numpy as np  # Bar x-positions and palette sampling indices.
import pandas as pd  # All aggregation is delegated to pandas.
import plotly.express as px  # Plotly palette resolution in the colour map.
import seaborn as sns  # Seaborn palette sampling in the colour map.

from .specs import Encoding  # Public role container.
from ._context import (  # Internal contexts + small helper structs.
    BoxContext,
    FacetSpec,
    FingerprintContext,
    GridContext,
    JointContext,
    PlotContext,
    ScatterSpec,
)


# ===========================================================================
# Category ordering and grid layout (pure helpers)
# ===========================================================================
def ordered_categories(
    df: pd.DataFrame,
    column: str | None,
    category_orders: Mapping[str, Sequence[Any]],
) -> list[Any]:
    """Return the ordered categories for ``column`` (original ``_get_categories``).

    Rules
    -----
    * ``column is None`` -> ``[None]`` (lets callers treat "no dimension"
      uniformly with the faceted case).
    * An explicit order in ``category_orders`` is used verbatim (this is the
      original behaviour: it doubles as a subset selector).
    * Otherwise: order of first appearance, NaNs dropped.

    Parameters
    ----------
    df:
        Source frame.
    column:
        Column to read categories from, or ``None``.
    category_orders:
        Mapping ``column -> explicit order``.

    Returns
    -------
    list
        Ordered categories (or ``[None]``).
    """
    if column is None:  # The "absent dimension" sentinel.
        return [None]  # Single placeholder panel/series.
    if category_orders and column in category_orders:  # Explicit order given.
        return list(category_orders[column])  # Used verbatim (also acts as a subset).
    return list(dict.fromkeys(df[column].dropna().tolist()))  # Order of first appearance, no NaN.


def facet_pairs(row_order: Sequence[Any], col_order: Sequence[Any]) -> list[tuple[Any, Any]]:
    """Cartesian product of facet row/col values as row-major ``(row, col)`` pairs.

    Parameters
    ----------
    row_order, col_order:
        Ordered facet row / column values (each may be ``[None]``).

    Returns
    -------
    list of tuple
        One ``(row_value, col_value)`` per panel.
    """
    return [
        (row_val, col_val)  # Panel identity.
        for row_val in row_order  # Rows vary slowest (row-major).
        for col_val in col_order  # Columns vary fastest.
    ]


def grid_shape(
    n_panels: int,
    n_row_levels: int,
    n_col_levels: int,
    col_wrap: int | None,
    row_wrap: int | None,
) -> tuple[int, int]:
    """Compute ``(nrows, ncols)`` for the subplot grid.

    Parameters
    ----------
    n_panels:
        Number of facet pairs.
    n_row_levels, n_col_levels:
        Distinct facet row / column counts.
    col_wrap, row_wrap:
        Mutually-exclusive wrapping requests.

    Returns
    -------
    (int, int)
        ``(nrows, ncols)``.
    """
    if col_wrap is not None:  # Wrap into a fixed number of columns.
        ncols = min(int(col_wrap), n_panels)  # Never more columns than panels.
        nrows = math.ceil(n_panels / ncols)  # Enough rows to fit.
        return nrows, ncols
    if row_wrap is not None:  # Wrap into a fixed number of rows.
        nrows = min(int(row_wrap), n_panels)  # Never more rows than panels.
        ncols = math.ceil(n_panels / nrows)  # Enough columns to fit.
        return nrows, ncols
    return n_row_levels, n_col_levels  # Regular full grid.


# ===========================================================================
# Data preparation and aggregation
# ===========================================================================
@dataclass
class PreparedData:
    """Cleaned data split into the main and reference parts.

    Attributes
    ----------
    full:
        All retained rows (used for emptiness tests).
    main:
        Rows forming the stacked / line series.
    reference:
        Rows selected by :class:`~mypackage.plot.specs.ScatterSpec` (may be empty).
    """

    full: pd.DataFrame  # Main + reference.
    main: pd.DataFrame  # Primary series.
    reference: pd.DataFrame  # Reference overlay.


def prepare_data(
    data: pd.DataFrame,
    encoding: Encoding,
    facets: FacetSpec,
    scatter: ScatterSpec | None,
) -> PreparedData:
    """Select needed columns, drop incomplete rows, split off references.

    Improvement over the original: ``dropna`` covers only the *essential*
    columns (x, y, colour, facets, compare). The reference-selector column is
    excluded so valid main rows are not dropped merely because the selector is NA
    there.

    Parameters
    ----------
    data:
        Raw input frame.
    encoding:
        Column-to-role mapping.
    facets:
        Facet specification.
    scatter:
        Reference overlay spec, or ``None``.

    Returns
    -------
    PreparedData
        Cleaned, split data.
    """
    essential = [c for c in (encoding.x, encoding.y, encoding.color)
                 if c is not None]  # Must be present and non-NA (colour optional for bubbles).
    for extra in (facets.col, facets.row, encoding.compare_by):  # Optional grouping columns.
        if extra is not None:
            essential.append(extra)  # Also subject to the NA filter.

    selected = list(essential)  # Full set of columns to carry.
    if encoding.x_hierarchy is not None:  # Needed for brackets...
        selected.append(encoding.x_hierarchy)  # ...but intentionally NOT in the NA filter.
    if encoding.size is not None:  # Bubble-size value column (graph_type="bubble").
        selected.append(encoding.size)
    if encoding.background is not None:  # Bubble background class column.
        selected.append(encoding.background)

    scatter_col = scatter.column if scatter is not None else None  # Reference selector column.
    if scatter_col is not None:  # Carry it so we can split on it.
        selected.append(scatter_col)

    selected = list(dict.fromkeys(selected))  # De-duplicate, preserve order.
    df = data.loc[:, selected].copy()  # Restrict columns and detach from the source.
    df = df.dropna(subset=essential)  # Drop rows missing an essential value (selector excluded).

    if scatter is not None:  # A reference overlay was requested.
        is_ref = df[scatter_col].eq(scatter.value)  # Mask of reference rows.
        reference = df.loc[is_ref].copy()  # The overlay rows.
        main = df.loc[~is_ref].copy()  # The remaining (main) rows.
    else:  # No overlay.
        reference = df.iloc[0:0].copy()  # Empty frame with the right columns.
        main = df.copy()  # All rows are main.

    return PreparedData(full=df, main=main, reference=reference)  # Bundle the three views.


@dataclass
class Aggregator:
    """A precomputed ``sum`` table supporting O(1) lookups.

    Attributes
    ----------
    table:
        A :class:`pandas.Series` indexed by the present grouping columns, a scalar
        total when there are none, or ``None`` for an empty source.
    dim_names:
        Ordered names of the present dimensions (subset of
        ``{"facet_row", "facet_col", "x", "color", "compare"}``).
    """

    table: pd.Series | float | None  # Precomputed sums / scalar total / None.
    dim_names: tuple[str, ...]  # Dimensions actually grouped on.

    def get(self, **dim_values: Any) -> float | None:
        """Look up the sum for a fully-specified cell.

        Callers pass *all* dimension values; absent dimensions are ignored.
        A missing combination returns ``None`` (a gap), never ``0``.

        Parameters
        ----------
        **dim_values:
            One keyword per dimension name; extras ignored.

        Returns
        -------
        float or None
            The sum, or ``None`` if absent.
        """
        if not self.dim_names:  # No grouping -> scalar total (or None).
            return self.table
        key = tuple(dim_values[name] for name in self.dim_names)  # Index key in table order.
        lookup = key[0] if len(key) == 1 else key  # 1-level index wants a scalar key.
        try:  # ``.loc`` raises KeyError for absent combinations.
            value = self.table.loc[lookup]
        except KeyError:  # Combination not present.
            return None
        if isinstance(value, pd.Series):  # Defensive: partial match -> collapse.
            value = value.sum()
        return None if pd.isna(value) else float(value)  # Normalise NA -> None.


def build_aggregator(
    df: pd.DataFrame,
    value_column: str,
    dimensions: Iterable[tuple[str, str | None]],
) -> Aggregator:
    """Precompute the grouped-sum table for fast lookups.

    Parameters
    ----------
    df:
        Data to aggregate (main or reference).
    value_column:
        Numeric column summed.
    dimensions:
        Ordered ``(dim_name, column)`` pairs; ``column=None`` dimensions are
        skipped (e.g. colour for the reference overlay).

    Returns
    -------
    Aggregator
        Wrapper around the grouped sums.
    """
    present = [(name, col) for name, col in dimensions if col is not None]  # Active dimensions.
    dim_names = tuple(name for name, _ in present)  # Their names, in order.
    columns = [col for _, col in present]  # Their columns, in order.

    if not columns:  # Nothing to group on.
        total = float(df[value_column].sum()) if not df.empty else None  # Scalar total (or None).
        return Aggregator(table=total, dim_names=())

    table = (
        df.groupby(columns, dropna=False, observed=True)[value_column]  # Group once on all dims.
        .sum()  # Sum the value column per group.
    )
    return Aggregator(table=table, dim_names=dim_names)  # Wrap for O(1) lookups.


# ===========================================================================
# Colour map (original auxil._build_color_map, preserved) + compare styles
# ===========================================================================
def build_color_map(
    categories: Sequence[Any],
    palette: str | Sequence[Any] | Mapping[Any, Any] = "viridis",
    reverse: bool = False,
    backend: str = "matplotlib",
    default_color: str = "grey",
) -> dict[Any, Any]:
    """Return ``{category: color}``, backend-aware (original ``auxil`` logic).

    ``palette`` may be:

    * a **Mapping** ``{category: colour}`` -- forces explicit colours per category
      (e.g. the IPCC temperature-class colour code); any category missing from the
      mapping falls back to ``default_color``;
    * a Matplotlib/seaborn or Plotly palette **name**;
    * an explicit **list/tuple** of colours.

    Categories are sorted so colour assignment is stable, exactly as in the
    original.

    Parameters
    ----------
    categories:
        Colour categories (``None`` entries are dropped).
    palette:
        A per-category mapping, a palette name, or an explicit colour list.
    reverse:
        Reverse the sampled colours (matplotlib path; ignored for a mapping).
    backend:
        ``"matplotlib"`` (RGB tuples) or ``"plotly"`` (colour strings).
    default_color:
        Fallback colour for categories missing from a mapping ``palette``.

    Returns
    -------
    dict
        Mapping ``category -> colour``.
    """
    categories = [c for c in categories if c is not None]  # Drop the placeholder.
    categories = sorted(categories)  # Stable, alphabetical assignment (original behaviour).
    n = max(len(categories), 1)  # At least one colour to avoid empty-palette errors.

    if isinstance(palette, Mapping):  # Forced per-category colours (e.g. IPCC temperature classes).
        return {cat: palette.get(cat, default_color) for cat in categories}  # Default for any missing.

    if isinstance(palette, (list, tuple)):  # Explicit colour list/tuple.
        colors = list(palette)  # Copy.
        if len(colors) < n:  # Recycle if too few.
            colors = (colors * math.ceil(n / len(colors)))[:n]
        return dict(zip(categories, colors))  # Pair up.

    if backend == "matplotlib":  # Matplotlib/seaborn colours (RGB tuples).
        try:
            colors = sns.color_palette(palette, n_colors=n)  # Seaborn-resolvable name.
            if reverse:
                colors.reverse()
        except Exception:  # Fall back to a raw matplotlib colormap.
            cmap = plt.get_cmap(palette)
            vals = np.linspace(0.05, 0.95, n)
            colors = [cmap(v) for v in vals]
        return dict(zip(categories, colors))

    # Plotly backend: resolve qualitative / sequential / diverging / continuous.
    try:
        if hasattr(px.colors.qualitative, palette):  # Qualitative palette name.
            colors = getattr(px.colors.qualitative, palette)
        elif hasattr(px.colors.sequential, palette):  # Sequential palette name.
            seq = getattr(px.colors.sequential, palette)
            idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
            colors = [seq[i] for i in idx]
        elif hasattr(px.colors.diverging, palette):  # Diverging palette name.
            seq = getattr(px.colors.diverging, palette)
            idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
            colors = [seq[i] for i in idx]
        else:  # Continuous colourscale sampled across its range.
            colors = px.colors.sample_colorscale(palette, np.linspace(0.05, 0.95, n))
    except Exception:  # Robust fallback to Viridis.
        colors = px.colors.sample_colorscale("Viridis", np.linspace(0.05, 0.95, n))

    if len(colors) < n:  # Recycle if too few.
        colors = (colors * math.ceil(n / len(colors)))[:n]
    return dict(zip(categories, colors))


# Default style cycles (matching the original lists).
_DEFAULT_HATCHES = ["", "///", "\\\\\\", "...", "xxx", "++", "--", "oo"]  # Matplotlib hatch cycle.
_DEFAULT_PLOTLY_PATTERNS = ["", "/", "\\", ".", "x", "+", "-", "o"]  # Plotly pattern cycle (parallel).
_DEFAULT_LINESTYLES = ["-", "--", ":", "-."]  # Matplotlib line-style cycle.
_DEFAULT_DASHES = ["solid", "dash", "dot", "dashdot"]  # Plotly dash cycle (parallel).

# Single-character translation from a Matplotlib hatch to a Plotly pattern shape.
_HATCH_CHAR_TO_PLOTLY = {"/": "/", "\\": "\\", "|": "|", "-": "-", "+": "+", "x": "x", ".": "."}


def _hatch_to_pattern(hatch: str) -> str:
    """Translate a Matplotlib hatch (e.g. ``"///"``) to a Plotly pattern shape.

    Fixes the original bug where a Matplotlib hatch dict was copied verbatim into
    the Plotly ``pattern.shape`` (which only accepts single characters from a
    fixed set), silently producing invalid patterns.

    Parameters
    ----------
    hatch:
        Matplotlib hatch string (possibly empty).

    Returns
    -------
    str
        A valid Plotly pattern shape (empty if untranslatable).
    """
    for char in hatch:  # First recognised character wins.
        if char in _HATCH_CHAR_TO_PLOTLY:
            return _HATCH_CHAR_TO_PLOTLY[char]
    return ""  # Nothing Plotly understands.


@dataclass
class CompareStyles:
    """Per-compare-value style maps for both backends.

    Attributes
    ----------
    hatch, pattern:
        Matplotlib hatch / Plotly pattern per compare value.
    linestyle, dash:
        Matplotlib line style / Plotly dash per compare value.
    """

    hatch: dict[Any, str]  # Matplotlib bar hatches.
    pattern: dict[Any, str]  # Plotly bar patterns.
    linestyle: dict[Any, str]  # Matplotlib line styles.
    dash: dict[Any, str]  # Plotly dash styles.


def build_compare_styles(
    compare_order: Sequence[Any],
    compare_hatches: Sequence[str] | Mapping[Any, str] | None,
) -> CompareStyles:
    """Build the hatch/pattern/linestyle/dash maps for the compare dimension.

    When ``compare_hatches is None`` the original parallel default cycles are used
    unchanged. When the user supplies hatches (mapping or sequence), the Plotly
    patterns are *translated* from them so the two backends stay consistent.

    Parameters
    ----------
    compare_order:
        Ordered compare categories (or ``[None]`` when unused).
    compare_hatches:
        ``None``, a sequence of hatches, or a mapping ``value -> hatch``.

    Returns
    -------
    CompareStyles
        The four parallel maps.
    """
    if list(compare_order) == [None]:  # Compare dimension unused -> plain style.
        return CompareStyles(hatch={None: ""}, pattern={None: ""},
                             linestyle={None: "-"}, dash={None: "solid"})

    if compare_hatches is None:  # Built-in cycles (original behaviour).
        hatch = {comp: _DEFAULT_HATCHES[i % len(_DEFAULT_HATCHES)]
                 for i, comp in enumerate(compare_order)}
        pattern = {comp: _DEFAULT_PLOTLY_PATTERNS[i % len(_DEFAULT_PLOTLY_PATTERNS)]
                   for i, comp in enumerate(compare_order)}
    elif isinstance(compare_hatches, Mapping):  # Explicit per-value hatches.
        hatch = {comp: compare_hatches.get(comp, "") for comp in compare_order}
        pattern = {comp: _hatch_to_pattern(h) for comp, h in hatch.items()}  # Translated (bug fix).
    else:  # Bare sequence of hatches.
        hatches = list(compare_hatches)
        hatch = {comp: hatches[i % len(hatches)] for i, comp in enumerate(compare_order)}
        pattern = {comp: _hatch_to_pattern(h) for comp, h in hatch.items()}  # Translated (bug fix).

    linestyle = {comp: _DEFAULT_LINESTYLES[i % len(_DEFAULT_LINESTYLES)]
                 for i, comp in enumerate(compare_order)}
    dash = {comp: _DEFAULT_DASHES[i % len(_DEFAULT_DASHES)]
            for i, comp in enumerate(compare_order)}
    return CompareStyles(hatch=hatch, pattern=pattern, linestyle=linestyle, dash=dash)


# ===========================================================================
# Render plan
# ===========================================================================
@dataclass
class RenderPlan:
    """Everything precomputed that the backends consume.

    A backend's ``render`` then takes just ``(context, plan)``.
    """

    prepared: PreparedData  # Cleaned, split data.
    agg_main: Aggregator  # Fast lookup for the main series.
    agg_ref: Aggregator  # Fast lookup for the reference overlay.
    x_order: list[Any]  # Ordered x categories.
    color_order: list[Any]  # Ordered colour categories.
    compare_order: list[Any]  # Ordered compare categories (or [None]).
    facet_row_order: list[Any]  # Ordered facet-row values (or [None]).
    facet_col_order: list[Any]  # Ordered facet-col values (or [None]).
    panels: list[tuple[Any, Any]]  # (row, col) facet pairs, row-major.
    nrows: int  # Grid rows.
    ncols: int  # Grid columns.
    color_map: dict[Any, Any]  # Colour category -> colour.
    compare_styles: CompareStyles  # Hatch/pattern/linestyle/dash maps.
    xpos: np.ndarray  # Left origin of each x group (bars).
    centers: np.ndarray  # Centre of each x group (tick locations).
    n_compare: int  # Bars per group (>= 1).
    y_order: list[Any] | None = None  # Ordered y categories (bubble heatmap only).
    bubble_range: dict | None = None  # {scope key: (vmin, vmax, smin, smax)} for bubbles.
    bubble_background: tuple[str, dict] | None = None  # ("x"|"y", {category: class}) for the bands.
    bubble_axes: dict | None = None  # {scope key: (x_order, y_order)} for the bubble heatmap.


def build_render_plan(ctx: PlotContext) -> RenderPlan:
    """Compute the full :class:`RenderPlan` from a validated context.

    Parameters
    ----------
    ctx:
        Validated plot context.

    Returns
    -------
    RenderPlan
        The fully precomputed plan.
    """
    enc, facets = ctx.encoding, ctx.facets  # Shorthands.

    prepared = prepare_data(ctx.data, enc, facets, ctx.scatter)  # 1. Clean + split.

    x_order = ordered_categories(prepared.full, enc.x, ctx.order.category_orders)  # 2. Orders.
    if enc.color is None:  # Bubble heatmap without a colour column.
        color_order = [None]
    else:
        color_order = (list(ctx.order.color_order) if ctx.order.color_order is not None
                       else ordered_categories(prepared.main, enc.color, ctx.order.category_orders))
    if enc.compare_by is None:  # Compare unused.
        compare_order: list[Any] = [None]
    elif ctx.order.compare_order is not None:  # Explicit compare order.
        compare_order = list(ctx.order.compare_order)
    else:  # Inferred.
        compare_order = ordered_categories(prepared.main, enc.compare_by, ctx.order.category_orders)
    facet_row_order = ordered_categories(prepared.full, facets.row, ctx.order.category_orders)
    facet_col_order = ordered_categories(prepared.full, facets.col, ctx.order.category_orders)

    if enc.color is not None:  # 3. Colour subset (no-op without a colour column).
        prepared.main = prepared.main[prepared.main[enc.color].isin(color_order)].copy()

    panels = facet_pairs(facet_row_order, facet_col_order)  # 4. Panels + grid.
    nrows, ncols = grid_shape(len(panels), len(facet_row_order), len(facet_col_order),
                              facets.col_wrap, facets.row_wrap)

    color_map = build_color_map(color_order, ctx.style.palette,  # 5. Style maps.
                                reverse=ctx.style.reverse_palette, backend=ctx.backend)
    compare_styles = build_compare_styles(compare_order, ctx.style.compare_hatches)

    n_compare = max(len(compare_order), 1)  # 6. Bar x-positions.
    step = n_compare + ctx.style.group_gap  # Distance between group origins.
    xpos = np.arange(len(x_order)) * step  # Left origin per group.
    centers = xpos + (n_compare - 1) / 2  # Group centres (tick locations).

    main_dims = [  # 7. Aggregators. Main series groups on every active dimension.
        ("facet_row", facets.row), ("facet_col", facets.col),
        ("x", enc.x), ("color", enc.color), ("compare", enc.compare_by),
    ]
    ref_dims = [  # Reference overlay sums across colour -> colour dropped.
        ("facet_row", facets.row), ("facet_col", facets.col),
        ("x", enc.x), ("compare", enc.compare_by),
    ]
    agg_main = build_aggregator(prepared.main, enc.y, main_dims)
    agg_ref = build_aggregator(prepared.reference, enc.y, ref_dims)

    y_order = None  # Bubble heatmap: ordered y categories + global colour/size ranges.
    bubble_range = None
    bubble_background = None
    bubble_axes = None
    if ctx.graph_type == "bubble":
        gcols = [c for c in (ctx.facets.row, ctx.facets.col, enc.x, enc.y) if c is not None]
        vcols = [c for c in dict.fromkeys((enc.color, enc.size)) if c is not None]

        def _scope_keys(scope: str):
            """Yield (key, sub-frame) pairs for a normalisation scope."""
            if scope == "row" and ctx.facets.row is not None:
                for k, sub in prepared.full.groupby(ctx.facets.row, observed=True):
                    yield k, sub
            elif scope == "col" and ctx.facets.col is not None:
                for k, sub in prepared.full.groupby(ctx.facets.col, observed=True):
                    yield k, sub
            elif scope == "panel" and (ctx.facets.row is not None or ctx.facets.col is not None):
                by = [c for c in (ctx.facets.row, ctx.facets.col) if c is not None]
                for k, sub in prepared.full.groupby(by, observed=True):
                    yield (k if isinstance(k, tuple) else (k,)), sub
            else:  # "figure", or a scope whose facet dimension is absent.
                yield None, prepared.full

        bubble_range = {}  # {scope key: (vmin, vmax, smin, smax)}.
        for key, sub in _scope_keys(ctx.bubble_scope):
            if vcols:  # Aggregate to cells first: the scales describe drawn bubbles.
                sagg = sub.groupby(gcols, observed=True)[vcols].agg(ctx.bubble_agg)
            vmin, vmax = ((float(sagg[enc.color].min()), float(sagg[enc.color].max()))
                          if enc.color is not None else (float("nan"), float("nan")))
            smin, smax = ((float(sagg[enc.size].min()), float(sagg[enc.size].max()))
                          if enc.size is not None else (float("nan"), float("nan")))
            bubble_range[key] = (vmin, vmax, smin, smax)

        bubble_axes = {}  # {scope key: (x_order, y_order)}.
        for key, sub in _scope_keys(ctx.bubble_axes_scope):
            bubble_axes[key] = (ordered_categories(sub, enc.x, ctx.order.category_orders),
                                ordered_categories(sub, enc.y, ctx.order.category_orders))
        x_order = bubble_axes.get(None, (x_order, None))[0] if None in bubble_axes else x_order
        y_order = bubble_axes[None][1] if None in bubble_axes else \
            ordered_categories(prepared.full, enc.y, ctx.order.category_orders)
        if enc.background is not None:  # Infer the band axis: class constant per x, else per y.
            for axis_name, axis_col in (("x", enc.x), ("y", enc.y)):
                dep = prepared.full.groupby(axis_col, observed=True)[enc.background].nunique()
                if (dep <= 1).all():
                    mapping = (prepared.full.groupby(axis_col, observed=True)[enc.background]
                               .first().to_dict())
                    bubble_background = (axis_name, mapping)
                    break
            else:
                raise ValueError(
                    f"encoding.background {enc.background!r} is constant neither within each "
                    f"{enc.x!r} category nor within each {enc.y!r} category; background bands "
                    f"need a functional class -> category alignment.")

    return RenderPlan(
        prepared=prepared, agg_main=agg_main, agg_ref=agg_ref,
        x_order=x_order, color_order=color_order, compare_order=compare_order,
        facet_row_order=facet_row_order, facet_col_order=facet_col_order,
        panels=panels, nrows=nrows, ncols=ncols,
        color_map=color_map, compare_styles=compare_styles,
        xpos=xpos, centers=centers, n_compare=n_compare,
        y_order=y_order, bubble_range=bubble_range, bubble_background=bubble_background,
        bubble_axes=bubble_axes,
    )


def panel_title(ctx: PlotContext, row_val: Any, col_val: Any) -> str:
    """Compose one facet panel's title.

    Parameters
    ----------
    ctx:
        Plot context (labels + active facet dims).
    row_val, col_val:
        Facet values (either may be ``None``).

    Returns
    -------
    str
        Panel title (empty when nothing is faceted).
    """
    has_row = ctx.facets.row is not None  # Rows faceted?
    has_col = ctx.facets.col is not None  # Columns faceted?
    if not has_row and not has_col:  # Single panel.
        return ""
    if not has_row:  # Column faceting only.
        return str(ctx.labels.lookup(col_val, ctx.facets.col))
    if not has_col:  # Row faceting only.
        return str(ctx.labels.lookup(row_val, ctx.facets.row))
    return f"{ctx.labels.lookup(row_val, ctx.facets.row)} | {ctx.labels.lookup(col_val, ctx.facets.col)}"


def filter_panel(df: pd.DataFrame, ctx: PlotContext, row_val: Any, col_val: Any) -> pd.DataFrame:
    """Slice a frame to one facet panel (used only for the emptiness test now).

    Parameters
    ----------
    df:
        Frame to slice.
    ctx:
        Plot context (facet columns).
    row_val, col_val:
        Facet values.

    Returns
    -------
    pandas.DataFrame
        This panel's rows.
    """
    out = df  # Start from the full frame.
    if ctx.facets.row is not None:  # Row filter.
        out = out[out[ctx.facets.row].eq(row_val)]
    if ctx.facets.col is not None:  # Column filter.
        out = out[out[ctx.facets.col].eq(col_val)]
    return out


# ===========================================================================
# multiplot_grid helpers
# ===========================================================================
def filter_year(df: pd.DataFrame, line_by: str, year: Any) -> pd.DataFrame:
    """Return the cross-section of ``df`` where ``line_by == year``.

    Parameters
    ----------
    df:
        Frame to slice.
    line_by:
        Column holding the year/time value.
    year:
        Value to keep; ``None`` returns a copy of the whole frame.

    Returns
    -------
    pandas.DataFrame
        The (copied) cross-section.
    """
    if year is None:  # No selection -> keep everything.
        return df.copy()
    return df[df[line_by] == year].copy()  # Rows at the requested year.


def regression_color(color_map: Mapping[Any, Any], backend: str = "matplotlib") -> Any:
    """Return a representative colour averaged over the category palette.

    Used as a fallback/derived colour for a regression curve. Averages the RGB of
    the category colours; returns an RGBA tuple for Matplotlib or an ``rgb(...)``
    string for Plotly.

    Parameters
    ----------
    color_map:
        Mapping ``category -> colour``.
    backend:
        ``"matplotlib"`` or ``"plotly"``.

    Returns
    -------
    Any
        A single colour (tuple or string), or ``"black"`` if the map is empty.
    """
    import matplotlib.colors as mcolors  # Local import: only needed here.

    colors = list(color_map.values())  # The category colours.
    if not colors:  # Empty palette (e.g. ``by is None``).
        return "black"  # Neutral default.

    if backend == "matplotlib":  # Average in RGBA space, return a tuple.
        rgba = np.array([mcolors.to_rgba(c) for c in colors], dtype=float)  # N x 4 array.
        return tuple(rgba.mean(axis=0))  # Mean RGBA.

    def _to_rgb_tuple(c: Any) -> tuple[float, float, float]:  # Coerce any colour to 0-255 RGB.
        if isinstance(c, str) and c.startswith("rgb"):  # "rgb(r,g,b)" string.
            vals = c[c.find("(") + 1:c.find(")")].split(",")
            return tuple(float(v.strip()) for v in vals[:3])
        if isinstance(c, str) and c.startswith("#"):  # "#rrggbb" string.
            c = c.lstrip("#")
            if len(c) == 6:
                return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
        rgb = mcolors.to_rgb(c)  # Fallback via matplotlib (named colours, tuples...).
        return tuple(int(round(255 * v)) for v in rgb)

    rgb = np.array([_to_rgb_tuple(c) for c in colors], dtype=float).mean(axis=0)  # Mean RGB.
    rgb = np.clip(np.round(rgb), 0, 255).astype(int)  # Clamp to valid byte range.
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"  # Plotly colour string.


@dataclass
class GridPlan:
    """Precomputed pieces shared by the two :func:`multiplot_grid` backends.

    Attributes
    ----------
    variables:
        The variable columns, one per grid line.
    ncols:
        Number of variables (the "N").
    cats:
        Ordered ``by`` categories (``[None]`` when ``by is None``).
    color_map:
        Mapping ``category -> colour`` (backend-aware; empty when ``by is None``).
    """

    variables: list[Any]  # Variables, one per grid line.
    ncols: int  # Number of variables.
    cats: list[Any]  # Ordered colour categories (or [None]).
    color_map: dict[Any, Any]  # Category -> colour.


def build_grid_plan(ctx: GridContext) -> GridPlan:
    """Resolve variables, categories and the colour map for a grid call.

    Parameters
    ----------
    ctx:
        Validated grid context.

    Returns
    -------
    GridPlan
        The precomputed pieces.
    """
    variables = list(ctx.encoding.variables)  # Variables -> one per line.
    cats = ordered_categories(ctx.data, ctx.encoding.by, ctx.order.category_orders)  # by categories.
    if ctx.order.sort_categories and ctx.encoding.by is not None:  # Optional alphabetical sort.
        cats = sorted(cats)
    color_map = build_color_map(  # Backend-aware colour map (empty if by is None).
        cats, ctx.style.palette, reverse=ctx.style.reverse_palette, backend=ctx.backend
    )
    return GridPlan(variables=variables, ncols=len(variables), cats=cats, color_map=color_map)


# ===========================================================================
# multi_jointplot_subfig helpers
# ===========================================================================
def build_benchmark_color_map(
    benchmark: pd.DataFrame | None,
    benchmark_by: str | None = None,
    benchmark_cmap: Mapping[Any, Any] | None = None,
    default_color: str = "grey",
) -> dict[Any, Any]:
    """Return ``{benchmark_category: colour}`` for the benchmark groups.

    Mirrors the original ``auxil._build_benchmark_color_map``: an empty map when
    there is nothing to colour by, a uniform ``default_color`` when no explicit
    mapping is given, otherwise the forced ``benchmark_cmap`` with
    ``default_color`` as the per-category fallback.

    Parameters
    ----------
    benchmark:
        The benchmark frame (or ``None``).
    benchmark_by:
        Column whose categories are coloured.
    benchmark_cmap:
        Forced ``{category: colour}`` mapping (e.g. IPCC temperature colours).
    default_color:
        Fallback colour.

    Returns
    -------
    dict
        Mapping ``benchmark_category -> colour`` (possibly empty).
    """
    if benchmark is None or benchmark_by is None or benchmark_by not in benchmark.columns:
        return {}  # Nothing to colour by.
    cats = list(dict.fromkeys(benchmark[benchmark_by].dropna().tolist()))  # Order of appearance.
    if benchmark_cmap is None:  # No explicit colours: uniform default.
        return {cat: default_color for cat in cats}
    return {cat: benchmark_cmap.get(cat, default_color) for cat in cats}  # Forced, with fallback.


def compute_axis_limits(values_list, margin: float = 0.05):
    """Compute ``(lo, hi)`` axis limits spanning several 1-D arrays/series.

    Parameters
    ----------
    values_list:
        Iterable of array-likes (``None`` entries are ignored).
    margin:
        Fractional padding added on both ends.

    Returns
    -------
    (float, float) or None
        The padded limits, or ``None`` if no finite values were given.
    """
    arrs = []  # Collect the non-empty arrays.
    for v in values_list:
        if v is None:
            continue
        a = np.asarray(pd.Series(v).dropna())  # Drop NaNs robustly.
        if a.size > 0:
            arrs.append(a)
    if not arrs:  # Nothing to bound.
        return None

    vals = np.concatenate(arrs)  # All values together.
    vmin, vmax = float(np.min(vals)), float(np.max(vals))  # Raw extremes.
    if np.isclose(vmin, vmax):  # Degenerate (single value): pad symmetrically.
        delta = abs(vmin) * 0.05 if vmin != 0 else 1.0
        return vmin - delta, vmax + delta
    vrange = vmax - vmin  # Span.
    return vmin - margin * vrange, vmax + margin * vrange  # Padded limits.


@dataclass
class JointPlan:
    """Precomputed pieces shared across the jointplot panels.

    Attributes
    ----------
    pairs:
        The ``(x, y)`` variable pairs, one per panel.
    nrows, ncols:
        Subplot-grid shape.
    cats:
        Ordered ``by`` categories (``[None]`` when ``by is None``).
    color_map:
        Mapping ``category -> colour`` for the data.
    benchmark_color_map:
        Mapping ``benchmark_category -> colour`` (possibly empty).
    """

    pairs: list[tuple[str, str]]  # (x, y) pairs.
    nrows: int  # Grid rows.
    ncols: int  # Grid columns.
    cats: list[Any]  # Data colour categories.
    color_map: dict[Any, Any]  # Data colours.
    benchmark_color_map: dict[Any, Any]  # Benchmark colours.


def build_joint_plan(ctx: JointContext) -> JointPlan:
    """Resolve pairs, grid shape, categories and colour maps for a joint call.

    Parameters
    ----------
    ctx:
        Validated joint context.

    Returns
    -------
    JointPlan
        The precomputed pieces.
    """
    pairs = [tuple(p) for p in ctx.encoding.var_pairs]  # Normalise to tuples.
    n = len(pairs)  # Number of panels.

    if ctx.orientation == "horizontal":  # One row.
        ncols = n
    elif ctx.orientation == "vertical":  # One column.
        ncols = 1
    elif ctx.ncols is not None:  # Explicit grid width.
        ncols = min(int(ctx.ncols), n)
    else:  # Default grid: up to 3 wide.
        ncols = min(3, n)
    nrows = math.ceil(n / ncols)  # Enough rows to fit.

    cats = ordered_categories(ctx.data, ctx.encoding.by, ctx.order.category_orders)  # by categories.
    if ctx.order.sort_categories and ctx.encoding.by is not None:  # Optional sort.
        cats = sorted(cats)
    color_map = build_color_map(  # Data colours (name / list / forced dict).
        cats, ctx.style.palette, reverse=ctx.style.reverse_palette, backend="matplotlib"
    )
    benchmark_color_map = build_benchmark_color_map(  # Benchmark colours (forced dict supported).
        ctx.benchmark, ctx.encoding.benchmark_by, ctx.style.benchmark_cmap, ctx.style.benchmark_color
    )
    return JointPlan(pairs=pairs, nrows=nrows, ncols=ncols, cats=cats,
                     color_map=color_map, benchmark_color_map=benchmark_color_map)


# ===========================================================================
# grouped_boxplot helpers
# ===========================================================================
@dataclass
class BoxPlan:
    """Precomputed pieces shared by the two :func:`grouped_boxplot` backends.

    Attributes
    ----------
    xcats:
        Ordered ``x`` categories (one group of marks each).
    panels:
        One ``(facet row value, facet col value)`` pair per panel, row-major;
        ``[(None, None)]`` when nothing is faceted.
    nrows, ncols:
        Facet grid shape.
    bycats:
        Ordered data groups within each ``x`` (``[None]`` when ``by is None``).
    bbycats:
        Ordered benchmark groups (``[None]`` for an ungrouped benchmark; empty
        when there is no benchmark).
    color_map:
        Data group -> colour (empty when ``by is None``).
    benchmark_color_map:
        Benchmark group -> colour (empty when no benchmark / no ``benchmark_by``).
    n_data_slots, n_total_slots, n_bench_slots, n_slots:
        Number of sub-positions per ``x`` for the data, the optional pooled
        "total" mark (0 or 1), the benchmark, and in total (the marks are 1.0
        apart in the local slot grid).
    base_spacing:
        Distance between consecutive ``x`` centres (``n_slots + group_gap``).
    centers:
        The x-axis centre of each ``x`` category's slot block.
    """

    xcats: list[Any]  # Ordered x categories.
    bycats: list[Any]  # Ordered data groups (or [None]).
    bbycats: list[Any]  # Ordered benchmark groups (or [None]/[]).
    color_map: dict[Any, Any]  # Data group -> colour.
    benchmark_color_map: dict[Any, Any]  # Benchmark group -> colour.
    n_data_slots: int  # Data sub-positions per x.
    n_total_slots: int  # Pooled-total sub-positions per x (0 or 1).
    n_bench_slots: int  # Benchmark sub-positions per x.
    n_slots: int  # Total sub-positions per x.
    base_spacing: float  # Distance between x centres.
    centers: list[float]  # Per-x centre position.
    panels: list = field(default_factory=lambda: [(None, None)])  # (row, col) facet panels.
    nrows: int = 1  # Facet grid rows.
    ncols: int = 1  # Facet grid columns.


def build_box_plan(ctx: BoxContext) -> BoxPlan:
    """Resolve categories, colour maps and slot geometry for a boxplot call.

    Parameters
    ----------
    ctx:
        Validated boxplot context.

    Returns
    -------
    BoxPlan
        The precomputed pieces consumed by both renderers.
    """
    enc, sty, order = ctx.encoding, ctx.style, ctx.order  # Shorthands.

    xcats = ordered_categories(ctx.data, enc.x, order.category_orders)  # x axis groups.

    bycats = ordered_categories(ctx.data, enc.by, order.category_orders)  # Data groups (or [None]).
    if order.sort_categories and enc.by is not None:  # Optional alphabetical sort.
        bycats = sorted(bycats)
    color_map = (build_color_map(bycats, sty.palette, reverse=sty.reverse_palette, backend=ctx.backend)
                 if enc.by is not None else {})  # Empty when ungrouped (renderers use a default).

    if ctx.benchmark is None:  # No benchmark at all.
        bbycats: list[Any] = []
    else:  # Benchmark present: grouped or a single ungrouped block.
        bbycats = ordered_categories(ctx.benchmark, enc.benchmark_by, order.category_orders)
        if order.sort_categories and enc.benchmark_by is not None:
            bbycats = sorted(bbycats)
    benchmark_color_map = build_benchmark_color_map(  # Forced/uniform benchmark colours.
        ctx.benchmark, enc.benchmark_by, sty.benchmark_cmap, default_color=sty.benchmark_color
    )

    facet_row_order = ordered_categories(ctx.data, ctx.facets.row, order.category_orders)
    facet_col_order = ordered_categories(ctx.data, ctx.facets.col, order.category_orders)
    panels = facet_pairs(facet_row_order, facet_col_order)  # Panels + grid shape.
    nrows, ncols = grid_shape(len(panels), len(facet_row_order), len(facet_col_order),
                              ctx.facets.col_wrap, ctx.facets.row_wrap)

    n_data_slots = len(bycats) if enc.by is not None else 1  # Data sub-positions.
    n_total_slots = 1 if ctx.show_total else 0  # Pooled mark (ctx guarantees by is set).
    n_bench_slots = (len(bbycats) if enc.benchmark_by is not None else 1) if ctx.benchmark is not None else 0
    n_slots = n_data_slots + n_total_slots + n_bench_slots  # Total sub-positions per x.
    base_spacing = n_slots + sty.group_gap  # Centre-to-centre distance.
    centers = [i * base_spacing for i in range(len(xcats))]  # One centre per x category.

    return BoxPlan(
        xcats=xcats, bycats=bycats, bbycats=bbycats,
        color_map=color_map, benchmark_color_map=benchmark_color_map,
        n_data_slots=n_data_slots, n_total_slots=n_total_slots, n_bench_slots=n_bench_slots,
        n_slots=n_slots, base_spacing=base_spacing, centers=centers,
        panels=panels, nrows=nrows, ncols=ncols,
    )


# ===========================================================================
# fingerprint_diagram helpers
# ===========================================================================
@dataclass
class FingerprintPlan:
    """Precomputed geometry + labels for :func:`fingerprint_diagram`.

    Attributes
    ----------
    categories:
        Category names in display order.
    indicators:
        ``{category: [indicator columns]}`` in display order.
    angles:
        ``{indicator column: spoke angle}`` in degrees, clockwise from the top.
    cat_span:
        ``{category: (a_start, a_end)}`` sector span in the same angle convention.
    cat_color:
        ``{category: colour}``.
    code:
        ``{indicator column: short code}`` drawn on the spoke.
    name:
        ``{indicator column: display name}`` used in the legend.
    n_categories, n_indicators:
        Convenience counts.
    """

    categories: list[str]  # Ordered category names.
    indicators: dict[str, list[str]]  # category -> ordered indicator columns.
    angles: dict[str, float]  # indicator -> spoke angle (deg, clockwise from top).
    cat_span: dict[str, tuple[float, float]]  # category -> (a_start, a_end) degrees.
    cat_color: dict[str, Any]  # category -> colour.
    code: dict[str, str]  # indicator -> short code.
    name: dict[str, str]  # indicator -> display name.
    n_categories: int  # Number of categories.
    n_indicators: int  # Total number of indicators (spokes).
    has_data: bool = False  # Whether a focal band is mapped (benchmark given + data present).
    band_quantiles: tuple[float, ...] = ()  # Focal quantiles drawn (per indicator).
    band_radii: dict[str, list[float]] = field(default_factory=dict)  # indicator -> radius per quantile.


def build_fingerprint_plan(ctx: FingerprintContext) -> FingerprintPlan:
    """Resolve order, colours, codes and angular geometry for a fingerprint.

    Each category occupies an equal ``360 / n_categories`` sector; the indicators
    of a category are spread evenly inside it (``len + 1`` margins), so the layout
    generalises to any per-category indicator count.

    Parameters
    ----------
    ctx:
        Validated fingerprint context.

    Returns
    -------
    FingerprintPlan
    """
    enc, sty = ctx.encoding, ctx.style  # Shorthands.

    cats = list(enc.category_order) if enc.category_order is not None else list(enc.categories.keys())
    indicators = {c: list(enc.categories[c]) for c in cats}  # Ordered indicators per category.

    cat_color = build_color_map(cats, sty.palette, reverse=sty.reverse_palette, backend="matplotlib")

    names_map = dict(enc.names) if enc.names is not None else {}
    codes_map = dict(enc.codes) if enc.codes is not None else {}
    name = {}  # indicator -> display name (legend).
    code = {}  # indicator -> short code (spoke).
    for c in cats:
        for col in indicators[c]:
            name[col] = names_map.get(col, str(col))  # Fallback: the column itself.
            code[col] = codes_map.get(col, str(col))  # Fallback: the column itself.

    n_cat = len(cats)  # Sectors.
    incr = 360.0 / n_cat if n_cat else 360.0  # Degrees per category sector.
    angles: dict[str, float] = {}  # indicator -> spoke angle.
    cat_span: dict[str, tuple[float, float]] = {}  # category -> sector span.
    n_ind = 0
    for ci, c in enumerate(cats):
        a0, a1 = incr * ci, incr * (ci + 1)  # Sector span (clockwise from top).
        cat_span[c] = (a0, a1)
        members = indicators[c]
        for vi, col in enumerate(members):  # Even spread inside the sector.
            angles[col] = a0 + incr * (vi + 1) / (len(members) + 1)
        n_ind += len(members)

    # --- data layer: map the focal distribution onto each spoke via the benchmark fence ---
    ds = ctx  # Data-layer fields (iqr_mult, band_quantiles, na_sentinels) are flat on ctx.
    has_data = ctx.benchmark is not None and not ctx.data.empty
    band_radii: dict[str, list[float]] = {}
    if has_data:
        r_border = sty.r_border
        for col in (c for cat in cats for c in indicators[cat]):
            ref = _fp_clean(ctx.benchmark[col], ds.na_sentinels)  # Reference ensemble values.
            fence = _fp_fence(ref, ds.iqr_mult)  # (lo, hi) Tukey fence, or None.
            foc = _fp_clean(ctx.data[col], ds.na_sentinels)  # Focal values.
            if fence is None or foc.size == 0:
                continue  # No usable scale or no focal data: leave this spoke bare.
            lo, hi = fence
            radii = []
            for q in ds.band_quantiles:
                vq = float(np.nanquantile(foc, q))  # Focal q-th quantile.
                frac = (vq - lo) / (hi - lo)  # Position within the fence.
                radii.append(r_border * float(np.clip(frac, 0.0, 1.0)))  # -> radius.
            band_radii[col] = radii

    return FingerprintPlan(
        categories=cats, indicators=indicators, angles=angles, cat_span=cat_span,
        cat_color=cat_color, code=code, name=name, n_categories=n_cat, n_indicators=n_ind,
        has_data=has_data and bool(band_radii), band_quantiles=tuple(ds.band_quantiles),
        band_radii=band_radii,
    )


def _fp_clean(series, na_sentinels: bool) -> "np.ndarray":
    """Numeric values of ``series`` with NaNs (and optional sentinels) dropped."""
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if na_sentinels:  # IIASA-style sentinels: 0 and >= 1e20 mean "missing".
        v = np.where((v == 0) | (v >= 1e20), np.nan, v)
    return v[~np.isnan(v)]


def _fp_fence(values: "np.ndarray", iqr_mult: float) -> "tuple[float, float] | None":
    """Tukey fence ``(Q25 - k*IQR, Q75 + k*IQR)`` of ``values``; None if degenerate."""
    if values.size == 0:
        return None
    q25, q75 = float(np.nanquantile(values, 0.25)), float(np.nanquantile(values, 0.75))
    iqr = q75 - q25
    lo, hi = q25 - iqr_mult * iqr, q75 + iqr_mult * iqr
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:  # Degenerate (e.g. constant column).
        return None
    return lo, hi
