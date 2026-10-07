"""Private validated contexts + small internal helper structs.

Each entry point in ``plot.py`` validates its arguments into one frozen
``*Context`` bundle (built directly from the unified public :class:`Encoding` /
:class:`Layout` / :class:`Style` / :class:`Regression` in ``specs.py``), which
``prepare``/``render`` then consume. A handful of tiny cross-cutting structs
(:class:`FacetSpec`, :class:`OrderSpec`, :class:`ScatterSpec`, :class:`LabelSpec`,
:class:`DistSpec`, :class:`MarginalSpec`) remain here as internal plumbing.

None of this is part of the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from .specs import (  # Public classes referenced by the contexts / adapters.
    Encoding, Layout, Style, Regression, SaveSpec, FingerprintStyle, PlotResult,
)


@dataclass(frozen=True)
class FacetSpec:
    """Small-multiples (facet) grid configuration.

    Parameters
    ----------
    col, row:
        Columns mapped to facet columns / rows.
    col_wrap, row_wrap:
        Mutually exclusive. Flatten panels and wrap into this many columns
        (resp. rows). Intended for a single facet dimension.
    share_y:
        Whether all panels share one y-range.
    hide_empty:
        Whether empty panels are blanked instead of drawn.
    """

    col: str | None = None  # Facet-column variable.
    row: str | None = None  # Facet-row variable.
    col_wrap: int | None = None  # Wrap flattened panels into this many columns.
    row_wrap: int | None = None  # Wrap flattened panels into this many rows.
    share_y: bool = True  # Unify the y-range across visible panels.
    hide_empty: bool = True  # Blank panels with no data.


# ---------------------------------------------------------------------------
# Explicit ordering / subsetting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderSpec:
    """Explicit category orders and subsets.

    Parameters
    ----------
    category_orders:
        Mapping ``column -> ordered values`` (used verbatim when present,
        matching the original ``_get_categories`` behaviour).
    color_order:
        Explicit ordering/subset of the colour categories (renames the old
        ``color_by``, which was an order, not a column).
    compare_order:
        Explicit ordering of the ``compare_by`` categories (renames the old
        ``compare_by_order``).
    """

    category_orders: Mapping[str, Sequence[Any]] = field(default_factory=dict)  # Per-column orders.
    color_order: Sequence[Any] | None = None  # Colour order/subset, or None to infer (plot_allocations).
    compare_order: Sequence[Any] | None = None  # Compare order, or None to infer (plot_allocations).
    sort_categories: bool = False  # Sort the ``by`` categories alphabetically (multiplot_grid).


# ---------------------------------------------------------------------------
# Reference overlay
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScatterSpec:
    """Reference overlay (e.g. a "Total") drawn on top of the main series.

    Parameters
    ----------
    category:
        One-item mapping ``{column: value}`` selecting the reference rows.
    size, facecolor, edgecolor, linewidth:
        Marker styling for the bar-chart overlay.
    """

    category: Mapping[str, Any]  # {column: value} selecting reference rows (required).
    size: float = 90.0  # Marker area (matplotlib ``s``).
    facecolor: str = "white"  # Marker fill.
    edgecolor: str = "black"  # Marker edge.
    linewidth: float = 1.5  # Marker edge width.

    @property
    def column(self) -> str:
        """Return the single column name in :attr:`category`."""
        return next(iter(self.category))  # Validation guarantees exactly one item.

    @property
    def value(self) -> Any:
        """Return the single value in :attr:`category`."""
        return next(iter(self.category.values()))  # The reference value within :attr:`column`.


# ---------------------------------------------------------------------------
# Cosmetic styling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelSpec:
    """Display-name lookup for raw values.

    Supports a *flat* mapping ``{value: label}`` and a *nested* mapping
    ``{column: {value: label}}`` simultaneously; per-column entries win.
    """

    mapping: Mapping[Any, Any] = field(default_factory=dict)  # Flat and/or nested labels.

    def lookup(self, value: Any, column: str | None = None) -> Any:
        """Return the display label for ``value`` (optionally within ``column``).

        Parameters
        ----------
        value:
            Raw value to translate.
        column:
            Column the value comes from (a nested mapping for it wins).

        Returns
        -------
        Any
            The label, or ``value`` unchanged if none is configured.
        """
        if column is not None and column in self.mapping:  # A per-column override may exist.
            per_col = self.mapping[column]  # Candidate override.
            if isinstance(per_col, Mapping):  # Treat as nested only if it is itself a mapping.
                return per_col.get(value, value)  # Nested label or raw value.
        return self.mapping.get(value, value)  # Flat label or raw value.


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlotContext:
    """The full, validated configuration for one plot call.

    Bundling the specs keeps backend signatures short: validation produces a
    ``PlotContext``, planning consumes it.
    """

    data: pd.DataFrame  # Raw input frame.
    encoding: Encoding  # Column-to-role mapping.
    graph_type: str  # "bar" or "line".
    backend: str  # "matplotlib" or "plotly".
    facets: FacetSpec  # Faceting.
    order: OrderSpec  # Explicit orders.
    style: Style  # Cosmetics (unified).
    layout: Layout  # Titles/limits/legend (unified).
    labels: LabelSpec  # Display labels.
    scatter: ScatterSpec | None  # Reference overlay, or None.
    save: SaveSpec | None  # Save config, or None.
    bubble_sizes: tuple[float, float] = (30.0, 500.0)  # Marker area range (pt^2, mpl) for graph_type="bubble".
    bubble_cmap: str = "viridis"  # Continuous colormap for the bubble colour column.
    bubble_agg: str = "mean"  # Aggregation when several rows share an (x, y) cell.
    bubble_color: str = "steelblue"  # Single bubble colour when encoding.color is None.
    bubble_background_palette: Mapping[Any, str] | None = None  # {class: colour} for the background bands.
    bubble_background_alpha: float = 0.15  # Background band opacity.
    bubble_scope: str = "figure"  # Colour/size normalisation scope: figure | row | col | panel.
    bubble_axes_scope: str = "figure"  # Scope of the x/y categories: figure | row | col | panel.
@dataclass(frozen=True)
class DistSpec:
    """Distribution (column 2) configuration for :func:`multiplot_grid`.

    Parameters
    ----------
    kind:
        ``"kde"``, ``"hist"`` or ``"box"``.
    norm:
        Unit of the density axis (the new option): ``"density"``, ``"frequency"``
        (raw counts) or ``"probability"``. Applies to histograms; KDE is always a
        density and ``box`` ignores it. Drives both the matplotlib ``stat`` /
        Plotly ``histnorm`` and the axis label.
    n_points:
        Number of evaluation points for the Plotly KDE curve.
    show_data:
        Draw the data's own distribution. Set ``False`` for a benchmark-only
        distribution cell (requires a benchmark with ``show_benchmark=True``).
    show_benchmark:
        Draw the benchmark distribution behind the data (when a benchmark is
        given). Set ``False`` to suppress the grey benchmark overlay.
    group_alpha:
        Opacity of the per-``by`` distributions, drawn as a background layer
        (colour from ``plan.color_map`` -- the same map the per-group scatter
        regression reads, so dist and regression colours agree by
        construction). For kde / hist the curves are *unfilled* and the alpha
        applies to the line; for ``kind="box"`` it is the box face opacity.
        Also used when ``by`` is None.
    show_total:
        Additionally draw the *pooled* distribution (all ``by`` categories
        together) as a single unfilled curve on top, for comparison against the
        benchmark. Ignored (with a warning) for ``kind="box"``.
    total_color:
        Colour of the pooled curve.
    total_label:
        Legend label of the pooled curve (also passed through
        ``encoding.labels``, so both renaming mechanisms work).
    highlight_on:
        Where to pin the spotlighted points (``multiplot_grid``'s ``highlight``
        selection) inside the distribution cell: ``None`` (no markers,
        default), ``"group"`` (on each point's own per-``by`` curve, its group
        colour) or ``"total"`` (on the pooled curve; the marker keeps the
        group colour). kde / hist only.
    log_x:
        Log-scale the density axis (the cell's x-axis; kde / hist only,
        ignored with a warning for ``kind="box"`` whose x-axis is categorical).
        The axis is floored 3 decades below the cell's density peak to keep
        the KDE tails from stretching it.
    """

    kind: str = "kde"  # "kde" | "hist" | "box".
    norm: str = "density"  # "density" | "frequency" | "probability".
    n_points: int = 200  # Plotly KDE sampling resolution.
    show_data: bool = True  # Draw the data's own distribution.
    show_benchmark: bool = True  # Draw the benchmark distribution (when a benchmark is given).
    group_alpha: float = 0.5  # Line opacity (kde/hist) / face opacity (box) of the per-group layer.
    show_total: bool = False  # Draw the pooled all-groups distribution on top.
    total_color: str = "black"  # Colour of the pooled curve.
    total_label: str = "total"  # Legend label of the pooled curve.
    highlight_on: str | None = None  # Pin highlighted points on: None | "group" | "total".
    log_x: bool = False  # Log-scale the density axis (kde/hist).
@dataclass(frozen=True)
class GridContext:
    """Validated configuration for one ``multiplot_grid`` call (B2: unified).

    Holds the public :class:`Encoding` / :class:`Style` / :class:`Regression`
    directly; the former ``GridLayoutSpec`` fields (grid geometry *and* the shared
    top-legend controls) are now flat. ``dist`` stays an internal struct.
    """

    data: pd.DataFrame
    benchmark: pd.DataFrame | None
    encoding: Encoding  # Roles (variables / line_by / scatter_by / by / aggregate).
    style: Style  # Cosmetics (unified).
    dist: DistSpec  # Distribution-cell config (internal helper).
    regression: Regression  # Regression config.
    order: OrderSpec  # Category ordering (internal helper).
    labels: LabelSpec  # Display names (internal helper).
    year: Any = None  # Cross-section year value (or None for "all values").
    highlight: Mapping[str, Any] | None = None  # {column: value(s)} -- scatter-cell points drawn on top: bigger, hatched, black edge.
    highlight_lines: Mapping[str, Any] | None = None  # {column: value(s)} -- series-cell trajectories redrawn with a black outline.
    panel_letters: bool = False  # Bold a, b, c... tag in each cell's upper-left corner (row-major).
    backend: str = "matplotlib"
    save: SaveSpec | None = None
    # --- geometry / windowing (ex-GridLayoutSpec) ---
    variables_by: str = "row"
    columns: tuple[str, ...] = ("series", "dist", "scatter")
    year_range: tuple[float, float] = (2020.0, 2050.0)
    var_window: Mapping[Any, tuple[float, float]] = field(default_factory=dict)
    shared_xaxes: bool = False
    shared_yaxes: str = "variable"
    figsize_per_col: tuple[float, float] = (5.0, 3.6)
    cell_ratios: tuple[float, ...] | None = None  # Relative size of each cell along the cell axis.
    plotly_size_per_col: tuple[float, float] = (360, 300)
    # --- shared top legend + margins (generic) ---
    legend_y: float | None = None
    legend_ncol: int | None = None
    legend_max_per_row: int = 6
    legend_gap: float = 0.01
    legend_row_height: float = 0.05
    top_margin: float = 0.02
    bottom_margin: float = 0.0
    left_margin: float = 0.0


@dataclass(frozen=True)
class MarginalSpec:
    """Marginal-distribution configuration for :func:`multi_jointplot_subfig`.

    Parameters
    ----------
    show:
        Whether to draw the top (x) and right (y) marginal distributions.
    kind:
        ``"kde"`` or ``"hist"``.
    group_alpha:
        Line opacity of the per-``by`` marginal curves, drawn *unfilled* as a
        background layer (colour from ``plan.color_map`` -- the same map the
        scatter and its per-group regression read, so all agree by
        construction). Also used when ``by`` is None.
    show_total:
        Additionally draw the *pooled* marginal (all ``by`` categories
        together) as a single unfilled curve on top of both marginals, for
        comparison against the benchmark.
    total_color:
        Colour of the pooled curve.
    total_label:
        Legend label of the pooled curve (also passed through
        ``encoding.labels``, so both renaming mechanisms work).
    """

    show: bool = True  # Draw the marginal distributions.
    kind: str = "kde"  # "kde" | "hist".
    group_alpha: float = 0.5  # Line opacity of the per-group (background) marginals.
    show_total: bool = False  # Draw the pooled all-groups marginal on top.
    total_color: str = "black"  # Colour of the pooled curve.
    total_label: str = "total"  # Legend label of the pooled curve.
@dataclass(frozen=True)
class JointContext:
    """Validated configuration for one ``multi_jointplot_subfig`` call (B2: unified).

    Holds the public :class:`Encoding` / :class:`Style` / :class:`Regression`
    directly; the former ``JointLayoutSpec`` fields are now flat (populated from
    the function's keyword arguments). ``marginals`` stays an internal struct.
    """

    data: pd.DataFrame
    benchmark: pd.DataFrame | None
    encoding: Encoding  # Roles (var_pairs / by / benchmark_by).
    style: Style  # Cosmetics (unified).
    marginals: MarginalSpec  # Marginal-distribution config (internal helper).
    regression: Regression  # Regression config (data + benchmark).
    order: OrderSpec  # Category ordering (internal helper).
    labels: LabelSpec  # Display names (internal helper).
    save: SaveSpec | None = None
    # --- geometry / arrangement (ex-JointLayoutSpec) ---
    orientation: str | None = None
    ncols: int | None = None
    figsize_per_plot: tuple[float, float] = (4.8, 4.8)
    main_height_ratio: float = 4.0
    marginal_height_ratio: float = 1.15
    main_width_ratio: float = 4.0
    marginal_width_ratio: float = 1.15
    xlim_mode: str = "both"
    ylim_mode: str = "both"
    axis_margin: float = 0.05
    diag_line: bool = True
    symmetric_xy: bool = False  # Per panel: same x/y limits (union), so y = x is the symmetry axis.
    x_min: float | None = None  # Hard lower x limit (from Layout).
    x_max: float | None = None  # Hard upper x limit (from Layout).
    y_min: float | None = None  # Hard lower y limit (from Layout).
    y_max: float | None = None  # Hard upper y limit (from Layout).
    diag_quartiles: bool = False  # Also draw y = 0.75x / 0.5x / 0.25x guide lines (thin, light grey).
    highlight: Mapping[str, Any] | None = None  # {column: value(s)} -- points drawn on top: bigger, hatched, black edge.
    panel_letters: bool = False  # Bold a, b, c... tag in each panel's upper-left corner.
    diag_margin: float = 0.0
    legend_y: float = 1.02
    legend_ncol: int | None = None  # Explicit legend columns (None = auto-balance).
    legend_max_per_row: int = 6  # Above this many items, split into two balanced rows.


@dataclass(frozen=True)
class BoxContext:
    """Validated configuration for one ``grouped_boxplot`` call (B2: unified)."""

    data: pd.DataFrame
    benchmark: pd.DataFrame | None
    encoding: Encoding  # Column-to-role mapping (unified).
    kind: str  # "box" | "violin" | "strip".
    backend: str  # "matplotlib" | "plotly".
    style: Style  # Cosmetics (unified).
    layout: Layout  # Titles / limits / legend (unified).
    order: OrderSpec  # Category ordering (internal helper).
    labels: LabelSpec  # Display names (internal helper).
    save: SaveSpec | None = None
    figsize: tuple[float, float] = (8.0, 6.0)  # ex-BoxStyleSpec.
    plotly_size: tuple[int, int] = (900, 650)  # ex-BoxStyleSpec.
    show_legend: bool = True  # ex-BoxStyleSpec.
    show_total: bool = False  # Extra pooled (all ``by`` groups) mark per x slot, unfilled.
    total_color: str = "black"  # Colour of the pooled mark (lines / points).
    total_label: str = "total"  # Legend label of the pooled mark.
    facets: FacetSpec = field(default_factory=FacetSpec)  # Facet dims (from encoding + Layout).


@dataclass(frozen=True)
class FingerprintContext:
    """Validated configuration for one ``fingerprint_diagram`` call (B2: unified).

    Holds the public :class:`Encoding` and :class:`FingerprintStyle` directly; the
    former per-chart ``FingerprintLayoutSpec`` / ``FingerprintDataSpec`` are now
    flat fields (populated from the function's keyword arguments).
    """

    data: pd.DataFrame  # Focal frame (drawn as a band per spoke); indicators in columns.
    benchmark: pd.DataFrame | None  # Reference ensemble setting the radial scale, or None (scaffold).
    encoding: Encoding  # Structure (categories / names / codes / order).
    style: FingerprintStyle  # Geometry + palette.
    save: SaveSpec | None = None  # Save config, or None.
    panel_letters: bool = False  # Bold 'a' tag in the corner (from Layout).
    # --- titles / guide / legend / window (ex-FingerprintLayoutSpec) ---
    title: str | None = None
    footnotes: Sequence[str] = ()
    quantile_guide: bool = True
    quantile_labels: Sequence[str] = (
        r"$Q_{25}\!-\!1.5\,$IQR", r"$Q_{25}$", r"$Q_{50}$", r"$Q_{75}$",
        r"$Q_{75}\!+\!1.5\,$IQR",
    )
    legend: bool = True
    legend_loc: str = "right"
    legend_fontsize: float = 13.0
    xlim: tuple[float, float] = (-0.35, 1.65)
    ylim: tuple[float, float] = (-0.18, 1.18)
    # --- radial scale + mark (ex-FingerprintDataSpec) ---
    iqr_mult: float = 1.5
    band_quantiles: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    na_sentinels: bool = False
    mark: str = "band"
    band_alpha: float = 0.5
    band_lw: float = 2.0
    tick_lw: float = 4.0
    tick_scale: float = 0.4

