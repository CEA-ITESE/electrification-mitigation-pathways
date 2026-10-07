"""
Configuration dataclasses for the plotting API (explainability study).

Every group of related parameters from the original flat-kwargs signature is
collapsed into a small *frozen* dataclass. Freezing serves two purposes:

1. The instances are immutable, so a single shared default instance can safely be
   used as a function default argument (a mutable default would be shared state).
2. It documents intent: a spec says *what* to draw, never how it mutates.

Container-valued fields use ``field(default_factory=...)`` because mutable
literals cannot be dataclass defaults.
"""

from __future__ import annotations  # Annotations are strings -> no runtime cost, no forward-ref issues.

from dataclasses import dataclass, field  # ``dataclass`` builds boilerplate; ``field`` configures attributes.
from typing import Any, Mapping, Sequence  # Documentation / static-checking aliases only.

import pandas as pd  # Only used for type annotations of the data frame.


# ---------------------------------------------------------------------------
# Data-to-visual mapping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SaveSpec:
    """Whether and how to write the figure to disk.

    Parameters
    ----------
    path, name:
        Output directory and base filename (without extension); both required.
    extensions:
        Formats to write (vector pdf/svg recommended for publication).
    dpi:
        Raster resolution for png.
    """

    path: str  # Output directory.
    name: str  # Base filename (no extension).
    extensions: Sequence[str] = ("pdf", "png", "svg")  # Formats to emit.
    dpi: int = 300  # Raster resolution.


# ---------------------------------------------------------------------------
# Internal contract objects (input bundle + uniform result)
# ---------------------------------------------------------------------------
@dataclass
class PlotResult:
    """Uniform return type for both backends.

    The original returned ``(fig, axes)`` for Matplotlib but a bare ``fig`` for
    Plotly; a single type removes that asymmetry.

    Attributes
    ----------
    figure:
        Backend figure object.
    axes:
        Matplotlib axes array, or ``None`` for Plotly.
    files:
        Paths written to disk (empty if not saved).
    backend:
        Producing backend name.
    """

    figure: Any  # Backend-specific figure.
    axes: Any = None  # Matplotlib axes array, or None.
    files: list[str] = field(default_factory=list)  # Saved file paths.
    backend: str = ""  # Producing backend.


# ===========================================================================
# multiplot_grid: a separate N-rows x 3-columns diagnostic grid.
#
# Each evaluated variable occupies one line of the grid (when
# ``GridLayoutSpec.variables_by == "row"``; transposed to 3 x N when "col"):
#   * column 1 -- time series of the variable vs ``line_by``, coloured by ``by``,
#                 with the benchmark trajectories greyed in the background;
#   * column 2 -- the marginal distribution at the final ``year`` (kde / hist /
#                 box), drawn horizontally (variable on the y-axis), with the
#                 benchmark distribution greyed behind;
#   * column 3 -- a scatter of the variable vs ``scatter_by`` at the final
#                 ``year``, coloured by ``by``, with optional LOESS regression
#                 lines for the data and (optionally) the benchmark.
# ===========================================================================
@dataclass(frozen=True)
class Encoding:
    """Which column plays which role (+ ordering + display labels).

    One generic container for every chart. Each function reads the roles it
    needs and ignores the rest; it validates the ones it requires.

    Roles by chart (all optional here):

    - ``plot_allocations``: ``x``, ``y``, ``color``, ``compare_by``,
      ``x_hierarchy``, ``facet_col``, ``facet_row``.
    - ``multiplot_grid``: ``variables``, ``line_by``, ``scatter_by``, ``by``,
      ``line_id``, ``benchmark_line_id``, ``aggregate``.
    - ``multi_jointplot_subfig``: ``var_pairs``, ``by``, ``benchmark_by``.
    - ``grouped_boxplot``: ``x``, ``y``, ``by``, ``x_hierarchy``, ``benchmark_by``.
    - ``fingerprint_diagram``: ``categories``, ``names``, ``codes``,
      ``category_order``, ``explanations``, ``footnotes``.

    Ordering (``category_orders``, ``color_order``, ``compare_order``,
    ``sort_categories``) and display ``labels`` apply wherever relevant.

    Parameters
    ----------
    x:
        Horizontal-axis column: bar / line positions (allocations), category
        axis (boxplot).
    y:
        Value column: bar height / line value (allocations), boxed
        distribution (boxplot).
    color:
        Colour column for allocations (stacking within each ``x`` for bars,
        one series per value for lines). Grid / joint / box use ``by`` instead
        -- the builders homogenize both under the name ``color``.
    compare_by:
        Comparison column (allocations): side-by-side bars or one line style
        per value, within each ``x``.
    size:
        Column encoding bubble sizes (``plot_allocations`` with
        ``graph_type="bubble"``); ``None`` -> uniform bubbles.
    background:
        Categorical column colouring the background bands of the bubble
        heatmap: each x (or y) category takes the colour of its class through
        ``bubble_background_palette``. The band axis is inferred: the class
        must be constant within each x category (vertical bands) or within
        each y category (horizontal bands).
    x_hierarchy:
        Coarser grouping of the ``x`` values, drawn as labelled brackets under
        the axis (allocations bar charts, boxplot; matplotlib only).
    facet_col:
        Column to facet on: one subplot per value, across columns (allocations).
    facet_row:
        Column to facet on: one subplot per value, across rows (allocations).
    variables:
        Variable columns, one grid line (series + distribution + scatter cells)
        each (grid).
    line_by:
        Dual role (grid): column on the series-cell x-axis (e.g. the year)
        *and* the column the frame is sliced on -- at the function's ``year=``
        cross-section value -- to build the distribution and scatter cells.
    scatter_by:
        Scatter-cell x column (grid).
    by:
        Colour-grouping column (grid / joint / box): one colour per value for
        series, distributions, scatter points and boxes. Exposed as ``color``
        in the builders.
    line_id:
        Per-trajectory identifier column for the data lines (grid).
    benchmark_line_id:
        Benchmark trajectory identifier column(s) (grid).
    aggregate:
        Reduction applied to duplicate (group, ``line_by``) points before
        drawing series (grid): ``"mean"``, ``"median"`` or ``None`` (keep all).
    var_pairs:
        ``(x, y)`` column pairs, one jointplot panel each (joint).
    benchmark_by:
        Benchmark colour-grouping column (joint / box): per-category benchmark
        marks / marginals instead of a single grey layer.
    categories:
        ``{category: [indicator columns]}`` -- the spoke layout (fingerprint).
    names:
        ``{indicator: display name}`` for the legend (fingerprint).
    codes:
        ``{indicator: short code}`` printed on each spoke (fingerprint).
    category_order:
        Explicit category order around the diagram (fingerprint).
    explanations:
        Optional ``{indicator: note}`` shown in the legend (fingerprint).
    footnotes:
        Optional ``{indicator: footnote}`` (fingerprint).
    category_orders:
        ``{column: explicit value order}``; applies to any role column of the
        target chart (axes, facets, groups).
    color_order:
        Order *and* subset of the colour categories (``color`` / ``by``); also
        fixes the palette assignment, hence the shared ``color_map`` read by
        distributions and per-group regressions.
    compare_order:
        Explicit order of the ``compare_by`` values (allocations).
    sort_categories:
        Sort categories not otherwise ordered (default: order of appearance).
    labels:
        Display names, flat ``{value: label}`` and/or per-column
        ``{column: {value: label}}``; applied wherever values are printed
        (ticks, legends, titles, brackets).
    """

    # --- allocations / box roles ---
    x: str | None = None  # Horizontal-axis column.
    y: str | None = None  # Value column (bar height / line value / box distribution).
    color: str | None = None  # Colour (allocations); stacked within x for bars.
    compare_by: str | None = None  # Side-by-side / line-style comparison (allocations).
    x_hierarchy: str | None = None  # Coarse x grouping drawn as brackets.
    size: str | None = None  # Bubble-size value column (plot_allocations graph_type="bubble").
    background: str | None = None  # Class column colouring the bubble-heatmap background bands.
    facet_col: str | None = None  # Facet across columns (allocations).
    facet_row: str | None = None  # Facet across rows (allocations).
    # --- grid roles ---
    variables: Sequence[str] | None = None  # One variable per grid line.
    line_by: str | None = None  # Time-series x / year selector (grid).
    scatter_by: str | None = None  # Scatter x column (grid).
    by: str | None = None  # Colour grouping (grid / joint / box).
    line_id: str | None = None  # Per-trajectory id for the data (grid).
    benchmark_line_id: str | Sequence[str] | None = None  # Benchmark trajectory id(s) (grid).
    aggregate: str | None = "mean"  # Reduce duplicate (group, line_by): "mean"/"median"/None.
    # --- joint roles ---
    var_pairs: Sequence[tuple[str, str]] | None = None  # (x, y) pairs, one jointplot each.
    benchmark_by: str | None = None  # Benchmark colour grouping (joint / box).
    # --- fingerprint roles ---
    categories: Mapping[str, Sequence[str]] | None = None  # {category: [indicator columns]}.
    names: Mapping[str, str] | None = None  # {indicator: display name} (legend).
    codes: Mapping[str, str] | None = None  # {indicator: short code} on the spoke.
    category_order: Sequence[str] | None = None  # Explicit category order (fingerprint).
    explanations: Mapping[str, str] | None = None  # Optional per-indicator note.
    footnotes: Mapping[str, str] | None = None  # Optional per-indicator footnote.
    # --- ordering ---
    category_orders: Mapping[str, Sequence[Any]] = field(default_factory=dict)  # {column: order}.
    color_order: Sequence[Any] | None = None  # Colour-category order/subset.
    compare_order: Sequence[Any] | None = None  # compare_by order.
    sort_categories: bool = False  # Sort categories not otherwise ordered.
    # --- labels ---
    labels: Mapping[Any, Any] = field(default_factory=dict)  # Flat {value: label} and/or {column: {value: label}}.

    def lookup(self, value: Any, column: str | None = None) -> Any:
        """Display label for ``value`` (optionally within ``column``); raw value if none."""
        if column is not None and column in self.labels:  # Per-column override may exist.
            per_col = self.labels[column]
            if isinstance(per_col, Mapping):
                return per_col.get(value, value)
        return self.labels.get(value, value)
@dataclass(frozen=True)
class Layout:
    """Figure furniture shared by every chart: titles, legend, margins, sizing.

    Only *universal* knobs live here. Chart-structural options (grid ``columns`` /
    ``variables_by`` / ``year_range`` / ``var_window``; joint ``orientation`` /
    ratios; fingerprint guide) are keyword arguments on their function.

    Parameters
    ----------
    title:
        Figure suptitle.
    x_title:
        Shared x-axis title.
    y_title:
        Shared y-axis title (defaults to the ``y`` column's label).
    x_title_offset:
        Plotly x-title standoff (px).
    x_title_y:
        Matplotlib x-title vertical position (figure fraction).
    y_min:
        Hard lower y limit, or ``None`` (auto) (allocations, joint).
    y_max:
        Hard upper y limit, or ``None`` (auto) (allocations, joint).
    x_min:
        Hard lower x limit, or ``None`` (auto) (joint).
    x_max:
        Hard upper x limit, or ``None`` (auto) (joint).
    legend_y:
        Absolute legend anchor (figure fraction); ``None`` auto-hugs the plot
        top with a ``legend_gap`` gap.
    legend_ncol:
        Explicit legend column count; ``None`` auto-balances.
    legend_max_per_row:
        Above this many items (in auto mode), wrap into two balanced rows.
    legend_gap:
        Gap between the plot top and the legend (figure fraction).
    legend_row_height:
        Reserved height per legend row (figure fraction).
    top_margin:
        Extra top reserve above the legend (figure fraction).
    bottom_margin:
        Bottom reserve for the x title and ticks (figure fraction).
    left_margin:
        Left reserve for the y title (figure fraction).
    col_wrap:
        Facet column wrap (allocations): lay a *single* facet dimension out
        over this many columns. Ignored with a warning when both
        ``facet_col`` and ``facet_row`` are set (the two dimensions already
        determine the grid). Mutually exclusive with ``row_wrap``.
    row_wrap:
        Facet row wrap (allocations); same semantics as ``col_wrap``, over
        rows.
    share_x:
        Share the x-axis across panels / cells.
    share_y:
        Share the y-axis across panels; for the grid, ``"variable"`` (per
        variable line), ``"all"``, or a bool.
    hide_empty:
        Blank facet panels that have no data (allocations).
    panel_letters:
        Tag each cell / panel with a bold letter in its upper-left corner, row
        by row from ``a`` (a, b, c on the first row, d, e, f on the second,
        ...). Honoured by every chart; hidden facet panels don't consume a
        letter, and single-axes charts (boxplot, fingerprint) are tagged ``a``.
    wspace:
        Horizontal gap between facet panels (allocations). Matplotlib: the
        ``subplots_adjust`` fraction of the mean panel width (try 0.3--0.6 for
        long y tick labels). Plotly: passed as ``horizontal_spacing`` (fraction
        of the figure, typically 0.03--0.12). ``None`` -> backend default.
    hspace:
        Vertical gap between facet panels (allocations); same semantics per
        backend as ``wspace``. ``None`` -> backend default.
    figsize_per_panel:
        Matplotlib per-panel size (inches); ``None`` -> the chart's default.
    plotly_size_per_panel:
        Plotly per-panel size (pixels); ``None`` -> the chart's default.
    """

    title: str | None = None  # Figure suptitle.
    x_title: str | None = None  # Shared x-axis title.
    y_title: str | None = None  # Shared y-axis title (defaults to y label).
    x_title_offset: int = 10  # Plotly x-title standoff (px).
    x_title_y: float = 0.02  # Matplotlib x-title vertical position (figure fraction).
    y_min: float | None = None  # Lower y limit, or None.
    y_max: float | None = None  # Upper y limit, or None.
    x_min: float | None = None  # Hard lower x limit (joint).
    x_max: float | None = None  # Hard upper x limit (joint).
    legend_y: float | None = None  # Absolute legend anchor; None auto-hugs (gap = legend_gap).
    legend_ncol: int | None = None  # Explicit legend columns; None auto-balances.
    legend_max_per_row: int = 6  # Above this many items (auto), wrap to two balanced rows.
    legend_gap: float = 0.01  # Gap between the plot top and the legend (figure fraction).
    legend_row_height: float = 0.06  # Reserved height per legend row (figure fraction).
    top_margin: float = 0.04  # Extra top reserve above the legend (figure fraction).
    bottom_margin: float = 0.06  # Bottom reserve (x title + ticks), figure fraction.
    left_margin: float = 0.04  # Left reserve (y title), figure fraction.
    col_wrap: int | None = None  # Facet column wrap (allocations).
    row_wrap: int | None = None  # Facet row wrap (allocations).
    share_x: bool = False  # Share x across panels/cells.
    share_y: bool | str = True  # Share y across panels ("variable"/"all"/bool for the grid).
    hide_empty: bool = True  # Blank empty facet panels (allocations).
    panel_letters: bool = False  # Bold a, b, c... corner tags (all charts).
    wspace: float | None = None  # Horizontal gap between panels (allocations; mpl fraction / plotly spacing).
    hspace: float | None = None  # Vertical gap between panels (allocations; mpl fraction / plotly spacing).
    figsize_per_panel: tuple[float, float] | None = None  # Matplotlib per-panel inches; None -> chart default.
    plotly_size_per_panel: tuple[int, int] | None = None  # Plotly per-panel pixels; None -> chart default.
@dataclass(frozen=True)
class Style:
    """Cosmetics shared by every chart: palette + the line/bar/box/scatter/benchmark vocabulary.

    (The fingerprint's radial geometry is separate: see :class:`FingerprintStyle`.)

    Parameters
    ----------
    palette:
        Colour palette: a named colormap, an explicit colour sequence, or a
        forced ``{category: colour}`` mapping; ``None`` -> the chart's default.
        Together with ``Encoding.color_order`` it determines the shared
        ``color_map`` read by series, distributions and per-group regressions.
    reverse_palette:
        Reverse the sampled colours.
    line_mode:
        ``"lines"`` or ``"lines+markers"`` (allocations line charts).
    line_width:
        Line width.
    marker_size:
        Line-marker size.
    scatter_size:
        Scatter-cell marker size (grid).
    bar_width:
        Bar width (allocations).
    box_width:
        Box / violin width (boxplot).
    group_gap:
        Gap between category groups (allocations / boxplot).
    data_alpha:
        Data face opacity for boxes / violins (boxplot) -- the box analogue of
        the grid's ``dist_group_alpha``.
    alpha:
        Generic marker opacity (joint scatter).
    reference_line_color:
        Reference-overlay marker edge colour (allocations).
    reference_linewidth:
        Reference-overlay marker edge width (allocations).
    compare_hatches:
        Hatch patterns for ``compare_by`` values (allocations bars): a
        sequence or a ``{value: hatch}`` mapping.
    benchmark_color:
        Benchmark colour (distributions, scatter, boxes).
    benchmark_alpha:
        Benchmark distribution / scatter opacity.
    benchmark_line_alpha:
        Benchmark trajectory opacity (grid series cell).
    benchmark_line_width:
        Benchmark trajectory width (grid series cell).
    benchmark_scatter_size:
        Benchmark scatter marker size.
    benchmark_label:
        Benchmark legend label (default ``"benchmark"``); also passed through
        ``encoding.labels``, so ``labels={"benchmark": ...}`` renames it too.
    benchmark_cmap:
        Forced ``{benchmark category: colour}`` mapping (with ``benchmark_by``).
    benchmark_marginal_color:
        Line colour of the single-benchmark marginal (joint).
    show_points:
        Overlay the raw points on boxes / violins (boxplot).
    point_size:
        Overlaid point size (boxplot).
    point_alpha:
        Overlaid point opacity (boxplot).
    point_color:
        Overlaid point colour; ``None`` follows the group colour.
    point_jitter:
        Horizontal jitter of the overlaid points (boxplot).
    hierarchy_line_offset:
        Vertical offset of the ``x_hierarchy`` bracket lines (axes fraction).
    hierarchy_text_offset:
        Vertical offset of the ``x_hierarchy`` labels (axes fraction).
    hierarchy_room:
        Bottom room reserved when ``x_hierarchy`` is drawn (figure fraction).
    font_family:
        Font family, or ``None`` (matplotlib / plotly default).
    font_size:
        Base font size (pt).
    tick_label_size:
        Tick-label size in points for both axes; ``None`` (default) follows
        ``font_size``. Useful when long categorical labels collide (bubble
        heatmap rows, hierarchical x axes).
    rasterized:
        Draw the *data marks* (lines, bars, boxes, scatters, densities, spans)
        as an embedded bitmap in vector output, while text, axes, ticks and
        legends stay vector. Fixes PDF/EPS printing failures on figures with
        very many objects; the bitmap resolution follows the save DPI
        (``SaveSpec.dpi``, or ``dpi=`` in your own ``savefig``), so use >= 300
        for print. Matplotlib only (ignored by the Plotly backend).
    """

    palette: str | Sequence[Any] | Mapping[Any, Any] | None = None  # None -> the chart's default palette.
    reverse_palette: bool = False  # Reverse sampled colours.
    line_mode: str = "lines+markers"  # "lines" or "lines+markers" (allocations).
    line_width: float = 2.0  # Line width.
    marker_size: float = 5.0  # Marker size.
    scatter_size: float = 34.0  # Scatter marker size.
    bar_width: float = 0.9  # Bar width (allocations).
    box_width: float = 0.7  # Box/violin width (boxplot).
    group_gap: float = 0.9  # Gap between groups (allocations/box).
    data_alpha: float = 0.5  # Data fill opacity (box).
    alpha: float = 0.9  # Generic marker opacity (joint).
    reference_line_color: str = "black"  # Reference/scatter marker edge (allocations).
    reference_linewidth: float = 2.0  # Reference marker edge width.
    compare_hatches: Sequence[str] | Mapping[Any, str] | None = None  # Hatches for compare_by (allocations).
    benchmark_color: str = "grey"  # Benchmark colour.
    benchmark_alpha: float = 0.15  # Benchmark distribution/scatter opacity.
    benchmark_line_alpha: float = 0.5  # Benchmark trajectory opacity (grid).
    benchmark_line_width: float = 0.5  # Benchmark trajectory width (grid).
    benchmark_scatter_size: float = 10.0  # Benchmark scatter size.
    benchmark_label: str = "benchmark"  # Benchmark legend label.
    benchmark_cmap: Mapping[Any, Any] | None = None  # Forced {benchmark category: colour}.
    benchmark_marginal_color: str = "black"  # Single-benchmark marginal line colour (joint).
    show_points: bool = False  # Overlay points on boxes/violins (box).
    point_size: float = 36.0  # Overlaid point size (box).
    point_alpha: float = 0.9  # Overlaid point opacity (box).
    point_color: str | None = None  # Overlaid point colour, or None to follow the group.
    point_jitter: float = 0.3  # Overlaid point jitter (box).
    hierarchy_line_offset: float = -0.04  # x_hierarchy bracket line offset.
    hierarchy_text_offset: float = -0.03  # x_hierarchy label offset.
    hierarchy_room: float = 0.22  # Bottom room reserved for x_hierarchy.
    font_family: str | None = None  # Font family, or None.
    font_size: int = 11  # Base font size (pt).
    tick_label_size: int | None = None  # Tick-label size (pt); None -> follow font_size.
    rasterized: bool = False  # Rasterise the data marks in vector output (Matplotlib only).
@dataclass(frozen=True)
class Regression:
    """LOESS regression for the scatter cell (``multiplot_grid``) and joint scatter.

    Parameters
    ----------
    data:
        Draw a LOESS through the data scatter.
    benchmark:
        Draw a LOESS through the benchmark scatter.
    loess_frac:
        LOESS smoothing fraction (share of points in each local fit).
    robust_iter:
        Robustifying iterations (0 = ordinary local mean, no outlier
        down-weighting).
    per_group:
        One LOESS per ``by`` category instead of a pooled fit; colours come
        from the same ``color_map`` as the distributions. Implies the data
        fit: ``Regression(per_group=True)`` alone is enough (no need to also
        set ``data=True``).
    data_color:
        Pooled data-regression colour (ignored when ``per_group=True``).
    benchmark_color:
        Benchmark-regression colour.
    dash:
        Dash style of the regression lines.
    line_width:
        Width of the regression lines.
    """

    data: bool = False  # LOESS through the data scatter.
    benchmark: bool = False  # LOESS through the benchmark scatter.
    loess_frac: float = 0.66  # LOESS smoothing fraction.
    robust_iter: int = 0  # Robustifying iterations (0 = ordinary local mean, no outlier hypothesis).
    per_group: bool = False  # One LOESS per `by` category instead of a pooled fit.
    data_color: str = "black"  # Data regression colour.
    benchmark_color: str = "firebrick"  # Benchmark regression colour.
    dash: str = "--"  # Dash style.
    line_width: float = 3.0  # Line width.
@dataclass(frozen=True)
class FingerprintStyle:
    """Radial geometry + cosmetics for :func:`fingerprint_diagram` (kept separate).

    Parameters
    ----------
    palette:
        Category colours: named colormap, colour sequence, or forced
        ``{category: colour}`` mapping.
    reverse_palette:
        Reverse the sampled colours.
    r_border:
        Outer ring radius (whisker fence, Q25 - / Q75 + ``iqr_mult`` x IQR).
    r_q75:
        Upper-quartile ring radius.
    r_median:
        Median ring radius.
    r_q25:
        Lower-quartile ring radius.
    ring_lw_border:
        Outer ring line width.
    ring_lw_median:
        Median ring line width.
    ring_lw_quartile:
        Quartile ring line width.
    arc_lw:
        Category arc line width.
    arc_gap_deg:
        Gap at each end of a category arc (degrees).
    spoke_style:
        Spoke line style.
    spoke_color:
        Spoke colour.
    spoke_radius:
        Spoke length.
    show_center_marker:
        Draw the central dot.
    show_codes:
        Print the indicator codes at the spoke tips.
    code_fontsize:
        Spoke-code font size.
    code_radius:
        Spoke-code radial position.
    show_category_labels:
        Draw the curved category labels.
    category_label_fontsize:
        Category-label font size.
    category_label_radius:
        Category-label radial position.
    figsize:
        Figure size (inches).
    font_family:
        Font family, or ``None``.
    font_size:
        Base font size (pt).
    rasterized:
        Rasterise the diagram's marks in vector output, keeping text vector
        (see :attr:`Style.rasterized`).
    """

    palette: str | Sequence[Any] | Mapping[Any, Any] = "tab10"  # Category colours.
    reverse_palette: bool = False  # Reverse sampled colours.
    r_border: float = 0.40  # Outer ring radius.
    r_q75: float = 0.29  # Upper-quartile ring radius.
    r_median: float = 0.20  # Median ring radius.
    r_q25: float = 0.135  # Lower-quartile ring radius.
    ring_lw_border: float = 18.0  # Outer ring width.
    ring_lw_median: float = 9.0  # Median ring width.
    ring_lw_quartile: float = 2.5  # Quartile ring width.
    arc_lw: float = 14.0  # Category arc width.
    arc_gap_deg: float = 3.0  # Gap at each end of a category arc (deg).
    spoke_style: str = ":"  # Spoke line style.
    spoke_color: str = "silver"  # Spoke colour.
    spoke_radius: float = 0.39  # Spoke length.
    show_center_marker: bool = True  # Draw the central dot.
    show_codes: bool = True  # Draw the spoke codes.
    code_fontsize: float = 22.0  # Spoke-code font size.
    code_radius: float = 0.45  # Spoke-code radius.
    show_category_labels: bool = True  # Draw curved category labels.
    category_label_fontsize: float = 24.0  # Category-label font size.
    category_label_radius: float = 0.36  # Category-label radius.
    figsize: tuple[float, float] = (18.0, 12.0)  # Figure size (inches).
    font_family: str | None = None  # Font family, or None.
    font_size: int = 14  # Base font size (pt).
    rasterized: bool = False  # Rasterise the data marks in vector output (see Style.rasterized).
