"""
Rendering layer for the plotting API (explainability study).

Holds the two backends and the figure-export helpers. Key changes versus the
original:

* **No global state mutation.** The Matplotlib path runs inside
  :func:`matplotlib.rc_context`; the original wrote to ``plt.rcParams`` and
  called ``sns.set_theme`` globally, leaking style into the user's session.
* **Correct ``share_y`` + ``x_hierarchy``.** Y-limits are finalised *before* the
  hierarchy brackets are drawn, so brackets stay correctly positioned under a
  shared range.
* **Explicit Plotly limitation.** ``x_hierarchy`` triggers a warning instead of
  being silently dropped.
"""

from __future__ import annotations  # String annotations.

import math  # Trigonometry for the radial fingerprint geometry.
import os  # Path handling for saving.
import warnings  # To surface the Plotly x_hierarchy limitation.
from typing import Any  # Documentation alias.

import matplotlib as mpl  # rc_context + colour utilities.
import matplotlib.patches as mpatches  # Legend swatches, arcs, circles.
import matplotlib.patheffects as pe  # Black outline for highlighted trajectories.
import matplotlib.pyplot as plt  # Figure/axes + Line2D.
import matplotlib.text as mtext  # Base class for the curved category labels.
import numpy as np  # NaN handling, marker-size conversion.
import pandas as pd  # Panel data type.
import plotly.graph_objects as go  # Plotly trace primitives.
import seaborn as sns  # Source of the "white" style rcParams.
from plotly.subplots import make_subplots  # Plotly faceted grid.

from .prepare import (  # Plan + helpers (allocations, grid, joint).
    BoxPlan,
    FingerprintPlan,
    GridPlan,
    JointPlan,
    RenderPlan,
    compute_axis_limits,
    filter_panel,
    filter_year,
    panel_title,
    regression_color,
)
from ._context import (  # Internal context bundles.
    BoxContext,
    FingerprintContext,
    GridContext,
    JointContext,
    PlotContext,
)
from .specs import PlotResult  # Public result type.


# ===========================================================================
# Saving (original auxil._save_*, with defensive HTML handling)
# ===========================================================================
def save_matplotlib(fig, path: str, name: str, extensions, dpi: int) -> list[str]:
    """Write a Matplotlib figure to one file per extension.

    Parameters
    ----------
    fig:
        The figure to save.
    path, name:
        Output directory and base filename.
    extensions:
        Extensions to write.
    dpi:
        Raster resolution (png only).

    Returns
    -------
    list of str
        Paths written.
    """
    os.makedirs(path, exist_ok=True)  # Ensure directory.
    written: list[str] = []  # Collect paths.
    for ext in extensions:  # One file per format.
        out = os.path.join(path, f"{name}.{ext}")  # Full path.
        if ext.lower() == "png":  # Only raster needs dpi.
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        else:  # Vector formats.
            fig.savefig(out, bbox_inches="tight")
        written.append(out)  # Record.
    return written


def save_plotly(fig, path: str, name: str, extensions) -> list[str]:
    """Write a Plotly figure to one file per extension.

    HTML uses ``write_html`` (engine-free); other formats use ``write_image``
    via the kaleido engine.

    Parameters
    ----------
    fig:
        The figure to save.
    path, name:
        Output directory and base filename.
    extensions:
        Extensions to write.

    Returns
    -------
    list of str
        Paths written.
    """
    os.makedirs(path, exist_ok=True)  # Ensure directory.
    written: list[str] = []  # Collect paths.
    for ext in extensions:  # One file per format.
        out = os.path.join(path, f"{name}.{ext}")  # Full path.
        if ext.lower() == "html":  # Interactive export.
            fig.write_html(out)
        else:  # Static export (needs kaleido).
            fig.write_image(out, format=ext, engine="kaleido")
        written.append(out)  # Record.
    return written


# ===========================================================================
# Matplotlib backend
# ===========================================================================
def _rasterize_data(fig, enabled: bool) -> None:
    """Rasterise every data artist of ``fig`` (Style.rasterized), keeping text vector.

    Only the mark artists -- collections, patches, lines, images -- are flagged;
    titles, tick labels, annotations and legends are left as vector objects, so
    the output stays searchable and sharp while the object count collapses.
    Call once, after everything is drawn. The bitmap resolution is the DPI used
    when saving.
    """
    if not enabled:
        return
    for ax in fig.get_axes():  # SubFigure axes are included in fig.get_axes().
        for art in [*ax.collections, *ax.patches, *ax.lines, *ax.images]:
            art.set_rasterized(True)


def _style_axis(ax, tick_label_size: int | None = None) -> None:
    """Apply the shared publication styling to one axis.

    ``tick_label_size`` (points) overrides the inherited font size on both
    axes' tick labels; ``None`` leaves them alone.
    """
    ax.grid(axis="y", color="lightgrey", linewidth=0.75)  # Horizontal gridlines only.
    ax.grid(axis="x", visible=False)  # No vertical grid.
    ax.spines["top"].set_visible(False)  # Drop top spine.
    ax.spines["right"].set_visible(False)  # Drop right spine.
    ax.spines["left"].set_linewidth(1.4)  # Emphasise left spine.
    ax.spines["bottom"].set_linewidth(1.4)  # Emphasise bottom spine.
    ax.tick_params(axis="y", labelleft=True)  # Keep y labels (even when shared).
    if tick_label_size is not None:  # Explicit tick-label size (both axes).
        ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.axhline(0, color="black", linewidth=0.8)  # Baseline at y=0.


def _hide_axis(ax) -> None:
    """Blank an unused / empty axis completely."""
    ax.set_frame_on(False)  # No frame.
    ax.set_xticks([])  # No x ticks.
    ax.set_yticks([])  # No y ticks.
    ax.set_xlabel("")  # No x label.
    ax.set_ylabel("")  # No y label.
    ax.set_title("")  # No title.
    ax.grid(False)  # No grid.
    for spine in ax.spines.values():  # Hide every spine.
        spine.set_visible(False)


def _draw_bar_panel(ax, ctx: PlotContext, plan: RenderPlan, row_val: Any, col_val: Any) -> None:
    """Draw stacked, grouped bars (and reference markers) for one panel."""
    enc = ctx.encoding  # Shorthand.
    for xi, xval in enumerate(plan.x_order):  # Each x group.
        for ci, comp in enumerate(plan.compare_order):  # Each compare bar in the group.
            pos = plan.xpos[xi] + ci  # This bar's x position.
            pos_bottom = 0.0  # Running top of the positive stack.
            neg_bottom = 0.0  # Running bottom of the negative stack.
            for c in plan.color_order:  # Stack colour segments.
                val = plan.agg_main.get(facet_row=row_val, facet_col=col_val,
                                        x=xval, color=c, compare=comp)  # O(1) height.
                if val is None:  # No data here.
                    continue
                if val >= 0:  # Positive -> stack up.
                    bottom = pos_bottom
                    pos_bottom += val
                else:  # Negative -> stack down.
                    bottom = neg_bottom
                    neg_bottom += val
                ax.bar(pos, val, bottom=bottom, width=ctx.style.bar_width,  # The segment.
                       color=plan.color_map[c], edgecolor="black", linewidth=0.6,
                       hatch=plan.compare_styles.hatch.get(comp, ""))

    if ctx.scatter is not None:  # Reference markers.
        for xi, xval in enumerate(plan.x_order):
            for ci, comp in enumerate(plan.compare_order):
                val = plan.agg_ref.get(facet_row=row_val, facet_col=col_val, x=xval, compare=comp)
                if val is None:
                    continue
                pos = plan.xpos[xi] + ci  # Aligns with the bar (ci==0 when no compare).
                ax.scatter(pos, val, s=ctx.scatter.size, facecolor=ctx.scatter.facecolor,
                           edgecolor=ctx.scatter.edgecolor, linewidth=ctx.scatter.linewidth, zorder=5)

    ax.set_xticks(plan.centers)  # Ticks at group centres.
    ax.set_xticklabels([ctx.labels.lookup(v, enc.x) for v in plan.x_order], rotation=0)  # Labels.


def _bubble_key(ctx: PlotContext, plan: RenderPlan, scope: str, row_val, col_val, store: dict):
    """Scope key for one panel, falling back to the figure-wide key when absent."""
    if scope == "row" and ctx.facets.row is not None:
        key = row_val
    elif scope == "col" and ctx.facets.col is not None:
        key = col_val
    elif scope == "panel" and (ctx.facets.row is not None or ctx.facets.col is not None):
        key = tuple(v for v, c in ((row_val, ctx.facets.row), (col_val, ctx.facets.col))
                    if c is not None)
    else:
        key = None
    return key if key in store else None  # Absent dimension -> the figure-wide entry.


def _bubble_bg_palette(ctx: PlotContext, plan: RenderPlan) -> dict:
    """Resolve {class: colour} for the background bands (user palette + tab10 fallback)."""
    _, mapping = plan.bubble_background
    classes = list(dict.fromkeys(mapping.values()))
    user = dict(ctx.bubble_background_palette or {})
    cycle = [mpl.cm.tab10(i % 10) for i in range(len(classes))]
    return {c: user.get(c, cycle[i]) for i, c in enumerate(classes)}


def _draw_bubble_panel(ax, ctx: PlotContext, plan: RenderPlan, row_val: Any, col_val: Any) -> None:
    """Draw one bubble-heatmap panel: one bubble per (x, y) cell.

    Colour and area encode ``encoding.color`` / ``encoding.size`` with the
    figure-wide scales precomputed in ``plan.bubble_range`` so every facet
    reads on the same colormap and size scale.
    """
    enc = ctx.encoding  # Shorthand.
    rng_key = _bubble_key(ctx, plan, ctx.bubble_scope, row_val, col_val, plan.bubble_range)
    vmin, vmax, smin, smax = plan.bubble_range[rng_key]
    ax_key = _bubble_key(ctx, plan, ctx.bubble_axes_scope, row_val, col_val, plan.bubble_axes)
    x_order, y_order = plan.bubble_axes[ax_key]
    pdf = filter_panel(plan.prepared.full, ctx, row_val, col_val)
    vcols = [c for c in dict.fromkeys((enc.color, enc.size)) if c is not None]
    cell = pdf.groupby([enc.x, enc.y], observed=True)[vcols].agg(ctx.bubble_agg).reset_index()
    xi = {v: i for i, v in enumerate(x_order)}
    yi = {v: i for i, v in enumerate(y_order)}
    cell = cell[cell[enc.x].isin(xi) & cell[enc.y].isin(yi)]
    if not cell.empty:
        s0, s1 = ctx.bubble_sizes
        if enc.size is not None and smax > smin:  # Linear area scale on the global range.
            sizes = s0 + (cell[enc.size].values - smin) / (smax - smin) * (s1 - s0)
        else:
            sizes = np.full(len(cell), 0.5 * (s0 + s1))
        colour_kw = (dict(c=cell[enc.color].values, cmap=ctx.bubble_cmap, vmin=vmin, vmax=vmax)
                     if enc.color is not None else dict(color=ctx.bubble_color))
        ax.scatter([xi[v] for v in cell[enc.x]], [yi[v] for v in cell[enc.y]],
                   s=sizes, edgecolor="black", linewidth=0.6, alpha=0.9, zorder=3,
                   **colour_kw)
    for xg in range(len(x_order)):  # Vertical guides (immune to the shared axis styling).
        ax.axvline(xg, color="0.9", linewidth=0.7, zorder=0)
    if plan.bubble_background is not None:  # Class-coloured background bands.
        axis_name, mapping = plan.bubble_background
        pal = _bubble_bg_palette(ctx, plan)
        cats = x_order if axis_name == "x" else y_order
        span = ax.axvspan if axis_name == "x" else ax.axhspan
        for i, cat in enumerate(cats):
            cls = mapping.get(cat)
            if cls is not None:
                span(i - 0.5, i + 0.5, color=pal[cls], alpha=ctx.bubble_background_alpha,
                     linewidth=0, zorder=0.3)
    ax.set_xticks(range(len(x_order)),
                  [str(ctx.labels.lookup(v, enc.x)) for v in x_order],
                  rotation=getattr(ctx.style, "xtick_rotation", 0) or 0)
    ax.set_yticks(range(len(y_order)),
                  [str(ctx.labels.lookup(v, enc.y)) for v in y_order])
    ax.set_xlim(-0.6, len(x_order) - 0.4)  # Breathing room around the outer bubbles.
    ax.set_ylim(-0.6, len(y_order) - 0.4)
    ax.grid(True, which="major", color="0.9", linewidth=0.7, zorder=0)


def _draw_line_panel(ax, ctx: PlotContext, plan: RenderPlan, row_val: Any, col_val: Any) -> None:
    """Draw one line per ``(color, compare)`` series (and any reference line)."""
    enc = ctx.encoding  # Shorthand.
    x_numeric = np.arange(len(plan.x_order))  # Evenly spaced numeric x.
    marker = "o" if ctx.style.line_mode == "lines+markers" else None  # Optional markers.

    for c in plan.color_order:  # Colour family.
        for comp in plan.compare_order:  # Compare style.
            ys = [plan.agg_main.get(facet_row=row_val, facet_col=col_val, x=xval, color=c, compare=comp)
                  for xval in plan.x_order]  # Series values (None where missing).
            if all(v is None for v in ys):  # Empty series.
                continue
            ax.plot(x_numeric, [np.nan if v is None else v for v in ys],  # Gaps -> NaN.
                    color=plan.color_map[c], linestyle=plan.compare_styles.linestyle.get(comp, "-"),
                    marker=marker, linewidth=2)

    if ctx.scatter is not None:  # Reference line.
        for comp in plan.compare_order:
            ys = [plan.agg_ref.get(facet_row=row_val, facet_col=col_val, x=xval, compare=comp)
                  for xval in plan.x_order]
            if all(v is None for v in ys):
                continue
            ax.plot(x_numeric, [np.nan if v is None else v for v in ys],
                    color=ctx.style.reference_line_color,
                    linestyle=plan.compare_styles.linestyle.get(comp, "-"),
                    linewidth=ctx.style.reference_linewidth, zorder=5)

    ax.set_xticks(x_numeric)  # Ticks per category.
    ax.set_xticklabels([ctx.labels.lookup(v, enc.x) for v in plan.x_order], rotation=0)  # Labels.


def _draw_hierarchy(ax, ctx: PlotContext, plan: RenderPlan, panel_df: pd.DataFrame,
                    lo: float, hi: float) -> None:
    """Draw x-hierarchy bracket annotations beneath a bar panel.

    Called *after* the y-limits are finalised; positions are relative to the
    final range ``(lo, hi)`` and placed just below ``lo`` (in the reserved room),
    so they stay correct under any shared-range rescaling.

    Parameters
    ----------
    ax:
        Target axis.
    ctx, plan:
        Context and plan.
    panel_df:
        This panel's rows (to map each x to its sector).
    lo, hi:
        Final data y-limits *before* the bracket room was added.
    """
    enc = ctx.encoding  # Shorthand.
    if enc.x_hierarchy is None or ctx.graph_type == "line":  # Bar charts only.
        return

    sector_of = {}  # x value -> sector label.
    for xval in plan.x_order:  # Inspect each x.
        vals = panel_df.loc[panel_df[enc.x].eq(xval), enc.x_hierarchy].dropna().unique()  # Sector(s).
        sector_of[xval] = vals[0] if len(vals) else ""  # First, or empty.

    sectors = []  # Contiguous runs (label, i_start, i_end).
    current, start = None, 0  # Run accumulator.
    for i, xval in enumerate(plan.x_order):
        sec = sector_of[xval]
        if current is None:  # Open first run.
            current, start = sec, i
        elif sec != current:  # Close previous, open new.
            sectors.append((current, start, i - 1))
            current, start = sec, i
    sectors.append((current, start, len(plan.x_order) - 1))  # Close last run.

    span = hi - lo  # Range used to scale offsets.
    line_y = lo + ctx.style.hierarchy_line_offset * span  # Bracket line just below the data.
    text_y = lo + (ctx.style.hierarchy_line_offset + ctx.style.hierarchy_text_offset) * span  # Label.

    for sec, i0, i1 in sectors:  # One bracket per run.
        x0 = plan.centers[i0] - plan.n_compare / 2  # Left edge.
        x1 = plan.centers[i1] + plan.n_compare / 2  # Right edge.
        ax.plot([x0, x1], [line_y, line_y], color="black", lw=1.0, clip_on=False)  # Bracket.
        ax.text((x0 + x1) / 2, text_y, ctx.labels.lookup(sec, enc.x_hierarchy),  # Label.
                ha="center", va="top", clip_on=False)


def _legend_handles(ctx: PlotContext, plan: RenderPlan) -> list:
    """Build a factored legend: colours, then compare styles, then reference."""
    handles = [mpatches.Patch(color=plan.color_map[c],  # One swatch per colour.
                              label=str(ctx.labels.lookup(c, ctx.encoding.color)))
               for c in plan.color_order]

    if ctx.encoding.compare_by is not None:  # Compare sub-legend.
        if ctx.graph_type == "bar":  # Hatched swatches.
            handles.extend(mpatches.Patch(facecolor="white", edgecolor="black",
                                          hatch=plan.compare_styles.hatch.get(comp, ""),
                                          label=str(ctx.labels.lookup(comp, ctx.encoding.compare_by)))
                           for comp in plan.compare_order)
        else:  # Line samples.
            handles.extend(plt.Line2D([0], [0], color="black",
                                      linestyle=plan.compare_styles.linestyle.get(comp, "-"),
                                      linewidth=2,
                                      label=str(ctx.labels.lookup(comp, ctx.encoding.compare_by)))
                           for comp in plan.compare_order)

    if ctx.scatter is not None:  # Reference entry.
        ref_label = str(ctx.labels.lookup(ctx.scatter.value, ctx.scatter.column))
        if ctx.graph_type == "bar":  # Hollow marker.
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                      markerfacecolor=ctx.scatter.facecolor,
                                      markeredgecolor=ctx.scatter.edgecolor,
                                      markeredgewidth=ctx.scatter.linewidth,
                                      markersize=np.sqrt(ctx.scatter.size), label=ref_label))
        else:  # Thick line.
            handles.append(plt.Line2D([0], [0], color=ctx.style.reference_line_color,
                                      linestyle="-", linewidth=ctx.style.reference_linewidth,
                                      label=ref_label))
    return handles


def _legend_ncol(n_items: int, layout) -> int:
    """Number of columns for a top legend.

    Explicit ``layout.legend_ncol`` wins; otherwise a single row up to
    ``layout.legend_max_per_row`` items, then two balanced rows (``ceil(n/2)``).
    """
    if layout.legend_ncol is not None:  # Explicit override.
        return max(1, layout.legend_ncol)
    if n_items > layout.legend_max_per_row:  # Too many: split into two balanced rows.
        return int(np.ceil(n_items / 2))
    return max(1, n_items)  # Single row.


def _finalize_top_legend(fig, handles, layout, labels=None, title=None) -> None:
    """Reserve top room, run ``tight_layout`` with the configured margins, and
    place a balanced legend just above the plot.

    The axes block is fitted into a rectangle whose top leaves room for the
    legend band (``legend_row_height`` per row) plus ``top_margin``; the legend
    is then anchored ``legend_gap`` above that top (so the gap is explicit and
    figure-size independent). Passing ``layout.legend_y`` (a float) restores the
    old fixed ``upper center`` anchor. With no handles, only the layout is set.
    """
    n = len(handles)  # Number of legend entries.
    ncol = _legend_ncol(n, layout) if n else 1  # Balanced columns.
    n_rows = int(np.ceil(n / ncol)) if n else 0  # Legend rows actually used.
    legend_band = layout.legend_row_height * n_rows  # Top room for the legend.

    top = max(0.5, 1.0 - layout.top_margin - legend_band)  # Axes-block top (keep it sane).
    fig.tight_layout(rect=[layout.left_margin, layout.bottom_margin, 1.0, top])  # Fit the panels.

    if not n:  # Nothing to place.
        return
    if layout.legend_y is None:  # Auto: hug the plot, gap = legend_gap.
        anchor, loc = top + layout.legend_gap, "lower center"
    else:  # Absolute anchor (legacy behaviour).
        anchor, loc = layout.legend_y, "upper center"
    kwargs = dict(handles=handles, frameon=False, loc=loc, bbox_to_anchor=(0.5, anchor),
                  ncol=ncol, handlelength=1.8, columnspacing=1.2)
    if labels is not None:  # Boxplot passes labels separately; allocations bake them into handles.
        kwargs["labels"] = labels
    if title:  # Optional legend title (e.g. the grid's `by` column name).
        kwargs["title"] = title
    fig.legend(**kwargs)


def render_matplotlib(ctx: PlotContext, plan: RenderPlan) -> PlotResult:
    """Render the full faceted figure with Matplotlib.

    Parameters
    ----------
    ctx, plan:
        Validated context and precomputed plan.

    Returns
    -------
    PlotResult
        Figure, axes array and any written file paths.
    """
    rc = dict(sns.axes_style("white"))  # Seaborn "white" rcParams (no global side effect).
    rc["font.size"] = ctx.style.font_size  # Requested base font size.
    if ctx.style.font_family is not None:  # Optional family override.
        rc["font.family"] = ctx.style.font_family

    with mpl.rc_context(rc):  # Scoped style: leaves global rcParams untouched.
        fig, axes = plt.subplots(  # Build the grid.
            nrows=plan.nrows, ncols=plan.ncols,
            figsize=((ctx.layout.figsize_per_panel or (7.0, 5.0))[0] * plan.ncols,
                     (ctx.layout.figsize_per_panel or (7.0, 5.0))[1] * plan.nrows),
            squeeze=False, sharey=ctx.facets.share_y,
        )

        drawn: list[tuple[Any, Any, Any, pd.DataFrame]] = []  # Visible (ax, row, col, panel_df).

        for k, (row_val, col_val) in enumerate(plan.panels):  # Row-major layout.
            ri, ci = divmod(k, plan.ncols)  # Grid position.
            ax = axes[ri, ci]  # Target axis.
            panel_df = filter_panel(plan.prepared.full, ctx, row_val, col_val)  # Emptiness test.
            if ctx.facets.hide_empty and panel_df.empty:  # Blank empty panels.
                _hide_axis(ax)
                continue
            if ctx.graph_type == "bar":  # Dispatch on graph type.
                _draw_bar_panel(ax, ctx, plan, row_val, col_val)
            elif ctx.graph_type == "bubble":
                _draw_bubble_panel(ax, ctx, plan, row_val, col_val)
            else:
                _draw_line_panel(ax, ctx, plan, row_val, col_val)
            _style_axis(ax, ctx.style.tick_label_size)  # Shared styling.
            if ctx.graph_type == "bubble" and ci > 0 \
                    and ctx.bubble_axes_scope not in ("col", "panel"):
                ax.tick_params(axis="y", labelleft=False)  # Shared categorical y: label column 1 only.
            ax.set_title(panel_title(ctx, row_val, col_val), fontsize=ctx.style.font_size + 1,
                         fontweight="bold", pad=8)  # Panel title.
            drawn.append((ax, row_val, col_val, panel_df))  # Remember for the limit pass.

        if ctx.layout.panel_letters:  # Bold a, b, c... on visible panels only.
            for idx, (ax_, _, _, _) in enumerate(drawn):
                ax_.text(0.02, 0.98, _panel_letter(idx), transform=ax_.transAxes,
                         fontweight="bold", fontsize=ctx.style.font_size + 2,
                         va="top", ha="left", zorder=10)

        for k in range(len(plan.panels), plan.nrows * plan.ncols):  # Hide trailing cells.
            ri, ci = divmod(k, plan.ncols)
            _hide_axis(axes[ri, ci])

        # Finalise y-limits BEFORE drawing hierarchy brackets (the original bug).
        y_min, y_max = ctx.layout.y_min, ctx.layout.y_max  # Optional hard limits.
        has_hier = ctx.encoding.x_hierarchy is not None and ctx.graph_type == "bar"  # Need room?

        if ctx.facets.share_y and drawn:  # Unify across visible panels.
            lo = min(ax.get_ylim()[0] for ax, *_ in drawn)  # Lowest bottom.
            hi = max(ax.get_ylim()[1] for ax, *_ in drawn)  # Highest top.
            if y_min is not None:
                lo = y_min
            if y_max is not None:
                hi = y_max
            room = ctx.style.hierarchy_room * (hi - lo) if has_hier else 0.0  # Bracket room.
            for ax, row_val, col_val, panel_df in drawn:  # Apply shared range to all.
                ax.set_ylim(lo - room, hi)
                if ctx.graph_type != "bubble":  # Bubble: the left column carries the labels.
                    ax.tick_params(axis="y", labelleft=True)
                _draw_hierarchy(ax, ctx, plan, panel_df, lo, hi)  # Brackets vs final (lo, hi).
        else:  # Independent y-axes.
            for ax, row_val, col_val, panel_df in drawn:
                lo, hi = ax.get_ylim()  # This panel's range.
                if y_min is not None:
                    lo = y_min
                if y_max is not None:
                    hi = y_max
                room = ctx.style.hierarchy_room * (hi - lo) if has_hier else 0.0
                ax.set_ylim(lo - room, hi)
                _draw_hierarchy(ax, ctx, plan, panel_df, lo, hi)

        # Figure-level decorations.
        fig.suptitle(ctx.layout.title or "", fontsize=ctx.style.font_size + 4, fontweight="bold", y=0.995)
        fig.supxlabel(ctx.layout.x_title if ctx.layout.x_title is not None else "",
                      fontweight="bold", y=ctx.layout.x_title_y)  # Configurable x-title position.
        fig.supylabel(ctx.layout.y_title if ctx.layout.y_title is not None
                      else ctx.labels.lookup(ctx.encoding.y), fontweight="bold", x=0.01)

        if ctx.graph_type == "bubble":  # Continuous scales: colorbar + size-reference legend.
            lay_ = ctx.layout  # Same margin vocabulary as the bar/line legend finalizer.
            band = lay_.legend_row_height if ctx.encoding.size is not None else 0.0
            block_top = max(0.5, 1.0 - lay_.top_margin - band)  # Axes-block top.
            fig.subplots_adjust(top=block_top, bottom=lay_.bottom_margin,  # Before the colorbars.
                                left=max(lay_.left_margin, 0.08),
                                wspace=lay_.wspace, hspace=lay_.hspace)
            groups: dict = {}  # {scope key: [axes]} -- one colour/size key per group.
            for ax_, rv, cv, _ in drawn:
                groups.setdefault(
                    _bubble_key(ctx, plan, ctx.bubble_scope, rv, cv, plan.bubble_range),
                    []).append(ax_)
            single = len(groups) == 1
            for key, gaxes in groups.items():
                vmin, vmax, smin, smax = plan.bubble_range[key]
                if ctx.encoding.color is not None:  # No colour column -> no colorbar.
                    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                                               cmap=ctx.bubble_cmap)
                    cbar = fig.colorbar(sm, ax=gaxes, shrink=0.85, pad=0.02)
                    cbar.set_label(str(ctx.labels.lookup(ctx.encoding.color)),
                                   fontsize=ctx.style.font_size)
                if ctx.encoding.size is None or not smax > smin:
                    continue
                s0, s1 = ctx.bubble_sizes  # 3 reference bubbles for this group.
                refs = [smin, 0.5 * (smin + smax), smax]
                handles = [plt.matplotlib.lines.Line2D(
                    [], [], marker="o", linestyle="", color="gray",
                    markeredgecolor="black", markeredgewidth=0.6,
                    markersize=np.sqrt(s0 + (r - smin) / (smax - smin) * (s1 - s0)),
                    label=f"{r:.3g}") for r in refs]
                if single:  # Figure-wide key: the usual top-centre placement.
                    has_bg = plan.bubble_background is not None
                    y_anchor = (block_top + lay_.legend_gap if lay_.legend_y is None
                                else lay_.legend_y)  # Explicit, figure-size independent gap.
                    anchor = (0.32 if has_bg else 0.5, y_anchor)
                else:  # Per-group key: anchored just above that group's panels.
                    boxes = [a.get_position() for a in gaxes]
                    anchor = (float(np.mean([(b.x0 + b.x1) / 2 for b in boxes])),
                              float(max(b.y1 for b in boxes)) + 0.012)
                fig.legend(handles=handles, loc="lower center", bbox_to_anchor=anchor,
                           ncol=3, frameon=False, fontsize=ctx.style.font_size - 1,
                           title=str(ctx.labels.lookup(ctx.encoding.size)),
                           title_fontsize=ctx.style.font_size - 1)
            if plan.bubble_background is not None:  # Discrete class legend for the bands.
                pal = _bubble_bg_palette(ctx, plan)
                patches = [mpatches.Patch(facecolor=colr, alpha=ctx.bubble_background_alpha + 0.25,
                                          label=str(ctx.labels.lookup(cls, ctx.encoding.background)))
                           for cls, colr in pal.items()]
                bg_x = 0.68 if (ctx.encoding.size is not None and len(groups) == 1) else 0.5
                bg_y = (block_top + lay_.legend_gap if lay_.legend_y is None else lay_.legend_y)
                fig.legend(handles=patches, loc="lower center",
                           bbox_to_anchor=(bg_x, bg_y),
                           ncol=min(len(patches), 4), frameon=False,
                           title=str(ctx.labels.lookup(ctx.encoding.background)))
        else:
            handles = _legend_handles(ctx, plan)  # Factored legend (labels baked into the handles).
            _finalize_top_legend(fig, handles, ctx.layout)  # Balanced legend + margin-driven layout.
            if ctx.layout.wspace is not None or ctx.layout.hspace is not None:
                fig.subplots_adjust(wspace=ctx.layout.wspace, hspace=ctx.layout.hspace)

        _rasterize_data(fig, ctx.style.rasterized)  # Printable vector output.
        files: list[str] = []  # Saved paths.
        if ctx.save is not None:  # Optional export.
            files = save_matplotlib(fig, ctx.save.path, ctx.save.name, ctx.save.extensions, ctx.save.dpi)

    return PlotResult(figure=fig, axes=axes, files=files, backend="matplotlib")  # Uniform result.


# ===========================================================================
# Plotly backend
# ===========================================================================
def _hide_empty_axis_plotly(fig: go.Figure, ri: int, ci: int, ncols: int) -> None:
    """Blank an empty Plotly subplot cell and clear its title."""
    idx = (ri - 1) * ncols + (ci - 1)  # Subplot titles are ordered annotations.
    if idx < len(fig.layout.annotations):  # Guard.
        fig.layout.annotations[idx].text = ""  # Clear title.
    fig.update_xaxes(showticklabels=False, ticks="", showline=False, showgrid=False,
                     zeroline=False, row=ri, col=ci)  # Strip x-axis.
    fig.update_yaxes(showticklabels=False, ticks="", showline=False, showgrid=False,
                     zeroline=False, row=ri, col=ci)  # Strip y-axis.


def _series_label(ctx: PlotContext, c: Any, comp: Any) -> str:
    """Compose a combined ``colour . compare`` legend label."""
    color_label = ctx.labels.lookup(c, ctx.encoding.color)  # Colour label.
    if comp is None:  # No compare dimension.
        return str(color_label)
    return f"{color_label} \u00b7 {ctx.labels.lookup(comp, ctx.encoding.compare_by)}"  # "colour . compare".


def render_plotly(ctx: PlotContext, plan: RenderPlan) -> PlotResult:
    """Render the full faceted figure with Plotly.

    Parameters
    ----------
    ctx, plan:
        Validated context and precomputed plan.

    Returns
    -------
    PlotResult
        Figure (no axes) and any written file paths.
    """
    if ctx.encoding.x_hierarchy is not None:  # Matplotlib-only feature.
        warnings.warn("x_hierarchy is not supported by the Plotly backend and will be ignored; "
                      "use backend='matplotlib' for hierarchy brackets.", stacklevel=2)

    subplot_titles = [panel_title(ctx, r, c) for r, c in plan.panels]  # One per panel.
    subplot_titles += [""] * (plan.nrows * plan.ncols - len(subplot_titles))  # Pad trailing cells.

    fig = make_subplots(rows=plan.nrows, cols=plan.ncols,  # The faceted grid.
                        subplot_titles=subplot_titles, shared_yaxes=ctx.facets.share_y,
                        horizontal_spacing=ctx.layout.wspace, vertical_spacing=ctx.layout.hspace)

    shown: set = set()  # Legend de-duplication across panels.

    n_letter = 0  # Visible-panel counter for Layout.panel_letters.
    for k, (row_val, col_val) in enumerate(plan.panels):  # Row-major.
        ri, ci = divmod(k, plan.ncols)  # 0-based.
        ri, ci = ri + 1, ci + 1  # Plotly is 1-based.
        panel_df = filter_panel(plan.prepared.full, ctx, row_val, col_val)  # Emptiness test.
        if ctx.facets.hide_empty and panel_df.empty:  # Blank empty panels.
            _hide_empty_axis_plotly(fig, ri, ci, plan.ncols)
            continue
        if ctx.layout.panel_letters:  # Bold a, b, c... on visible panels only.
            fig.add_annotation(text=f"<b>{_panel_letter(n_letter)}</b>", xref="x domain",
                               yref="y domain", x=0.02, y=0.98, xanchor="left", yanchor="top",
                               showarrow=False, font=dict(size=ctx.style.font_size + 3),
                               row=ri, col=ci)  # ri/ci are already 1-based in this loop.
            n_letter += 1

        if ctx.graph_type == "bubble":  # One markers trace per panel, shared coloraxis.
            enc_ = ctx.encoding
            rng_key = _bubble_key(ctx, plan, ctx.bubble_scope, row_val, col_val, plan.bubble_range)
            vmin, vmax, smin, smax = plan.bubble_range[rng_key]
            k_idx = list(plan.bubble_range).index(rng_key)
            caxis = "coloraxis" if k_idx == 0 else f"coloraxis{k_idx + 1}"
            vcols = [c for c in dict.fromkeys((enc_.color, enc_.size)) if c is not None]
            cell = panel_df.groupby([enc_.x, enc_.y], observed=True)[vcols] \
                           .agg(ctx.bubble_agg).reset_index()
            if not cell.empty:
                s0, s1 = ctx.bubble_sizes  # Same area scale as Matplotlib (pt^2)...
                if enc_.size is not None and smax > smin:
                    areas = s0 + (cell[enc_.size].values - smin) / (smax - smin) * (s1 - s0)
                else:
                    areas = np.full(len(cell), 0.5 * (s0 + s1))
                px = np.sqrt(np.abs(areas))  # ...converted to marker diameters (px).
                marker = dict(size=px, line=dict(color="black", width=0.6))
                if enc_.color is not None:
                    marker.update(color=cell[enc_.color].values, coloraxis=caxis)
                else:
                    marker.update(color=ctx.bubble_color)
                if plan.bubble_background is not None:  # Class bands below the markers.
                    axis_name, mapping = plan.bubble_background
                    pal = _bubble_bg_palette(ctx, plan)
                    cats = plan.x_order if axis_name == "x" else plan.y_order
                    for i, cat in enumerate(cats):
                        cls = mapping.get(cat)
                        if cls is None:
                            continue
                        colr = mpl.colors.to_hex(pal[cls])
                        common = dict(type="rect", fillcolor=colr,
                                      opacity=ctx.bubble_background_alpha,
                                      line_width=0, layer="below", row=ri, col=ci)
                        if axis_name == "x":
                            fig.add_shape(x0=i - 0.5, x1=i + 0.5, yref="y domain",
                                          y0=0, y1=1, **common)
                        else:
                            fig.add_shape(y0=i - 0.5, y1=i + 0.5, xref="x domain",
                                          x0=0, x1=1, **common)
                        if _legend_once(fig, f"bg_{cls}"):  # One legend entry per class.
                            fig.add_trace(go.Scatter(
                                x=[None], y=[None], mode="markers",
                                marker=dict(symbol="square", size=12, color=colr,
                                            opacity=min(ctx.bubble_background_alpha + 0.25, 1.0)),
                                name=str(ctx.labels.lookup(cls, enc_.background)),
                                legendgroup=f"bg_{cls}", showlegend=True),
                                row=ri, col=ci)
                fig.add_trace(go.Scatter(
                    x=[str(ctx.labels.lookup(v, enc_.x)) for v in cell[enc_.x]],
                    y=[str(ctx.labels.lookup(v, enc_.y)) for v in cell[enc_.y]],
                    mode="markers",
                    marker=marker,
                    showlegend=False,
                    customdata=cell[vcols].values if vcols else None,
                    hovertemplate="%{x} / %{y}"
                                  + "".join(f"<br>{c}: %{{customdata[{i}]:.3g}}"
                                            for i, c in enumerate(vcols))
                                  + "<extra></extra>"),
                    row=ri, col=ci)
            continue  # Bubble panels skip the bar/line machinery below.

        for c in plan.color_order:  # Colour family.
            for comp in plan.compare_order:  # Compare split.
                xs, vals = [], []  # Non-missing points.
                for xval in plan.x_order:
                    val = plan.agg_main.get(facet_row=row_val, facet_col=col_val,
                                            x=xval, color=c, compare=comp)
                    if val is None:
                        continue
                    xs.append(ctx.labels.lookup(xval, ctx.encoding.x))
                    vals.append(val)
                if not vals:  # Empty series.
                    continue
                name = _series_label(ctx, c, comp)  # Combined legend label.
                key = (ctx.graph_type, c, comp)  # Legend identity.
                if ctx.graph_type == "bar":  # Bar trace.
                    trace = go.Bar(x=xs, y=vals, name=name,
                                   marker=dict(color=plan.color_map[c],
                                               pattern=dict(shape=plan.compare_styles.pattern.get(comp, ""))),
                                   offsetgroup=str(comp), legendgroup=str(key),
                                   showlegend=key not in shown)
                else:  # Line trace.
                    trace = go.Scatter(x=xs, y=vals,
                                       mode="lines+markers" if ctx.style.line_mode == "lines+markers" else "lines",
                                       name=name,
                                       line=dict(color=plan.color_map[c],
                                                 dash=plan.compare_styles.dash.get(comp, "solid")),
                                       legendgroup=str(key), showlegend=key not in shown)
                fig.add_trace(trace, row=ri, col=ci)  # Place it.
                shown.add(key)  # Mark legend shown.

        if ctx.scatter is not None:  # Reference overlay.
            for comp in plan.compare_order:
                xs, vals = [], []
                for xval in plan.x_order:
                    val = plan.agg_ref.get(facet_row=row_val, facet_col=col_val, x=xval, compare=comp)
                    if val is None:
                        continue
                    xs.append(ctx.labels.lookup(xval, ctx.encoding.x))
                    vals.append(val)
                if not vals:
                    continue
                ref_base = ctx.labels.lookup(ctx.scatter.value, ctx.scatter.column)  # Reference label base.
                name = (str(ref_base) if comp is None
                        else f"{ref_base} \u00b7 {ctx.labels.lookup(comp, ctx.encoding.compare_by)}")
                key = ("reference", ctx.scatter.value, comp)
                if ctx.graph_type == "bar":  # Reference markers.
                    trace = go.Scatter(x=xs, y=vals, mode="markers", name=name,
                                       marker=dict(size=np.sqrt(ctx.scatter.size),
                                                   color=ctx.scatter.facecolor,
                                                   line=dict(color=ctx.scatter.edgecolor,
                                                             width=ctx.scatter.linewidth)),
                                       legendgroup=str(key), showlegend=key not in shown)
                else:  # Reference line.
                    trace = go.Scatter(x=xs, y=vals, mode="lines", name=name,
                                       line=dict(color=ctx.style.reference_line_color,
                                                 width=ctx.style.reference_linewidth,
                                                 dash=plan.compare_styles.dash.get(comp, "solid")),
                                       legendgroup=str(key), showlegend=key not in shown)
                fig.add_trace(trace, row=ri, col=ci)
                shown.add(key)

    for k in range(len(plan.panels), plan.nrows * plan.ncols):  # Hide trailing cells.
        ri, ci = divmod(k, plan.ncols)
        _hide_empty_axis_plotly(fig, ri + 1, ci + 1, plan.ncols)

    y_range = None  # Auto-range by default.
    if ctx.layout.y_min is not None or ctx.layout.y_max is not None:  # A hard limit was set.
        y_values = plan.prepared.full[ctx.encoding.y]  # Data extremes for the open end.
        y_range = [ctx.layout.y_min if ctx.layout.y_min is not None else float(y_values.min()),
                   ctx.layout.y_max if ctx.layout.y_max is not None else float(y_values.max())]

    if ctx.graph_type == "bubble" and ctx.encoding.color is not None:  # Colour scale(s).
        keys = list(plan.bubble_range)
        n_keys = len(keys)
        for i, key in enumerate(keys):
            vmin, vmax, _, _ = plan.bubble_range[key]
            name = "coloraxis" if i == 0 else f"coloraxis{i + 1}"
            title = str(ctx.labels.lookup(ctx.encoding.color))
            bar = dict(title=title)
            if n_keys > 1:  # Stack one colorbar per group along the right edge.
                height = 1.0 / n_keys
                bar = dict(len=height * 0.8, y=1.0 - (i + 0.5) * height, yanchor="middle",
                           title=dict(text=f"{title}<br>{key}", font=dict(size=10)))
            fig.update_layout(**{name: dict(colorscale=ctx.bubble_cmap, cmin=vmin, cmax=vmax,
                                            colorbar=bar)})

    fig.update_layout(  # Figure-level styling.
        template="simple_white",
        width=(ctx.layout.plotly_size_per_panel or (450, 360))[0] * plan.ncols,
        height=(ctx.layout.plotly_size_per_panel or (450, 360))[1] * plan.nrows,
        title=dict(text=ctx.layout.title, font=dict(size=ctx.style.font_size + 4), x=0.5, xanchor="center"),
        barmode="relative" if ctx.graph_type == "bar" else None,
        font=dict(family=ctx.style.font_family, size=ctx.style.font_size),
        legend=dict(orientation="h", yanchor="bottom",
                    y=ctx.layout.legend_y if ctx.layout.legend_y is not None else 1.02,
                    xanchor="center", x=0.5),
        margin=dict(t=110),
    )
    if ctx.style.tick_label_size is not None:  # Explicit tick-label size (both axes).
        fig.update_xaxes(tickfont=dict(size=ctx.style.tick_label_size))
        fig.update_yaxes(tickfont=dict(size=ctx.style.tick_label_size))
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black",  # Shared x styling.
                     showgrid=(ctx.graph_type == "bubble"), gridwidth=0.75, gridcolor="lightgrey",
                     title_text=ctx.layout.x_title if ctx.layout.x_title is not None else "",
                     title_standoff=ctx.layout.x_title_offset)
    fig.update_yaxes(range=y_range, showline=True, linewidth=2, linecolor="black",  # Shared y styling.
                     showgrid=True, gridwidth=0.75, gridcolor="lightgrey",
                     zeroline=True, zerolinecolor="black", zerolinewidth=0.8,
                     title_text=ctx.layout.y_title if ctx.layout.y_title is not None
                     else str(ctx.labels.lookup(ctx.encoding.y)))

    files: list[str] = []  # Saved paths.
    if ctx.save is not None:  # Optional export.
        files = save_plotly(fig, ctx.save.path, ctx.save.name, ctx.save.extensions)

    return PlotResult(figure=fig, axes=None, files=files, backend="plotly")  # Uniform result.


# ===========================================================================
# multiplot_grid: shared helpers
# ===========================================================================
# Distribution-axis unit ("norm") translated per backend, plus its axis label.
_NORM_MPL = {"density": "density", "frequency": "count", "probability": "probability"}  # sns ``stat``.
_NORM_PLOTLY = {"density": "probability density", "frequency": "", "probability": "probability"}  # histnorm.
_NORM_LABEL = {"density": "Density", "frequency": "Frequency", "probability": "Probability"}  # Axis text.


def _dist_axis_label(kind: str, norm: str) -> str:
    """Return the density-axis label for the distribution column.

    KDE is always a density; ``box`` has no density axis; histograms follow the
    chosen ``norm``.
    """
    if kind == "kde":  # KDE is a density regardless of ``norm``.
        return "Density"
    if kind == "box":  # Box has no density axis.
        return ""
    return _NORM_LABEL.get(norm, "Density")  # Histogram label from the norm.


def _loess(x, y, frac: float, it: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a LOESS smooth ``(xs, ys)`` of ``y`` on ``x`` (statsmodels).

    statsmodels is imported lazily so the package only depends on it when a
    regression is actually requested.

    Parameters
    ----------
    x, y:
        Predictor / response arrays.
    frac:
        LOESS smoothing fraction.
    it:
        Number of robustifying iterations. ``0`` (default) fits the ordinary
        local mean through *all* points; larger values down-weight large-residual
        points as outliers (statsmodels' own default is ``3``), which on a
        bimodal cloud collapses the curve onto the denser regime.

    Returns
    -------
    (ndarray, ndarray)
        Sorted ``x`` grid and the smoothed ``y``.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess  # Lazy import.

    out = lowess(endog=np.asarray(y, dtype=float), exog=np.asarray(x, dtype=float),
                 frac=frac, it=it, return_sorted=True)  # N x 2 sorted by x.
    return out[:, 0], out[:, 1]  # xs, ys.


def _benchmark_ids(encoding) -> list[str]:
    """Normalise ``GridEncoding.benchmark_line_id`` to a list of columns (or [])."""
    bid = encoding.benchmark_line_id  # May be None, a str, or a sequence.
    if bid is None:  # No benchmark trajectory id.
        return []
    if isinstance(bid, str):  # A single column.
        return [bid]
    return list(bid)  # Already a sequence of columns.


def _grid_ax(axes, variables_by: str, j0: int, which: int):
    """Return the Matplotlib axis for variable ``j0`` (0-based) and cell ``which``.

    ``which``: 0 = time series, 1 = distribution, 2 = scatter. The ``variables_by``
    orientation decides whether variables run down the rows or across the columns.
    """
    if variables_by == "col":  # 3 rows x N cols: plot type on the row.
        return axes[which, j0]
    return axes[j0, which]  # N rows x 3 cols: plot type on the column.


def _grid_rowcol(variables_by: str, j1: int, which: int) -> tuple[int, int]:
    """Return the 1-based Plotly ``(row, col)`` for variable ``j1`` and cell ``which``."""
    if variables_by == "col":  # 3 x N.
        return which + 1, j1
    return j1, which + 1  # N x 3.


def _grid_axis_style_mpl(ax) -> None:
    """Apply the shared spine/grid styling used throughout the grid (matplotlib)."""
    ax.grid(axis="y", color="lightgrey", linewidth=0.75)  # Horizontal gridlines.
    ax.grid(axis="x", visible=False)  # No vertical grid.
    ax.spines["top"].set_visible(False)  # Drop top spine.
    ax.spines["right"].set_visible(False)  # Drop right spine.
    ax.spines["left"].set_linewidth(1.5)  # Emphasise left spine.
    ax.spines["bottom"].set_linewidth(1.5)  # Emphasise bottom spine.


def _apply_var_window_mpl(ax, ctx: GridContext, var: Any, axis: str = "y") -> None:
    """Apply per-variable limits to the variable axis, or clamp its bottom to >= 0."""
    window = ctx.var_window  # Mapping variable -> (lo, hi).
    if var in window:  # Explicit window for this variable.
        if axis == "y":
            ax.set_ylim(window[var])
        else:
            ax.set_xlim(window[var])
    else:  # Auto: clamp the bottom to 0 only for variables that are themselves non-negative.
        if axis == "y":
            frames = [f for f in (ctx.data, ctx.benchmark)
                      if f is not None and var in f.columns]
            signed = any(float(np.nanmin(f[var].values)) < 0 for f in frames if len(f))
            if not signed:  # Non-negative quantity: keep the padding from dipping below zero.
                bottom, _ = ax.get_ylim()
                ax.set_ylim(bottom=max(0, bottom))


# ===========================================================================
# multiplot_grid: Matplotlib backend
# ===========================================================================
def _agg_line(frame, line_by: str, var: str, agg: str | None):
    """Ordered ``(x, y)`` for one time-series line.

    Reduces multiple rows per ``line_by`` value with ``agg`` (``"mean"`` /
    ``"median"``), so a category holding several scenarios becomes a single clean
    line instead of a polyline zig-zagging through every same-year point. With
    ``agg=None`` the raw points are kept (legacy). Returns ``(None, None)`` when
    there is nothing to draw.
    """
    g = frame[[line_by, var]].dropna()
    if g.empty:
        return None, None
    if agg in ("mean", "median"):  # Collapse duplicates at each x.
        g = g.groupby(line_by, as_index=False)[var].agg(agg)
    g = g.sort_values(line_by)  # X must be monotonic for a sensible line.
    return g[line_by].to_numpy(), g[var].to_numpy()


def _grid_sharey(lay) -> str:
    """Matplotlib ``sharey`` mode from ``GridLayoutSpec.shared_yaxes``.

    ``False`` / ``"none"`` -> independent; ``"all"`` -> the whole grid; ``True``
    or ``"variable"`` (default) -> per variable: each grid row in row-mode and
    each grid column in col-mode is one variable, so its cells share the y-axis.
    """
    sy = lay.shared_yaxes
    if sy is False or sy in ("none", None):
        return "none"
    if sy == "all":
        return "all"
    return "row" if lay.variables_by == "row" else "col"  # Per-variable.


def _grid_shared_yaxes_plotly(lay):
    """Plotly ``make_subplots(shared_yaxes=...)`` value from ``shared_yaxes``."""
    sy = lay.shared_yaxes
    if sy is False or sy in ("none", None):
        return False
    if sy == "all":
        return True
    return "rows" if lay.variables_by == "row" else "columns"  # Per-variable.


def _mpl_cell_series(ax, ctx: GridContext, plan: GridPlan, var, bench_df) -> None:
    """Draw the time-series cell (``"series"``) for one variable on ``ax``.

    The benchmark trajectories (when present) are greyed in the background; the
    data is one ungrouped line, or one coloured family per ``by`` category (one
    line per ``line_id`` trajectory when set).
    """
    enc, lay, sty = ctx.encoding, ctx, ctx.style  # Shorthands.
    data = ctx.data  # The user's frame.
    cats, color_map = plan.cats, plan.color_map  # Categories + colours.
    by, line_by = enc.by, enc.line_by  # Roles.
    bench_ids = _benchmark_ids(enc)  # Benchmark trajectory id columns.
    year = ctx.year  # Cross-section year (may be None).
    year_lo, year_hi = lay.year_range  # Time-series x-limits / benchmark window.
    year_cut = year if year is not None else year_hi  # Upper bound for the trajectories.
    agg = enc.aggregate  # Reduction for multiple rows per (group, line_by).

    if bench_df is not None and bench_ids:  # Faint benchmark trajectories in the background.
        bback = bench_df[(bench_df[line_by] >= year_lo) & (bench_df[line_by] <= year_cut)]
        for _, g in bback.groupby(bench_ids):  # One faint line per benchmark trajectory.
            x, y = _agg_line(g, line_by, var, agg)  # Collapse duplicate years -> no vertical spikes.
            if x is None:
                continue
            ax.plot(x, y, color=sty.benchmark_color,  # Legend entry only when no scatter cell
                    alpha=sty.benchmark_line_alpha, lw=sty.benchmark_line_width, zorder=1,
                    label=(str(ctx.labels.lookup(sty.benchmark_label))  # already provides one.
                           if "scatter" not in ctx.columns else None))

    if by is None:  # Single ungrouped series (collapsed over duplicate years).
        x, y = _agg_line(data[data[line_by] <= year_cut], line_by, var, agg)
        if x is not None:
            ax.plot(x, y, color="C0", lw=sty.line_width, marker="o", ms=sty.marker_size)
    else:  # One coloured family per ``by`` category.
        for cat in cats:
            if enc.line_id is not None:  # One line per individual trajectory.
                d = data.loc[(data[by] == cat) & (data[line_by] <= year_cut),
                             [enc.line_id, line_by, var]].dropna()
                for _, g in d.groupby(enc.line_id):
                    x, y = _agg_line(g, line_by, var, agg)
                    if x is None:
                        continue
                    ax.plot(x, y, color=color_map[cat], lw=sty.line_width,
                            marker="o", ms=sty.marker_size, label=str(cat))
            else:  # One aggregate line per category (mean/median over scenarios).
                sub = data.loc[(data[by] == cat) & (data[line_by] <= year_cut), [line_by, var]]
                x, y = _agg_line(sub, line_by, var, agg)
                if x is None:
                    continue
                ax.plot(x, y, color=color_map[cat], lw=sty.line_width,
                        marker="o", ms=sty.marker_size, label=str(cat))

    v_label = ctx.labels.lookup(var)  # Variable display name.
    ax.set_title(v_label, fontsize=12, pad=8)  # Variable title.
    ax.set_xlabel(ctx.labels.lookup(line_by))  # Time axis.
    ax.set_ylabel(v_label)  # Variable axis.
    ax.set_xlim([year_lo, year_hi])  # Configurable time window.
    # ---- highlighted trajectories (black outline, on top) ----
    if ctx.highlight_lines:
        h_col, h_vals = next(iter(ctx.highlight_lines.items()))
        h_vals = [h_vals] if isinstance(h_vals, str) or not hasattr(h_vals, "__iter__") else list(h_vals)
        dsel = data.loc[data[h_col].isin(h_vals) & (data[line_by] <= year_cut)]
        peff = [pe.Stroke(linewidth=sty.line_width + 2.5, foreground="black"), pe.Normal()]
        cat_iter = [None] if by is None else [c for c in cats if (dsel[by] == c).any()]
        for cat in cat_iter:
            dcat = dsel if cat is None else dsel.loc[dsel[by] == cat]
            colr = "C0" if cat is None else color_map[cat]
            groups = dcat.groupby(enc.line_id) if enc.line_id is not None else [(None, dcat)]
            for _, g in groups:
                x, y = _agg_line(g, line_by, var, agg)
                if x is None:
                    continue
                ax.plot(x, y, color=colr, lw=sty.line_width + 0.6, marker="o",
                        ms=sty.marker_size, path_effects=peff, zorder=3)

    _apply_var_window_mpl(ax, ctx, var, axis="y")  # Per-variable y window.
    _grid_axis_style_mpl(ax)  # Shared spine/grid styling.


def _mpl_cell_dist(ax, ctx: GridContext, plan: GridPlan, var, bench_df) -> None:
    """Draw the distribution cell (``"dist"``) for one variable on ``ax``.

    Horizontal (variable on the y-axis). The benchmark distribution is drawn when
    ``dist.show_benchmark`` and a benchmark is present; the data's own
    distribution when ``dist.show_data``.
    """
    enc, sty, dist = ctx.encoding, ctx.style, ctx.dist  # Shorthands.
    data = ctx.data  # The user's frame.
    cats, color_map = plan.cats, plan.color_map  # Categories + colours.
    by, line_by = enc.by, enc.line_by  # Roles.
    year = ctx.year  # Cross-section year.
    stat = _NORM_MPL.get(dist.norm, "density")  # Histogram stat from the norm.
    dens_label = _dist_axis_label(dist.kind, dist.norm)  # Density-axis label.
    v_label = ctx.labels.lookup(var)  # Variable display name.
    row2 = filter_year(data, line_by, year)  # Final-year data slice.

    if dist.show_benchmark and bench_df is not None:  # Benchmark distribution behind, in grey.
        bench2 = filter_year(bench_df, line_by, year)[[var]].dropna()
        if not bench2.empty:
            if dist.kind == "kde":
                sns.kdeplot(data=bench2, y=var, ax=ax, color=sty.benchmark_color,
                            lw=1.8, fill=True, alpha=sty.benchmark_alpha, legend=False, zorder=1)
            elif dist.kind == "hist":
                sns.histplot(data=bench2, y=var, ax=ax, color=sty.benchmark_color,
                             alpha=sty.benchmark_alpha, stat=stat, element="step",
                             fill=True, legend=False, zorder=1)
            elif dist.kind == "box":
                ax.boxplot(bench2[var].values, vert=True, positions=[0], widths=0.45,
                           patch_artist=True,
                           boxprops=dict(facecolor=sty.benchmark_color, alpha=sty.benchmark_alpha),
                           medianprops=dict(color="black"), zorder=1)

    def _one_dist(sub: pd.DataFrame, colr, position: int) -> None:
        """Draw one background distribution (kde/hist/box) in ``colr``.

        kde / hist are unfilled curves with ``group_alpha`` on the line; box
        keeps a face fill at ``group_alpha`` (no curve to speak of). Per-group
        colours come from ``plan.color_map`` -- the same map the per-group
        scatter regression reads, so both agree by construction.
        """
        if dist.kind == "kde":  # Unfilled curve, alpha on the line.
            sns.kdeplot(data=sub, y=var, ax=ax, color=colr, lw=2.0, fill=False,
                        alpha=dist.group_alpha, legend=False, zorder=2)
        elif dist.kind == "hist":  # NB: y=var (horizontal), fixing the original x/y mix-up.
            sns.histplot(data=sub, y=var, ax=ax, color=colr, stat=stat,
                         alpha=dist.group_alpha, element="step", fill=False,
                         linewidth=2.0, legend=False, zorder=2)
        elif dist.kind == "box":
            ax.boxplot(sub[var].values, vert=True, positions=[position], widths=0.45,
                       patch_artist=True,
                       boxprops=dict(facecolor=colr, alpha=dist.group_alpha),
                       medianprops=dict(color="black"), zorder=2)

    if dist.show_data:  # The data's own distribution(s), filled, as a background layer.
        if by is None:  # Single ungrouped distribution.
            sub = row2[[var]].dropna()
            if not sub.empty:
                _one_dist(sub, "C0", position=1)
        else:  # One filled distribution per category.
            for k, cat in enumerate(cats):
                sub = row2.loc[row2[by] == cat, [var]].dropna()
                if not sub.empty:
                    _one_dist(sub, color_map[cat], position=k + 1)

        if dist.kind == "box" and by is not None:  # Category ticks for the box variant.
            ax.set_xticks(range(1, len(cats) + 1))
            ax.set_xticklabels([str(ctx.labels.lookup(c, by)) for c in cats], rotation=45, ha="right")

    if dist.show_total and dist.kind != "box":  # Pooled all-groups curve on top (kde/hist only).
        pooled = row2[[var]].dropna()  # All ``by`` categories together.
        if not pooled.empty:
            t_label = str(ctx.labels.lookup(dist.total_label))  # Legend entry (kwarg + labels).
            if dist.kind == "kde":
                sns.kdeplot(data=pooled, y=var, ax=ax, color=dist.total_color, lw=2.2,
                            fill=False, legend=False, zorder=3, label=t_label)
            else:  # "hist": unfilled step outline over the filled groups.
                sns.histplot(data=pooled, y=var, ax=ax, color=dist.total_color, stat=stat,
                             element="step", fill=False, linewidth=2.0, legend=False, zorder=3,
                             label=t_label)

    # ---- highlighted points pinned on a density curve ----
    if ctx.highlight and dist.highlight_on and dist.kind != "box":
        h_col, h_vals = next(iter(ctx.highlight.items()))
        h_vals = [h_vals] if isinstance(h_vals, str) or not hasattr(h_vals, "__iter__") else list(h_vals)
        hsub = row2.loc[row2[h_col].isin(h_vals)].dropna(subset=[var])
        drew_hl = False
        for _, prow in hsub.iterrows():
            v = float(prow[var])
            cat = prow[by] if by is not None else None
            colr = color_map.get(cat, "C0") if cat is not None else "C0"
            if dist.highlight_on == "total" or by is None:  # Pooled reference curve.
                sample = row2[var].dropna().values
            else:  # The point's own group curve.
                sample = row2.loc[row2[by] == cat, var].dropna().values
            d = _density_at(sample, v, dist.kind, dist.norm)
            if d is None:
                continue
            ax.scatter([d], [v], color=colr, alpha=0.9, s=2.5 * sty.scatter_size, zorder=3.4)
            ax.scatter([d], [v], s=2.5 * sty.scatter_size, hatch="///", facecolor="none",
                       edgecolor="black", linewidth=1.0, zorder=3.5)
            drew_hl = True
        if drew_hl:  # Legend swatch (deduplicated by label with the scatter cell's).
            ax.add_patch(mpatches.Rectangle((np.nan, np.nan), 0, 0, facecolor="none",
                                            edgecolor="black", hatch="///", linewidth=1.0,
                                            label=str(ctx.labels.lookup("Highlight"))))

    title_year = f"in {year}" if year is not None else "all values"  # Title suffix.
    ax.set_title(f"{v_label}, distribution {title_year}", fontsize=11, pad=8)
    ax.set_ylabel(v_label)  # Variable on the y-axis.
    if dist.kind != "box":  # Density axis label (x).
        ax.set_xlabel(dens_label)
        if dist.log_x:  # Log density axis, floored 3 decades below the peak.
            ax.set_xscale("log")
            xmax = ax.get_xlim()[1]
            if xmax > 0:
                ax.set_xlim(left=xmax * 1e-3)
    _apply_var_window_mpl(ax, ctx, var, axis="y")  # Same variable window as the row.
    _grid_axis_style_mpl(ax)


def _mpl_cell_scatter(ax, ctx: GridContext, plan: GridPlan, var, bench_df) -> None:
    """Draw the scatter cell (``"scatter"``) at the final year for one variable.

    Benchmark scatter greyed behind; data coloured by ``by``; optional LOESS fits
    for the data and (separately) the benchmark.
    """
    enc, sty, reg = ctx.encoding, ctx.style, ctx.regression  # Shorthands.
    data = ctx.data  # The user's frame.
    cats, color_map = plan.cats, plan.color_map  # Categories + colours.
    by, line_by, scatter_by = enc.by, enc.line_by, enc.scatter_by  # Roles.
    year = ctx.year  # Cross-section year.
    data_reg_color = reg.data_color if reg.data_color is not None else regression_color(color_map, "matplotlib")
    v_label = ctx.labels.lookup(var)  # Variable display name.
    s_label = ctx.labels.lookup(scatter_by)  # Scatter-axis display name.
    row2 = filter_year(data, line_by, year)  # Final-year data slice.
    bench_year = filter_year(bench_df, line_by, year) if bench_df is not None else None

    if bench_df is not None:  # Benchmark scatter behind, in grey.
        bench3 = bench_year[[scatter_by, var]].dropna()
        if not bench3.empty:
            ax.scatter(bench3[scatter_by], bench3[var], color=sty.benchmark_color,
                       alpha=sty.benchmark_alpha, s=sty.benchmark_scatter_size, zorder=1,
                       label=str(ctx.labels.lookup(sty.benchmark_label)))

    if by is None:  # Single ungrouped scatter.
        sub = row2[[scatter_by, var]].dropna()
        ax.scatter(sub[scatter_by], sub[var], color="C0", s=sty.scatter_size, alpha=0.9, zorder=2)
    else:  # One coloured scatter per category.
        for cat in cats:
            sub = row2.loc[row2[by] == cat, [scatter_by, var]].dropna()
            if sub.empty:
                continue
            ax.scatter(sub[scatter_by], sub[var], color=color_map[cat], s=sty.scatter_size,
                       alpha=0.9, zorder=2, label=str(cat))

    # ---- highlighted points (on top of the data scatter) ----
    if ctx.highlight:
        h_col, h_vals = next(iter(ctx.highlight.items()))
        h_vals = [h_vals] if isinstance(h_vals, str) or not hasattr(h_vals, "__iter__") else list(h_vals)
        hsub = row2.loc[row2[h_col].isin(h_vals)].dropna(subset=[scatter_by, var])
        if not hsub.empty:  # Bigger, lightly hatched (colour stays visible), black edge.
            if by is None:
                ax.scatter(hsub[scatter_by], hsub[var], color="C0", alpha=0.9,
                           s=2.5 * sty.scatter_size, zorder=4.4)
            else:
                for cat in dict.fromkeys(hsub[by].dropna().tolist()):
                    tmp = hsub.loc[hsub[by] == cat]
                    ax.scatter(tmp[scatter_by], tmp[var], color=color_map[cat], alpha=0.9,
                               s=2.5 * sty.scatter_size, zorder=4.4)
            ax.scatter(hsub[scatter_by], hsub[var], s=2.5 * sty.scatter_size, hatch="///",
                       facecolor="none", edgecolor="black", linewidth=1.0, zorder=4.5)
            ax.add_patch(mpatches.Rectangle(  # Invisible carrier for the hatched legend swatch.
                (np.nan, np.nan), 0, 0, facecolor="none", edgecolor="black", hatch="///",
                linewidth=1.0, label=str(ctx.labels.lookup("Highlight"))))

    if reg.data or reg.per_group:  # per_group implies the data fit. # Optional LOESS through the data scatter.
        if reg.per_group and by is not None:  # One curve per category (each regime separately).
            drew = False
            for cat in cats:
                rsub = row2.loc[row2[by] == cat, [scatter_by, var]].dropna()
                if len(rsub) > 2 and rsub[scatter_by].nunique() > 1:
                    try:
                        xs, ys = _loess(rsub[scatter_by], rsub[var], reg.loess_frac, reg.robust_iter)
                        ax.plot(xs, ys, linestyle=reg.dash, linewidth=reg.line_width,
                                color=color_map[cat], zorder=4)
                        drew = True
                    except Exception:  # Never let a smoothing failure kill the figure.
                        pass
            if drew:  # One neutral legend entry for the style (colours are the groups').
                ax.plot([], [], linestyle=reg.dash, linewidth=reg.line_width, color="dimgray",
                        label=str(ctx.labels.lookup("LOESS (data)")))
        else:  # Single pooled curve through all data points.
            rsub = row2[[scatter_by, var]].dropna()
            if not rsub.empty and rsub[scatter_by].nunique() > 1:
                try:
                    xs, ys = _loess(rsub[scatter_by], rsub[var], reg.loess_frac, reg.robust_iter)
                    ax.plot(xs, ys, linestyle=reg.dash, linewidth=reg.line_width,
                            color=data_reg_color, zorder=4,
                            label=str(ctx.labels.lookup("LOESS (data)")))
                except Exception:
                    pass

    if reg.benchmark and bench_df is not None:  # Optional LOESS through the benchmark scatter.
        bsub = bench_year[[scatter_by, var]].dropna()
        if not bsub.empty and bsub[scatter_by].nunique() > 1:
            try:
                xs, ys = _loess(bsub[scatter_by], bsub[var], reg.loess_frac, reg.robust_iter)
                ax.plot(xs, ys, linestyle=reg.dash, linewidth=reg.line_width,
                        color=reg.benchmark_color, zorder=3,
                        label=str(ctx.labels.lookup("LOESS (benchmark)")))
            except Exception:
                pass

    ax.set_title(f"{v_label} vs\n{s_label}", fontsize=11, pad=8)  # Scatter title (wrapped after "vs").
    ax.set_xlabel(s_label)  # Scatter x.
    ax.set_ylabel(v_label)  # Variable y.
    _apply_var_window_mpl(ax, ctx, var, axis="y")  # Same variable window.
    _grid_axis_style_mpl(ax)


def render_multiplot_matplotlib(ctx: GridContext, plan: GridPlan) -> PlotResult:
    """Render the N x 3 (or 3 x N) diagnostic grid with Matplotlib.

    Parameters
    ----------
    ctx, plan:
        Validated grid context and precomputed plan.

    Returns
    -------
    PlotResult
        Figure, axes array and any written file paths.
    """
    lay = ctx  # Grid layout fields are flat on ctx.
    benchmark = ctx.benchmark  # The comparison/reference frame, or None.
    variables = plan.variables  # Variables, one per grid line.
    ncols = plan.ncols  # Number of variables (the "N").
    by = ctx.encoding.by  # Colour grouping (used by the shared legend below).

    columns = list(lay.columns)  # Selected diagnostic cells, in order.
    pos = {kind: i for i, kind in enumerate(columns)}  # Cell name -> column index.
    ncells = len(columns)  # The grid's second dimension (was hard-coded 3).

    # Grid shape depends on the orientation; the cell axis now has ``ncells`` slots.
    if lay.variables_by == "col":  # ncells rows x N cols.
        nrows_g, ncols_g = ncells, ncols
        fig_w, fig_h = lay.figsize_per_col[0] * ncols, lay.figsize_per_col[1] * ncells
    else:  # N rows x ncells cols.
        nrows_g, ncols_g = ncols, ncells
        fig_w, fig_h = lay.figsize_per_col[0] * ncells, lay.figsize_per_col[1] * ncols

    with mpl.rc_context(dict(sns.axes_style("white"))):  # Scoped "white" style (no global side effects).
        gs_kw = {}  # Relative cell sizes along the cell axis (width when cells are columns).
        if lay.cell_ratios is not None:
            key = "height_ratios" if lay.variables_by == "col" else "width_ratios"
            gs_kw[key] = list(lay.cell_ratios)
        fig, axes = plt.subplots(nrows=nrows_g, ncols=ncols_g, figsize=(fig_w, fig_h),
                                 squeeze=False, sharex=lay.shared_xaxes, sharey=_grid_sharey(lay),
                                 gridspec_kw=gs_kw)

        for j, var in enumerate(variables):  # One line of the grid per variable.
            # Benchmark overlays are per-variable: a variable absent from the
            # benchmark is plotted normally, just without its benchmark overlay.
            bench_df = benchmark if (benchmark is not None and var in benchmark.columns) else None
            if benchmark is not None and bench_df is None:  # Present overall but missing this variable.
                warnings.warn(f"Variable {var!r} is absent from the benchmark; "
                              f"its benchmark overlay is skipped.", stacklevel=2)

            if "series" in pos:  # Cell: time series vs line_by.
                _mpl_cell_series(_grid_ax(axes, lay.variables_by, j, pos["series"]),
                                 ctx, plan, var, bench_df)
            if "dist" in pos:  # Cell: distribution at the final year.
                _mpl_cell_dist(_grid_ax(axes, lay.variables_by, j, pos["dist"]),
                               ctx, plan, var, bench_df)
            if "scatter" in pos:  # Cell: scatter at the final year.
                _mpl_cell_scatter(_grid_ax(axes, lay.variables_by, j, pos["scatter"]),
                                  ctx, plan, var, bench_df)

        # Y labelling: show the numeric tick labels on every cell (per-variable y
        # sharing hides them by default), and keep the indicator name only on the
        # first cell of each variable (so it isn't repeated on dist / scatter).
        for j in range(ncols):
            for ci in range(ncells):
                cell = _grid_ax(axes, lay.variables_by, j, ci)
                cell.tick_params(labelleft=True)  # Re-show y numbers even when y is shared.
                if ci != 0:  # Drop the redundant indicator name off non-first cells.
                    cell.set_ylabel("")

        # One shared legend above the grid, with gap + two-row wrapping controls
        # (same machinery as plot_allocations via _finalize_top_legend).
        handles, labels = [], []
        for ax in axes.flat:  # Collect across every axis.
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        seen, uniq_h, uniq_l = set(), [], []
        for h, l in zip(handles, labels):  # Keep the first occurrence of each label.
            if l and l not in seen:
                uniq_h.append(h)
                uniq_l.append(l)
                seen.add(l)
        _finalize_top_legend(fig, uniq_h, lay, labels=uniq_l,
                             title=str(ctx.labels.lookup(by)) if by is not None else None)

        if ctx.panel_letters:  # Bold a, b, c... tags, row-major over the grid.
            for idx, ax_ in enumerate(np.atleast_2d(axes).ravel()):
                ax_.text(0.02, 0.98, _panel_letter(idx), transform=ax_.transAxes,
                         fontweight="bold", fontsize=12, va="top", ha="left", zorder=10)

        _rasterize_data(fig, ctx.style.rasterized)  # Printable vector output.
        files: list[str] = []  # Saved paths.
        if ctx.save is not None:  # Optional export.
            files = save_matplotlib(fig, ctx.save.path, ctx.save.name, ctx.save.extensions, ctx.save.dpi)

    return PlotResult(figure=fig, axes=axes, files=files, backend="matplotlib")  # Uniform result.


# ===========================================================================
# multiplot_grid: Plotly backend
# ===========================================================================
def _panel_letter(i: int) -> str:
    """0-based panel index -> a, b, ..., z, aa, ab, ... (bold corner tags)."""
    letters = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        letters = chr(97 + r) + letters
    return letters


def _legend_once(fig, group: str) -> bool:
    """True if no visible legend entry exists yet for ``group`` (Plotly grids).

    Grid cells repeat per variable; this keeps one legend entry per concept
    (total, LOESS data / benchmark) instead of one per variable line.
    """
    return not any(t.legendgroup == group and t.showlegend for t in fig.data)


def _kde_curve(values: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(grid, density)`` from a Gaussian KDE, or ``None`` if degenerate."""
    from scipy.stats import gaussian_kde  # Lazy import (Plotly KDE only).

    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]  # Drop NaNs.
    if len(np.unique(vals)) <= 1:  # KDE needs spread.
        return None
    grid = np.linspace(vals.min(), vals.max(), n_points)  # Evaluation grid.
    return grid, gaussian_kde(vals)(grid)  # Grid + density.


def _density_at(sample, v: float, kind: str, norm: str) -> float | None:
    """Height of ``sample``'s kde / hist curve at ``v`` -- pins highlight markers.

    kde: ``gaussian_kde`` with the Scott bandwidth (the same estimator seaborn
    and ``_kde_curve`` use, so the marker sits on the drawn curve). hist:
    ``numpy`` "auto" bins in the ``norm`` unit -- identical to ``_hist_steps``
    (Plotly); the Matplotlib hist is binned by seaborn, so the marker can sit a
    touch off the drawn step there.
    """
    from scipy.stats import gaussian_kde  # Lazy import (same as _kde_curve).
    vals = np.asarray(sample, dtype=float)
    vals = vals[~np.isnan(vals)]
    if kind == "kde":
        if vals.size < 2 or np.unique(vals).size < 2:  # KDE needs spread.
            return None
        return float(gaussian_kde(vals)(v)[0])
    counts, edges = np.histogram(vals, bins="auto", density=(norm == "density"))
    if norm == "probability":  # Fraction of observations per bin.
        counts = counts / max(counts.sum(), 1)
    if not edges[0] <= v <= edges[-1]:  # Outside the sample range: baseline.
        return 0.0
    idx = min(max(int(np.searchsorted(edges, v, side="right")) - 1, 0), len(counts) - 1)
    return float(counts[idx])


def _hist_steps(values, norm: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(value_steps, edge_steps)`` for a horizontal unfilled hist outline.

    Plotly has no native unfilled step histogram, so both the per-group hist
    curves and the pooled ("total") overlay are drawn as Scatter polylines.
    Bins are ``numpy`` "auto", computed per subset. ``norm`` is the package
    unit ("density" / "frequency" / "probability").
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]  # Drop NaNs.
    if vals.size == 0:
        return None
    counts, edges = np.histogram(vals, bins="auto", density=(norm == "density"))
    if norm == "probability":  # Fraction of observations per bin.
        counts = counts / max(counts.sum(), 1)
    xs = [0.0]  # Value axis (horizontal): start on the baseline.
    ys = [edges[0]]  # Variable axis (vertical).
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):  # One step per bin.
        xs += [c, c]
        ys += [e0, e1]
    xs.append(0.0)  # Close back onto the baseline.
    ys.append(edges[-1])
    return np.asarray(xs), np.asarray(ys)


def _plotly_cell_series(fig, ctx: GridContext, plan: GridPlan, var, bench_df,
                        row: int, col: int, shown: set, legend: bool) -> None:
    """Draw the time-series cell (``"series"``) for one variable into ``fig``.

    ``legend`` is True only for the legend-bearing variable line (the first), so
    each ``by`` category contributes a single legend entry; ``shown`` de-duplicates
    categories across lines.
    """
    enc, lay, sty = ctx.encoding, ctx, ctx.style  # Shorthands.
    data = ctx.data  # The user's frame.
    cats, color_map = plan.cats, plan.color_map  # Categories + colours.
    by, line_by = enc.by, enc.line_by  # Roles.
    bench_ids = _benchmark_ids(enc)  # Benchmark trajectory id columns.
    year = ctx.year  # Cross-section year.
    year_lo, year_hi = lay.year_range  # Time window.
    year_cut = year if year is not None else year_hi  # Upper bound for the trajectories.
    agg = enc.aggregate  # Reduction for multiple rows per (group, line_by).

    if bench_df is not None and bench_ids:  # Faint benchmark trajectories behind.
        bback = bench_df[(bench_df[line_by] >= year_lo) & (bench_df[line_by] <= year_cut)]
        first = True
        for _, g in bback.groupby(bench_ids):
            x, y = _agg_line(g, line_by, var, agg)  # Collapse duplicate years -> no spikes.
            if x is None:
                continue
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                     line=dict(color=sty.benchmark_color, width=sty.benchmark_line_width),
                                     opacity=sty.benchmark_line_alpha, name=str(ctx.labels.lookup(sty.benchmark_label)),
                                     legendgroup="benchmark", showlegend=(legend and first)),
                          row=row, col=col)
            first = False

    if by is None:  # Single series (collapsed over duplicate years).
        x, y = _agg_line(data[data[line_by] <= year_cut], line_by, var, agg)
        if x is not None:
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                     line=dict(width=sty.line_width),
                                     marker=dict(size=sty.marker_size + 2), showlegend=False),
                          row=row, col=col)
    else:  # One family per category (one trace per trajectory if line_id is set).
        for cat in cats:
            if enc.line_id is not None:
                d = data.loc[(data[by] == cat) & (data[line_by] <= year_cut),
                             [enc.line_id, line_by, var]].dropna()
                first_cat = cat not in shown
                for _, g in d.groupby(enc.line_id):
                    x, y = _agg_line(g, line_by, var, agg)
                    if x is None:
                        continue
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                             line=dict(width=sty.line_width, color=color_map[cat]),
                                             marker=dict(size=sty.marker_size + 2, color=color_map[cat]),
                                             name=str(cat), legendgroup=f"group_{cat}",
                                             showlegend=(legend and first_cat)),
                                  row=row, col=col)
                    first_cat = False
                shown.add(cat)
            else:
                sub = data.loc[(data[by] == cat) & (data[line_by] <= year_cut), [line_by, var]]
                x, y = _agg_line(sub, line_by, var, agg)
                if x is None:
                    continue
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                         line=dict(width=sty.line_width, color=color_map[cat]),
                                         marker=dict(size=sty.marker_size + 2, color=color_map[cat]),
                                         name=str(cat), legendgroup=f"group_{cat}",
                                         showlegend=(legend and cat not in shown)),
                              row=row, col=col)
                shown.add(cat)

    fig.update_xaxes(title_text=str(ctx.labels.lookup(line_by)), range=[year_lo, year_hi], row=row, col=col)
    # ---- highlighted trajectories (emulated outline: black under-trace) ----
    if ctx.highlight_lines:
        h_col, h_vals = next(iter(ctx.highlight_lines.items()))
        h_vals = [h_vals] if isinstance(h_vals, str) or not hasattr(h_vals, "__iter__") else list(h_vals)
        dsel = data.loc[data[h_col].isin(h_vals) & (data[line_by] <= year_cut)]
        cat_iter = [None] if by is None else [c for c in cats if (dsel[by] == c).any()]
        for cat in cat_iter:
            dcat = dsel if cat is None else dsel.loc[dsel[by] == cat]
            colr = "#1f77b4" if cat is None else color_map[cat]
            groups = dcat.groupby(enc.line_id) if enc.line_id is not None else [(None, dcat)]
            for _, g in groups:
                x, y = _agg_line(g, line_by, var, agg)
                if x is None:
                    continue
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines",  # The outline underneath.
                                         line=dict(color="black", width=sty.line_width + 2.5),
                                         showlegend=False, hoverinfo="skip"),
                              row=row, col=col)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines",  # The coloured line on top.
                                         line=dict(color=colr, width=sty.line_width + 0.6),
                                         showlegend=False),
                              row=row, col=col)

    fig.update_yaxes(title_text=str(ctx.labels.lookup(var)), row=row, col=col)
    if var in lay.var_window:  # Per-variable window on the variable (y) axis.
        fig.update_yaxes(range=list(lay.var_window[var]), row=row, col=col)


def _plotly_cell_dist(fig, ctx: GridContext, plan: GridPlan, var, bench_df,
                      row: int, col: int) -> None:
    """Draw the distribution cell (``"dist"``) for one variable into ``fig``.

    Horizontal (variable on the y-axis). Benchmark behind when ``dist.show_benchmark``
    and a benchmark is present; the data's own distribution when ``dist.show_data``.
    """
    enc, lay, sty, dist = ctx.encoding, ctx, ctx.style, ctx.dist  # Shorthands.
    data = ctx.data  # The user's frame.
    cats, color_map = plan.cats, plan.color_map  # Categories + colours.
    by, line_by = enc.by, enc.line_by  # Roles.
    year = ctx.year  # Cross-section year.
    histnorm = _NORM_PLOTLY.get(dist.norm, "probability density")  # Plotly histnorm from norm.
    dens_label = _dist_axis_label(dist.kind, dist.norm)  # Density-axis label.
    row2 = filter_year(data, line_by, year)  # Final-year data slice.
    dens_peaks: list[float] = []  # Cell density maxima, for the log-axis floor (dist.log_x).

    if dist.show_benchmark and bench_df is not None:  # Benchmark distribution behind.
        bench2 = filter_year(bench_df, line_by, year)[[var]].dropna()
        if not bench2.empty:
            if dist.kind == "hist":
                fig.add_trace(go.Histogram(y=bench2[var], name=str(ctx.labels.lookup(sty.benchmark_label)),
                                           legendgroup="benchmark", showlegend=False,
                                           marker=dict(color=sty.benchmark_color),
                                           opacity=sty.benchmark_alpha, histnorm=histnorm),
                              row=row, col=col)
                bsteps = _hist_steps(bench2[var].values, dist.norm)  # Peak only (go.Histogram bins internally).
                if bsteps is not None:
                    dens_peaks.append(float(bsteps[0].max()))
            elif dist.kind == "box":
                fig.add_trace(go.Box(y=bench2[var], name=str(ctx.labels.lookup(sty.benchmark_label)),
                                     legendgroup="benchmark", showlegend=False,
                                     marker=dict(color=sty.benchmark_color),
                                     line=dict(color=sty.benchmark_color), opacity=sty.benchmark_alpha),
                              row=row, col=col)
            elif dist.kind == "kde":
                curve = _kde_curve(bench2[var].values, dist.n_points)
                if curve is not None:  # Horizontal: density on x, variable on y.
                    grid, dens = curve
                    dens_peaks.append(float(dens.max()))
                    fig.add_trace(go.Scatter(x=dens, y=grid, mode="lines", fill="tozerox",
                                             line=dict(color=sty.benchmark_color, width=2),
                                             opacity=sty.benchmark_alpha, name=str(ctx.labels.lookup(sty.benchmark_label)),
                                             legendgroup="benchmark", showlegend=False),
                                  row=row, col=col)

    if dist.show_data:  # The data's own distribution(s), filled, as a background layer.
        cat_iter = [None] if by is None else cats  # Unify the two cases.
        for cat in cat_iter:  # One filled distribution per category (or one ungrouped).
            sub = row2[[var]].dropna() if by is None else row2.loc[row2[by] == cat, [var]].dropna()
            if sub.empty:
                continue
            colr = "#1f77b4" if by is None else color_map[cat]  # Colour (plan.color_map = regression's map).
            nm = None if by is None else str(cat)  # Legend name.
            grp = None if by is None else f"group_{cat}"  # Legend group.
            if dist.kind == "hist":  # Unfilled step outline (Plotly has no native one), alpha on the line.
                steps = _hist_steps(sub[var].values, dist.norm)
                if steps is not None:
                    xs, ys = steps
                    dens_peaks.append(float(xs.max()))
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                             line=dict(width=2.0, color=colr),
                                             opacity=dist.group_alpha,
                                             name=nm, legendgroup=grp, showlegend=False),
                                  row=row, col=col)
            elif dist.kind == "box":  # Face fill at group_alpha (no curve to speak of).
                fig.add_trace(go.Box(y=sub[var], marker=dict(color=colr), line=dict(color=colr),
                                     opacity=dist.group_alpha,
                                     name=nm, legendgroup=grp, showlegend=False),
                              row=row, col=col)
            elif dist.kind == "kde":  # Unfilled curve, alpha on the line.
                curve = _kde_curve(sub[var].values, dist.n_points)
                if curve is not None:
                    grid, dens = curve
                    dens_peaks.append(float(dens.max()))
                    fig.add_trace(go.Scatter(x=dens, y=grid, mode="lines",
                                             line=dict(width=2.0, color=colr),
                                             opacity=dist.group_alpha, name=nm,
                                             legendgroup=grp, showlegend=False),
                                  row=row, col=col)

    if dist.show_total and dist.kind != "box":  # Pooled all-groups curve on top (kde/hist only).
        pooled = row2[[var]].dropna()  # All ``by`` categories together.
        if not pooled.empty:
            if dist.kind == "kde":
                curve = _kde_curve(pooled[var].values, dist.n_points)
                if curve is not None:
                    grid, dens = curve
                    dens_peaks.append(float(dens.max()))
                    fig.add_trace(go.Scatter(x=dens, y=grid, mode="lines",
                                             line=dict(width=2.2, color=dist.total_color),
                                             name=str(ctx.labels.lookup(dist.total_label)),
                                             legendgroup="total",
                                             showlegend=_legend_once(fig, "total")),
                                  row=row, col=col)
            else:  # "hist": unfilled step outline over the filled groups.
                steps = _hist_steps(pooled[var].values, dist.norm)
                if steps is not None:
                    xs, ys = steps
                    dens_peaks.append(float(xs.max()))
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                             line=dict(width=2.0, color=dist.total_color),
                                             name=str(ctx.labels.lookup(dist.total_label)),
                                             legendgroup="total",
                                             showlegend=_legend_once(fig, "total")),
                                  row=row, col=col)

    # ---- highlighted points pinned on a density curve (no hatch in Plotly) ----
    if ctx.highlight and dist.highlight_on and dist.kind != "box":
        h_col, h_vals = next(iter(ctx.highlight.items()))
        h_vals = [h_vals] if isinstance(h_vals, str) or not hasattr(h_vals, "__iter__") else list(h_vals)
        hsub = row2.loc[row2[h_col].isin(h_vals)].dropna(subset=[var])
        h_label = str(ctx.labels.lookup("Highlight"))
        for _, prow in hsub.iterrows():
            v = float(prow[var])
            cat = prow[by] if by is not None else None
            colr = color_map.get(cat, "#1f77b4") if cat is not None else "#1f77b4"
            if dist.highlight_on == "total" or by is None:
                sample = row2[var].dropna().values
            else:
                sample = row2.loc[row2[by] == cat, var].dropna().values
            d = _density_at(sample, v, dist.kind, dist.norm)
            if d is None:
                continue
            fig.add_trace(go.Scatter(x=[d], y=[v], mode="markers",
                                     marker=dict(color=colr, size=14,
                                                 line=dict(color="black", width=1.5)),
                                     name=h_label, legendgroup="highlight",
                                     showlegend=_legend_once(fig, "highlight")),
                          row=row, col=col)

    fig.update_xaxes(title_text=dens_label, row=row, col=col)
    if dist.log_x and dist.kind != "box" and dens_peaks:  # Log density axis, floored 3 decades below the peak.
        top = math.log10(max(dens_peaks))
        fig.update_xaxes(type="log", range=[top - 3.0, top + 0.08], row=row, col=col)
    fig.update_yaxes(title_text=str(ctx.labels.lookup(var)), row=row, col=col)
    if var in lay.var_window:  # Per-variable window on the variable (y) axis.
        fig.update_yaxes(range=list(lay.var_window[var]), row=row, col=col)


def _plotly_cell_scatter(fig, ctx: GridContext, plan: GridPlan, var, bench_df,
                         row: int, col: int) -> None:
    """Draw the scatter cell (``"scatter"``) at the final year for one variable."""
    enc, lay, sty, reg = ctx.encoding, ctx, ctx.style, ctx.regression  # Shorthands.
    data = ctx.data  # The user's frame.
    cats, color_map = plan.cats, plan.color_map  # Categories + colours.
    by, line_by, scatter_by = enc.by, enc.line_by, enc.scatter_by  # Roles.
    year = ctx.year  # Cross-section year.
    row2 = filter_year(data, line_by, year)  # Final-year data slice.
    bench_year = filter_year(bench_df, line_by, year) if bench_df is not None else None

    if bench_df is not None:  # Benchmark scatter behind (final-year slice, consistent with mpl).
        bench3 = bench_year[[scatter_by, var]].dropna()
        if not bench3.empty:
            fig.add_trace(go.Scatter(x=bench3[scatter_by], y=bench3[var], mode="markers",
                                     marker=dict(color=sty.benchmark_color,
                                                 size=sty.benchmark_scatter_size / 2,
                                                 opacity=sty.benchmark_alpha),
                                     name=str(ctx.labels.lookup(sty.benchmark_label)), legendgroup="benchmark",
                                     showlegend=False),
                          row=row, col=col)

    if by is None:  # Single scatter.
        sub = row2[[scatter_by, var]].dropna()
        fig.add_trace(go.Scatter(x=sub[scatter_by], y=sub[var], mode="markers",
                                 marker=dict(size=sty.scatter_size / 2, opacity=0.9), showlegend=False),
                      row=row, col=col)
    else:  # One scatter per category.
        for cat in cats:
            sub = row2.loc[row2[by] == cat, [scatter_by, var]].dropna()
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(x=sub[scatter_by], y=sub[var], mode="markers",
                                     marker=dict(color=color_map[cat], size=sty.scatter_size / 2,
                                                 opacity=0.9),
                                     name=str(cat), legendgroup=f"group_{cat}", showlegend=False),
                          row=row, col=col)

    # ---- highlighted points (on top; Plotly scatter markers cannot hatch) ----
    if ctx.highlight:
        h_col, h_vals = next(iter(ctx.highlight.items()))
        h_vals = [h_vals] if isinstance(h_vals, str) or not hasattr(h_vals, "__iter__") else list(h_vals)
        hsub = row2.loc[row2[h_col].isin(h_vals)].dropna(subset=[scatter_by, var])
        if not hsub.empty:
            h_label = str(ctx.labels.lookup("Highlight"))
            cat_iter = [None] if by is None else list(dict.fromkeys(hsub[by].dropna().tolist()))
            for cat in cat_iter:
                tmp = hsub if cat is None else hsub.loc[hsub[by] == cat]
                colr = "#1f77b4" if cat is None else color_map[cat]
                fig.add_trace(go.Scatter(x=tmp[scatter_by], y=tmp[var], mode="markers",
                                         marker=dict(color=colr, size=14,
                                                     line=dict(color="black", width=1.5)),
                                         name=h_label, legendgroup="highlight",
                                         showlegend=_legend_once(fig, "highlight")),
                              row=row, col=col)

    if reg.data or reg.per_group:  # per_group implies the data fit. # LOESS through the data scatter.
        loess_label = str(ctx.labels.lookup("LOESS (data)"))  # Legend entry (renamable via labels).
        if reg.per_group and by is not None:  # One curve per category (each regime separately).
            drew = False
            for cat in cats:
                rsub = row2.loc[row2[by] == cat, [scatter_by, var]].dropna()
                if len(rsub) > 2 and rsub[scatter_by].nunique() > 1:
                    try:
                        xs, ys = _loess(rsub[scatter_by], rsub[var], reg.loess_frac, reg.robust_iter)
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                                 line=dict(color=color_map[cat], width=reg.line_width,
                                                           dash="dash" if reg.dash == "--" else "solid"),
                                                 name=loess_label, legendgroup="loess_data",
                                                 showlegend=False),
                                      row=row, col=col)
                        drew = True
                    except Exception:
                        pass
            if drew and _legend_once(fig, "loess_data"):  # One neutral entry for the style.
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                         line=dict(color="dimgray", width=reg.line_width,
                                                   dash="dash" if reg.dash == "--" else "solid"),
                                         name=loess_label, legendgroup="loess_data",
                                         showlegend=True),
                              row=row, col=col)
        else:  # Single pooled curve.
            rsub = row2[[scatter_by, var]].dropna()
            if not rsub.empty and rsub[scatter_by].nunique() > 1:
                try:
                    xs, ys = _loess(rsub[scatter_by], rsub[var], reg.loess_frac, reg.robust_iter)
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                             line=dict(color=reg.data_color or "black",
                                                       width=reg.line_width,
                                                       dash="dash" if reg.dash == "--" else "solid"),
                                             name=loess_label, legendgroup="loess_data",
                                             showlegend=_legend_once(fig, "loess_data")),
                                  row=row, col=col)
                except Exception:
                    pass

    if reg.benchmark and bench_df is not None:  # LOESS through the benchmark scatter (new option).
        bsub = bench_year[[scatter_by, var]].dropna()
        if not bsub.empty and bsub[scatter_by].nunique() > 1:
            try:
                xs, ys = _loess(bsub[scatter_by], bsub[var], reg.loess_frac, reg.robust_iter)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                         line=dict(color=reg.benchmark_color, width=reg.line_width,
                                                   dash="dash" if reg.dash == "--" else "solid"),
                                         name=str(ctx.labels.lookup("LOESS (benchmark)")),
                                         legendgroup="loess_benchmark",
                                         showlegend=_legend_once(fig, "loess_benchmark")),
                              row=row, col=col)
            except Exception:
                pass

    fig.update_xaxes(title_text=str(ctx.labels.lookup(scatter_by)), row=row, col=col)
    fig.update_yaxes(title_text=str(ctx.labels.lookup(var)), row=row, col=col)
    if var in lay.var_window:  # Per-variable window on the variable (y) axis.
        fig.update_yaxes(range=list(lay.var_window[var]), row=row, col=col)


def render_multiplot_plotly(ctx: GridContext, plan: GridPlan) -> PlotResult:
    """Render the N x 3 (or 3 x N) diagnostic grid with Plotly.

    The distribution column is drawn horizontally (variable on the y-axis) and the
    scatter uses the final-year benchmark slice, so both backends are consistent.

    Parameters
    ----------
    ctx, plan:
        Validated grid context and precomputed plan.

    Returns
    -------
    PlotResult
        Figure (no axes) and any written file paths.
    """
    enc, lay = ctx.encoding, ctx  # Shorthands.
    dist = ctx.dist  # Distribution spec (kind/norm/toggles).
    benchmark = ctx.benchmark  # The comparison/reference frame, or None.
    variables = plan.variables  # Variables, one per grid line.
    by, line_by, scatter_by = enc.by, enc.line_by, enc.scatter_by  # Roles.
    ncols = plan.ncols  # Number of variables (the "N").
    year = ctx.year  # Cross-section year.

    columns = list(lay.columns)  # Selected diagnostic cells, in order.
    pos = {kind: i for i, kind in enumerate(columns)}  # Cell name -> column index.
    ncells = len(columns)  # The grid's second dimension (was hard-coded 3).

    nrows_g, ncols_g = (ncells, ncols) if lay.variables_by == "col" else (ncols, ncells)  # Grid shape.

    def _cell_title(v, kind):  # Per-cell subplot title.
        if kind == "series":
            return str(ctx.labels.lookup(v))
        if kind == "dist":
            return (f"{ctx.labels.lookup(v)} distribution"
                    + (f" ({line_by}={year})" if year is not None else ""))
        return f"{ctx.labels.lookup(v)} vs<br>{ctx.labels.lookup(scatter_by)}"  # scatter (wrapped after "vs")

    # Subplot titles in the grid's row-major order, only for the selected cells.
    if lay.variables_by == "col":  # ncells rows (cells) x N cols (variables).
        titles = [_cell_title(v, k) for k in columns for v in variables]
    else:  # N rows (variables) x ncells cols (cells).
        titles = [_cell_title(v, k) for v in variables for k in columns]

    sub_kw = {}  # Relative cell sizes along the cell axis (Plotly wants normalised fractions).
    if lay.cell_ratios is not None:
        frac = [r / sum(lay.cell_ratios) for r in lay.cell_ratios]
        sub_kw["row_heights" if lay.variables_by == "col" else "column_widths"] = frac
    fig = make_subplots(rows=nrows_g, cols=ncols_g, subplot_titles=titles,
                        shared_xaxes=lay.shared_xaxes, shared_yaxes=_grid_shared_yaxes_plotly(lay),
                        horizontal_spacing=0.05, vertical_spacing=0.10, **sub_kw)

    shown: set = set()  # Legend de-duplication across variable lines.

    for j0, var in enumerate(variables):  # One line per variable.
        # Benchmark overlays are per-variable: a variable absent from the
        # benchmark is plotted normally, just without its benchmark overlay.
        bench_df = benchmark if (benchmark is not None and var in benchmark.columns) else None
        if benchmark is not None and bench_df is None:  # Present overall but missing this variable.
            warnings.warn(f"Variable {var!r} is absent from the benchmark; "
                          f"its benchmark overlay is skipped.", stacklevel=2)

        if "series" in pos:  # Cell: time series vs line_by.
            r, c = _grid_rowcol(lay.variables_by, j0 + 1, pos["series"])
            _plotly_cell_series(fig, ctx, plan, var, bench_df, r, c, shown, j0 == 0)
        if "dist" in pos:  # Cell: distribution at the final year.
            r, c = _grid_rowcol(lay.variables_by, j0 + 1, pos["dist"])
            _plotly_cell_dist(fig, ctx, plan, var, bench_df, r, c)
        if "scatter" in pos:  # Cell: scatter at the final year.
            r, c = _grid_rowcol(lay.variables_by, j0 + 1, pos["scatter"])
            _plotly_cell_scatter(fig, ctx, plan, var, bench_df, r, c)

        for ci in range(ncells):  # Y numbers on every cell; name only on the first.
            r, c = _grid_rowcol(lay.variables_by, j0 + 1, ci)
            fig.update_yaxes(showticklabels=True, row=r, col=c)  # Even when y is shared.
            if ci != 0:
                fig.update_yaxes(title_text="", row=r, col=c)  # Drop the repeated indicator name.

    if dist.kind == "hist":  # Overlay the histograms.
        fig.update_layout(barmode="overlay")

    fig.update_layout(template="simple_white",  # Figure-level styling.
                      width=lay.plotly_size_per_col[0] * ncols_g,
                      height=lay.plotly_size_per_col[1] * nrows_g,
                      legend_title=str(ctx.labels.lookup(by)) if by is not None else "")
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", showgrid=False)  # Axis styling.
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black",
                     showgrid=True, gridwidth=0.75, gridcolor="lightgrey")

    if ctx.panel_letters:  # Bold a, b, c... tags, row-major over the subplot grid.
        for idx in range(nrows_g * ncols_g):
            r_, c_ = divmod(idx, ncols_g)
            fig.add_annotation(text=f"<b>{_panel_letter(idx)}</b>", xref="x domain", yref="y domain",
                               x=0.02, y=0.98, xanchor="left", yanchor="top", showarrow=False,
                               font=dict(size=14), row=r_ + 1, col=c_ + 1)

    files: list[str] = []  # Saved paths.
    if ctx.save is not None:  # Optional export.
        files = save_plotly(fig, ctx.save.path, ctx.save.name, ctx.save.extensions)

    return PlotResult(figure=fig, axes=None, files=files, backend="plotly")  # Uniform result.


# ===========================================================================
# multi_jointplot_subfig: Matplotlib backend (SubFigure-based)
# ===========================================================================
def _draw_marginal(ax, frame: pd.DataFrame, var: str, color, fill: bool = False,
                   kind: str = "kde", alpha: float = 0.5, orient: str = "x", lw: float = 2.0,
                   zorder: float = 2.0) -> None:
    """Draw one marginal distribution of ``var`` on ``ax``.

    Parameters
    ----------
    ax:
        Target marginal axis.
    frame:
        Source data.
    var:
        Variable column.
    color:
        Line (and fill) colour.
    fill:
        Whether to fill under the curve.
    kind:
        ``"kde"`` or ``"hist"``.
    alpha:
        Opacity.
    orient:
        ``"x"`` (top marginal) or ``"y"`` (right marginal).
    lw:
        Line width.
    zorder:
        Explicit stacking order (benchmark=1 < group fills=2 < contours=2.5 <
        pooled total=3, mirroring the grid's distribution cell).
    """
    if frame is None or frame.empty or var not in frame.columns:  # Nothing to draw.
        return
    sub = frame[[var]].dropna()  # Drop NaNs.
    if sub.empty:
        return

    if kind == "kde":
        if sub[var].nunique() <= 1:  # KDE needs spread.
            return
        if orient == "x":  # Distribution along the x-axis (top marginal).
            sns.kdeplot(data=sub, x=var, ax=ax, color=color, lw=lw, fill=fill, alpha=alpha,
                        legend=False, zorder=zorder)
        else:  # Distribution along the y-axis (right marginal).
            sns.kdeplot(data=sub, y=var, ax=ax, color=color, lw=lw, fill=fill, alpha=alpha,
                        legend=False, zorder=zorder)
    elif kind == "hist":
        if orient == "x":
            sns.histplot(data=sub, x=var, ax=ax, color=color, alpha=alpha, stat="density",
                         element="step", fill=fill, linewidth=lw, legend=False, zorder=zorder)
        else:
            sns.histplot(data=sub, y=var, ax=ax, color=color, alpha=alpha, stat="density",
                         element="step", fill=fill, linewidth=lw, legend=False, zorder=zorder)
    else:
        raise ValueError("marginal kind must be 'kde' or 'hist'.")


def _style_joint_axes(ax_main, ax_top, ax_right, x_label: str, y_label: str, title: str | None) -> None:
    """Apply the shared styling to one jointplot panel's three axes."""
    ax_main.set_xlabel(x_label)  # Central plot axis labels.
    ax_main.set_ylabel(y_label)
    if title is not None:
        ax_main.set_title(title, fontsize=11, pad=6)
    ax_main.grid(axis="y", color="lightgrey", linewidth=0.75)  # Light grid.
    ax_main.grid(axis="x", color="lightgrey", linewidth=0.75, alpha=0.5)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_linewidth(1.4)
    ax_main.spines["bottom"].set_linewidth(1.4)
    for ax in (ax_top, ax_right):  # Marginals: bare, no labels/ticks/spines.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="y", labelleft=False)
        ax.grid(False)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)


def _draw_joint_panel(subfig, ctx: JointContext, plan: JointPlan, xvar: str, yvar: str,
                      bench_df: pd.DataFrame | None, legend_handles: dict,
                      benchmark_legend_handles: dict, extra_legend_handles: dict) -> dict:
    """Draw one jointplot-like panel (scatter + marginals) inside a SubFigure.

    Parameters
    ----------
    subfig:
        The Matplotlib SubFigure hosting this panel.
    ctx, plan:
        Validated context and precomputed plan.
    xvar, yvar:
        The variables on the x / y axes.
    bench_df:
        The benchmark frame for this panel, or ``None`` (already filtered to
        variables present, for robustness).
    legend_handles, benchmark_legend_handles:
        Shared dicts accumulating one handle per data / benchmark category.

    Returns
    -------
    dict
        The three axes ``{"ax_main", "ax_top", "ax_right"}``.
    """
    lay, sty, marg, reg = ctx, ctx.style, ctx.marginals, ctx.regression  # Shorthands (joint layout flat on ctx).
    by, benchmark_by = ctx.encoding.by, ctx.encoding.benchmark_by  # Roles.
    color_map, bench_cmap = plan.color_map, plan.benchmark_color_map  # Colour maps.

    gs = subfig.add_gridspec(2, 2,  # 2x2: marginals around the central scatter.
                             width_ratios=[lay.main_width_ratio, lay.marginal_width_ratio],
                             height_ratios=[lay.marginal_height_ratio, lay.main_height_ratio],
                             wspace=0.03, hspace=0.03)
    ax_top = subfig.add_subplot(gs[0, 0])  # Top marginal (x).
    ax_main = subfig.add_subplot(gs[1, 0], sharex=ax_top)  # Central scatter.
    ax_right = subfig.add_subplot(gs[1, 1], sharey=ax_main)  # Right marginal (y).
    subfig.add_subplot(gs[0, 1]).axis("off")  # Empty corner.

    # --- data prep --------------------------------------------------------
    main_cols = [xvar, yvar] + ([by] if by is not None and by in ctx.data.columns else [])
    if ctx.highlight:  # Keep the spotlight column (validated upstream) for the selection.
        h_col = next(iter(ctx.highlight))
        if h_col not in main_cols:
            main_cols.append(h_col)
    dsub = ctx.data[main_cols].dropna(subset=[xvar, yvar] + ([by] if by in main_cols else [])).copy()  # Clean data subset.

    bsub = None  # Clean benchmark subset (or None).
    if bench_df is not None:
        bcols = [xvar, yvar] + ([benchmark_by] if benchmark_by is not None
                                and benchmark_by in bench_df.columns else [])
        bsub = bench_df[bcols].dropna().copy()

    # --- axis limits (note: y from y_values -- fixes the original copy/paste bug) ---
    x_values, y_values = [], []
    if lay.xlim_mode in {"data", "both"}:
        x_values.append(dsub[xvar] if not dsub.empty else None)
    if lay.xlim_mode in {"benchmark", "both"} and bsub is not None:
        x_values.append(bsub[xvar] if not bsub.empty else None)
    if lay.ylim_mode in {"data", "both"}:
        y_values.append(dsub[yvar] if not dsub.empty else None)
    if lay.ylim_mode in {"benchmark", "both"} and bsub is not None:
        y_values.append(bsub[yvar] if not bsub.empty else None)
    xlim = compute_axis_limits(x_values, margin=lay.axis_margin)  # X range.
    ylim = compute_axis_limits(y_values, margin=lay.axis_margin)  # Y range (from y_values!).
    if ctx.symmetric_xy and xlim is not None and ylim is not None:  # Union: y = x is the symmetry axis.
        lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        xlim = ylim = (lo, hi)
    if xlim is not None:  # Hard Layout overrides win over everything above.
        xlim = (ctx.x_min if ctx.x_min is not None else xlim[0],
                ctx.x_max if ctx.x_max is not None else xlim[1])
    if ylim is not None:
        ylim = (ctx.y_min if ctx.y_min is not None else ylim[0],
                ctx.y_max if ctx.y_max is not None else ylim[1])

    # --- benchmark scatter (behind) --------------------------------------
    if bsub is not None and not bsub.empty:
        if benchmark_by is not None and benchmark_by in bsub.columns:  # Per-category benchmark.
            for bcat in dict.fromkeys(bsub[benchmark_by].dropna().tolist()):
                tmp = bsub.loc[bsub[benchmark_by] == bcat]
                if tmp.empty:
                    continue
                sc = ax_main.scatter(tmp[xvar], tmp[yvar],
                                     color=bench_cmap.get(bcat, sty.benchmark_color),
                                     alpha=sty.benchmark_alpha, s=sty.benchmark_scatter_size, zorder=1)
                benchmark_legend_handles.setdefault(bcat, sc)  # One handle per benchmark category.
        else:  # Single undifferentiated benchmark.
            sc = ax_main.scatter(bsub[xvar], bsub[yvar], color=sty.benchmark_color,
                                 alpha=sty.benchmark_alpha, s=sty.benchmark_scatter_size, zorder=1)
            extra_legend_handles.setdefault(str(ctx.labels.lookup(sty.benchmark_label)), sc)

    # --- data scatter -----------------------------------------------------
    if by is None:  # Single ungrouped scatter.
        sc = ax_main.scatter(dsub[xvar], dsub[yvar], color="C0", alpha=sty.alpha,
                             s=sty.scatter_size, zorder=2)
        legend_handles.setdefault("data", sc)
    else:  # One colour per category.
        for cat in dict.fromkeys(dsub[by].dropna().tolist()):
            tmp = dsub.loc[dsub[by] == cat]
            if tmp.empty:
                continue
            sc = ax_main.scatter(tmp[xvar], tmp[yvar], color=color_map[cat], alpha=sty.alpha,
                                 s=sty.scatter_size, zorder=2, label=str(cat))
            legend_handles.setdefault(cat, sc)

    # --- highlighted points (on top of the data scatter) -------------------
    if ctx.highlight:
        h_col, h_vals = next(iter(ctx.highlight.items()))
        h_vals = [h_vals] if isinstance(h_vals, str) or not hasattr(h_vals, "__iter__") else list(h_vals)
        hsub = dsub.loc[dsub[h_col].isin(h_vals)]
        if not hsub.empty:  # Bigger, lightly hatched (colour stays visible), black edge.
            h_kw = dict(s=2.5 * sty.scatter_size, hatch="///", facecolor="none",
                        edgecolor="black", linewidth=1.0, zorder=4.5)
            if by is None:
                ax_main.scatter(hsub[xvar], hsub[yvar], color="C0", alpha=sty.alpha,
                                s=2.5 * sty.scatter_size, zorder=4.4)
                ax_main.scatter(hsub[xvar], hsub[yvar], **h_kw)
            else:
                for cat in dict.fromkeys(hsub[by].dropna().tolist()):
                    tmp = hsub.loc[hsub[by] == cat]
                    ax_main.scatter(tmp[xvar], tmp[yvar], color=color_map[cat], alpha=sty.alpha,
                                    s=2.5 * sty.scatter_size, zorder=4.4)
                ax_main.scatter(hsub[xvar], hsub[yvar], **h_kw)
            extra_legend_handles.setdefault(  # Hatched swatch, transparent face, black edge.
                str(ctx.labels.lookup("Highlight")),
                mpatches.Patch(facecolor="none", edgecolor="black", hatch="///", linewidth=1.0))

    # --- regressions (data + optional benchmark) -------------------------
    if reg.data or reg.per_group:  # per_group implies the data fit. # LOESS through the data.
        loess_label = str(ctx.labels.lookup("LOESS (data)"))  # Legend entry (renamable via labels).
        if reg.per_group and by is not None:  # One curve per category (each regime separately).
            drew = False
            for cat in dict.fromkeys(dsub[by].dropna().tolist()):
                rsub = dsub.loc[dsub[by] == cat, [xvar, yvar]].dropna()
                if len(rsub) > 2 and rsub[xvar].nunique() > 1:
                    try:
                        xs, ys = _loess(rsub[xvar], rsub[yvar], reg.loess_frac, reg.robust_iter)
                        ax_main.plot(xs, ys, linestyle=reg.dash, linewidth=reg.line_width,
                                     color=color_map[cat], zorder=4)
                        drew = True
                    except Exception:  # Never let a smoothing failure kill the figure.
                        pass
            if drew:  # One neutral legend entry for the style (colours are the groups').
                proxy, = ax_main.plot([], [], linestyle=reg.dash, linewidth=reg.line_width,
                                      color="dimgray")
                extra_legend_handles.setdefault(loess_label, proxy)
        else:  # Single pooled curve through all data points.
            rsub = dsub[[xvar, yvar]].dropna()
            if not rsub.empty and rsub[xvar].nunique() > 1:
                try:
                    xs, ys = _loess(rsub[xvar], rsub[yvar], reg.loess_frac, reg.robust_iter)
                    ln, = ax_main.plot(xs, ys, linestyle=reg.dash, linewidth=reg.line_width,
                                       color=reg.data_color or "black", zorder=4)
                    extra_legend_handles.setdefault(loess_label, ln)
                except Exception:
                    pass
    if reg.benchmark and bsub is not None and not bsub.empty:  # LOESS through the benchmark.
        rb = bsub[[xvar, yvar]].dropna()
        if not rb.empty and rb[xvar].nunique() > 1:
            try:
                xs, ys = _loess(rb[xvar], rb[yvar], reg.loess_frac, reg.robust_iter)
                ln, = ax_main.plot(xs, ys, linestyle=reg.dash, linewidth=reg.line_width,
                                   color=reg.benchmark_color, zorder=3.5)
                extra_legend_handles.setdefault(str(ctx.labels.lookup("LOESS (benchmark)")), ln)
            except Exception:
                pass

    # --- x = y diagonal (dashed, bold) + optional quartile guides ---------
    if (lay.diag_line or lay.diag_quartiles) and xlim is not None and ylim is not None:
        lo = min(xlim[0], ylim[0])  # Common lower bound.
        hi = max(xlim[1], ylim[1])  # Common upper bound.
        if np.isfinite(lo) and np.isfinite(hi):
            if not np.isclose(lo, hi):  # Optionally extend the line a touch.
                dr = hi - lo
                lo -= lay.diag_margin * dr
                hi += lay.diag_margin * dr
            if lay.diag_line:
                ax_main.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5,
                             color="black", alpha=0.8, zorder=1.5)
            if lay.diag_quartiles:  # y = 0.75x / 0.5x / 0.25x: thinner, lighter grey.
                for frac in (0.75, 0.5, 0.25):
                    ax_main.plot([lo, hi], [frac * lo, frac * hi], linestyle="--",
                                 linewidth=1.0, color="darkgray", alpha=0.8, zorder=1.4)
                # Percentage tags at each line's upper-right end: 25/50/75 just above
                # their line, 100 just below y = x (above would clip at the axes top).
                x_anchor = xlim[1] - 0.02 * (xlim[1] - xlim[0])
                y_off = 0.012 * (ylim[1] - ylim[0])
                tagged = [(1.0, "100", "black", -1)] if lay.diag_line else []
                tagged += [(0.75, "75", "dimgray", +1), (0.5, "50", "dimgray", +1),
                           (0.25, "25", "dimgray", +1)]
                for frac, tag, colr, side in tagged:
                    ax_main.text(x_anchor, frac * x_anchor + side * y_off, tag, fontsize=8,
                                 color=colr, ha="right", va="bottom" if side > 0 else "top",
                                 alpha=0.9, zorder=1.6, clip_on=True)

    # --- marginals --------------------------------------------------------
    if marg.show:
        if bsub is not None and not bsub.empty:  # Benchmark marginals first (behind, zorder=1).
            if benchmark_by is not None and benchmark_by in bsub.columns:  # Per-category.
                for bcat in dict.fromkeys(bsub[benchmark_by].dropna().tolist()):
                    tmp = bsub.loc[bsub[benchmark_by] == bcat]
                    if tmp.empty:
                        continue
                    bcolor = bench_cmap.get(bcat, sty.benchmark_color)
                    _draw_marginal(ax_top, tmp, xvar, bcolor, kind=marg.kind,
                                   alpha=sty.benchmark_alpha, orient="x", lw=1.8, zorder=1)
                    _draw_marginal(ax_right, tmp, yvar, bcolor, kind=marg.kind,
                                   alpha=sty.benchmark_alpha, orient="y", lw=1.8, zorder=1)
            else:  # Single benchmark: grey filled area + black line on top.
                for ax_m, vv, orient in ((ax_top, xvar, "x"), (ax_right, yvar, "y")):
                    _draw_marginal(ax_m, bsub, vv, sty.benchmark_color, fill=True,
                                   kind=marg.kind, alpha=sty.benchmark_alpha, orient=orient,
                                   lw=0.0, zorder=1)
                    _draw_marginal(ax_m, bsub, vv, sty.benchmark_marginal_color, fill=False,
                                   kind=marg.kind, alpha=1.0, orient=orient, lw=1.8, zorder=1)

        def _group_marginal(frame: pd.DataFrame, colr) -> None:
            """One background marginal curve (top + right) in ``colr``.

            Unfilled; ``group_alpha`` applies to the line itself. Colours come
            from ``plan.color_map`` -- the same map the scatter and its
            per-group regression read, so all agree by construction.
            """
            for ax_m, vv, orient in ((ax_top, xvar, "x"), (ax_right, yvar, "y")):
                _draw_marginal(ax_m, frame, vv, colr, fill=False, kind=marg.kind,
                               alpha=marg.group_alpha, orient=orient, lw=2.0, zorder=2)

        if by is None:  # Data marginals (single), filled like a group.
            _group_marginal(dsub, "C0")
        else:  # Data marginals (per category), filled, as a background layer.
            for cat in dict.fromkeys(dsub[by].dropna().tolist()):
                tmp = dsub.loc[dsub[by] == cat]
                if not tmp.empty:
                    _group_marginal(tmp, color_map[cat])

        if marg.show_total:  # Pooled all-groups curve on top of both marginals.
            for ax_m, vv, orient in ((ax_top, xvar, "x"), (ax_right, yvar, "y")):
                _draw_marginal(ax_m, dsub, vv, marg.total_color, fill=False, kind=marg.kind,
                               alpha=1.0, orient=orient, lw=2.2 if marg.kind == "kde" else 2.0,
                               zorder=3)
            proxy, = ax_main.plot([], [], color=marg.total_color,
                                  lw=2.2 if marg.kind == "kde" else 2.0)  # Legend proxy.
            extra_legend_handles.setdefault(str(ctx.labels.lookup(marg.total_label)), proxy)

    # --- limits + styling -------------------------------------------------
    if xlim is not None:
        ax_main.set_xlim(*xlim)
        ax_top.set_xlim(*xlim)
    if ylim is not None:
        ax_main.set_ylim(*ylim)
        ax_right.set_ylim(*ylim)

    x_label = str(ctx.labels.lookup(xvar))  # Display names.
    y_label = str(ctx.labels.lookup(yvar))
    _style_joint_axes(ax_main, ax_top, ax_right, x_label, y_label, title=f"{y_label} vs {x_label}")

    return {"ax_main": ax_main, "ax_top": ax_top, "ax_right": ax_right}  # The panel's axes.


def render_multi_jointplot(ctx: JointContext, plan: JointPlan) -> PlotResult:
    """Render the multi-panel jointplot figure (Matplotlib SubFigures).

    Parameters
    ----------
    ctx, plan:
        Validated context and precomputed plan.

    Returns
    -------
    PlotResult
        Figure, the list of per-panel axes dicts, and any written file paths.
    """
    benchmark = ctx.benchmark  # Shorthand.
    nrows, ncols = plan.nrows, plan.ncols  # Grid shape.

    with mpl.rc_context(dict(sns.axes_style("white"))):  # Scoped "white" style (no global side effects).
        fig = plt.figure(figsize=(ctx.figsize_per_plot[0] * ncols,
                                  ctx.figsize_per_plot[1] * nrows),
                         constrained_layout=True)
        subfigs = fig.subfigures(nrows=nrows, ncols=ncols)  # One SubFigure per panel.

        # Normalise the SubFigure container to a 2-D array for uniform indexing.
        subfigs = np.atleast_2d(subfigs)
        if nrows == 1 and ncols == 1:
            subfigs = subfigs.reshape(1, 1)
        elif ncols == 1:
            subfigs = subfigs.reshape(nrows, 1)
        elif nrows == 1:
            subfigs = subfigs.reshape(1, ncols)

        legend_handles: dict = {}  # One handle per data category.
        benchmark_legend_handles: dict = {}  # One handle per benchmark category.
        extra_legend_handles: dict = {}  # Benchmark (single), LOESS and marginal-total entries.
        panel_axes: list[dict] = []  # The per-panel axes dicts.

        for idx, (xvar, yvar) in enumerate(plan.pairs):  # One panel per variable pair.
            subfig = subfigs[idx // ncols, idx % ncols]  # Target SubFigure.

            # Per-panel benchmark robustness: drop the benchmark for this panel if
            # either variable is absent from it (plot the data panel regardless).
            bench_df = benchmark
            if benchmark is not None and not ({xvar, yvar} <= set(benchmark.columns)):
                bench_df = None
                missing = [v for v in (xvar, yvar) if v not in benchmark.columns]
                warnings.warn(f"Variable(s) {missing} absent from the benchmark; "
                              f"benchmark overlay skipped for panel ({xvar}, {yvar}).", stacklevel=2)

            panel_axes.append(_draw_joint_panel(subfig, ctx, plan, xvar, yvar, bench_df,
                                                legend_handles, benchmark_legend_handles,
                                                extra_legend_handles))
            if ctx.panel_letters:  # Bold a, b, c... tag, one per panel.
                panel_axes[-1]["ax_main"].text(0.02, 0.98, _panel_letter(idx),
                                               transform=panel_axes[-1]["ax_main"].transAxes,
                                               fontweight="bold", fontsize=12, va="top",
                                               ha="left", zorder=10)

        for idx in range(len(plan.pairs), nrows * ncols):  # Blank any unused panels.
            subfigs[idx // ncols, idx % ncols].subplots().axis("off")

        # Shared legend: data categories, then benchmark categories.
        handles = list(legend_handles.values()) + list(benchmark_legend_handles.values()) \
            + list(extra_legend_handles.values())
        labels = [str(k) for k in legend_handles] \
            + [f"{ctx.labels.lookup(ctx.style.benchmark_label)}: {k}" for k in benchmark_legend_handles] \
            + list(extra_legend_handles)  # Keys are already final display labels.
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       title=str(ctx.labels.lookup(ctx.encoding.by)) if ctx.encoding.by is not None else "",
                       bbox_to_anchor=(0.5, ctx.legend_y),
                       ncol=_legend_ncol(len(labels), ctx), frameon=False)

        _rasterize_data(fig, ctx.style.rasterized)  # Printable vector output.
        files: list[str] = []  # Saved paths.
        if ctx.save is not None:  # Optional export.
            files = save_matplotlib(fig, ctx.save.path, ctx.save.name, ctx.save.extensions, ctx.save.dpi)

    return PlotResult(figure=fig, axes=panel_axes, files=files, backend="matplotlib")  # Uniform result.


# ===========================================================================
# grouped_boxplot: categorical distribution comparison (box / violin / strip)
# ===========================================================================
def _swarm_offsets(n: int, max_width: float) -> np.ndarray:
    """Symmetric horizontal offsets approximating a compact swarm.

    Not a true collision-avoidance swarm: points are spread evenly across
    ``[-max_width, max_width]`` and reordered so the densest region sits around
    the centre. Visually effective and aligned with the manual box positions.
    """
    if n <= 1:
        return np.array([0.0])  # A single point sits on the centre.
    offsets = np.linspace(-max_width, max_width, n)  # Even spread.
    center = (n - 1) / 2  # Index of the middle point.
    order = np.argsort(np.abs(np.arange(n) - center))  # Fill from the centre outwards.
    return offsets[order]


def _box_points_mpl(ax, vals, xpos: float, color, sty, zorder: int = 5) -> None:
    """Overlay the individual ``vals`` as a compact swarm around ``xpos``."""
    vals = np.asarray(vals)
    if vals.size == 0:
        return
    vals_sorted = np.sort(vals)  # Sorting makes the swarm shape stable.
    offsets = _swarm_offsets(len(vals_sorted), max_width=sty.box_width * sty.point_jitter)
    ax.scatter(np.full(len(vals_sorted), xpos) + offsets, vals_sorted,
               s=sty.point_size, alpha=sty.point_alpha, color=color, zorder=zorder, linewidths=0)


def _box_mark_mpl(ax, vals, xpos: float, color, fill_alpha: float, kind: str, sty,
                  point_zorder: int, fill: bool = True, line_color: str = "black") -> None:
    """Draw one box / violin / strip of ``vals`` centred at ``xpos`` in ``color``.

    ``fill=False`` draws an unfilled mark with all lines in ``line_color`` (used
    by the pooled "total" mark); the default is the filled mark with black lines.
    """
    vals = np.asarray(vals)
    if vals.size == 0:  # Nothing to draw for an empty (x, group) cell.
        return
    point_color = sty.point_color if sty.point_color is not None else color

    if kind == "box":
        ax.boxplot(vals, positions=[xpos], widths=sty.box_width, patch_artist=True,
                   manage_ticks=False,
                   boxprops=(dict(facecolor=color, alpha=fill_alpha, edgecolor=line_color) if fill
                             else dict(facecolor="none", edgecolor=line_color, linewidth=1.6)),
                   medianprops=dict(color=line_color),
                   whiskerprops=dict(color=line_color), capprops=dict(color=line_color))
        if sty.show_points:  # Optional swarm overlay.
            _box_points_mpl(ax, vals, xpos, point_color, sty, zorder=point_zorder)

    elif kind == "violin":
        parts = ax.violinplot([vals], positions=[xpos], widths=sty.box_width,
                              showmeans=False, showmedians=True, showextrema=False)
        for body in parts["bodies"]:  # Colour each violin body.
            body.set_facecolor(color if fill else "none")
            body.set_alpha(fill_alpha if fill else 1.0)
            body.set_edgecolor(line_color)
            if not fill:
                body.set_linewidth(1.6)
        if "cmedians" in parts:  # Median bar (black, or line_color for the unfilled mark).
            parts["cmedians"].set_color(line_color)
        if sty.show_points:
            _box_points_mpl(ax, vals, xpos, point_color, sty, zorder=point_zorder)

    else:  # kind == "strip": the points are the mark.
        _box_points_mpl(ax, vals, xpos, point_color, sty, zorder=point_zorder)


def _draw_box_hierarchy(ax, ctx: BoxContext, plan: BoxPlan, lo: float, hi: float) -> None:
    """Draw x-hierarchy bracket annotations beneath the axis (matplotlib only).

    Mirrors :func:`_draw_hierarchy` (plot_allocations): contiguous runs of ``x``
    sharing an ``x_hierarchy`` value get a bracket placed just below ``lo`` in the
    reserved room, so positions stay correct under any y-limit change.
    """
    enc = ctx.encoding  # Shorthand.
    if enc.x_hierarchy is None:  # Nothing to annotate.
        return

    sector_of = {}  # x value -> its coarse sector label.
    for xval in plan.xcats:
        vals = ctx.data.loc[ctx.data[enc.x].eq(xval), enc.x_hierarchy].dropna().unique()
        sector_of[xval] = vals[0] if len(vals) else ""  # First, or empty.

    sectors = []  # Contiguous runs (label, i_start, i_end).
    current, start = None, 0
    for i, xval in enumerate(plan.xcats):
        sec = sector_of[xval]
        if current is None:  # Open first run.
            current, start = sec, i
        elif sec != current:  # Close previous, open new.
            sectors.append((current, start, i - 1))
            current, start = sec, i
    sectors.append((current, start, len(plan.xcats) - 1))  # Close last run.

    span = hi - lo  # Range used to scale offsets.
    line_y = lo + ctx.style.hierarchy_line_offset * span  # Bracket line just below the data.
    text_y = lo + (ctx.style.hierarchy_line_offset + ctx.style.hierarchy_text_offset) * span  # Label.

    for sec, i0, i1 in sectors:  # One bracket per run, spanning the full slot blocks.
        x0 = plan.centers[i0] - 0.5  # Left edge of the first block.
        x1 = plan.centers[i1] + (plan.n_slots - 1) + 0.5  # Right edge of the last block.
        ax.plot([x0, x1], [line_y, line_y], color="black", lw=1.0, clip_on=False)  # Bracket.
        ax.text((x0 + x1) / 2, text_y, str(ctx.labels.lookup(sec, enc.x_hierarchy)),  # Label.
                ha="center", va="top", clip_on=False)


def render_scatter_explain(jctx: JointContext, jplan: JointPlan, rows_ctx: list,
                           layout, style, figsize: tuple[float, float],
                           width_ratios: tuple[float, float], save) -> PlotResult:
    """Composite renderer: per row, one joint panel (left) + two allocation panels (right).

    Reuses :func:`_draw_joint_panel` and :func:`_draw_bar_panel` /
    :func:`_draw_line_panel` verbatim; only the arrangement and the two
    figure-level half-width legends are new. Matplotlib only.
    """
    rc = dict(sns.axes_style("white"))
    rc["font.size"] = style.font_size
    if style.font_family is not None:
        rc["font.family"] = style.font_family
    n_rows = len(rows_ctx)

    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows))
        sfs = np.atleast_2d(fig.subfigures(n_rows, 2, width_ratios=list(width_ratios),
                                           wspace=0.01, hspace=0.12)).reshape(n_rows, 2)

        legend_handles: dict = {}  # Accumulated across rows (deduplicated by label).
        benchmark_legend_handles: dict = {}
        extra_legend_handles: dict = {}
        axes: list[dict] = []
        n_letter = 0  # Visible-panel counter for Layout.panel_letters.
        for ri, ((left_sf, right_sf), row) in enumerate(zip(sfs, rows_ctx)):
            # ---- left: the joint panel (all its options come with it) ----
            xvar, yvar = jplan.pairs[ri]
            bench_df = jctx.benchmark  # Per-panel robustness, as in render_multi_jointplot.
            if bench_df is not None and not ({xvar, yvar} <= set(bench_df.columns)):
                missing = [v for v in (xvar, yvar) if v not in bench_df.columns]
                warnings.warn(f"Variable(s) {missing} absent from the benchmark; "
                              f"benchmark overlay skipped for panel ({xvar}, {yvar}).",
                              stacklevel=2)
                bench_df = None
            joint_axes = _draw_joint_panel(left_sf, jctx, jplan, xvar, yvar, bench_df,
                                           legend_handles, benchmark_legend_handles,
                                           extra_legend_handles)

            # ---- right: two allocation panels side by side (hidden slots stay empty) ----
            right_axes = right_sf.subplots(1, 2)
            visible = [(ax, slot) for ax, slot in zip(right_axes, row) if slot is not None]
            for ax, slot in zip(right_axes, row):
                if slot is None:  # Hidden panel: keep the slot, drop the axis.
                    ax.axis("off")
                    continue
                pctx, pplan, ptitle = slot
                if pctx.graph_type == "bar":
                    _draw_bar_panel(ax, pctx, pplan, None, None)
                else:
                    _draw_line_panel(ax, pctx, pplan, None, None)
                _style_axis(ax)
                ax.set_title(ptitle, fontsize=style.font_size + 1, fontweight="bold", pad=8)
            if layout.share_y and len(visible) > 1:  # Unify the row's visible y ranges.
                lo = min(a.get_ylim()[0] for a, _ in visible)
                hi = max(a.get_ylim()[1] for a, _ in visible)
                for ax, _ in visible:
                    ax.set_ylim(lo, hi)
            for ax, (pctx, pplan, _) in visible:  # Brackets need final limits.
                lo, hi = ax.get_ylim()
                has_hier = pctx.encoding.x_hierarchy is not None and pctx.graph_type == "bar"
                room = style.hierarchy_room * (hi - lo) if has_hier else 0.0
                ax.set_ylim(lo - room, hi)
                _draw_hierarchy(ax, pctx, pplan, pctx.data, lo, hi)

            top = 0.86 if ri == 0 else 0.94  # Only the first row gives room to the legends.
            left_sf.subplots_adjust(top=top, right=0.985)
            right_sf.subplots_adjust(top=top, left=0.05, right=0.99, wspace=0.28)

            if layout.panel_letters:  # Row-major over *visible* panels only.
                joint_axes["ax_main"].text(
                    0.02, 0.98, _panel_letter(n_letter), transform=joint_axes["ax_main"].transAxes,
                    fontweight="bold", fontsize=12, va="top", ha="left", zorder=10)
                n_letter += 1
                for ax, _ in visible:
                    ax.text(0.02, 0.98, _panel_letter(n_letter), transform=ax.transAxes,
                            fontweight="bold", fontsize=12, va="top", ha="left", zorder=10)
                    n_letter += 1
            axes.append({"joint": joint_axes, "alloc_x": right_axes[0],
                         "alloc_y": right_axes[1]})

        # ---- two legends, each over its half of the figure ----
        l_handles = list(legend_handles.values()) + list(benchmark_legend_handles.values()) \
            + list(extra_legend_handles.values())
        l_labels = [str(k) for k in legend_handles] \
            + [f"{jctx.labels.lookup(jctx.style.benchmark_label)}: {k}"
               for k in benchmark_legend_handles] \
            + list(extra_legend_handles)
        if l_handles:
            fig.legend(l_handles, l_labels, loc="lower center",
                       bbox_to_anchor=(width_ratios[0] / (2 * sum(width_ratios)), 0.99),
                       ncol=_legend_ncol(len(l_labels), layout), frameon=False,
                       title=str(jctx.labels.lookup(jctx.encoding.by))
                       if jctx.encoding.by is not None else "")
        first = next(slot for row in rows_ctx for slot in row if slot is not None)
        r_handles = _legend_handles(*first[:2])  # Same vocabulary on every panel.
        if r_handles:
            fig.legend(handles=r_handles, loc="lower center",
                       bbox_to_anchor=(1 - width_ratios[1] / (2 * sum(width_ratios)), 0.99),
                       ncol=_legend_ncol(len(r_handles), layout), frameon=False)

        if layout.title:
            fig.suptitle(layout.title, fontsize=style.font_size + 4, fontweight="bold", y=1.06)

        _rasterize_data(fig, style.rasterized)  # Printable vector output.
        files: list[str] = []
        if save is not None:
            files = save_matplotlib(fig, save.path, save.name, save.extensions, save.dpi)

    return PlotResult(figure=fig, axes=axes, files=files, backend="matplotlib")


def _filter_box_panel(df: pd.DataFrame, ctx: BoxContext, row_val: Any, col_val: Any) -> pd.DataFrame:
    """Slice a frame to one facet panel (boxplot: facets live on the context)."""
    out = df
    if ctx.facets.row is not None:
        out = out[out[ctx.facets.row].eq(row_val)]
    if ctx.facets.col is not None:
        out = out[out[ctx.facets.col].eq(col_val)]
    return out


def _draw_box_panel(ax, ctx: BoxContext, plan: BoxPlan,
                    data: pd.DataFrame, benchmark: pd.DataFrame | None) -> tuple[list, list]:
    """Draw one boxplot panel (all x slots) and return its tick positions / labels."""
    enc, sty, kind = ctx.encoding, ctx.style, ctx.kind
    xcats, bycats, bbycats = plan.xcats, plan.bycats, plan.bbycats
    color_map, bcm = plan.color_map, plan.benchmark_color_map
    xtick_positions, xtick_labels = [], []

    for i, xcat in enumerate(xcats):  # One slot block per x category.
        center = plan.centers[i]
        xtick_positions.append(center + (plan.n_slots - 1) / 2 if plan.n_slots > 1 else center)
        xtick_labels.append(str(ctx.labels.lookup(xcat, enc.x)))
        pos = 0  # Sub-position within the block.

        # ---- data marks ----
        if enc.by is None:  # Single ungrouped distribution.
            vals = data.loc[data[enc.x] == xcat, enc.y].dropna().values
            _box_mark_mpl(ax, vals, center + pos, "C0", sty.data_alpha, kind, sty, point_zorder=5)
            pos += 1
        else:  # One mark per data group.
            for cat in bycats:
                vals = data.loc[(data[enc.x] == xcat) & (data[enc.by] == cat), enc.y].dropna().values
                _box_mark_mpl(ax, vals, center + pos, color_map[cat], sty.data_alpha, kind, sty,
                              point_zorder=5)
                pos += 1

        # ---- pooled "total" mark (all data groups together, unfilled) ----
        if ctx.show_total:
            vals = data.loc[data[enc.x] == xcat, enc.y].dropna().values
            _box_mark_mpl(ax, vals, center + pos, ctx.total_color, 1.0, kind, sty,
                          point_zorder=5, fill=False, line_color=ctx.total_color)
            pos += 1

        # ---- benchmark marks ----
        if benchmark is not None:
            if enc.benchmark_by is None:  # Single ungrouped benchmark block.
                vals = benchmark.loc[benchmark[enc.x] == xcat, enc.y].dropna().values
                _box_mark_mpl(ax, vals, center + pos, sty.benchmark_color, sty.benchmark_alpha,
                              kind, sty, point_zorder=4)
                pos += 1
            else:  # One mark per benchmark group.
                for bcat in bbycats:
                    vals = benchmark.loc[(benchmark[enc.x] == xcat)
                                         & (benchmark[enc.benchmark_by] == bcat), enc.y].dropna().values
                    _box_mark_mpl(ax, vals, center + pos, bcm.get(bcat, sty.benchmark_color),
                                  sty.benchmark_alpha, kind, sty, point_zorder=4)
                    pos += 1

    return xtick_positions, xtick_labels


def _box_legend_entries(ctx: BoxContext, plan: BoxPlan) -> tuple[list, list]:
    """Legend handles/labels: data groups, pooled total, then benchmark groups."""
    enc, sty = ctx.encoding, ctx.style
    handles, labels = [], []
    if enc.by is not None:
        for cat in plan.bycats:
            handles.append(mpatches.Patch(color=plan.color_map[cat], alpha=sty.data_alpha))
            labels.append(str(ctx.labels.lookup(cat, enc.by)))
    else:
        handles.append(mpatches.Patch(color="C0", alpha=sty.data_alpha))
        labels.append("data")
    if ctx.show_total:  # Unfilled patch matching the pooled mark.
        handles.append(mpatches.Patch(facecolor="none", edgecolor=ctx.total_color, linewidth=1.6))
        labels.append(str(ctx.labels.lookup(ctx.total_label)))
    if ctx.benchmark is not None:
        if enc.benchmark_by is not None:
            for bcat in plan.bbycats:
                handles.append(mpatches.Patch(color=plan.benchmark_color_map.get(bcat, sty.benchmark_color),
                                              alpha=sty.benchmark_alpha))
                labels.append(f"{ctx.labels.lookup(sty.benchmark_label)}: "
                              f"{ctx.labels.lookup(bcat, enc.benchmark_by)}")
        else:
            handles.append(mpatches.Patch(color=sty.benchmark_color, alpha=sty.benchmark_alpha))
            labels.append(str(ctx.labels.lookup(sty.benchmark_label)))
    return handles, labels


def render_box_matplotlib(ctx: BoxContext, plan: BoxPlan) -> PlotResult:
    """Render :func:`grouped_boxplot` with Matplotlib.

    For each ``x`` category, data groups are drawn first (coloured by the
    palette), then the benchmark groups beside them (grey / benchmark colours).
    ``x_hierarchy`` brackets are drawn after the y-limits are finalised. With
    ``facet_col`` / ``facet_row`` set, the same slot layout is repeated in every
    panel of the facet grid (shared category order, so panels are comparable).
    """
    enc, sty, lay = ctx.encoding, ctx.style, ctx.layout  # Shorthands.
    faceted = ctx.facets.row is not None or ctx.facets.col is not None

    rc = {"font.size": sty.font_size}  # Base typography.
    if sty.font_family is not None:
        rc["font.family"] = sty.font_family

    with mpl.rc_context({**dict(sns.axes_style("white")), **rc}):  # Scoped style (no side effects).
        figsize = ((ctx.figsize[0] * plan.ncols, ctx.figsize[1] * plan.nrows) if faceted
                   else ctx.figsize)
        fig, axes = plt.subplots(nrows=plan.nrows, ncols=plan.ncols, figsize=figsize,
                                 squeeze=False, sharey=ctx.facets.share_y)
        lx = lay.x_title if lay.x_title is not None else str(ctx.labels.lookup(enc.x))  # Axis titles.
        ly = lay.y_title if lay.y_title is not None else str(ctx.labels.lookup(enc.y))
        drawn = []  # Visible axes (facet panels can be empty).

        for k, (row_val, col_val) in enumerate(plan.panels):  # Row-major layout.
            ri, ci = divmod(k, plan.ncols)
            ax = axes[ri, ci]
            sub = _filter_box_panel(ctx.data, ctx, row_val, col_val)
            bsub = (_filter_box_panel(ctx.benchmark, ctx, row_val, col_val)
                    if ctx.benchmark is not None else None)
            if faceted and ctx.facets.hide_empty and sub.empty:  # Blank empty panels.
                _hide_axis(ax)
                continue
            xtick_positions, xtick_labels = _draw_box_panel(ax, ctx, plan, sub, bsub)
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel("")  # The x title is placed at figure level (vs x_hierarchy).
            ax.set_ylabel(ly if (not faceted or ci == 0) else "")
            ax.set_title(panel_title(ctx, row_val, col_val) if faceted
                         else (lay.title if lay.title is not None else f"{ly}, by {lx}"), pad=8)

            if lay.y_min is not None or lay.y_max is not None:  # Optional hard y-limits.
                cur = ax.get_ylim()
                ax.set_ylim(lay.y_min if lay.y_min is not None else cur[0],
                            lay.y_max if lay.y_max is not None else cur[1])

            ax.grid(axis="y", color="lightgrey", linewidth=0.75)  # Styling (no y=0 baseline).
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            if sty.tick_label_size is not None:
                ax.tick_params(axis="both", labelsize=sty.tick_label_size)

            if enc.x_hierarchy is not None:  # Brackets after limits are finalised.
                lo, hi = ax.get_ylim()
                _draw_box_hierarchy(ax, ctx, plan, lo, hi)
                ax.set_ylim(lo - sty.hierarchy_room * (hi - lo), hi)  # Room below for the labels.
            drawn.append(ax)

        if lay.panel_letters:  # Bold a, b, c... over the visible panels (row-major).
            for idx, ax_ in enumerate(drawn):
                ax_.text(0.02, 0.98, _panel_letter(idx) if faceted else "a",
                         transform=ax_.transAxes, fontweight="bold",
                         fontsize=sty.font_size + 2, va="top", ha="left", zorder=10)

        fig.supxlabel(lx, fontweight="bold", y=lay.x_title_y)  # Figure-level x title.
        if faceted and lay.title is not None:
            fig.suptitle(lay.title, fontsize=sty.font_size + 4, fontweight="bold")

        handles, labels = _box_legend_entries(ctx, plan) if ctx.show_legend else ([], [])
        _finalize_top_legend(fig, handles, lay, labels=labels,  # Balanced legend + layout.
                             title=str(ctx.labels.lookup(enc.by)) if enc.by is not None else None)
        if lay.wspace is not None or lay.hspace is not None:  # Explicit panel spacing.
            fig.subplots_adjust(wspace=lay.wspace, hspace=lay.hspace)

        _rasterize_data(fig, ctx.style.rasterized)  # Printable vector output.
        files: list[str] = []
        if ctx.save is not None:
            files = save_matplotlib(fig, ctx.save.path, ctx.save.name, ctx.save.extensions, ctx.save.dpi)

    return PlotResult(figure=fig, axes=(axes if faceted else axes[0, 0]), files=files,
                      backend="matplotlib")



def render_box_plotly(ctx: BoxContext, plan: BoxPlan) -> PlotResult:
    """Render :func:`grouped_boxplot` with Plotly (grouped boxes / violins / strips).

    Grouping across ``x`` and side-by-side placement of the data and benchmark
    groups is handled by Plotly's ``boxmode/violinmode="group"`` and per-group
    ``offsetgroup``. ``x_hierarchy`` is a Matplotlib-only feature and is ignored
    here (with a warning), mirroring :func:`plot_allocations`.
    """
    enc, sty, lay = ctx.encoding, ctx.style, ctx.layout  # Shorthands.
    data, benchmark, kind = ctx.data, ctx.benchmark, ctx.kind  # Frames + mark.
    xcats, bycats, bbycats = plan.xcats, plan.bycats, plan.bbycats  # Ordered categories.
    color_map, bcm = plan.color_map, plan.benchmark_color_map  # Colour maps.

    if enc.x_hierarchy is not None:  # Not supported by Plotly.
        warnings.warn("x_hierarchy brackets are a Matplotlib-only feature; the Plotly backend "
                      "ignores encoding.x_hierarchy.", stacklevel=2)

    faceted = ctx.facets.row is not None or ctx.facets.col is not None
    if faceted:  # One subplot per facet panel (shared category order).
        fig = make_subplots(rows=plan.nrows, cols=plan.ncols,
                            subplot_titles=[panel_title(ctx, rv, cv) for rv, cv in plan.panels],
                            shared_yaxes=ctx.facets.share_y,
                            horizontal_spacing=lay.wspace, vertical_spacing=lay.hspace)
    else:
        fig = go.Figure()
    cell: dict = {}  # add_trace target (row/col) for the panel being drawn.
    msize = max(3, int(np.sqrt(sty.point_size)))  # Marker size (area -> radius-ish).

    def _add_mark(xv, yv, name, color, fill_alpha, group):
        """Add one grouped trace (box / violin / strip) for a category."""
        pcolor = sty.point_color if sty.point_color is not None else color  # Point colour.
        marker = dict(color=pcolor, size=msize, opacity=sty.point_alpha)
        if kind == "violin":
            fig.add_trace(go.Violin(showlegend=first_panel, x=xv, y=yv, name=name, legendgroup=group, offsetgroup=group,
                                    scalegroup=group, line_color=color, fillcolor=color,
                                    opacity=fill_alpha, meanline_visible=False,
                                    points="all" if sty.show_points else False,
                                    jitter=sty.point_jitter, pointpos=0, marker=marker), **cell)
        elif kind == "strip":  # Invisible box, points only.
            fig.add_trace(go.Box(showlegend=first_panel, x=xv, y=yv, name=name, legendgroup=group, offsetgroup=group,
                                 boxpoints="all", pointpos=0, jitter=sty.point_jitter,
                                 fillcolor="rgba(0,0,0,0)", line=dict(color="rgba(0,0,0,0)"),
                                 marker=marker), **cell)
        else:  # box
            fig.add_trace(go.Box(showlegend=first_panel, x=xv, y=yv, name=name, legendgroup=group, offsetgroup=group,
                                 line=dict(color=color), fillcolor=color, opacity=fill_alpha,
                                 boxpoints="all" if sty.show_points else False,
                                 pointpos=0, jitter=sty.point_jitter, marker=marker), **cell)

    for k, (row_val, col_val) in enumerate(plan.panels):  # One pass per facet panel.
        ri, ci = divmod(k, plan.ncols)
        cell = dict(row=ri + 1, col=ci + 1) if faceted else {}
        first_panel = (k == 0)  # One legend entry per concept, from the first panel.
        pdata = _filter_box_panel(data, ctx, row_val, col_val)
        pbench = (_filter_box_panel(benchmark, ctx, row_val, col_val)
                  if benchmark is not None else None)
        if faceted and ctx.facets.hide_empty and pdata.empty:  # Nothing to draw here.
            continue

        # ---- data ----
        if enc.by is None:
            d = pdata[[enc.x, enc.y]].dropna()
            _add_mark(d[enc.x], d[enc.y], "data", "#1f77b4", sty.data_alpha, "data")
        else:
            for cat in bycats:
                tmp = pdata.loc[pdata[enc.by] == cat, [enc.x, enc.y]].dropna()
                if tmp.empty:
                    continue
                _add_mark(tmp[enc.x], tmp[enc.y], str(ctx.labels.lookup(cat, enc.by)),
                          color_map[cat], sty.data_alpha, f"data_{cat}")

        # ---- pooled "total" mark (all data groups together, unfilled) ----
        if ctx.show_total:
            d = pdata[[enc.x, enc.y]].dropna()
            if not d.empty:
                tname = str(ctx.labels.lookup(ctx.total_label))
                tmark = dict(color=ctx.total_color, size=msize, opacity=sty.point_alpha)
                if kind == "violin":
                    fig.add_trace(go.Violin(showlegend=first_panel, x=d[enc.x], y=d[enc.y], name=tname, legendgroup="total",
                                            offsetgroup="total", scalegroup="total",
                                            line_color=ctx.total_color, fillcolor="rgba(0,0,0,0)",
                                            meanline_visible=False,
                                            points="all" if sty.show_points else False,
                                            jitter=sty.point_jitter, pointpos=0, marker=tmark), **cell)
                elif kind == "strip":  # Points only, in the total colour.
                    fig.add_trace(go.Box(showlegend=first_panel, x=d[enc.x], y=d[enc.y], name=tname, legendgroup="total",
                                         offsetgroup="total", boxpoints="all", pointpos=0,
                                         jitter=sty.point_jitter, fillcolor="rgba(0,0,0,0)",
                                         line=dict(color="rgba(0,0,0,0)"), marker=tmark), **cell)
                else:  # box
                    fig.add_trace(go.Box(showlegend=first_panel, x=d[enc.x], y=d[enc.y], name=tname, legendgroup="total",
                                         offsetgroup="total", fillcolor="rgba(0,0,0,0)",
                                         line=dict(color=ctx.total_color, width=1.6),
                                         boxpoints="all" if sty.show_points else False,
                                         pointpos=0, jitter=sty.point_jitter, marker=tmark), **cell)

        # ---- benchmark ----
        if pbench is not None:
            if enc.benchmark_by is None:
                b = pbench[[enc.x, enc.y]].dropna()
                _add_mark(b[enc.x], b[enc.y], str(ctx.labels.lookup(sty.benchmark_label)),
                          sty.benchmark_color, sty.benchmark_alpha, "benchmark")
            else:
                for bcat in bbycats:
                    tmp = pbench.loc[pbench[enc.benchmark_by] == bcat, [enc.x, enc.y]].dropna()
                    if tmp.empty:
                        continue
                    _add_mark(tmp[enc.x], tmp[enc.y],
                              f"{ctx.labels.lookup(sty.benchmark_label)}: {ctx.labels.lookup(bcat, enc.benchmark_by)}",
                              bcm.get(bcat, sty.benchmark_color), sty.benchmark_alpha, f"benchmark_{bcat}")

    lx = lay.x_title if lay.x_title is not None else str(ctx.labels.lookup(enc.x))  # Axis titles.
    ly = lay.y_title if lay.y_title is not None else str(ctx.labels.lookup(enc.y))
    size = ((ctx.plotly_size[0] * plan.ncols, ctx.plotly_size[1] * plan.nrows) if faceted
            else ctx.plotly_size)
    fig.update_layout(template="simple_white", width=size[0], height=size[1],
                      boxmode="group", violinmode="group", showlegend=ctx.show_legend,
                      title=lay.title if lay.title is not None else f"{ly} by {lx}",
                      legend_title=str(ctx.labels.lookup(enc.by)) if enc.by is not None else "")
    if faceted:  # Axis titles on the outer panels only.
        fig.update_xaxes(title_text=lx, row=plan.nrows)
        fig.update_yaxes(title_text=ly, col=1)
    else:
        fig.update_layout(xaxis_title=lx, yaxis_title=ly)
    if sty.font_family is not None or sty.font_size:  # Typography.
        fig.update_layout(font=dict(family=sty.font_family, size=sty.font_size))
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", showgrid=False,
                     categoryorder="array", categoryarray=list(xcats),
                     tickmode="array", tickvals=list(xcats),
                     ticktext=[str(ctx.labels.lookup(c, enc.x)) for c in xcats])
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black", showgrid=True,
                     gridwidth=0.75, gridcolor="lightgrey")
    if lay.y_min is not None and lay.y_max is not None:  # Plotly needs both ends.
        fig.update_yaxes(range=[lay.y_min, lay.y_max])

    if ctx.layout.panel_letters:  # Bold a, b, c... (row-major) or a single 'a'.
        for k in range(len(plan.panels) if faceted else 1):
            ri, ci = divmod(k, plan.ncols)
            fig.add_annotation(text=f"<b>{_panel_letter(k) if faceted else 'a'}</b>",
                               xref="x domain", yref="y domain",
                               x=0.02, y=0.98, xanchor="left", yanchor="top",
                               showarrow=False, font=dict(size=sty.font_size + 3),
                               **(dict(row=ri + 1, col=ci + 1) if faceted else {}))

    files: list[str] = []
    if ctx.save is not None:
        files = save_plotly(fig, ctx.save.path, ctx.save.name, ctx.save.extensions)

    return PlotResult(figure=fig, axes=None, files=files, backend="plotly")


# ===========================================================================
# fingerprint_diagram: radial "model fingerprint" dashboard (Matplotlib only)
# ===========================================================================
def _fp_point(angle_deg: float, radius: float) -> tuple[float, float]:
    """Cartesian point on the unit canvas for a polar position.

    ``angle_deg`` is measured clockwise from the top; ``radius`` is in canvas
    units from the centre ``(0.5, 0.5)`` (``0.5`` reaches the axes edge).
    """
    rad = math.radians(90.0 - angle_deg)  # Clockwise-from-top -> standard math angle.
    return (radius * math.cos(rad) + 0.5, radius * math.sin(rad) + 0.5)


def _fp_point_ex(angle_deg: float, radius: float, dx: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two endpoints of a cross-tick of half-length ``dx``, perpendicular to
    the spoke at ``(angle_deg, radius)``.
    """
    rad = math.radians(90.0 - angle_deg)
    ux, uy = math.cos(rad), math.sin(rad)  # Radial unit vector.
    px, py = radius * ux + 0.5, radius * uy + 0.5  # Point on the spoke.
    return ((px + dx * (-uy), py + dx * ux),  # Perpendicular = (-uy, ux).
            (px - dx * (-uy), py - dx * ux))


class CurvedText(mtext.Text):
    """A text object whose characters follow an arbitrary curve.

    Adapted from a well-known StackOverflow answer
    (https://stackoverflow.com/a/42014041) for the curved category labels.
    Each character is an individual :class:`matplotlib.text.Text` re-positioned
    and rotated on every draw to track the curve given by ``x``/``y``.
    """

    def __init__(self, x, y, text, axes, **kwargs):
        super().__init__(x[0], y[0], " ", **kwargs)
        axes.add_artist(self)
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()
        self.__Characters = []
        for c in text:
            if c == " ":  # Render spaces as an invisible 'a' (keeps spacing).
                t = mtext.Text(0, 0, "a")
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0, 0, c, **kwargs)
            t.set_ha("center")
            t.set_rotation(0)
            t.set_zorder(self.__zorder + 1)
            self.__Characters.append((c, t))
            axes.add_artist(t)

    def set_zorder(self, zorder):  # Keep characters above the curve artist.
        super().set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c, t in self.__Characters:
            t.set_zorder(self.__zorder + 1)

    def draw(self, renderer, *args, **kwargs):  # Do not draw self; update characters.
        self.update_positions(renderer)

    def update_positions(self, renderer):
        """Re-place and rotate each character along the curve."""
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        figW, figH = self.axes.get_figure().get_size_inches()
        _, _, w, h = self.axes.get_position().bounds
        aspect = ((figW * w) / (figH * h)) * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

        x_fig, y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
                (i, j) for i, j in zip(self.__x, self.__y)
            ]))
        )
        x_fig_dist = (x_fig[1:] - x_fig[:-1])
        y_fig_dist = (y_fig[1:] - y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist ** 2 + y_fig_dist ** 2)
        l_fig = np.insert(np.cumsum(r_fig_dist), 0, 0)
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]), (x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)

        total_w = 0.0  # Measure the text to centre it along the path.
        for c, t in self.__Characters:
            t.set_rotation(0)
            t.set_va("center")
            total_w += t.get_window_extent(renderer=renderer).width
        rel_pos = max(5.0, (l_fig[-1] - total_w) / 2.0)  # Centred start offset.
        for c, t in self.__Characters:
            t.set_rotation(0)
            t.set_va("center")
            bbox1 = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height
            if rel_pos + w / 2 > l_fig[-1]:  # Drop characters that no longer fit.
                t.set_alpha(0.0)
                rel_pos += w
                continue
            elif c != " ":
                t.set_alpha(1.0)
            il = np.where(rel_pos + w / 2 >= l_fig)[0][-1]
            ir = np.where(rel_pos + w / 2 <= l_fig)[0][0]
            if ir == il:
                ir += 1
            used = l_fig[il] - rel_pos
            rel_pos = l_fig[il]
            fraction = (w / 2 - used) / r_fig_dist[il]
            x = self.__x[il] + fraction * (self.__x[ir] - self.__x[il])
            y = self.__y[il] + fraction * (self.__y[ir] - self.__y[il])
            t.set_va(self.get_va())
            bbox2 = t.get_window_extent(renderer=renderer)
            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0] - bbox1d[0])
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad) * aspect],
                [-math.sin(rad) / aspect, math.cos(rad)],
            ])
            drp = np.dot(dr, rot_mat)
            t.set_position(np.array([x, y]) + drp)
            t.set_rotation(degs[il])
            t.set_va("center")
            t.set_ha("center")
            rel_pos += w - used


def _fp_legend(ax, ctx: FingerprintContext, plan: FingerprintPlan) -> None:
    """Draw the code -> name legend, grouped by category and colour-coded."""
    lay, sty = ctx, ctx.style
    if lay.legend_loc == "bottom":  # Simple stacked block beneath the diagram.
        x0, y0, dy = 0.0, -0.12, -0.035
    else:  # Default: a column to the right of the circle.
        x0, y0, dy = 1.02, 1.05, -0.032
    y = y0
    for c in plan.categories:
        ax.text(x0, y, c, fontsize=lay.legend_fontsize + 2, fontweight="bold",
                color=plan.cat_color[c], ha="left", va="top", transform=ax.transData)
        y += dy
        for col in plan.indicators[c]:
            ax.text(x0 + 0.015, y, f"{plan.code[col]}  —  {plan.name[col]}",
                    fontsize=lay.legend_fontsize, color="black", ha="left", va="top",
                    transform=ax.transData)
            y += dy
        y += dy * 0.4  # Small gap between categories.


def render_fingerprint_matplotlib(ctx: FingerprintContext, plan: FingerprintPlan) -> PlotResult:
    """Render the fingerprint *scaffold* (rings, arcs, spokes, labels, guide).

    The data/benchmark layer (mapping values to radii and drawing a model's
    fingerprint) is added separately; this builds the empty, fully-labelled
    diagram for an arbitrary number of categories and indicators.
    """
    enc, sty, lay = ctx.encoding, ctx.style, ctx  # Shorthands (layout fields flat on ctx).
    col_of_cat = {col: c for c in plan.categories for col in plan.indicators[c]}  # Reverse map.

    rc = {"font.size": sty.font_size}  # Base typography.
    if sty.font_family is not None:
        rc["font.family"] = sty.font_family

    with mpl.rc_context({**dict(sns.axes_style("white")), **rc}):  # Scoped style.
        fig, ax = plt.subplots(figsize=sty.figsize)
        ax.set_aspect("equal")  # Keep the rings circular regardless of figsize.

        # --- reference rings (white disks stacked smallest-on-top -> ring outlines) ---
        ax.add_patch(mpatches.Circle((0.5, 0.5), sty.r_border, facecolor="white",
                                     edgecolor="k", lw=sty.ring_lw_border, zorder=0.0))
        ax.add_patch(mpatches.Circle((0.5, 0.5), sty.r_q75, facecolor="white",
                                     edgecolor="k", lw=sty.ring_lw_quartile, zorder=0.1))
        ax.add_patch(mpatches.Circle((0.5, 0.5), sty.r_median, facecolor="white",
                                     edgecolor="k", lw=sty.ring_lw_median, zorder=0.2))
        ax.add_patch(mpatches.Circle((0.5, 0.5), sty.r_q25, facecolor="white",
                                     edgecolor="k", lw=sty.ring_lw_quartile, zorder=0.3))

        # --- outer category arcs (diameter 1 -> radius 0.5), one per sector ---
        for c in plan.categories:
            a0, a1 = plan.cat_span[c]  # Sector span (clockwise from top).
            theta1 = 90.0 - a1 + sty.arc_gap_deg  # Convert to math angle (CCW from +x).
            theta2 = 90.0 - a0 - sty.arc_gap_deg
            ax.add_patch(mpatches.Arc((0.5, 0.5), 1.0, 1.0, theta1=theta1, theta2=theta2,
                                      color=plan.cat_color[c], lw=sty.arc_lw, zorder=0.5))

        if sty.show_center_marker:  # Central dot.
            ax.plot([0.5], [0.5], "ko", ms=12, zorder=5)

        # --- spokes + codes ---
        for c in plan.categories:
            for col in plan.indicators[c]:
                a = plan.angles[col]
                px, py = _fp_point(a, sty.spoke_radius)
                ax.plot([0.5, px], [0.5, py], linestyle=sty.spoke_style,
                        color=sty.spoke_color, zorder=1)
                if sty.show_codes:
                    cx, cy = _fp_point(a, sty.code_radius)
                    ax.text(cx, cy, plan.code[col], fontsize=sty.code_fontsize,
                            color=plan.cat_color[c], ha="center", va="center", zorder=6)

        # --- focal band per spoke (data vs benchmark): min..max capsule + quantile ticks ---
        if plan.has_data:
            ds = ctx  # Data-layer fields (iqr_mult, mark, band_*, tick_*) are flat on ctx.
            for c in plan.categories:
                n = len(plan.indicators[c])
                for col in plan.indicators[c]:
                    radii = plan.band_radii.get(col)
                    if not radii:
                        continue  # No usable scale / no focal data for this spoke.
                    a = plan.angles[col]
                    color = plan.cat_color[c]
                    pts_a, pts_b = [], []
                    for r in radii:  # Cross-tick at each focal quantile.
                        dx = ds.tick_scale * r / n  # Proportional half-length.
                        pa, pb = _fp_point_ex(a, r, dx)
                        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=ds.tick_lw, zorder=9)
                        pts_a.append(pa)
                        pts_b.append(pb)
                    i_lo, i_hi = int(np.argmin(radii)), int(np.argmax(radii))  # Min/max quantile.
                    poly = [pts_a[i_lo], pts_a[i_hi], pts_b[i_hi], pts_b[i_lo]]  # Capsule corners.
                    ax.add_patch(mpatches.Polygon(poly, closed=True, facecolor=color,
                                                  edgecolor=color, alpha=ds.band_alpha,
                                                  lw=ds.band_lw, zorder=8))

        # --- curved category labels on the inner arc ---
        if sty.show_category_labels:
            R = sty.category_label_radius
            for c in plan.categories:
                a0, a1 = plan.cat_span[c]
                th = np.linspace(90.0 - a0, 90.0 - a1, 200)  # Math angles across the sector.
                xs = R * np.cos(np.radians(th)) + 0.5
                ys = R * np.sin(np.radians(th)) + 0.5
                mid = math.radians(90.0 - (a0 + a1) / 2.0)  # Sector mid (math angle).
                if math.sin(mid) < 0:  # Bottom half: reverse so text stays upright.
                    xs, ys = xs[::-1], ys[::-1]
                CurvedText(x=xs, y=ys, text=str(c), axes=ax, color=plan.cat_color[c],
                           fontsize=sty.category_label_fontsize, va="center", ha="center")

        # --- central quantile guide ---
        if lay.quantile_guide:
            radii = [0.015, sty.r_q25, sty.r_median, sty.r_q75, sty.r_border]  # lo -> hi (fence).
            for label, r in zip(lay.quantile_labels, radii):
                gx, gy = _fp_point(0.0, r)  # Straight up.
                ax.text(gx, gy, label, fontsize=sty.font_size + 4, ha="center", va="center",
                        zorder=10, bbox=dict(boxstyle="round", ec="white", fc="white"))
            top, _ = _fp_point(0.0, sty.r_q75 + 0.02)
            ax.annotate("", xy=_fp_point(0.0, sty.r_q75 + 0.02), xytext=_fp_point(0.0, sty.r_median),
                        arrowprops=dict(shrink=0.05, facecolor="k"), zorder=9)
            ax.annotate("", xy=_fp_point(0.0, sty.r_q25 - 0.02), xytext=_fp_point(0.0, sty.r_median),
                        arrowprops=dict(shrink=0.05, facecolor="k"), zorder=9)

        if lay.legend:  # Grouped code -> name legend.
            _fp_legend(ax, ctx, plan)

        for i, note in enumerate(lay.footnotes):  # Italic footnotes, bottom-left.
            ax.text(lay.xlim[0] + 0.02, lay.ylim[0] + 0.02 + 0.03 * (len(lay.footnotes) - 1 - i),
                    note, fontsize=sty.font_size + 2, ha="left", va="center", style="italic", zorder=10)

        if lay.title is not None:
            ax.set_title(lay.title, fontsize=sty.font_size + 8, fontweight="bold")

        ax.set_xlim(list(lay.xlim))
        ax.set_ylim(list(lay.ylim))
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        if ctx.panel_letters:  # Single-axes chart: tag 'a'.
            ax.text(0.02, 0.98, "a", transform=ax.transAxes, fontweight="bold",
                    fontsize=sty.font_size + 2, va="top", ha="left", zorder=10)

        _rasterize_data(fig, ctx.style.rasterized)  # Printable vector output.
        files: list[str] = []
        if ctx.save is not None:
            files = save_matplotlib(fig, ctx.save.path, ctx.save.name, ctx.save.extensions, ctx.save.dpi)

    return PlotResult(figure=fig, axes=ax, files=files, backend="matplotlib")
