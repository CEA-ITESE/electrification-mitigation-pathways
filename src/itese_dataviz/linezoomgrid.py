#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pandera.pandas as pa
import numpy as np
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic import model_validator
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from typing import Final
from matplotlib.patches import ConnectionPatch

from itese_dataviz.schemas import SourceDatasetSchema, ReferenceDatasetSchema
from itese_dataviz.histgrid import HistGridPlot

class LineZoomHistGridPlot(HistGridPlot):

    # optional parameters
    col_total: str | None = None

    _REQUIRED_KEYS: Final[tuple[str, ...]] = (
        "Grid",
        "ColorStrip",
        "Yaxis",
        "Zoom Grid"
    )

    _ALLOWED_OPTIONS: Final[tuple[str, ...]] = (
        "rasterized",
    )

    @model_validator(mode="after")
    def preprocess_pipeline(self) -> "LineZoomHistGridPlot":
        super().preprocess_pipeline()

        if self.col_total is None:
            for col in self.col_names.keys():
                if 'total' in col.lower():
                    self.col_total = col
                
        if self.col_total is None:
            raise ValueError(
                "Fill field 'col_total' or set a column name with 'total' to set it as the total column"
            )

        return self

    @model_validator(mode="after")
    def preprocess_reference_data(self) -> "LineZoomHistGridPlot":
        super().preprocess_reference_data()
        return self

    @model_validator(mode="after")
    def _preprocess_plot(self) -> "LineZoomHistGridPlot":
        return self

    def plot(self) -> None:
        """Generates the full multi-panel plot with overview line/boxplots on top

        and detailed technology mix zoom panels on the bottom.
        """
        # Color and style configuration
        zoom_box_color = "lightgrey"
        zoom_line_color = "#9ca3af"
        box_color = "#e5e7eb"
        line_color = "#9ca3af"
        median_color = "#374151"

        policy_palette = mpl.cm.coolwarm_r
        norm = mpl.colors.Normalize(
            vmin=self.data[self.display["ColorStrip"]].min(),
            vmax=self.data[self.display["ColorStrip"]].max(),
        )

        # Relative dimensions setup
        base_width = 0.48
        marker_ratio = 1.40
        bottomboxplot_width = 0.2
        halo_width = bottomboxplot_width * 1.06
        marker_half_width = (base_width * marker_ratio) / 2.0

        zoom_grid = self.display["Zoom Grid"]
        num_zoom_cols = len(zoom_grid)

        # Figure layout initialization
        self._fig = plt.figure(figsize=(18, 11), layout="constrained")
        gs = self._fig.add_gridspec(
            3,
            num_zoom_cols,
            hspace=0.1,
            wspace=0.01,
            width_ratios=[1] * num_zoom_cols,
            height_ratios=[1, 1, 0.02], # TODO : variable parameters
        )

        # Top main plot (Overview lines + boxplots)
        ax_top_main = self._fig.add_subplot(gs[0, :])

        # Background trend lines
        sns.lineplot(
            data=self.data,
            x=self.display["Grid"],
            y=self.col_total,
            hue=self.display["ColorStrip"],
            units="Case",
            estimator=None,
            palette=policy_palette,
            hue_norm=norm,
            alpha=0.3,
            linewidth=1.4,
            ax=ax_top_main,
            legend=False,
            zorder=1,
        )

        unique_periods = self.data["Period"].unique()

        # Reference scatter points
        for p in unique_periods:
            data_ref_p = self.reference_data[self.reference_data["Year"] == p]
            if not data_ref_p.empty:
                jitter = np.random.uniform(-0.19, 0.19, size=len(data_ref_p))
                sns.scatterplot(
                    data=data_ref_p,
                    x=data_ref_p["Year"] + jitter,
                    y=self.col_total,
                    color="white",
                    edgecolor="#2B2B2B",
                    linewidth=0.5,
                    s=16,
                    alpha=0.4,
                    ax=ax_top_main,
                    zorder=2,  # Placed behind boxplots
                )

        # White halo and main boxplots
        for p in unique_periods:
            data_p = self.data[self.data["Period"] == p][self.col_total].dropna()
            if not data_p.empty:
                # White background halo boxplot (Wider width and thicker lines for complete surrounding halo)
                ax_top_main.boxplot(
                    data_p,
                    positions=[p],
                    widths=1.08,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    zorder=3,
                    boxprops=dict(
                        facecolor="white", edgecolor="white", linewidth=2.5, alpha=0.2
                    ),
                    medianprops=dict(color="white", linewidth=4.0),
                    whiskerprops=dict(color="white", linewidth=4.0),
                    capprops=dict(color="white", linewidth=4.0),
                )
                # Main overlay boxplot
                ax_top_main.boxplot(
                    data_p,
                    positions=[p],
                    widths=1.03,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    zorder=4,
                    boxprops=dict(facecolor=box_color, edgecolor=line_color, alpha=0.4),
                    medianprops=dict(color=median_color, linewidth=2),
                    whiskerprops=dict(color=line_color),
                    capprops=dict(color=line_color),
                )

        # Top plot axis boundaries and styling
        up_top = 1.05 * self.data[self.col_total].max()
        up_bottom = 0.95 * self.data[self.col_total].min()

        if not self.reference_data.empty:
            up_ref_top = 1.10 * self.reference_data[self.col_total].quantile(0.95)
            up_ref_bottom = 0.90 * self.reference_data[self.col_total].quantile(0.05)
            top_limit = max(up_top, up_ref_top)
            bottom_limit = min(up_bottom, up_ref_bottom)
        else:
            top_limit, bottom_limit = up_top, up_bottom

        ax_top_main.set_xlim(
            min(self.row_filters["Period"]) - 2, max(self.row_filters["Period"]) + 2
        )
        ax_top_main.set_ylim(bottom_limit, top_limit)
        ax_top_main.set_xlabel("")
        ax_top_main.set_title(f"{self.display['Yaxis']}", fontsize=14)
        ax_top_main.set_ylabel(f"{self.col_total}")
        ax_top_main.grid(
            True, axis="both", linestyle="--", linewidth=0.2, alpha=0.3, zorder=0
        )

        # Bottom zoom plots (Technology mix panels)
        self._melt_indicators()
        self._drop_na()

        axes_bot: list[plt.Axes] = []
        q_low_list: list[float] = []
        q_high_list: list[float] = []
        p_low_list: list[float] = []
        p_high_list: list[float] = []

        for i, p in enumerate(zoom_grid):
            ax_bot = self._fig.add_subplot(
                gs[1, i], sharey=axes_bot[0] if i > 0 else None
            )
            axes_bot.append(ax_bot)

            # Filter category data for current period
            data_p = self.data[
                (self.data["Period"] == p) & (self.data["Technology"] != self.col_total)
            ]
            data_ref_p = self.reference_data[
                (self.reference_data["Year"] == p)
                & (self.reference_data["Technology"] != self.col_total)
            ]

            # Calculate per-plot quantiles (0.05 and 0.95) to compute global bottom Y-limits
            q_low_list.append(data_ref_p[self.display["Yaxis"]].dropna().quantile(0.05))
            q_high_list.append(data_ref_p[self.display["Yaxis"]].dropna().quantile(0.95))
            p_low_list.append(data_p[self.display["Yaxis"]].dropna().min())
            p_high_list.append(data_p[self.display["Yaxis"]].dropna().max())

            if not data_p.empty:
                # Map categorical technology names to numeric X positions
                tech_categories = list(data_p["Technology"].unique())
                tech_to_x = {tech: idx for idx, tech in enumerate(tech_categories)}

            # A. Draw exact relative horizontal lines for policies
            for _, row in data_p.iterrows():
                x_center = tech_to_x[row["Technology"]]
                y_val = row[self.display["Yaxis"]]
                color = policy_palette(norm(row["ClimatePolicy"]))

                ax_bot.hlines(
                    y=y_val,
                    xmin=x_center - marker_half_width * 1,
                    xmax=x_center - marker_half_width * 0,
                    colors=color,
                    linewidth=0.5,
                    zorder=1,
                    alpha=0.6
                )

            # Draw white halo boxplot
            sns.boxplot(
                data=data_p,
                x="Technology",
                y=self.display["Yaxis"],
                color="white",
                width=halo_width,
                boxprops=dict(edgecolor="white", linewidth=2.5, alpha=0.6),
                whiskerprops=dict(color="white", linewidth=3),
                capprops=dict(color="white", linewidth=3),
                medianprops=dict(color="white", linewidth=3),
                showfliers=False,
                ax=ax_bot,
                zorder=3,
            )

            # Main boxplot
            sns.boxplot(
                data=data_p,
                x="Technology",
                y=self.display["Yaxis"],
                color="lightgrey",
                width=bottomboxplot_width,
                boxprops=dict(facecolor=box_color, edgecolor=line_color, alpha=0.7),
                medianprops=dict(color=median_color, linewidth=1.5),
                whiskerprops=dict(color=line_color),
                capprops=dict(color=line_color),
                linewidth=0.7,
                showfliers=False,
                ax=ax_bot,
                zorder=3,
            )

            # Reference stripplot
            if not data_ref_p.empty:
                # already defined plot objects
                n_collections_before = len(ax_bot.collections)

                sns.stripplot(
                    data=data_ref_p,
                    x="Technology",
                    y=self.display["Yaxis"],
                    color="white",
                    edgecolor="#2B2B2B",
                    linewidth=0.5,
                    size=3,
                    alpha=0.4,
                    jitter=0.05,
                    ax=ax_bot,
                    zorder=2,
                )

                shift = 0.25
                for collection in ax_bot.collections[n_collections_before:]:
                    offsets = collection.get_offsets()
                    offsets[:, 0] += shift  # Décalage sur l'axe X (inverser si horizontal)
                    collection.set_offsets(offsets)

            # Subplot formatting
            ax_bot.set_title(f"{self.display['Grid']} = {p}", fontsize=10)
            ax_bot.tick_params(axis="x", rotation=90)
            ax_bot.set_xlabel("")

            if i == 0:
                ax_bot.set_ylabel("Technological Mix")
                ax_bot.tick_params(axis="y", labelleft=True, left=False)
            else:
                ax_bot.set_ylabel("")
                ax_bot.tick_params(axis="y", labelleft=False, left=False)

            # Apply dashed box frame around bottom plots
            for spine in ax_bot.spines.values():
                spine.set_visible(True)
                spine.set_color(zoom_box_color)
                spine.set_linestyle("--")
                spine.set_linewidth(0.8)

            ax_bot.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Apply unified Y-limits for bottom plots using global min/max of per-plot quantiles
        if q_low_list and q_high_list:
            min_bottom_q = min(q_low_list + p_low_list) * 0.90
            max_bottom_q = max(q_high_list + p_high_list) * 1.10
            axes_bot[0].set_ylim(min_bottom_q, max_bottom_q)

        if num_zoom_cols >= 3:
            cbar_ax = self._fig.add_subplot(gs[2, 1:-1])  # Occupe les colonnes centrales
        else:
            cbar_ax = self._fig.add_subplot(gs[2, :])  # Occupe toute la largeur

        sm = mpl.cm.ScalarMappable(cmap=policy_palette, norm=norm)
        cbar = self._fig.colorbar(
            sm, cax=cbar_ax, orientation="horizontal", label=f"Climate Policy (%)"
        )

        # Zoom connection patches
        for i, p in enumerate(zoom_grid):
            data_p = self.data[
                (self.data["Period"] == p) & (self.data["Technology"] == self.col_total)
            ][self.display["Yaxis"]]

            if not data_p.empty:
                q1, q3 = data_p.quantile([0.25, 0.75])

                con_l = ConnectionPatch(
                    xyA=(p - 0.6, q1),
                    xyB=(0, 1),
                    coordsA="data",
                    coordsB="axes fraction",
                    axesA=ax_top_main,
                    axesB=axes_bot[i],
                    color=zoom_line_color,
                    alpha=0.2,
                )
                con_r = ConnectionPatch(
                    xyA=(p + 0.6, q3),
                    xyB=(1, 1),
                    coordsA="data",
                    coordsB="axes fraction",
                    axesA=ax_top_main,
                    axesB=axes_bot[i],
                    color=zoom_line_color,
                    alpha=0.2,
                )
                self._fig.add_artist(con_l)
                self._fig.add_artist(con_r)

        if self.rasterized:
            ax_top_main.set_rasterized(True)
            for ax in axes_bot:
                ax.set_rasterized(True)
