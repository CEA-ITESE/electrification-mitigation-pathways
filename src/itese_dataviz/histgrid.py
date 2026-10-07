#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict, Field
from pydantic import model_validator
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from typing import Final

from itese_dataviz.schemas import SourceDatasetSchema, ReferenceDatasetSchema
from itese_dataviz.base import BasePlot

class HistGridPlot(BasePlot):

    # otpional parameters
    rasterized: bool = False

    _ALLOWED_OPTIONS: Final[tuple[str, ...]] = (
        "rasterized",
    )

    @model_validator(mode="after")
    def preprocess_pipeline(self) -> "HistGridPlot":
        """
        Preprocess data by filtering rows/columns, cleaning values, and validating schemas.
        """
        super().preprocess_pipeline()

        self._aggregate_columns(mapping=self.col_names, target="source", aggregate="sum")

        # Replace region names based on the provided mapping
        if self.region_map:
            self.data["Region"] = self.data["Region"].replace(self.region_map)

        # Filter columns
        self._col_filters = set(self.col_names.keys())

        col_filters = set()
        col_filters.update(self._col_filters)
        if self.index_col:
            col_filters.update(self.index_col)
        if self._col_filters:
            filtered_cols = self.data.columns.isin(col_filters)
            self.data = self.data.loc[:, filtered_cols]

        if self.row_filters:
            for col, values in self.row_filters.items():
                self.data = self.data[self.data[col].isin(values)]

        # Filter rows
        if self.row_filters:
            for col, values in self.row_filters.items():
                self.data = self.data[self.data[col].isin(values)]

        # Set NaN values
        if not self.nan_values is None:
            self.data = self._replace_na(self.data, self.nan_values)

        return self

    @model_validator(mode="after")
    def preprocess_reference_data(self) -> "HistGridPlot":
        super().preprocess_reference_data()

        self._aggregate_columns(mapping=self.reference_col_names, target="reference", aggregate="sum")

        # Filter columns
        col_filters = set()
        col_filters.update(self.reference_index_col)

        # Complete columns mapping with default value
        for col in self._col_filters:
            if col not in self.reference_map_col:
                self.reference_map_col[col] = col
        col_filters.update(list(self.reference_map_col.keys()))

        if col_filters:
            filtered_cols = self.reference_data.columns.isin(col_filters)
            self.reference_data = self.reference_data.loc[:, filtered_cols]

        # Filter rows
        if self.reference_row_filters:
            for col, values in self.reference_row_filters.items():
                self.reference_data = self.reference_data[self.reference_data[col].isin(values)]

        # Set NaN values
        if not self.reference_nan_values is None:
            self.reference_data = self._replace_na(self.reference_data, self.reference_nan_values)

        return self

    @model_validator(mode="after")
    def _preprocess_plot(self) -> "HistGridPlot":

        self._melt_indicators()
        self._drop_na()
        self._convert_categorical()

        return self


    def plot(self):

        def apply_dynamic_scatter_s(
            g: sns.FacetGrid, target_data_width: float = 0.6, num_categories: int = 4, scatter_zorder: int = 1,
        ) -> None:
            """Dynamically adjust the 's' parameter of scatterplot markers (marker='_')

            to match a precise data width, updating automatically upon window resize.

            Args:
                g: Seaborn FacetGrid containing the scatterplots.
                target_data_width: Desired span of line traits in data units (e.g.,
                0.6).
                num_categories: Total number of categories on the x-axis.
            """

            def update_sizes(event=None):
                fig = g.fig

                # Safely trigger canvas update across all backends (GUI, Agg, Jupyter)
                if hasattr(fig.canvas, "draw_idle"):
                    fig.canvas.draw_idle()

                for ax in g.axes.flat:
                    if ax is None:
                        continue

                    # 1. Get real width of the axis in points (1 pt = 1/72 inch)
                    bbox = ax.get_window_extent().transformed(
                        fig.dpi_scale_trans.inverted()
                    )
                    ax_width_in_pts = bbox.width * 72.0

                    # 2. Compute trait length in points
                    pts_per_category = ax_width_in_pts / max(1, num_categories)
                    trait_length_pt = target_data_width * pts_per_category

                    # 3. Area s is length squared
                    target_s = trait_length_pt**2

                    # 4. Apply new size to scatterplot collections
                    for collection in ax.collections:
                        if not (
                            hasattr(collection, "_sizes")
                            and len(collection.get_sizes())
                        ):
                            continue

                        # Filter 1: Check zorder match
                        if collection.get_zorder() != scatter_zorder:
                            continue

                        if hasattr(collection, "_sizes") and len(
                            collection.get_sizes()
                        ):
                            collection.set_sizes([target_s])

            # Initial calculation
            update_sizes()

            # Connect to draw_event so 's' rescales dynamically on window resize
            g.fig.canvas.mpl_connect("draw_event", update_sizes)


        # TODO put the add_stripplot function outside
        def add_stripplot(data: pd.DataFrame, color: str | None = None, **kwargs) -> None:
            """Custom plotting function to overlay a high-performance stripplot on each facet.

            Args:
                data: The subset of the dataframe corresponding to the current facet.
                color: Optional color argument passed by Seaborn's mapping engine.
                **kwargs: Additional keyword arguments passed by map_dataframe.
            """


            # Capture the active axis for the current facet iteration
            ax = plt.gca()

            # Retrieve the period key to slice the reference scenario dataset
            current_year = data[self.display['Grid']].iloc[0]
            subset = self.reference_data[self.reference_data["Year"] == current_year]

            # Render points instantaneously using randomized horizontal jitter
            sns.stripplot(
                data=subset,
                x="Technology",
                y=self.display['Yaxis'],
                color="white",
                edgecolor="#2B2B2B",
                linewidth=0.5,
                size=3,
                alpha=0.4,
                jitter=0.15,
                ax=ax,
                zorder=2
            )

            # De-clutter the layout by clearing out default individual x-axis titles
            ax.set_xlabel("")

        def add_swarm(data, color, **kwargs):
            ax = plt.gca()
            
            year = data[self.display['Grid']].iloc[0]   # current facet
            
            subset = self.reference_data[self.reference_data["Year"] == year]
            
            sns.swarmplot(
                data=subset,
                x="Technology",
                y=self.display['Yaxis'],
                color="black",
                size=2,
                alpha=0.2,
                ax=ax,
                zorder=2
            )
            # x-labels configuration
            for ax in self._fig.axes.flat:
                #ax.tick_params(axis='x', rotation=90 )
                ax.set_xlabel("")

        # Order columns
        self.data["Technology"] = pd.Categorical(self.data["Technology"], categories=self.col_names.keys(), ordered=True)
        
        # # TODO: Add more customization options for the plot (e.g., color palette, size, etc.)
        palette_color = mpl.cm.coolwarm_r
        norm = mpl.colors.Normalize(
            vmin=self.data[self.display['ColorStrip']].min(),
            vmax=self.data[self.display['ColorStrip']].max()
        )
        self._fig = sns.FacetGrid(
            self.data,
            col=self.display['Grid'],
            col_wrap=3,
            height=4,
            sharey=False
        )
        box_color = "#e5e7eb"
        line_color = "#9ca3af"
        median_color = "#374151"
        white_halo = [path_effects.withStroke(linewidth=4, foreground="white")]

        self._fig.map_dataframe(
            sns.scatterplot,
            x="Technology",
            y=self.display['Yaxis'],
            hue=self.display['ColorStrip'],
            #palette="RdYlGn",
            palette = palette_color,
            hue_norm=norm,
            marker="_",      # ⭐ vertical lines
            s=500,           # controls line length
            linewidth=0.5,
            edgecolor=None,
            alpha=0.6,
            legend=False,
            zorder=1
        )

        if not self.reference_data is None:
            self._fig.map_dataframe(add_stripplot)

        #Plotting the boxplot graphs
        self._fig.map_dataframe(
            sns.boxplot,
            x="Technology",
            y=self.display['Yaxis'],
            color="white",
            width=0.58,  # Légèrement plus large
            boxprops=dict(edgecolor="white", alpha=0.6),
            whiskerprops=dict(color="white", linewidth=3),
            capprops=dict(color="white", linewidth=3),
            medianprops=dict(color="white", linewidth=3),
            showfliers=False,
            zorder=3
        )
        self._fig.map_dataframe(
            sns.boxplot,
            x="Technology",
            y=self.display['Yaxis'],
            color = "lightgrey",
            width=0.55,
            boxprops=dict(
                facecolor=box_color, edgecolor=line_color, alpha=0.4),
            medianprops=dict(color=median_color, linewidth=1.5),
            whiskerprops=dict(color=line_color),
            capprops=dict(color=line_color),
            linewidth=0.7,
            showfliers=False,
            zorder=3
        )

        apply_dynamic_scatter_s(
            self._fig, target_data_width=0.80, num_categories=len(self._col_filters)
        )


        # x-labels configuration
        for ax in self._fig.axes.flat:
            ax.tick_params(axis='x', rotation=90 )
            ax.set_xlabel("")
            ax.set_ylabel("")

        self._fig.fig.supylabel(
            self.display['Yaxis']
        )

        # Create space at the bottom
        self._fig.fig.subplots_adjust(bottom=0.25)

        # Add a dedicated axis for the colorbar
        cbar_ax = self._fig.fig.add_axes([0.2, 0.01, 0.6, 0.01])
        # [left, bottom, width, height]
        #Add a colorbar for policy
        norm = mpl.colors.Normalize(
            vmin=self.data.ClimatePolicy.min(),
            vmax=self.data.ClimatePolicy.max()
        )
        sm = mpl.cm.ScalarMappable(
            cmap=palette_color,
            norm=norm
        )
        sm.set_array([])
        cbar = self._fig.fig.colorbar(
            sm,
            cax=cbar_ax,
            orientation="horizontal",
            #fraction=0.05,
            #pad=0.08
        )
        cbar.set_label("Climate Policy (%)") # TODO set variable

        # Vertical limit of graph
        top = self.data[self.display['Yaxis']].max()
        bottom = self.data[self.display['Yaxis']].min()
        if not self.reference_data is None:
            ref_top = float(1.15 * self.reference_data[['Year', 'Technology', self.display['Yaxis']]].groupby(['Year', 'Technology'])[self.display['Yaxis']].quantile(0.95).max())
            ref_bottom = float(0.85 * self.reference_data[['Year', 'Technology', self.display['Yaxis']]].groupby(['Year', 'Technology'])[self.display['Yaxis']].quantile(0.05).min())
            
            top = max(ref_top, top)
            bottom = min (ref_bottom, bottom)

        # Improving readiness
        self._fig.fig.subplots_adjust(bottom=0.25, left=0.1, right=0.98, top=0.92, wspace=0.15) # increased margin to see the x labels
        for ax in self._fig.axes.flat:
            # remove top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # lighten grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            ax.set_ylim(top=top, bottom=bottom)

            if self.rasterized:
                ax.set_rasterized(True)

        # plt.show()