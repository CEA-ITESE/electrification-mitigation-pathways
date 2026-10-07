#!/usr/bin/env python
# -*- coding: utf-8 -*-

# load upset modules
from __future__ import annotations
from collections import defaultdict, Counter
# from typing import TYPE_CHECKING
from typing import Literal
import math
import matplotlib.pyplot as plt
import numpy as np
from pydantic import ConfigDict, validate_call

# if TYPE_CHECKING:
from collections.abc import Sequence
from matplotlib.figure import Figure

# load shapiq class
from shapiq.interaction_values import InteractionValues
from shapiq.plot._config import BLUE, RED

LabelsKeys = Literal["count_label", "y_label", "remainder", "empty_coalition", "grand_coalition"]
DEFAULT_LABELS: dict[LabelsKeys, str] = {
    "count_label": "Feature Occurence",
    "y_label": "Interaction Value",
    "remainder": "Others",
    "empty_coalition": "Empty coalition",
    "grand_coalition": "Grand coalition",
}

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def upset_plot(
    interaction_values: InteractionValues,
    *,
    n_interactions: int = 20,
    feature_names: Sequence[str] | None = None,
    color_matrix: bool = False,
    all_features: bool = True,
    hide_empty_coalition: bool  = False,
    hide_grand_coalition: bool  = False,
    show_remainder: bool  = False,
    show_counts: bool | Literal["absolute", "proportion"] = False,
    labels: dict[LabelsKeys, str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> Figure | None:
    """Plots the upset plot.

    UpSet plots[1]_ can be used to visualize the interactions between features. The plot consists of
    two parts: the upper part shows the interaction values as bars, and the lower part shows the
    interactions as a matrix. Originally, the UpSet plot was introduced by Lex et al. (2014)[1]_.
    For a more detailed explanation about the plots, see the references or the original
    [documentation](https://upset.app/).

    An example of this plot is shown below.

    .. image:: /_static/images/upset_plot.png
        :width: 600
        :align: center

    Args:
        interaction_values: The interaction values as an ``InteractionValues`` object.
        feature_names: The names of the features. Defaults to ``None``. If ``None``, the features
            will be named with their index.
        n_interactions: The number of top interactions to plot. Defaults to ``20``. Note this number
            is completely arbitrary and can be adjusted to the user's needs.
        color_matrix: Whether to color the matrix (red for positive values, blue for negative) or
            not (black). Defaults to ``False``.
        all_features: Whether to plot all ``n_players`` features or only the features that are
            present in the top interactions. Defaults to ``True``.
        hide_empty_coalition: Whether to hide the empty coalition (coalition with no features)
            from the plot. If ``True``, it is excluded from calculations and counting.
            Defaults to ``False``.
        hide_grand_coalition: Whether to hide the grand coalition (coalition containing all features)
            from the plot. If ``True``, it is excluded from calculations and counting.
            Defaults to ``False``
        show_remainder: Whether to show an additional bar representing the sum of remaining non-plotted
            interactions. Defaults to ``False``.
        show_counts: Whether to display a count barplot in the bottom right of the plot.
            If ``True`` or ``"absolute"``, shows a simple count barplot. If ``"proportion"``,
            the bars are colored according to the proportion of positive and negative Shapley
            values. Defaults to ``False``.
        labels: Custom labels dictionary mapping ``LabelsKeys`` to string values to override
            default plot text. Defaults to ``None``.
        figsize: The size of the figure. Defaults to ``None``. If ``None``, the size will be set
            automatically depending on the number of features.
        show: Whether to show the plot. Defaults to ``False``.

    Returns:
        If ``show`` is ``True``, the function returns ``None``. Otherwise, it returns a tuple with
        the figure and the axis of the plot.

    References:
        .. [1] Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot, Hanspeter Pfister. UpSet: Visualization of Intersecting Sets IEEE Transactions on Visualization and Computer Graphics (InfoVis), 20(12): 1983--1992, doi:10.1109/TVCG.2014.2346248, 2014.

    """

    # prepare data ---------------------------------------------------------------------------------
    values = interaction_values.values
    values_ids: dict[int, tuple[int, ...]] = {}
    empty_coalition_id = None
    interactions_count = {f:Counter({'n': 0, 'p': 0, 's': 0, 'n_val': 0, 'p_val': 0}) for f in np.arange(interaction_values.n_players)}
    grand_coalition_id = None
    empty_coalition_id = None
    for k, v in interaction_values.interaction_lookup.items():
        values_ids[v] = k
        # extract empty coalition
        if not k:
            empty_coalition_id = v
            empty_coalition_value = values[v]
        # extract grand coalition
        if len(k) == interaction_values.n_players:
            grand_coalition_id = v
            grand_coalition_value = values[v]
    # sort index interactions by absolute value 
    # and filter by empty and grand coalition if needed
    values_abs = abs(values)
    idx = values_abs.argsort()[::-1]
    if hide_empty_coalition | hide_grand_coalition:
        filter_ids = np.full(len(values), True)
        if hide_empty_coalition & (empty_coalition_id is not None):
            filter_ids = filter_ids & (idx != empty_coalition_id)
        if hide_grand_coalition & (grand_coalition_id is not None):
            filter_ids = filter_ids & (idx != grand_coalition_id)
        idx = idx[filter_ids]
    if n_interactions > 0:
        if show_remainder:
            idx_remainder = idx[n_interactions:]
            remainder_value = values[idx_remainder].sum()
        idx = idx[:n_interactions]
    else:
        idx = idx
    values = values[idx]
    interactions: list[tuple[int, ...]] = [values_ids[i] for i in idx]
    # interaction counter
    for i, interaction in enumerate(interactions):
        for feat in interaction:
            if values[i] > 0:
                interactions_count[int(feat)]['p'] += 1
                interactions_count[int(feat)]['p_val'] += values[i]
                interactions_count[int(feat)]['s'] += 1
            else:
                interactions_count[feat]['n'] += 1
                interactions_count[feat]['n_val'] -= values[i]
                interactions_count[int(feat)]['s'] += 1
    max_count = max([c['s'] for c in interactions_count.values()])
    labels = DEFAULT_LABELS | (labels or {})

    # prepare feature names ------------------------------------------------------------------------
    if all_features:
        features = set(range(interaction_values.n_players))
    else:
        features = {feature for interaction in interactions for feature in interaction}
    n_features = len(features)
    feature_pos = {feature: n_features - 1 - i for i, feature in enumerate(features)}
    if feature_names is None:
        feature_names = [f"Feature {feature}" for feature in features]
    else:
        feature_names = [feature_names[feature] for feature in features]

    # create figure --------------------------------------------------------------------------------
    layout = [
            [0],
            [1]
        ]
    height_upper, height_lower = 5, n_features * 0.75
    height = height_upper + height_lower
    height_ratio = [height_upper, height_lower]
    
    width_left, width_right = 8.5, 1.5
    width = width_left + width_right
    width_ratio = [width_left, width_right]

    gridspec = {"height_ratios": height_ratio}

    # graph horizontal bar chart
    if show_counts:
        layout = [
            [0, '.'],
            [1, 2]
        ]

        gridspec["width_ratios"] =  width_ratio
        gridspec["wspace"] =  0

    if figsize is None:
        figsize = (width, height)
    else:
        if figsize[1] is None:
            figsize = (figsize[0], height)
        if figsize[0] is None:
            figsize = (width, figsize[1])

    fig, ax = plt.subplot_mosaic(
        layout,
        figsize=figsize,
        gridspec_kw=gridspec,
        sharex=False)
    ax[0].sharex(ax[1])

    # plot the upset plot
    for x_pos, interaction in enumerate(interactions):
        color = RED.hex if values[x_pos] >= 0 else BLUE.hex

        # plot upper part
        bar = ax[0].bar(x_pos, values[x_pos], color=color)
        label = [f"{values[x_pos]:.2f}"]
        ax[0].bar_label(bar, label, label_type="edge", color="black", fontsize=12, padding=3)

        # plot lower part
        # plot the matrix in the background
        ax[1].plot(
            [x_pos for _ in range(n_features)],
            list(range(n_features)),
            color="lightgray",
            marker="o",
            markersize=15,
            linewidth=0,
        )
        # add the interaction to the matrix
        y_pos = [feature_pos[feature] for feature in interaction]
        ax[1].plot(
            [x_pos for _ in range(len(interaction))],
            y_pos,
            color="black" if not color_matrix else color,
            marker="o",
            markersize=15,
            linewidth=1.5,
        )

    if show_remainder:
        # plot remainder value
        bar = ax[0].bar(len(interactions), remainder_value, color="black", alpha=0.4)
        label = [f"{remainder_value:.2f}"]
        ax[0].bar_label(bar, label, label_type="edge", color="black", fontsize=12, padding=3)

        # add remainder name
        rect = bar[0]
        x_center = rect.get_x() + rect.get_width() / 2.0
        pad = 10

        if remainder_value >= 0:
            y_opposite = rect.get_y()  # baseline of the bar
            va_align = "top"  # write under the bar
            pad = - pad  # offset to write under the bar
        else:
            y_opposite = rect.get_y() + rect.get_height() # top of the bar
            va_align = "bottom"
            pad = pad

        other_label_text = labels['remainder']
        ax[0].annotate(
            other_label_text,
            xy=(x_center, y_opposite),  # Anchor point in data coordinates
            xytext=(0, pad),  # Offset in points (pixels)
            textcoords="offset points",  # Ensures pad is in points, not data units
            ha="center",
            va=va_align,
            color="black",
            fontsize=12,
        )

    if hide_empty_coalition & (empty_coalition_id is not None):
        ax[0].text(
            0.98,  # just inside the right border
            empty_coalition_value,  # Y: Data coordinate value
            f"{empty_coalition_value:.2f} {labels['empty_coalition']}",
            color="lightgray",
            fontsize=11,
            va="bottom",  # on top of the line
            ha="left",  # Left-aligned so text grows to the right
            transform=ax[0].get_yaxis_transform(),  # Blend: X in axes fraction, Y in data
            clip_on=False,  # Prevent clipping outside plot area
        )
        ax[0].axhline(y=empty_coalition_value, color='lightgray', linestyle=':')
        
    if  hide_grand_coalition & (grand_coalition_id is not None):
        ax[0].text(
            0.98,  # just inside the right border
            grand_coalition_value,  # Y: Data coordinate value
            f"{grand_coalition_value:.2f} {labels['grand_coalition']}",
            color="slategrey",
            fontsize=11,
            va="bottom",  # on top of the line
            ha="left",  # Left-aligned so text grows to the right
            transform=ax[0].get_yaxis_transform(),  # Blend: X in axes fraction
            clip_on=False,  # Prevent clipping outside plot area
        )
        ax[0].axhline(y=grand_coalition_value, color='slategrey', linestyle=':')

    # counter bar plot
    if show_counts:
        if show_counts == "proportion":
            p_arr = np.array([interactions_count[f]['p_val'] for f in reversed(np.arange(n_features))])
            n_arr = np.array([interactions_count[f]['n_val'] for f in reversed(np.arange(n_features))])
            total_arr = p_arr + n_arr
            s_arr = np.array([interactions_count[f]['s'] for f in reversed(np.arange(n_features))])

            p_bars = list(np.divide(p_arr, total_arr, out=np.zeros_like(p_arr), where=total_arr!=0) * s_arr)
            n_bars = list(np.divide(n_arr, total_arr, out=np.zeros_like(n_arr), where=total_arr!=0) * s_arr)
        else:
            # bar lenghts
            p_bars = [interactions_count[f]['p'] for f in reversed(np.arange(n_features))]
            n_bars = [interactions_count[f]['n'] for f in reversed(np.arange(n_features))]
            
        # positive part
        bars = ax[2].barh(
            range(n_features),
            p_bars,
            height=0.7,
            color=RED.hex,
            align='center',
            edgecolor='none'
        )
        # negative part
        bars = ax[2].barh(
            range(n_features),
            n_bars,
            height=0.7,
            color=BLUE.hex,
            align='center',
            left=p_bars,
            edgecolor='none'
        )

        # graduation
        ax[2].set_xticks(range(0, math.ceil(max_count / 2) * 2 + 1, 2))

    # beautify upper plot --------------------------------------------------------------------------
    min_max = (min(values), max(values))
    delta = (min_max[1] - min_max[0]) * 0.1
    ax[0].set_ylim(min_max[0] - delta, min_max[1] + delta)
    ax[0].set_ylabel(labels['y_label'])
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["bottom"].set_visible(False)
    ax[0].axhline(0, color="black", linewidth=0.5)  # add line at 0

    # beautify lower plot --------------------------------------------------------------------------
    ax[1].set_ylim(-1, n_features)
    ax[1].yaxis.set_ticks(range(n_features))
    ax[1].set_yticklabels(reversed(feature_names))
    ax[1].tick_params(axis="y", length=0)  # remove y-ticks
    ax[1].set_xticks([])  # remove x-axis
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)

    # beautify right plot --------------------------------------------------------------------------
    if show_counts:
        # align matrix xith barplot
        ax[2].set_ylim(ax[1].get_ylim())
        ax[2].set_ylim(-1, n_features)
        ax[2].set_xlim(0, math.ceil(max_count / 2) * 2)

        # ticks
        ax[2].xaxis.tick_top()
        ax[2].xaxis.set_label_position('top')
        ax[2].tick_params(axis='x', width=0.5, length=4)
        ax[2].tick_params(axis="y", length=0)    # no ticks on Y
        ax[2].set_yticks([])                     # no labels on  Y

        # spines
        ax[2].spines['top'].set_linewidth(0.5)
        ax[2].spines['top'].set_position(('data', n_features-0.5))

        ax[2].set_xlabel(labels['count_label'])

        ax[2].spines["top"].set_visible(True)
        ax[2].spines["right"].set_visible(False)
        ax[2].spines["left"].set_visible(False) 
        ax[2].spines["bottom"].set_visible(False) 

    # background shading
    for i in range(n_features):
        if i % 2 == 0:
            ax[1].axhspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.25, zorder=0, lw=0)

            if show_counts:
                ax[2].axhspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.25, zorder=0, lw=0)

    # adjust whitespace
    plt.subplots_adjust(hspace=0.0)
    plt.subplots_adjust(wspace=0.0)
    plt.tight_layout()

    if not show:
        return fig
    plt.show()
    return None


class CeaInteractionValues(InteractionValues):

    def plot_upset(self, *, show: bool = True, **kwargs):

        return upset_plot(self, show=show, **kwargs)




if __name__ == '__main__':
    # import shapiq
    import pandas as pd
    import numpy as np
    import plotly.io as pio
    import seaborn as sns
    import plotly.express as px

    import os

    try:
        os.remove('figures.svg')
    except FileNotFoundError:
        pass


    ### PLOTTING STYLES ###

    # hvplot.extension('plotly')

    pio.templates['draft'] = pio.templates['simple_white']
    pio.templates['draft'].layout.xaxis = dict(showline=True,
                                                linewidth=2,
                                                linecolor='black',
                                                showgrid=False)
    pio.templates['draft'].layout.yaxis = dict(showline=True,
                                                linewidth=2,
                                                linecolor='black',
                                                showgrid=True,
                                                gridwidth=0.75,
                                                gridcolor='light grey'
                                            )
                                            #, griddash='dash')
    pio.templates['draft'].layout.height = 800
    pio.templates['draft'].layout.width = 800
    pio.templates.default = 'draft'
    sns.set_theme(style='white')

    def siq_plot(df, n_players, valuecol='Contribution to PV'):
    
        """
        """
        
        values = df[valuecol].values
        index = 'FSII'
        max_order = df['order'].max()
        n_players = n_players
        min_order = 0
        ks = list(map(tuple, list(df['keys'])))
        idx = list(df.index)
        interaction_lookup = {ks[i]: idx[i] for i in range(len(idx))}
        estimated = False
        baseline_value = values.sum()
        
        ivs = CeaInteractionValues(values=values,
                                    index=index,
                                    max_order=max_order,
                                    n_players=n_players,
                                    min_order=min_order,
                                    baseline_value=baseline_value,
                                    interaction_lookup=interaction_lookup,
                                    estimated=False,
                                    )

        return ivs

    def interact_plot(indices, features,
                    climatestep='0-100',
                    method='Shapley-Taylor',
                    mode='relative',
                    attribute='VAR_FIn',
                    process='Nuclear',
                    commodity='ELC',
                    region='World',
                    period=2050,
                    n_interactions=12,
                    max_order=None):

        max_order = len(features) if max_order is None else max_order

        siq = indices[(indices['ClimateStep']==climatestep) & (indices['method']==method) & (indices['mode']==mode) &\
                    (indices['Attribute']==attribute) & (indices['Process']==process) & (indices['Period']==period) &\
                    (indices['Commodity']==commodity) & (indices['Region']==region) & (indices['coalition']!='Total') &\
                    (indices['Order']<=max_order)].reset_index(drop=True)
        
        ivs = siq_plot(siq, 7)
        fig = ivs.plot_upset(n_interactions=n_interactions, feature_names=features, color_matrix=True, show=False, 
            hide_empty_coalition = True, graph_counts = True)
        return fig, siq, ivs

    SAVEFIG = False
    READRAW = False


    indices = pd.read_parquet('./data/indices_20250611.parquet')
    indices.loc[~indices['keys'].isna(), 'keys'] = indices.loc[~indices['keys'].isna(), 'keys'].apply(tuple)

    p = ['Share_DMD_*', 'ELC']
    climatestep = '0-100'
    method = 'Shapley-Taylor'
    mode = 'absolute'
    attribute = 'VAR_FIn'
    process = p[0]
    commodity = p[1]
    period = 2050
    n_interactions = 10
    max_order = None
    feats = ['Net Zero Obj.'] + ['SocioEco.',
                                'Geol. Seq. Pot.',
                                'Sust. Bio. Pot.',
                                'Fav. Nuc. Eco.',
                                'Fav. H2 Eco.',
                                'Gbl H2 Trade']

    f, siq, ivs = interact_plot(indices, feats, max_order=max_order, mode=mode, method=method,
                                process=process, commodity=commodity, attribute=attribute,
                                n_interactions=n_interactions)

    f.savefig('figures.svg', format='svg')


