import copy
import plotly.io as pio
import seaborn as sns
import numpy as np


def apply_theme() -> None:
    """Configure Plotly and Seaborn defaults for publication-ready figures."""
    # Deepcopy to prevent mutating the built-in 'simple_white' template in memory
    draft_template = copy.deepcopy(pio.templates["simple_white"])

    draft_template.layout.xaxis = dict(
        showline=True, linewidth=2, linecolor="black", showgrid=False
    )
    draft_template.layout.yaxis = dict(
        showline=True,
        linewidth=2,
        linecolor="black",
        showgrid=True,
        gridwidth=0.75,
        gridcolor="light grey",
    )
    draft_template.layout.height = 800
    draft_template.layout.width = 800

    pio.templates["draft"] = draft_template
    pio.templates.default = "draft"

    sns.set_theme(style="white")

    np.set_printoptions(legacy='1.25')


# Automatically apply styling upon module import
apply_theme()