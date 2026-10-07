"""Plotting domain: faceted allocation plots, diagnostic grids and jointplots.

The public API lives in :mod:`mypackage.plot.plot`; the internals are split into
:mod:`~mypackage.plot.specs` (data structures), :mod:`~mypackage.plot.prepare`
(computation) and :mod:`~mypackage.plot.render` (rendering).
"""

from .plot import *  # noqa: F401,F403 -- re-export the public API (entry points + spec objects).
from . import encodings as encodings  # Redundant alias = explicit re-export (PEP 484 convention Pylance honours).
from .plot import __all__ as _plot_all

__all__ = [*_plot_all, "encodings"]  # Same `*` surface as the module + the builders submodule.
