from .plot import *  # noqa: F401,F403 -- re-export the plotting public API at the package root.
from . import plot as plot  # Explicit re-export so `import mypackage as pt; pt.plot...` type-checks.
from .plot import encodings as encodings  # Builders reachable as pt.encodings too (both import styles).
from .plot import __all__ as _plot_all

__all__ = [*_plot_all, "plot", "encodings"]
