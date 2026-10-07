"""Interactions domain: cooperative-game interaction indices and rank distances.

The public API lives in :mod:`mypackage.interactions.interactions`; the helpers
(player names, the nshap value-function bridge, the weighted Kendall-tau) live in
:mod:`mypackage.interactions.utils`.
"""

from .interactions import *  # noqa: F401,F403 -- re-export the public API.
from .interactions import __all__  # noqa: F401 -- expose the same `*` surface as the module.
