"""
Internal helpers for :mod:`mypackage.interactions`.

Contains: player-name generation, the value-function bridge to ``nshap`` (with a
fast O(1) coalition lookup and a user-controlled missing-coalition policy), the
conversion of ``nshap`` outputs to tidy records, and the weighted Kendall-tau
distance used to compare two allocations.

Compatibility note
-------------------
``nshap`` 0.2.0 calls ``np.math.factorial``; ``np.math`` was removed in NumPy
>= 2. We restore the alias on import (harmless when it already exists) so the
exact ``nshap`` implementations keep working -- preserving result stability with
the user's current pipeline.
"""

from __future__ import annotations  # Modern annotation syntax.

import math  # Standard-library factorials etc.
import string  # Uppercase alphabet for player names.
from itertools import product  # Cartesian products for names and coalitions.
from typing import Any, Callable, Iterable, Mapping, Sequence  # Documentation aliases.

import numpy as np  # Numerics.
import pandas as pd  # DataFrames / Series.

if not hasattr(np, "math"):  # NumPy >= 2 removed the ``np.math`` alias that nshap 0.2.0 relies on.
    np.math = math  # Restore it so nshap's exact implementations keep working.

import nshap  # Cooperative-game interaction indices (imported after the shim).


# ---------------------------------------------------------------------------
# Player names
# ---------------------------------------------------------------------------
def generate_player_names(n: int, letters: str = string.ascii_uppercase) -> list[str]:
    """Generate ``n`` spreadsheet-style names: ``A, B, ..., Z, AA, AB, ...``.

    Parameters
    ----------
    n:
        Number of names to generate.
    letters:
        Alphabet to draw from (kwarg for generalisation; default A-Z).

    Returns
    -------
    list of str
        ``n`` names in lexicographic order.
    """
    names: list[str] = []  # Accumulated names.
    length = 1  # Current name length (1, then 2, ...).
    while len(names) < n:  # Keep widening until we have enough.
        for combo in product(letters, repeat=length):  # All combos of the current length.
            names.append("".join(combo))  # Join characters into a name.
            if len(names) == n:  # Stop as soon as we have n.
                return names
        length += 1  # Not enough yet: try longer names.
    return names  # (Reached only if n == 0.)


# ---------------------------------------------------------------------------
# Argument normalisation
# ---------------------------------------------------------------------------
def as_list(x: Any) -> list[Any]:
    """Return ``[]`` for ``None``, ``[x]`` for a scalar/str, else ``list(x)``."""
    if x is None:  # No value.
        return []
    if isinstance(x, str):  # A single string is one item, not characters.
        return [x]
    if isinstance(x, Iterable):  # Any other iterable.
        return list(x)
    return [x]  # A lone scalar.


# ---------------------------------------------------------------------------
# Value-function bridge to nshap
# ---------------------------------------------------------------------------
def build_value_lookup(
    frame: pd.DataFrame,
    players: Sequence[str],
    value_col: str,
    inference: float | None,
) -> tuple[dict[frozenset, float], int, int]:
    """Build an O(1) ``{coalition: value}`` lookup from a coalition table.

    Each row encodes a coalition by 0/1 presence in the ``players`` columns; the
    coalition key is the ``frozenset`` of *player indices* present (matching the
    indices nshap passes to the value function).

    Parameters
    ----------
    frame:
        Rows of one game (one value per coalition).
    players:
        Ordered player columns (0/1 presence).
    value_col:
        Column holding the characteristic-function value ``v(S)``.
    inference:
        Missing-coalition policy. ``None`` requires the table to be complete
        (all ``2**n`` coalitions present) and raises otherwise; a number is used
        as ``v(S)`` for any coalition absent from the table.

    Returns
    -------
    (dict, int, int)
        The lookup, the number of players ``n``, and the count of missing
        coalitions (0 when complete).

    Raises
    ------
    ValueError
        If a coalition is duplicated, or if coalitions are missing and
        ``inference is None``.
    """
    n = len(players)  # Number of players.
    sub = frame[list(players) + [value_col]].copy()  # Only the relevant columns.

    present: dict[frozenset, float] = {}  # coalition (frozenset of indices) -> value.
    for row in sub.itertuples(index=False):  # Iterate rows as tuples (fast).
        flags = row[:n]  # The 0/1 presence flags.
        key = frozenset(i for i, f in enumerate(flags) if f)  # Indices of present players.
        if key in present:  # The same coalition must not appear twice in one game.
            raise ValueError(f"Duplicated coalition {sorted(key)} in the value table; "
                             f"did you forget a groupby column?")
        present[key] = float(row[n])  # Store its value.

    expected = 1 << n  # 2**n coalitions in a complete game.
    n_missing = expected - len(present)  # How many are absent.
    if n_missing > 0 and inference is None:  # No fill policy -> the table must be complete.
        raise ValueError(f"{n_missing} of {expected} coalitions are missing and inference=None. "
                         f"Pass inference=<constant> (e.g. 0) to fill missing coalitions.")
    return present, n, n_missing


def make_vfunc(present: Mapping[frozenset, float], inference: float | None) -> Callable:
    """Return an nshap-compatible value function backed by an O(1) lookup.

    Parameters
    ----------
    present:
        Mapping ``frozenset(player indices) -> value``.
    inference:
        Fill value for coalitions absent from ``present`` (used only when not
        ``None``; completeness is enforced upstream otherwise).

    Returns
    -------
    callable
        ``v_func(x, S)`` ignoring ``x`` and returning ``v(S)``.
    """
    def v_func(x, S):  # nshap calls v_func(x, S); x (the data row) is unused here.
        key = frozenset(S)  # Coalition as a frozenset of indices.
        value = present.get(key)  # O(1) lookup.
        if value is None:  # Absent coalition.
            return float(inference)  # Filled with the inference constant.
        return value  # Stored value.
    return v_func


def nshap_records(interaction: Mapping[tuple, float], players: Sequence[str]) -> list[dict]:
    """Convert an nshap interaction dict to tidy records (one per coalition).

    Parameters
    ----------
    interaction:
        ``{tuple_of_player_indices: value}`` as returned by nshap.
    players:
        Ordered player columns (to translate indices to names).

    Returns
    -------
    list of dict
        Records with ``order``, ``coalition`` (comma-joined names), ``keys``
        (the original index tuple) and ``value``. The empty coalition is dropped.
    """
    records = []  # Accumulated rows.
    for key, value in interaction.items():  # Each coalition and its index value.
        if len(key) == 0:  # Skip the empty coalition (no contribution meaning).
            continue
        records.append({
            "order": len(key),  # Interaction order = coalition size.
            "coalition": ", ".join(players[i] for i in key),  # Human-readable coalition.
            "keys": tuple(key),  # Original index tuple (stable identifier).
            "value": float(value),  # The index value.
        })
    return records


# ---------------------------------------------------------------------------
# Value-based distances between two allocations
# ---------------------------------------------------------------------------
def _aligned_vectors(alloc_a: Mapping[Any, float],
                     alloc_b: Mapping[Any, float]) -> tuple[np.ndarray, np.ndarray]:
    """Align two allocations over the union of their coalitions.

    A coalition absent from an allocation is given the value ``0`` there (the same
    missing rule as :func:`weighted_kendall_tau`), so the two returned vectors are
    aligned element-wise over the union.

    Parameters
    ----------
    alloc_a, alloc_b:
        ``{coalition: value}`` mappings.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The two value vectors over the shared coalition universe.
    """
    universe = list(dict.fromkeys(list(alloc_a) + list(alloc_b)))  # Union, stable order.
    va = np.array([float(alloc_a.get(c, 0.0)) for c in universe])  # Missing -> 0 in A.
    vb = np.array([float(alloc_b.get(c, 0.0)) for c in universe])  # Missing -> 0 in B.
    return va, vb


def mse(alloc_a: Mapping[Any, float], alloc_b: Mapping[Any, float]) -> float:
    """Mean squared error between two allocations (missing coalition -> value 0)."""
    va, vb = _aligned_vectors(alloc_a, alloc_b)  # Aligned over the union.
    if va.size == 0:  # No coalitions -> zero distance.
        return 0.0
    return float(np.mean((va - vb) ** 2))  # Mean of squared differences.


def rmse(alloc_a: Mapping[Any, float], alloc_b: Mapping[Any, float]) -> float:
    """Root mean squared error between two allocations (missing -> value 0)."""
    return float(np.sqrt(mse(alloc_a, alloc_b)))  # Square root of the MSE.


def mae(alloc_a: Mapping[Any, float], alloc_b: Mapping[Any, float]) -> float:
    """Mean absolute error between two allocations (missing -> value 0)."""
    va, vb = _aligned_vectors(alloc_a, alloc_b)  # Aligned over the union.
    if va.size == 0:
        return 0.0
    return float(np.mean(np.abs(va - vb)))  # Mean of absolute differences.


def chebyshev(alloc_a: Mapping[Any, float], alloc_b: Mapping[Any, float]) -> float:
    """Chebyshev (max absolute) distance between two allocations (missing -> 0)."""
    va, vb = _aligned_vectors(alloc_a, alloc_b)  # Aligned over the union.
    if va.size == 0:
        return 0.0
    return float(np.max(np.abs(va - vb)))  # Largest absolute difference.


def euclidean(alloc_a: Mapping[Any, float], alloc_b: Mapping[Any, float]) -> float:
    """Euclidean (L2) distance between two allocations (missing -> value 0)."""
    va, vb = _aligned_vectors(alloc_a, alloc_b)  # Aligned over the union.
    if va.size == 0:
        return 0.0
    return float(np.sqrt(np.sum((va - vb) ** 2)))  # L2 norm of the difference.


def manhattan(alloc_a: Mapping[Any, float], alloc_b: Mapping[Any, float]) -> float:
    """Manhattan (L1) distance between two allocations (missing -> value 0)."""
    va, vb = _aligned_vectors(alloc_a, alloc_b)  # Aligned over the union.
    if va.size == 0:
        return 0.0
    return float(np.sum(np.abs(va - vb)))  # L1 norm of the difference.


# ---------------------------------------------------------------------------
# Weighted Kendall-tau distance between two allocations
# ---------------------------------------------------------------------------
def _strict_positions(values: Mapping[Any, float], universe: Sequence[Any]) -> dict[Any, int]:
    """Assign strict 1-based positions over ``universe`` for one allocation.

    Present coalitions come first, ordered by descending ``|value|``; coalitions
    missing from ``values`` are placed last (the "maximum rank" rule). Ties are
    broken deterministically by ``str(coalition)`` so the *same* tie-break is used
    for both allocations -- pairs of coalitions tied this way never count as
    inversions.

    Parameters
    ----------
    values:
        ``{coalition: value}`` for one allocation (may omit some of ``universe``).
    universe:
        The full set of coalitions (union of both allocations).

    Returns
    -------
    dict
        ``{coalition: position}`` with position 1 = most important.
    """
    present = sorted((c for c in universe if c in values),  # Present coalitions, best first.
                     key=lambda c: (-abs(values[c]), str(c)))
    missing = sorted((c for c in universe if c not in values), key=str)  # Missing -> last.
    order = present + missing  # Full strict ordering.
    return {c: i + 1 for i, c in enumerate(order)}  # 1-based positions.


def weighted_kendall_tau(
    alloc_a: Mapping[Any, float],
    alloc_b: Mapping[Any, float],
    normalize: bool = True,
    weight: str = "max",
    position_weights: Any = None,
) -> float:
    """Element- (and optionally position-) weighted Kendall-tau distance.

    Implements the weighted Kendall's tau of Kumar & Vassilvitskii, *Generalized
    Distances between Rankings* (WWW 2010). With element weights only this is
    eq. (3); with position weights it is eq. (5)/(11) (without the element-
    similarity term ``D_ij``):

        ``K = sum over pairs (i, j) of  w_i w_j * pbar_i pbar_j * [a, b order (i, j) oppositely]``

    Every pair of coalitions ranked in opposite order by the two allocations
    contributes the product of their **element weights** ``w`` (the magnitudes of
    the interaction values) times the product of their **position weights**
    ``pbar`` (which let disagreements near the top of the ranking cost more).

    Missing coalitions (different maximum order and/or cutoff), per your rule: a
    coalition absent from an allocation gets the **maximum rank** (placed last)
    **and the value 0** there, so its element weight comes only from the
    allocation in which it is present.

    Position weights (Kumar-Vassilvitskii)
    --------------------------------------
    With ``delta_k`` the cost of swapping positions ``k-1`` and ``k`` (default
    ``1``), let ``p_1 = 1`` and ``p_k = p_{k-1} + delta_k``. For each coalition,
    ``pbar = (p_r - p_s) / (r - s)`` (``1`` if ``r == s``) is the average position
    cost of moving it from its position ``r`` in allocation ``a`` (the reference)
    to its position ``s`` in allocation ``b``. Uniform ``delta`` gives ``pbar = 1``
    and collapses to the element-weighted form.

    Parameters
    ----------
    alloc_a, alloc_b:
        ``{coalition: value}`` mappings. ``alloc_a`` is the position-weight
        reference (relevant only when ``position_weights`` is set).
    normalize:
        Divide by the total pair weight to get a distance in ``[0, 1]``
        (``normalize=False`` returns the raw KV ``K``).
    weight:
        Element-weight combination of the two magnitudes (missing side = 0):
        ``"max"`` (default), ``"mean"`` or ``"sum"``.
    position_weights:
        ``None`` (default) for uniform swap costs (no position weighting); a
        **callable** ``delta(k) -> float`` for ``k = 2..n``; or a **sequence** of
        swap costs interpreted as ``delta_2, ..., delta_n`` (shorter sequences are
        padded with their last value). Larger costs near the top make head-of-
        ranking disagreements costlier.

    Returns
    -------
    float
        The (optionally normalised) weighted Kendall-tau distance.
    """
    universe = list(dict.fromkeys(list(alloc_a) + list(alloc_b)))  # Union, stable order.
    n = len(universe)  # Number of coalitions ranked.
    if n < 2:  # Fewer than two coalitions -> no pair -> zero distance.
        return 0.0

    pos_a = _strict_positions(alloc_a, universe)  # Strict positions under A (reference).
    pos_b = _strict_positions(alloc_b, universe)  # Strict positions under B.

    def magnitude(alloc: Mapping[Any, float], c: Any) -> float:  # |value|, or 0 if absent.
        return abs(alloc[c]) if c in alloc else 0.0

    w: dict[Any, float] = {}  # Element weight per coalition (missing side contributes 0).
    for c in universe:
        va, vb = magnitude(alloc_a, c), magnitude(alloc_b, c)  # Per-allocation magnitudes.
        if weight == "max":  # Single present value when present in only one allocation.
            w[c] = max(va, vb)
        elif weight == "mean":  # Average of the two (missing = 0).
            w[c] = 0.5 * (va + vb)
        elif weight == "sum":  # Sum of the two (missing = 0).
            w[c] = va + vb
        else:
            raise ValueError("weight must be 'max', 'mean' or 'sum'.")

    # Position-weight factor pbar per coalition (1.0 everywhere when uniform).
    if position_weights is None:  # Uniform swap costs -> no position weighting.
        pbar = {c: 1.0 for c in universe}
    else:
        def delta(k: int) -> float:  # Swap cost between positions k-1 and k (k = 2..n).
            if callable(position_weights):  # A user function of the position.
                return float(position_weights(k))
            seq = list(position_weights)  # A sequence delta_2..delta_n.
            idx = k - 2  # Position k maps to index k-2.
            return float(seq[idx]) if idx < len(seq) else float(seq[-1])  # Pad with the last value.
        prefix = [0.0] * (n + 1)  # 1-based prefix sums p_1..p_n.
        prefix[1] = 1.0  # p_1 = 1.
        for k in range(2, n + 1):
            prefix[k] = prefix[k - 1] + delta(k)  # p_k = p_{k-1} + delta_k.
        pbar = {}  # Average position cost per coalition.
        for c in universe:
            r, s = pos_a[c], pos_b[c]  # Positions in the reference (A) and in B.
            pbar[c] = 1.0 if r == s else (prefix[r] - prefix[s]) / (r - s)  # KV average cost.

    eff = {c: w[c] * pbar[c] for c in universe}  # Effective weight e = w * pbar (factorises KV eq. 11).

    num = 0.0  # Weighted inverted-pair mass (the numerator).
    den = 0.0  # Total pair-weight mass (for normalisation).
    for i in range(n):  # All unordered pairs (i, j), i < j.
        ci = universe[i]
        for j in range(i + 1, n):
            cj = universe[j]
            wij = eff[ci] * eff[cj]  # Product of effective weights.
            den += wij  # Accumulate total mass.
            if (pos_a[ci] - pos_a[cj]) * (pos_b[ci] - pos_b[cj]) < 0:  # Opposite order -> inverted.
                num += wij  # Inverted pair contributes its weight.
    if normalize:  # Normalise to [0, 1] (KV itself does not normalise).
        return num / den if den > 0 else 0.0
    return num  # Raw weighted Kendall tau K.
