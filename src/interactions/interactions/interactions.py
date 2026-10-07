"""
Cooperative-game interaction indices and allocation distances.

Public functions
----------------
* :func:`load_game`        -- build a toy cooperative game (convex / non-convex).
* :func:`compute_indices`  -- interaction / contribution indices from a value
  table, via ``nshap`` (+ Harsanyi dividends), generalised over indicators and
  context groups, with a user-controlled missing-coalition policy.
* :func:`compute_distances`-- pairwise distances between allocation methods
  (a menu of rank- and value-based metrics), over arbitrary context groups.

Helpers (player names, the nshap value-function bridge, the weighted Kendall-tau
itself) live in :mod:`mypackage.interactions.utils`.
"""

from __future__ import annotations  # Modern annotation syntax.

from itertools import combinations  # Unordered method pairs.
from typing import Any, Mapping, Sequence  # Documentation aliases.

import numpy as np  # Numerics.
import pandas as pd  # DataFrames.

# Importing the helpers module also applies the ``np.math`` shim and imports nshap.
from . import utils as iu
from .utils import (  # noqa: F401 (re-export)
    chebyshev,
    euclidean,
    generate_player_names,
    mae,
    manhattan,
    mse,
    rmse,
    weighted_kendall_tau,
)
import nshap  # Safe to import here: the shim ran inside interactions_utils.

# Distance registry: key -> (callable, accepts the Kendall-tau-specific options).
# Rank-based: the element-/position-weighted Kendall tau. Value-based: MSE, RMSE,
# MAE, Chebyshev (each aligns the two allocations on the union, missing -> 0).
_DISTANCES: dict[str, tuple[Any, bool]] = {
    "weighted_kendall_tau": (weighted_kendall_tau, True),  # Uses normalize / weight / position_weights.
    "mse": (mse, False),
    "rmse": (rmse, False),
    "mae": (mae, False),
    "chebyshev": (chebyshev, False),
    "euclidean": (euclidean, False),
    "manhattan": (manhattan, False),
}

#: Names of the distances available to :func:`compute_distances`.
DISTANCE_METRICS: tuple[str, ...] = tuple(_DISTANCES)

__all__ = ["load_game", "compute_indices", "compute_distances",
           "generate_player_names", "weighted_kendall_tau", "DISTANCE_METRICS"]

# Method registry: key -> (nshap callable, needs the max-order argument, display label).
# Harsanyi dividends are the Moebius transform, which nshap provides directly.
_METHODS: dict[str, tuple[Any, bool, str]] = {
    "shapley": (nshap.shapley_values, False, "Shapley"),
    "harsanyi": (nshap.moebius_transform, False, "Harsanyi dividends"),
    "shapley_interaction": (nshap.shapley_interaction_index, True, "Shapley Interaction"),
    "shapley_taylor": (nshap.shapley_taylor, True, "Shapley-Taylor"),
    "n_shapley": (nshap.n_shapley_values, True, "n-Shapley"),
    "faith_shapley": (nshap.faith_shap, True, "Faith-Shapley"),
    "banzhaf_interaction": (nshap.banzhaf_interaction_index, True, "Banzhaf Interaction"),
    "faith_banzhaf": (nshap.faith_banzhaf, True, "Faith-Banzhaf"),
}


# ===========================================================================
# load_game
# ===========================================================================
def load_game(
    nplayers: int = 2,
    convex: bool = True,
    *,
    returns_coeff: float = 0.5,
    returns_exp: float = 1.5,
    demands: Sequence[float] | None = None,
    demands_base: float = 2.0,
    nonconvex_base: float = 25.0,
    value_col: str = "f(S)",
    letters: str | None = None,
) -> pd.DataFrame:
    """Build the full coalition table of a small cooperative game.

    The returned frame has one 0/1 column per player (presence in the coalition)
    and one ``value_col`` column with the characteristic function ``v(S)``. All
    the formerly hard-coded constants are exposed as keyword arguments.

    Parameters
    ----------
    nplayers:
        Number of players ``n`` (the table has ``2**n`` rows).
    convex:
        If ``True``, ``v(S) = d.S + returns_coeff * (d.S) ** returns_exp`` (a
        super-additive / convex-style payoff). If ``False``, the non-convex
        ``v(S) = nonconvex_base - d.S`` for non-empty ``S`` (and ``0`` for the
        empty coalition), where ``d.S`` is the summed demand of ``S``.
    returns_coeff, returns_exp:
        Coefficient and exponent of the convex term.
    demands:
        Per-player demands. If ``None``, ``demands_base ** arange(n)`` is used.
    demands_base:
        Base of the default geometric demands.
    nonconvex_base:
        Constant of the non-convex payoff.
    value_col:
        Name of the value column.
    letters:
        Alphabet for player names (defaults to A-Z).

    Returns
    -------
    pandas.DataFrame
        ``2**n`` rows; columns = player names + ``value_col``.
    """
    names = (generate_player_names(nplayers) if letters is None
             else generate_player_names(nplayers, letters))  # Player column names.

    coalitions = np.array(list(__import__("itertools").product([0, 1], repeat=nplayers)),
                          dtype=int) if nplayers > 0 else np.zeros((1, 0), dtype=int)  # All 2**n rows.
    df = pd.DataFrame(coalitions, columns=names)  # 0/1 presence table.

    demand_vec = (np.asarray(demands, dtype=float) if demands is not None
                  else demands_base ** np.arange(nplayers))  # Per-player demands.

    summed = coalitions @ demand_vec if nplayers > 0 else np.zeros(len(df))  # d.S for each coalition.
    if convex:  # Convex / increasing-returns payoff.
        values = summed + returns_coeff * (summed ** returns_exp)
    else:  # Non-convex payoff: a constant minus the summed demand (0 for the empty coalition).
        counts = coalitions.sum(axis=1) if nplayers > 0 else np.zeros(len(df))
        values = np.where(counts > 0, nonconvex_base - summed, 0.0)

    df[value_col] = values  # Attach the characteristic function.
    return df


# ===========================================================================
# compute_indices
# ===========================================================================
def compute_indices(
    df: pd.DataFrame,
    players: Sequence[str],
    value: str | Sequence[str],
    *,
    groupby: str | Sequence[str] | None = None,
    order: int = -1,
    method: str | Sequence[str] | None = None,
    value_mode: str | Sequence[str] | None = None,
    cutoff: Mapping[str, float] | None = None,
    inference: float | None = None,
) -> pd.DataFrame:
    """Compute interaction / contribution indices from a coalition value table.

    Generalises the original: the player columns and value column(s) are explicit
    arguments, several indicators can be processed at once, and a ``groupby`` lets
    you iterate over context combinations in a single call (no external loop).

    Parameters
    ----------
    df:
        Long table. Within each ``groupby`` context it must contain one row per
        coalition, with 0/1 presence in the ``players`` columns and the
        characteristic-function value in each ``value`` column.
    players:
        Player columns (0/1 presence), in the desired order.
    value:
        Value column, or a list of indicator columns to iterate over.
    groupby:
        Context column(s); indices are computed independently per group, and the
        group keys are carried into the output. ``None`` treats ``df`` as one
        game.
    order:
        Maximum interaction order for the order-aware methods. ``-1`` uses ``n``.
    method:
        Method key(s); ``None`` computes all of: ``shapley``, ``harsanyi``,
        ``shapley_interaction``, ``shapley_taylor``, ``n_shapley``,
        ``faith_shapley``, ``banzhaf_interaction``, ``faith_banzhaf``.
    value_mode:
        ``"absolute"``, ``"relative"`` (share of the per-method total), or both
        (``None``).
    cutoff:
        Optional ``{mode: threshold}``; within each context/indicator/method/mode,
        coalitions with ``|value| <= threshold`` are merged into a single
        ``"Multiple"`` row.
    inference:
        Missing-coalition policy. ``None`` (default) requires complete games and
        raises otherwise; a constant (e.g. ``0``) fills any missing coalition's
        ``v(S)`` with that constant.

    Returns
    -------
    pandas.DataFrame
        Tidy long output with columns: the ``groupby`` columns (if any),
        ``indicator, method, mode, order, coalition, keys, contribution,
        abs_contribution``.

    Raises
    ------
    KeyError
        If a referenced column is absent.
    ValueError
        On invalid options, or on missing coalitions when ``inference is None``.
    """
    players = list(players)  # Materialise.
    indicators = iu.as_list(value)  # One or several indicator columns.
    group_cols = iu.as_list(groupby)  # Context columns (possibly empty).
    methods = list(_METHODS) if method is None else iu.as_list(method)  # Methods to run.
    modes = ["absolute", "relative"] if value_mode is None else iu.as_list(value_mode)  # Output modes.

    # --- validation -------------------------------------------------------
    unknown = [m for m in methods if m not in _METHODS]  # Reject unknown methods.
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; valid: {sorted(_METHODS)}.")
    bad_modes = [m for m in modes if m not in ("absolute", "relative")]  # Reject unknown modes.
    if bad_modes:
        raise ValueError(f"value_mode must be 'absolute' / 'relative'; got {bad_modes}.")
    referenced = players + indicators + group_cols  # All required columns.
    missing_cols = [c for c in dict.fromkeys(referenced) if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in df: {missing_cols}.")

    n = len(players)  # Number of players.
    max_order = n if order == -1 else int(order)  # Resolve the max order.
    x = np.zeros((1, n))  # Dummy data row for nshap.

    # --- iterate over context groups x indicators -------------------------
    groups = [(None, df)] if not group_cols else list(df.groupby(group_cols, dropna=False))
    records: list[dict] = []  # Absolute-value records.

    for gkey, gdf in groups:  # Each context combination.
        gvals = (list(gkey) if isinstance(gkey, tuple) else [gkey]) if group_cols else []  # Group values.
        for indicator in indicators:  # Each indicator column.
            present, _, _ = iu.build_value_lookup(gdf, players, indicator, inference)  # Value lookup.
            v_func = iu.make_vfunc(present, inference)  # nshap value function.
            for mkey in methods:  # Each requested method.
                func, needs_n, label = _METHODS[mkey]  # Registry entry.
                interaction = func(x, v_func, max_order) if needs_n else func(x, v_func)  # Run nshap.
                for rec in iu.nshap_records(interaction, players):  # Tidy each coalition.
                    rec.update(method=label, indicator=indicator)  # Tag method + indicator.
                    for k, val in zip(group_cols, gvals):  # Tag the context.
                        rec[k] = val
                    records.append(rec)

    if not records:  # Nothing produced (e.g. empty df).
        return pd.DataFrame(columns=group_cols + ["indicator", "method", "mode", "order",
                                                  "coalition", "keys", "contribution", "abs_contribution"])

    base = pd.DataFrame(records)  # Absolute contributions, long.
    base = base.rename(columns={"value": "contribution"})  # Final value column name.
    ctx_cols = group_cols + ["indicator", "method"]  # Columns identifying one allocation.

    # --- build the requested modes ---------------------------------------
    frames = []  # One frame per mode.
    if "absolute" in modes:  # Absolute contributions as-is.
        a = base.copy()
        a["mode"] = "absolute"
        frames.append(a)
    if "relative" in modes:  # Share of the per-allocation total.
        r = base.copy()
        denom = r.groupby(ctx_cols)["contribution"].transform("sum")  # Total per allocation.
        r["contribution"] = np.where(denom != 0, r["contribution"] / denom, np.nan)  # Normalise.
        r["mode"] = "relative"
        frames.append(r)
    out = pd.concat(frames, axis=0, ignore_index=True)  # All modes together.

    # --- optional cutoff aggregation -------------------------------------
    if cutoff:  # Merge small coalitions into a single "Multiple" row, per allocation/mode.
        out = _apply_cutoff(out, ctx_cols, cutoff)

    # --- per-order Total rows + |contribution| ---------------------------
    totals = (out.groupby(ctx_cols + ["mode", "order"], dropna=False)["contribution"]
              .sum().reset_index())  # Sum across coalitions, per order.
    totals["coalition"] = "Total"  # Mark as totals.
    totals["keys"] = None  # Totals have no single coalition key.
    out = pd.concat([out, totals], axis=0, ignore_index=True, sort=False)  # Append totals.
    out["abs_contribution"] = out["contribution"].abs()  # Convenience magnitude column.

    ordered_cols = group_cols + ["indicator", "method", "mode", "order",
                                 "coalition", "keys", "contribution", "abs_contribution"]
    return out[ordered_cols]  # Tidy column order.


def _apply_cutoff(out: pd.DataFrame, ctx_cols: Sequence[str], cutoff: Mapping[str, float]) -> pd.DataFrame:
    """Merge sub-threshold coalitions into one ``"Multiple"`` row per allocation/mode.

    Parameters
    ----------
    out:
        Long contributions (with a ``mode`` column).
    ctx_cols:
        Columns identifying one allocation (group + indicator + method).
    cutoff:
        ``{mode: threshold}``; rows with ``|contribution| <= threshold`` in that
        mode are aggregated.

    Returns
    -------
    pandas.DataFrame
        The frame with small coalitions collapsed.
    """
    kept = []  # Rebuilt groups.
    for _, grp in out.groupby(list(ctx_cols) + ["mode"], dropna=False):  # Per allocation x mode.
        mode = grp["mode"].iloc[0]  # This group's mode.
        thr = cutoff.get(mode)  # Threshold for that mode (or None).
        if thr is None:  # No cutoff for this mode -> keep as-is.
            kept.append(grp)
            continue
        small = grp["contribution"].abs() <= thr  # Sub-threshold coalitions.
        big = grp[~small]  # Kept individually.
        if small.any():  # Collapse the small ones into a single row.
            row = grp[small].iloc[0].copy()  # Template row (keeps context columns).
            row["contribution"] = grp.loc[small, "contribution"].sum()  # Summed mass.
            row["coalition"] = "Multiple"  # Bucket label.
            row["order"] = "Multiple"  # Mixed orders.
            row["keys"] = None
            kept.append(pd.concat([big, row.to_frame().T], ignore_index=True))
        else:
            kept.append(grp)
    return pd.concat(kept, axis=0, ignore_index=True, sort=False)  # Reassemble.


# ===========================================================================
# compute_distances
# ===========================================================================
def compute_distances(
    df: pd.DataFrame,
    *,
    metrics: str | Sequence[str] = "weighted_kendall_tau",
    method_col: str = "method",
    coalition_col: str = "coalition",
    value_col: str = "contribution",
    groupby: str | Sequence[str] | None = None,
    methods: Sequence[str] | Sequence[Sequence[str]] | None = None,
    exclude_coalitions: Sequence[str] = ("Total",),
    normalize: bool = True,
    weight: str = "max",
    position_weights: Any = None,
) -> pd.DataFrame:
    """Pairwise distances between allocation methods, over arbitrary contexts.

    For each context group, each method's rows form an allocation
    ``{coalition: value}``; the requested distance(s) are then computed for every
    requested method pair. This typically consumes the output of
    :func:`compute_indices` directly, with no external looping.

    Available distances (see :data:`DISTANCE_METRICS`)
    --------------------------------------------------
    * ``"weighted_kendall_tau"`` -- rank-based Kumar-Vassilvitskii distance
      (honours ``normalize`` / ``weight`` / ``position_weights``);
    * ``"mse"``, ``"rmse"``, ``"mae"``, ``"chebyshev"``, ``"euclidean"``,
      ``"manhattan"`` -- value-based distances on the contribution vectors.

    All distances align the two allocations on the union of their coalitions, with
    a coalition absent from one allocation taken as value ``0`` there (the same
    missing rule throughout).

    Parameters
    ----------
    df:
        Long table of indices (e.g. from :func:`compute_indices`).
    metrics:
        One distance name, or several. Each produces a row per (context, pair).
    method_col, coalition_col, value_col:
        Columns holding the method label, the coalition identifier and the value.
    groupby:
        Context column(s) defining one comparison each (e.g. ``["indicator",
        "mode"]`` plus any scenario columns). ``None`` compares over the whole
        frame.
    methods:
        Which methods to compare. ``None`` -> all present (all unordered pairs);
        a flat list -> all unordered pairs among them; a list of sub-lists ->
        only the unordered pairs *within* each sub-list. Entries must match the
        values in ``method_col``.
    exclude_coalitions:
        Coalition labels to drop before comparing (default drops the ``"Total"``
        rows added by :func:`compute_indices`).
    normalize, weight, position_weights:
        Options forwarded to
        :func:`~mypackage.interactions.utils.weighted_kendall_tau`; ignored by the
        value-based distances.

    Returns
    -------
    pandas.DataFrame
        Tidy long output, one row per (context, method pair, metric): the
        ``groupby`` columns plus ``method_a, method_b, n_coalitions, metric,
        distance``.

    Raises
    ------
    KeyError
        If a referenced column is absent.
    ValueError
        If an unknown metric is requested.
    """
    metric_list = iu.as_list(metrics)  # One or several distances.
    unknown = [m for m in metric_list if m not in _DISTANCES]  # Reject unknown metrics.
    if unknown:
        raise ValueError(f"Unknown metric(s) {unknown}; valid: {list(DISTANCE_METRICS)}.")

    group_cols = iu.as_list(groupby)  # Context columns.
    referenced = [method_col, coalition_col, value_col] + group_cols  # Required columns.
    missing = [c for c in dict.fromkeys(referenced) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in df: {missing}.")

    work = df[~df[coalition_col].isin(set(exclude_coalitions))].copy()  # Drop excluded coalitions.

    # Decide the pair list per group from the ``methods`` argument.
    def pairs_for(present: list[str]) -> list[tuple[str, str]]:
        if methods is None:  # All unordered pairs among present methods.
            return list(combinations(sorted(present), 2))
        if methods and all(isinstance(m, (list, tuple)) for m in methods):  # List of sub-lists.
            out_pairs: list[tuple[str, str]] = []
            for sub in methods:  # Pairs within each sub-list (only methods present here).
                sub_present = [m for m in sub if m in present]
                out_pairs += list(combinations(sub_present, 2))
            return out_pairs
        flat = [m for m in methods if m in present]  # Flat list: pairs among the requested ones.
        return list(combinations(flat, 2))

    def distance(metric: str, a: Mapping, b: Mapping) -> float:  # Dispatch one metric on a pair.
        func, takes_kt_opts = _DISTANCES[metric]  # Registry entry.
        if takes_kt_opts:  # Kendall tau takes the rank-distance options.
            return func(a, b, normalize=normalize, weight=weight, position_weights=position_weights)
        return func(a, b)  # Value-based distances take just the two allocations.

    groups = [(None, work)] if not group_cols else list(work.groupby(group_cols, dropna=False))
    rows: list[dict] = []  # Output rows.

    for gkey, gdf in groups:  # Each comparison context.
        gvals = (list(gkey) if isinstance(gkey, tuple) else [gkey]) if group_cols else []  # Group values.
        # Build each method's allocation {coalition: value}.
        alloc = {m: dict(zip(mdf[coalition_col], mdf[value_col]))
                 for m, mdf in gdf.groupby(method_col, dropna=False)}
        present_methods = list(alloc)  # Methods available in this context.
        for ma, mb in pairs_for(present_methods):  # Each requested pair.
            if ma not in alloc or mb not in alloc:  # Skip pairs missing here.
                continue
            n_coal = len(set(alloc[ma]) | set(alloc[mb]))  # Coalitions involved.
            for metric in metric_list:  # One row per requested metric.
                row = {k: v for k, v in zip(group_cols, gvals)}  # Context.
                row.update(method_a=ma, method_b=mb, n_coalitions=n_coal,
                           metric=metric, distance=distance(metric, alloc[ma], alloc[mb]))
                rows.append(row)

    cols = group_cols + ["method_a", "method_b", "n_coalitions", "metric", "distance"]
    return pd.DataFrame(rows, columns=cols)  # Tidy long pairwise distances.
