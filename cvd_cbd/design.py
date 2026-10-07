"""Explicitly new DoE heuristics; not recovered historical selection logic."""
from itertools import product
import numpy as np
import pandas as pd
from .config import FEATURES
from .data import evaluate_conditions
from .surrogate import predict_surrogate


def factorial_candidates(cfg, levels=None):
    n = levels or cfg.doe_grid_levels_per_other_factor
    rows = product(np.linspace(*cfg.aspect_ratio_range, n),
                   np.linspace(*cfg.viscosity_mpa_s_range, n), cfg.doe_temperature_levels_c)
    return pd.DataFrame(rows, columns=FEATURES)


def select_screening_design(model, cfg, importance):
    """Deterministic sensitivity-weighted farthest-point selection.

    Validation permutation importance sets feature-space weights (floor=10%).
    A response coordinate balances response coverage. This is exploratory,
    NOT statistically optimal, not D-optimal, and not replication/power analysis.
    """
    candidates = factorial_candidates(cfg)
    p = predict_surrogate(model, candidates, cfg)
    lo, hi = np.asarray(cfg.bounds).T
    scaled = (candidates.to_numpy() - lo) / (hi - lo)
    imp = importance.set_index("feature").loc[FEATURES, "validation_mae_increase_mean"].to_numpy()
    weights = np.maximum(imp, 0)
    weights = weights / weights.sum() if weights.sum() else np.full(3, 1 / 3)
    weights = 0.1 + 0.7 * weights
    span = np.ptp(p)
    response = (p - p.min()) / span if span > 0 else np.zeros(len(p))
    coordinates = np.column_stack([scaled * np.sqrt(weights), 0.5 * response])
    first = int(np.argmin(np.sum((scaled - 0.5) ** 2, axis=1)))
    selected = [first]
    distance = np.full(len(candidates), np.inf)
    while len(selected) < cfg.doe_runs:
        new_distance = np.sum((coordinates - coordinates[selected[-1]]) ** 2, axis=1)
        distance = np.minimum(distance, new_distance)
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    result = evaluate_conditions(candidates.iloc[selected].reset_index(drop=True), cfg)
    result["surrogate_sc_proxy"] = p[selected]
    rng = np.random.default_rng(cfg.seed + 7)
    result.insert(0, "suggested_run_order", rng.permutation(np.arange(1, len(result) + 1)))
    result.insert(0, "selection_index", np.arange(1, len(result) + 1))
    result["selection_status"] = "new_heuristic_not_historical_or_validated"
    return result.sort_values("suggested_run_order"), len(candidates)
