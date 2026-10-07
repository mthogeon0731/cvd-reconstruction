"""Synthetic simulator data. No historical measurements are embedded."""
import numpy as np
import pandas as pd
from scipy.stats import qmc
from .config import FEATURES, TARGET
from .physics import coefficients, analytic_profile


def validate_features(frame, cfg, require_domain=True):
    missing = [c for c in FEATURES if c not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(frame) == 0:
        raise ValueError("At least one input row is required")
    a = frame[FEATURES].to_numpy(dtype=float)
    if np.any(~np.isfinite(a)):
        raise ValueError("Feature values must be numeric and finite")
    if np.any(a[:, :2] <= 0) or np.any(a[:, 2] <= -273.15):
        raise ValueError("Nonphysical feature input")
    if require_domain:
        lo, hi = np.asarray(cfg.bounds).T
        if np.any((a < lo) | (a > hi)):
            raise ValueError("Out-of-domain prediction refused. Change config and retrain; RF cannot extrapolate.")
    return frame[FEATURES].astype(float)


def evaluate_conditions(frame, cfg):
    x = validate_features(frame, cfg, require_domain=False)
    c = coefficients(*(x[k].to_numpy() for k in FEATURES), cfg)
    result = x.copy()
    for key, value in c.items():
        result[key] = value
    result[TARGET] = analytic_profile(1.0, c["da"], c["bottom_biot"])
    result["sc_proxy_percent"] = 100 * result[TARGET]
    result["data_origin"] = "synthetic_uncalibrated"
    return result


def generate_synthetic(cfg):
    unit = qmc.LatinHypercube(d=len(FEATURES), seed=cfg.seed).random(cfg.n_samples)
    lo, hi = np.asarray(cfg.bounds).T
    x = pd.DataFrame(qmc.scale(unit, lo, hi), columns=FEATURES)
    result = evaluate_conditions(x, cfg)
    result.insert(0, "sample_id", [f"synthetic_{i:05d}" for i in range(len(result))])
    return result
