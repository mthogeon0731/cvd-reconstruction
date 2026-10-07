"""Optional early-proposal RF surrogate. This is NOT the image-thickness regressor."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .config import FEATURES, TARGET
from .data import validate_features


def regression_metrics(y, prediction):
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    return {"r2": float(r2_score(y, p)) if len(y) >= 2 and np.var(y) > 0 else None,
            "mae_fraction": float(mean_absolute_error(y, p)),
            "rmse_fraction": float(np.sqrt(mean_squared_error(y, p))),
            "max_abs_error_fraction": float(np.max(np.abs(y - p)))}


def train_surrogate(data, cfg):
    x = validate_features(data, cfg)
    y = data[TARGET].astype(float)
    if np.any(~np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("SC target must be finite in [0,1]")
    train_val, test = train_test_split(np.arange(len(data)), test_size=0.2, random_state=cfg.seed)
    train, validation = train_test_split(train_val, test_size=0.25, random_state=cfg.seed + 1)
    model = RandomForestRegressor(n_estimators=cfg.n_estimators, min_samples_leaf=cfg.min_samples_leaf,
                                  random_state=cfg.seed, n_jobs=1)
    model.fit(x.iloc[train], y.iloc[train])
    importance = permutation_importance(model, x.iloc[validation], y.iloc[validation],
                                       scoring="neg_mean_absolute_error", n_repeats=10,
                                       random_state=cfg.seed + 2, n_jobs=1)
    importance_df = pd.DataFrame({"feature": FEATURES, "validation_mae_increase_mean": importance.importances_mean,
                                  "validation_mae_increase_std": importance.importances_std})
    predictions = data.iloc[test][["sample_id", *FEATURES, TARGET]].copy()
    predictions["predicted_sc_proxy"] = model.predict(x.iloc[test])
    split = np.full(len(data), "train", dtype=object)
    split[validation], split[test] = "validation", "test"
    metrics = {"interpretation": "Synthetic simulator approximation only; NOT experimental validation.",
               "train_count": len(train), "validation_count": len(validation), "test_count": len(test),
               "train": regression_metrics(y.iloc[train], model.predict(x.iloc[train])),
               "test": regression_metrics(y.iloc[test], predictions["predicted_sc_proxy"]),
               "target_units": "fraction, 0..1"}
    return model, metrics, importance_df, predictions, split


def predict_surrogate(model, frame, cfg):
    return model.predict(validate_features(frame, cfg, require_domain=True))
