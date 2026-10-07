"""Condition-held-out thickness regression scaffold; no historical fit is claimed."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .imaging import FEATURE_COLUMNS, MATCH_BASES
from .input_contracts import validate_pixel_scale,scalar_pixel_scale,reject_axis_pixel_calibration
from .io_utils import read_csv,validate_columns


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1729
    test_fraction: float = 0.25
    cv_splits: int = 3
    equal_condition_weight: bool = True
    allow_synthetic: bool = False


def condition_weights(groups: Sequence[str]) -> np.ndarray:
    """Weights sum to sample count, with equal total mass for each condition."""
    group = pd.Series(np.asarray(groups))
    weights = 1.0 / group.map(group.value_counts()).to_numpy(dtype=float)
    return weights * len(weights) / weights.sum()


def _metrics(y: np.ndarray, prediction: np.ndarray, groups: Sequence[str], weighted: bool) -> dict:
    # Lazy import avoids the modeling -> condition_weights dependency at import.
    # Share the vector/group contract while retaining legacy weighted estimands.
    from .modeling import metrics
    metrics(y, prediction, groups)
    y, prediction = np.asarray(y, float).reshape(-1), np.asarray(prediction, float).reshape(-1)
    if not len(y):
        return {'rmse_nm': None, 'mae_nm': None, 'r2': None, 'mean_absolute_percentage_error': None}
    weights = condition_weights(groups) if weighted else None
    r2 = float(r2_score(y, prediction, sample_weight=weights)) if len(y) >= 2 and np.ptp(y) > 0 else None
    relative_valid = np.abs(y) >= 1e-6
    return {
        "rmse_nm": float(np.sqrt(mean_squared_error(y, prediction, sample_weight=weights))),
        "mae_nm": float(mean_absolute_error(y, prediction, sample_weight=weights)),
        "r2": r2,
        "mean_absolute_percentage_error": float(np.average(np.abs((prediction[relative_valid]-y[relative_valid])/y[relative_valid]), weights=weights[relative_valid] if weights is not None else None) * 100) if relative_valid.any() else None,
    }


def _models(seed: int) -> dict:
    return {
        "KRR": (
            Pipeline([("scale", StandardScaler()), ("model", KernelRidge(kernel="rbf"))]),
            {"model__alpha": [0.1, 1.0, 10.0], "model__gamma": [0.01, 0.1]},
        ),
        "RF": (
            Pipeline([("model", RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1))]),
            {"model__min_samples_leaf": [1, 3], "model__max_depth": [None, 8]},
        ),
        "SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {"model__C": [10.0, 100.0], "model__epsilon": [0.1, 1.0], "model__gamma": ["scale", 0.01]},
        ),
    }


def _fit(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray, weighted: bool) -> Pipeline:
    fit_args = {}
    if weighted:
        weight = condition_weights(groups)
        fit_args["model__sample_weight"] = weight
        if "scale" in pipeline.named_steps:
            fit_args["scale__sample_weight"] = weight
    return pipeline.fit(X, y, **fit_args)


def _read_table(features: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        validate_columns(features, 'legacy feature table')
        return features.copy()
    return read_csv(features, dtype={c: str for c in ("sample_id", "image_id", "condition_id", "measurement_id")})


def _validate_table(data: pd.DataFrame, columns: Sequence[str], allow_synthetic: bool) -> None:
    validate_columns(data, 'legacy feature table')
    reject_axis_pixel_calibration(data.columns)
    required = {"sample_id", "condition_id", "image_id", "measurement_id", "match_basis", "data_origin", "pixel_size_um", "thickness_nm", *columns}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing feature-table columns: {sorted(missing)}")
    if data.empty or not columns:
        raise ValueError("Training requires measured rows and explicit feature_ columns")
    data['pixel_size_um'] = validate_pixel_scale(data.pixel_size_um, 'pixel_size_um')
    if len(columns) != len(set(columns)) or any(not c.startswith("feature_") for c in columns):
        raise ValueError("Use unique feature_ columns; identifiers and targets cannot be predictors")
    for name in ("sample_id", "condition_id", "image_id", "measurement_id", "match_basis", "data_origin"):
        if data[name].isna().any() or data[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {name}; unmatched or missing labels cannot be trained")
        data[name] = data[name].astype(str)
    if data.sample_id.duplicated().any() or data.measurement_id.duplicated().any():
        raise ValueError("sample_id and measurement_id must each be unique; do not duplicate FE-SEM labels across tiles")
    if data.groupby("image_id").condition_id.nunique().max() != 1:
        raise ValueError("The same original image cannot cross condition groups")
    if not data.match_basis.isin(MATCH_BASES).all():
        raise ValueError(f"match_basis must be one of {sorted(MATCH_BASES)}")
    if not data.data_origin.isin(["real_measurement", "synthetic_demo"]).all():
        raise ValueError("data_origin must be real_measurement or synthetic_demo")
    if data.data_origin.nunique() != 1:
        raise ValueError("Never mix real measurements with a synthetic demo for validation")
    if data.data_origin.eq("synthetic_demo").any() and not allow_synthetic:
        raise ValueError("Synthetic demo data require allow_synthetic=True and do not validate microscopy")
    try:
        values = data[[*columns, "thickness_nm", "pixel_size_um"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features, calibration and thickness must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Missing/non-finite features, labels or calibration; no silent imputation")
    if (values[:, -2:] <= 0).any():
        raise ValueError("thickness_nm and pixel_size_um must be positive")
    if not np.allclose(values[:, -1], values[0, -1], rtol=0.01, atol=0):
        raise ValueError("Pixel-scale features require matched calibration within 1%; resample/calibrate upstream")


def fit_grouped_thickness(features: pd.DataFrame | str | Path, output_dir: str | Path,
                          feature_columns: Sequence[str] | None = None,
                          config: TrainingConfig | None = None,
                          test_conditions: Sequence[str] | None = None) -> dict:
    """Select KRR/RF/SVR using TRAIN-only grouped CV; evaluate winner once on test.

    Both scalers and models are fitted inside each fold. All ROIs from an original
    condition remain together. Artifacts retain the split and the model trained
    only on training conditions; no silent all-data refit occurs.
    """
    config = config or TrainingConfig()
    if not 0 < config.test_fraction < 1 or config.cv_splits < 2:
        raise ValueError("Require 0 < test_fraction < 1 and cv_splits >= 2")
    data = _read_table(features).reset_index(drop=True)
    columns = list(feature_columns) if feature_columns is not None else [c for c in data if c.startswith("feature_")]
    _validate_table(data, columns, config.allow_synthetic)
    groups = data.condition_id.to_numpy()
    unique = np.unique(groups)
    if len(unique) < config.cv_splits + 2:
        raise ValueError(f"Need at least {config.cv_splits+2} distinct conditions: inner CV plus >=2 held-out conditions")
    X, y = data[columns].to_numpy(dtype=float), data.thickness_nm.to_numpy(dtype=float)
    if test_conditions is None:
        n_test = max(2, math.ceil(config.test_fraction * len(unique)))
        if len(unique)-n_test < config.cv_splits:
            raise ValueError("test_fraction leaves too few training conditions for grouped CV")
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=n_test, random_state=config.seed).split(X, y, groups))
    else:
        requested = set(map(str, test_conditions))
        if len(requested) < 2 or not requested.issubset(set(unique)):
            raise ValueError("test_conditions must name >=2 existing conditions")
        test_idx = np.flatnonzero(np.isin(groups, list(requested)))
        train_idx = np.flatnonzero(~np.isin(groups, list(requested)))
        if len(np.unique(groups[train_idx])) < config.cv_splits:
            raise ValueError("Too few remaining training conditions for grouped CV")
    target = Path(output_dir)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"Training output must be a new or empty directory: {target}")
    train_groups, test_groups = groups[train_idx], groups[test_idx]
    if set(train_groups).intersection(test_groups):
        raise RuntimeError("Unexpected condition leakage")
    Xt, yt = X[train_idx], y[train_idx]
    folds = list(GroupKFold(n_splits=config.cv_splits).split(Xt, yt, train_groups))
    fold_labels = np.full(len(train_idx), -1)
    for fold_id, (fit_idx, val_idx) in enumerate(folds):
        if set(train_groups[fit_idx]).intersection(train_groups[val_idx]):
            raise RuntimeError("Unexpected inner-CV condition leakage")
        fold_labels[val_idx] = fold_id
    candidates, best = [], None
    for name, (base, grid) in _models(config.seed).items():
        for candidate_id, parameters in enumerate(ParameterGrid(grid)):
            oof = np.full(len(yt), np.nan)
            for fit_idx, val_idx in folds:
                pipeline = clone(base).set_params(**parameters)
                _fit(pipeline, Xt[fit_idx], yt[fit_idx], train_groups[fit_idx], config.equal_condition_weight)
                oof[val_idx] = pipeline.predict(Xt[val_idx])
            if not np.isfinite(oof).all():
                raise ValueError(f"{name} produced non-finite CV predictions")
            score = _metrics(yt, oof, train_groups, config.equal_condition_weight)
            row = {"model": name, "candidate_id": candidate_id, "parameters": json.dumps(parameters, sort_keys=True), **{f"cv_{k}": v for k, v in score.items()}}
            candidates.append(row)
            if best is None or score["rmse_nm"] < best["score"]:
                best = {"name": name, "parameters": parameters, "base": base, "score": score["rmse_nm"], "oof": oof.copy()}
    selected = _fit(clone(best["base"]).set_params(**best["parameters"]), Xt, yt, train_groups, config.equal_condition_weight)
    prediction = selected.predict(X[test_idx])
    if not np.isfinite(prediction).all():
        raise ValueError("Selected model produced non-finite held-out predictions")
    origin = str(data.data_origin.iloc[0])
    report = {
        "data_origin": origin,
        "scientific_validation": "NOT VALIDATION: synthetic demonstration" if origin == "synthetic_demo" else "Exploratory held-out condition evaluation; verify FE-SEM registration and acquisition confounders",
        "selected_model": best["name"], "selected_parameters": best["parameters"],
        "selection_metric": "condition-balanced grouped-CV RMSE (nm)" if config.equal_condition_weight else "row-weighted grouped-CV RMSE (nm)",
        "selected_cv_rmse_nm": best["score"],
        "test_condition_balanced": _metrics(y[test_idx], prediction, test_groups, True),
        "test_row_weighted": _metrics(y[test_idx], prediction, test_groups, False),
        "n_train_samples": len(train_idx), "n_test_samples": len(test_idx),
        "train_conditions": sorted(set(train_groups)), "test_conditions": sorted(set(test_groups)),
        "no_condition_leakage": True, "model_fitted_on": "training conditions only",
        "config": asdict(config), "feature_columns": columns,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "limitations": ["Feature proxies are not a calibrated optical thickness law", "CV selection scores are optimistic; held-out data are used only once", "No historical R-squared or error-reduction claim has been reproduced", "No calibrated prediction interval is provided"],
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    split = data[["sample_id", "image_id", "condition_id", "measurement_id"]].copy()
    split["split"], split["cv_validation_fold"] = "test", -1
    split.loc[train_idx, "split"] = "train"
    split.loc[train_idx, "cv_validation_fold"] = fold_labels
    split.to_csv(target / "split_assignments.csv", index=False)
    pd.DataFrame(candidates).sort_values("cv_rmse_nm").to_csv(target / "cv_results.csv", index=False)
    predictions = data.iloc[test_idx][["sample_id", "image_id", "condition_id", "measurement_id", "thickness_nm"]].copy()
    predictions["predicted_thickness_nm"] = prediction
    predictions["error_nm"] = prediction - y[test_idx]
    predictions["absolute_percentage_error"] = np.abs(prediction-y[test_idx])/y[test_idx] * 100
    predictions["negative_unphysical_prediction"] = prediction < 0
    predictions.to_csv(target / "test_predictions.csv", index=False)
    oof = data.iloc[train_idx][["sample_id", "condition_id", "thickness_nm"]].copy()
    oof["cv_validation_fold"], oof["predicted_thickness_nm"] = fold_labels, best["oof"]
    oof.to_csv(target / "selected_cv_predictions.csv", index=False)
    bundle = {
        "pipeline": selected, "feature_columns": columns, "report": report,
        "feature_min": Xt.min(axis=0), "feature_max": Xt.max(axis=0),
        "pixel_size_um": float(data.pixel_size_um.iloc[0]),
    }
    joblib.dump(bundle, target / "selected_model.joblib")
    (target / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (target / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    return report


def predict_thickness(model_path: str | Path, features: pd.DataFrame | str | Path,
                      output_csv: str | Path | None = None, *,
                      trust_model: bool = False) -> pd.DataFrame:
    """Predict from an explicitly trusted joblib artifact; flags extrapolation.

    Set trust_model=True only for artifacts you created or independently trust.
    Joblib/pickle can execute arbitrary code during loading.
    """
    if not trust_model:
        raise ValueError("Joblib can execute arbitrary code. Only load a trusted model: set trust_model=True or CLI --trust-model")
    if output_csv is not None:
        output = Path(output_csv).resolve()
        inputs = [Path(model_path).resolve()]
        if not isinstance(features, pd.DataFrame):
            inputs.append(Path(features).resolve())
        if output in inputs:
            raise ValueError("Prediction output must not overwrite a model or input feature file")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    bundle = joblib.load(model_path)  # Pickle-based: never load an untrusted file.
    reject_axis_pixel_calibration(bundle, 'saved model pixel calibration')
    model_scale = scalar_pixel_scale(bundle['pixel_size_um'], 'saved model pixel_size_um')
    data = _read_table(features)
    reject_axis_pixel_calibration(data.columns)
    required = {"sample_id", "pixel_size_um", *bundle["feature_columns"]}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError(f"Prediction table requires {sorted(required)} and at least one row")
    X = data[bundle["feature_columns"]].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Prediction features must be finite")
    scales = validate_pixel_scale(data.pixel_size_um, 'pixel_size_um')
    if not np.allclose(scales, model_scale, rtol=0.01, atol=0):
        raise ValueError("Prediction pixel calibration differs from training")
    result = data[[c for c in ("sample_id", "image_id", "condition_id") if c in data]].copy()
    prediction = bundle["pipeline"].predict(X)
    result["predicted_thickness_nm"] = prediction
    result["outside_training_feature_range"] = ((X < bundle["feature_min"]) | (X > bundle["feature_max"])).any(axis=1)
    result["negative_unphysical_prediction"] = prediction < 0
    result["model_data_origin"] = bundle["report"]["data_origin"]
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


def synthetic_feature_fixture(n_conditions: int = 12, regions_per_condition: int = 2,
                              seed: int = 1729) -> pd.DataFrame:
    """Artificial feature/label fixture, NOT microscopy or reconstructed FE-SEM."""
    if n_conditions < 1 or regions_per_condition < 1:
        raise ValueError("Condition and region counts must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for condition in range(n_conditions):
        base = rng.uniform(0.1, 0.9)
        for region in range(regions_per_condition):
            values = np.clip(base + rng.normal(0, 0.04, len(FEATURE_COLUMNS)), 0, 1)
            rows.append({
                "sample_id": f"synthetic-c{condition:03d}-r{region}",
                "image_id": f"synthetic-image-{condition:03d}", "condition_id": f"synthetic-c{condition:03d}",
                "measurement_id": f"SYNTHETIC-NOT-FESEM-{condition:03d}-{region}",
                "match_basis": "registered_roi", "data_origin": "synthetic_demo", "pixel_size_um": 0.5,
                "thickness_nm": 20 + 100 * values[0] + rng.normal(0, 1),
                **dict(zip(FEATURE_COLUMNS, values)),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("features_csv")
    fit.add_argument("output_dir")
    fit.add_argument("--seed", type=int, default=1729)
    fit.add_argument("--cv-splits", type=int, default=3)
    demo = sub.add_parser("demo")
    demo.add_argument("output_dir")
    predict = sub.add_parser("predict")
    predict.add_argument("model_path")
    predict.add_argument("features_csv")
    predict.add_argument("output_csv")
    predict.add_argument("--trust-model", action="store_true", help="Confirm that this pickle-based model comes from a trusted source; loading can execute code")
    args = parser.parse_args(argv)
    if args.command == "predict":
        output = predict_thickness(args.model_path, args.features_csv, args.output_csv, trust_model=args.trust_model)
        print(f"Predicted {len(output)} matched regions; inspect out-of-range flags")
    elif args.command == "demo":
        folder = Path(args.output_dir)
        data = synthetic_feature_fixture()
        report = fit_grouped_thickness(data, folder, config=TrainingConfig(allow_synthetic=True))
        data.to_csv(folder / "SYNTHETIC_features.csv", index=False)
        print(f"SYNTHETIC DEMO ONLY: {report['selected_model']} selected. No experimental validation.")
    else:
        report = fit_grouped_thickness(args.features_csv, args.output_dir, config=TrainingConfig(seed=args.seed, cv_splits=args.cv_splits))
        print(json.dumps({"selected_model": report["selected_model"], "held_out": report["test_condition_balanced"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
