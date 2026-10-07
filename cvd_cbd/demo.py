"""Reproducible synthetic smoke demo, with explicit provenance on every output."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
from .config import FEATURES
from .data import generate_synthetic, evaluate_conditions
from .surrogate import train_surrogate
from .design import select_screening_design, factorial_candidates
from .physics import analytic_profile, fdm_profile, coefficients, illustrative_da_at_sc
from .monte_carlo import noise_stress_test


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_demo(cfg, output, plots=True, monte_carlo_repeats=4000):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Choose a fresh --out to avoid overwriting work.")
    data = generate_synthetic(cfg)
    model, metrics, importance, predictions, split = train_surrogate(data, cfg)
    data["split"] = split
    design, candidate_count = select_screening_design(model, cfg, importance)
    # This 3x3x3 grid is an illustrative baseline; source does not define third factor.
    baseline = evaluate_conditions(factorial_candidates(cfg, levels=3), cfg)
    if np.ptp(baseline.sc_proxy.to_numpy()) < 1e-12:
        trials = pd.DataFrame(columns=["trial", "fixed_theory_r2", "ols_in_sample_r2", "ols_slope"])
        noise_summary = {"status": "not_applicable_constant_sc_design", "repeats": 0,
                         "reason": "No response span; R² stress test is undefined."}
    else:
        trials, noise_summary = noise_stress_test(baseline.sc_proxy, repeats=monte_carlo_repeats, seed=cfg.seed)
    data.to_csv(out / "synthetic_dataset.csv", index=False)
    predictions.to_csv(out / "surrogate_test_predictions.csv", index=False)
    importance.to_csv(out / "surrogate_validation_importance.csv", index=False)
    design.to_csv(out / "screened_design.csv", index=False)
    baseline.to_csv(out / "illustrative_factorial_baseline.csv", index=False)
    trials.to_csv(out / "monte_carlo_trials.csv", index=False)
    write_json(out / "monte_carlo_summary.json", noise_summary)
    write_json(out / "config_used.json", cfg.to_dict())
    write_json(out / "surrogate_metrics.json", metrics)
    joblib.dump({"model": model, "config": cfg.to_dict(), "sklearn_version": sklearn.__version__,
                 "status": "synthetic_uncalibrated"}, out / "synthetic_surrogate.joblib", compress=3)
    middle = np.mean(np.asarray(cfg.bounds), axis=1)
    c = coefficients(*middle, cfg)
    da = float(c["da"])
    bi = float(c["bottom_biot"])
    x, numeric = fdm_profile(da, bi, cfg.fdm_nodes)
    analytical = analytic_profile(x, da, bi)
    reactive_bi = float(c["surface_rate_m_s"] * c["depth_m"] / c["diffusivity_m2_s"])
    reactive = analytic_profile(x, da, reactive_bi)
    pd.DataFrame({"depth_fraction": x, "analytic_configured_end": analytical,
                  "fdm_configured_end": numeric, "analytic_equal_rate_reactive_bottom": reactive}).to_csv(
                      out / "reference_profiles.csv", index=False)
    errors = []
    for nodes in (51, 101, 201, 401):
        xx, uu = fdm_profile(da, bi, nodes)
        errors.append({"nodes": nodes, "max_abs_error": float(np.max(np.abs(uu - analytic_profile(xx, da, bi))))})
    manifest = {"status": "UNCALIBRATED SYNTHETIC DEMO; no original research data",
                "python": sys.version.split()[0], "platform": platform.system(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                             "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
                "seed": cfg.seed, "data_count": len(data), "doe_candidate_count": candidate_count,
                "doe_selected_count": len(design), "fdm_convergence": errors,
                "source_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(Path(__file__).parent.glob("*.py"))},
                "illustrative_sc_threshold": cfg.illustrative_sc_threshold,
                "no_flux_algebraic_da_threshold": float(illustrative_da_at_sc(cfg.illustrative_sc_threshold)),
                "threshold_warning": "Algebraic SC cutoff only; not empirically identified void threshold",
                "safe_model_loading": "Only load joblib from a trusted source and matching sklearn version"}
    write_json(out / "run_manifest.json", manifest)
    if plots:
        make_plots(out, cfg, predictions, importance, trials)
    return manifest


def make_plots(out, cfg, predictions, importance, trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.scatter(100 * predictions.sc_proxy, 100 * predictions.predicted_sc_proxy, s=7, alpha=.45)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set(xlabel="Analytic simulator SC proxy (%)", ylabel="RF proxy prediction (%)", title="Synthetic hold-out only")
    axes[1].bar(importance.feature, importance.validation_mae_increase_mean * 100,
                yerr=importance.validation_mae_increase_std * 100, color="#277c8e")
    axes[1].set(ylabel="Validation MAE increase (percentage points)", title="Permutation sensitivity")
    axes[1].tick_params(axis="x", rotation=20)
    axes[2].hist(trials.ols_in_sample_r2, bins=40, alpha=.7, label="OLS in-sample")
    axes[2].hist(trials.fixed_theory_r2, bins=40, alpha=.5, label="Fixed theory")
    axes[2].set(xlabel="R² under assumed noise", ylabel="Trial count", title="Monte Carlo design stress test")
    axes[2].legend(fontsize=8)
    fig.suptitle("UNCALIBRATED RECONSTRUCTION DEMO | no historical metrics reproduced")
    fig.savefig(out / "synthetic_demo.png", dpi=160)
    plt.close(fig)
    temps = np.linspace(*cfg.temperature_c_range, 81)
    viscosities = np.linspace(*cfg.viscosity_mpa_s_range, 81)
    tt, vv = np.meshgrid(temps, viscosities)
    frame = pd.DataFrame({"aspect_ratio": cfg.aspect_ratio_range[1], "viscosity_mpa_s": vv.ravel(), "temperature_c": tt.ravel()})
    sc = evaluate_conditions(frame, cfg).sc_proxy.to_numpy().reshape(tt.shape)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    cs = ax.contourf(tt, vv, 100 * sc, levels=np.linspace(0, 100, 21), cmap="viridis", vmin=0, vmax=100)
    if sc.min() < cfg.illustrative_sc_threshold < sc.max():
        line = ax.contour(tt, vv, sc, levels=[cfg.illustrative_sc_threshold], colors="white", linewidths=2)
        ax.clabel(line, fmt={cfg.illustrative_sc_threshold: "Illustrative SC cutoff"}, fontsize=8)
    ax.set(xlabel="Temperature (°C)", ylabel="Viscosity at this temperature (mPa s)",
           title=f"Uncalibrated SC proxy map | AR={cfg.aspect_ratio_range[1]:g}\nNot a validated process or void window")
    fig.colorbar(cs, ax=ax, label="SC proxy (%)")
    fig.savefig(out / "illustrative_process_map.png", dpi=160)
    plt.close(fig)
