"""Noise-aware design stress test; historic headline numbers are not targets."""
import numpy as np
import pandas as pd


def noise_stress_test(clean_sc, repeats=4000, noise_sd_fraction=0.05,
                      seed=20261006, target_r2=0.95, noise_kind="additive"):
    """Gaussian noise is an explicit demo model, not an interpretation of ±15%.

    additive: SD=5 percentage points at default 0.05; relative: SD=0.05*SC.
    Measurements are not clipped (clipping biases R²). Fixed-theory R² has no
    fitted parameters. OLS R² fits slope/intercept on each same noisy sample,
    so is in-sample and optimistic; it is not held-out validation.
    """
    truth = np.asarray(clean_sc, float)
    if truth.ndim != 1 or len(truth) < 3 or np.any(~np.isfinite(truth)) or np.any((truth < 0) | (truth > 1)):
        raise ValueError("clean_sc must contain >=3 finite SC fractions in [0,1]")
    if np.ptp(truth) < 1e-12:
        raise ValueError("Design has no response span; regression stress test is undefined")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not np.isfinite(noise_sd_fraction) or noise_sd_fraction < 0:
        raise ValueError("Noise SD must be finite and nonnegative")
    if not np.isfinite(target_r2) or not 0 <= target_r2 <= 1:
        raise ValueError("target_r2 must be in [0,1]")
    if noise_kind not in {"additive", "relative"}:
        raise ValueError("noise_kind must be additive or relative")
    rng = np.random.default_rng(seed)
    sd = noise_sd_fraction if noise_kind == "additive" else noise_sd_fraction * truth
    observed = truth[None, :] + rng.normal(size=(repeats, len(truth))) * sd
    centered_x = truth - truth.mean()
    centered_y = observed - observed.mean(axis=1, keepdims=True)
    slopes = centered_y @ centered_x / (centered_x @ centered_x)
    fitted = observed.mean(axis=1, keepdims=True) + slopes[:, None] * centered_x
    sst = (centered_y ** 2).sum(axis=1)
    if np.any(sst <= np.finfo(float).tiny):
        raise ValueError("Degenerate simulated measurements; R² undefined")
    fixed_r2 = 1 - ((observed - truth) ** 2).sum(axis=1) / sst
    ols_r2 = 1 - ((observed - fitted) ** 2).sum(axis=1) / sst
    trials = pd.DataFrame({"trial": np.arange(1, repeats + 1), "fixed_theory_r2": fixed_r2,
                           "ols_in_sample_r2": ols_r2, "ols_slope": slopes})
    summary = {"status": "synthetic_design_stress_test_not_historical_reproduction",
               "repeats": repeats, "n_design_points": len(truth), "seed": seed,
               "noise_kind": noise_kind, "noise_sd_fraction": noise_sd_fraction,
               "noise_warning": "SD is a chosen assumption; historical ±15% scatter is not enough to infer this SD.",
               "clean_sc_span_percentage_points": float(np.ptp(truth) * 100),
               "outside_physical_measurement_fraction": float(np.mean((observed < 0) | (observed > 1))),
               "target_r2": target_r2}
    for name, values in (("fixed_theory", fixed_r2), ("ols_in_sample", ols_r2)):
        successes = int(np.sum(values >= target_r2))
        probability = successes / repeats
        # Wilson interval remains nonzero-width for 0/n or n/n successes.
        z = 1.959963984540054
        denom = 1 + z*z/repeats
        center = (probability + z*z/(2*repeats)) / denom
        halfwidth = z * np.sqrt(probability*(1-probability)/repeats + z*z/(4*repeats*repeats)) / denom
        summary[name] = {"r2_median": float(np.median(values)),
                         "r2_q05": float(np.quantile(values, 0.05)),
                         "r2_q95": float(np.quantile(values, 0.95)),
                         "target_success_count": successes,
                         "target_success_fraction": probability,
                         "target_probability_wilson95": [float(max(0, center-halfwidth)), float(min(1, center+halfwidth))],
                         "monte_carlo_standard_error": float(np.sqrt(probability * (1 - probability) / repeats))}
    return trials, summary
