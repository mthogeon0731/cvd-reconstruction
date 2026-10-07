"""Render the README figure from a completed synthetic study (no model loading)."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    results = args.run / "SYNTHETIC_results"
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["origin"] != "SYNTHETIC" or manifest["status"] != "PASS_SOFTWARE_PIPELINE":
        raise ValueError("A completed SYNTHETIC study is required")
    sources = ["evaluation/selected_oof.csv", "similitude_cross_fitted/points.csv"]
    oof, sc = [pd.read_csv(results / name, float_precision="round_trip") for name in sources]
    if not oof.origin.eq("SYNTHETIC").all() or oof.measurement_id.duplicated().any():
        raise ValueError("Expected unique synthetic held-out ROI predictions")
    metrics = json.loads((results / "evaluation/metrics.json").read_text(encoding="utf-8"))
    theory = json.loads((results / "similitude_cross_fitted/report.json").read_text(encoding="utf-8"))
    labels = json.loads((results / "similitude_fesem/report.json").read_text(encoding="utf-8"))
    sources += ["evaluation/metrics.json", "similitude_cross_fitted/report.json",
                "similitude_fesem/report.json"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#34495e", "text.color": "#19364b"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, oof.condition_id.nunique()))
    for color, (_, group) in zip(colors, oof.groupby("condition_id", sort=True)):
        axes[0].scatter(group.label_nm, group.pred_nm, s=20, color=color, alpha=0.75,
                        edgecolor="white", linewidth=0.25)
    lo = float(min(oof.label_nm.min(), oof.pred_nm.min()))
    hi = float(max(oof.label_nm.max(), oof.pred_nm.max()))
    axes[0].plot([lo, hi], [lo, hi], color="#8292a0", lw=1, ls="--", label="Identity")
    axes[0].set(xlabel="Synthetic ROI label (nm)", ylabel="Held-out prediction (nm)",
                title="A  |  Thickness at held-out conditions")
    axes[0].text(0.04, 0.95, f"{len(oof)} ROIs · {oof.condition_id.nunique()} conditions\n"
                 "Inner RF / KRR / SVR selection", transform=axes[0].transAxes,
                 va="top", fontsize=9)
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].scatter(sc.phi, sc.SC_measured, s=32, color="#26858b", alpha=0.85,
                    label="Cross-fitted synthetic SC")
    axes[1].scatter(sc.phi, sc.SC_model, s=30, marker="x", linewidth=1.2,
                    color="#bb773d", label="Fixed theory at same windows")
    axes[1].set(xlabel="Reaction–diffusion modulus φ", ylabel="Step coverage (bottom / top)",
                title="B  |  SC and the shared physical model")
    axes[1].legend(loc="lower left", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Synthetic example · seed 0" if manifest["seed"] == 0 else
                 f"Synthetic example · seed {manifest['seed']}", fontsize=16, fontweight="bold")
    fig.supxlabel("Common physical generator · software consistency only · no experimental accuracy claim",
                  fontsize=9, color="#536678")
    args.out.mkdir(parents=True, exist_ok=True)
    figure = args.out / "synthetic-results.png"
    fig.savefig(figure, dpi=150, metadata={"Software": "cvd-reconstruction scripts/render_example.py"})
    plt.close(fig)
    summary = {
        "origin": "SYNTHETIC", "seed": manifest["seed"], "config": manifest["config"],
        "environment": {key: manifest[key] for key in ["python", "OS", "versions"]},
        "input_root": "<RUN>/SYNTHETIC_inputs",
        "input_creation": "cvd_cbd.synthetic.generate via cvd_cbd study demo; no external observations",
        "n_rois": len(oof), "n_conditions": int(oof.condition_id.nunique()),
        "nested_selection": metrics["nested_selection"],
        "similitude_cross_fitted": theory,
        "similitude_synthetic_labels": labels,
        "sources_sha256": {"SYNTHETIC_results/" + name:
                           hashlib.sha256((results / name).read_bytes()).hexdigest()
                           for name in sources},
        "figure_sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
        "interpretation": "Shared synthetic physical generator; not experimental validation",
    }
    (args.out / "synthetic-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(oof)} ROIs in {oof.condition_id.nunique()} conditions; seed {manifest['seed']}")


if __name__ == "__main__":
    main()
