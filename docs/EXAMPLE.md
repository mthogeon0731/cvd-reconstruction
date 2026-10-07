# Synthetic example and figure provenance

The representative figure is generated from an actual seed-0 study, not a hand-drawn performance curve. [The figure](assets/synthetic-results.png) is labeled **Synthetic example** and contains no external image. The [pipeline SVG](assets/pipeline.svg) is an authored diagram of the code paths described in [MODEL.md](MODEL.md).

## Rebuild

Follow the [installation](../README.md#quickstart), then run from the repository root:

```sh
python -m cvd_cbd study demo --config configs/synthetic_study.json --out outputs/portfolio_demo
python verification/verify_run.py --run outputs/portfolio_demo --out outputs/portfolio_demo_verification.json
python scripts/render_example.py --run outputs/portfolio_demo --out docs/assets
```

The study requires a new output directory. Inputs are created under `outputs/portfolio_demo/SYNTHETIC_inputs` by `cvd_cbd.synthetic.generate`, called from the study CLI. The generator constructs TIFF arrays, tables, labels and artificial scale evidence from formulas and a seeded RNG; it does not load a research photo or external dataset. The committed [configuration](../configs/synthetic_study.json) uses seed 0, six conditions and AR ladder 9, 13, 18, 25, 35, 49. Image inputs include `OD=thickness_nm/450`, camera noise SD 20 counts and label noise SD 0.4 nm.

## What is plotted

- Panel A reads `SYNTHETIC_results/evaluation/selected_oof.csv`: 144 unique labeled trench ROIs, six conditions. Every plotted prediction comes from an outer held-out condition, with RF/KRR/SVR selection inside its training folds. The dashed line is identity, not a fitted regression.
- Panel B reads `SYNTHETIC_results/similitude_cross_fitted/points.csv`: 36 condition/AR points from 72 paired trenches. The source column `SC_measured` holds **synthetic cross-fitted predicted SC** in this path; the legend explicitly avoids implying real measurements. The paired theoretical values use the same actual spatial observation support. No theoretical parameter is fitted to these SC values.
- The all-label final model is not used to plot evaluation. The separate `similitude_fesem/report.json` uses generated stand-in labels and is reported only as a different comparison below.

## Recorded values

| Fresh seed-0 output | Value |
|---|---:|
| Nested thickness R² | 0.9967539337015857 |
| Nested thickness RMSE (nm) | 2.657989667120674 |
| Cross-fitted SC / fixed theory R² | 0.9987861679926912 |
| Cross-fitted SC RMSE (fraction) | 0.010043235013578644 |
| Synthetic label SC / fixed theory R² | 0.9998866852151083 |
| Synthetic label SC RMSE (fraction) | 0.0030678354292110505 |

The historical ≈0.99989 refers to the synthetic label-derived comparison, not the cross-fitted ML SC result. Neither establishes experimental accuracy. Both paths share the generator's physics. No real OM or FE-SEM check was run.

[synthetic-summary.json](assets/synthetic-summary.json) is regenerated alongside the PNG. It includes settings, software versions, source hashes and figure hash, without copying raw runtime provenance or personal paths. It is a compact record, not a substitute for rerunning the inputs. Only this small PNG, authored SVGs and summary are tracked; generated TIFFs, fold/final models, intermediate tables and logs are ignored. PNG metadata contains the public generator name and resolution; no EXIF, owner contact or local path is embedded.
