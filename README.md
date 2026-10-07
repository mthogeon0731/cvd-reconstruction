# CVD/CBD reconstruction 0.2.2

[한국어](README.ko.md)

A source-guided reimplementation of a trench coating analysis workflow. This is a research software prototype with synthetic software validation. It is **not recovered laboratory code or a reproduction of measured experimental performance**. No real research observations, microscope images, pretrained models or historical output bundles are distributed.

The pipeline combines a steady 1-D reaction–diffusion model, independent planar-growth/viscosity calibration, registered TIFF metrology with REF/DARK correction, ROI features, nested condition-level RF/KRR/SVR evaluation, separately fitted final inference, bot/top step coverage (SC), fixed-theory comparison, uncertainty propagation and model-dependent aspect-ratio limits. See [model and limits](docs/MODEL.md).

## Install

Run from the repository root. Python >=3.10 is the declared package minimum; only the environment actually listed in [validation](docs/VALIDATION.md) has been checked for this publication copy. The pinned dependencies may impose a newer minimum and are not a promise of support for every Python version.

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install -r requirements-build-lock.txt
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
python -m pip check
```

## Minimal example and tests

```sh
python examples/minimal.py
python -m pytest tests -q
python scripts/check_repository.py
```

The product tests generate their synthetic inputs and models in temporary directories. No old audit folder or output fixture is required. Historical audit suites are a separate scope; see [external audits](docs/EXTERNAL_AUDITS.md). Large generated files are deliberately absent from Git.

## Synthetic data and full demo

```sh
python -m cvd_cbd study generate --config configs/synthetic_study.json --root outputs/generated_inputs
python -m cvd_cbd study validate --config configs/synthetic_study.json --root outputs/generated_inputs
python -m cvd_cbd study demo --config configs/synthetic_study.json --out outputs/demo
python verification/verify_run.py --run outputs/demo --out outputs/demo_verification.json
```

The demo creates its own inputs under `outputs/demo/SYNTHETIC_inputs` and results under `outputs/demo/SYNTHETIC_results`. Use a new output directory each time. [The 14-stage CLI procedure](docs/CLI.md) processes the separately generated inputs. `cvd-study` is the installed equivalent of `python -m cvd_cbd study`.

The generator creates 16-bit TIFFs from arrays, analytic property tables, synthetic region labels and explicitly artificial calibration evidence. Its OD = thickness / 450 law and shared reaction–diffusion physics are chosen assumptions. Historical synthetic R² around **0.9967** and similitude R² around **0.99989** test software consistency; they do not measure experimental accuracy or independently validate the physics.

## Real data and interpretation

The header-only CSVs in [templates](templates) describe the input schema; [input contracts](docs/INPUT_CONTRACTS.md) describe validation. `configs/real_study_template.json` contains unverified placeholders, including a window that must be corrected using measured observation locations. It is not a runnable real-data example.

Real OM/FE-SEM measurements, scale calibration, acquisition metadata, matched labels, independent splits, planar growth and viscosity are still required. A file hash proves byte identity, not scientific authenticity. Final-model training predictions are not independent evaluation. Only load joblib artifacts you trust: deserialization can execute code, and version changes require retraining.

## Review records and licensing

- [Current validation and exact commands](docs/VALIDATION.md)
- [Historical results and audit limitations](docs/EXTERNAL_AUDITS.md)
- [Public provenance, file scope and checks](docs/PROVENANCE.md)
- [Changes](CHANGELOG.md)
- [Rights, dependencies and license decision](docs/RIGHTS.md)

No project license has been selected. This review copy remains subject to the owner's separate publication decision. No GitHub Actions success or other unexecuted environment is claimed.
