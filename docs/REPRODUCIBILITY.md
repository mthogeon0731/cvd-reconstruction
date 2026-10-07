# Reproducing the synthetic study

Use the [Quickstart](../README.md#quickstart) with a new virtual environment and the committed build/runtime pins. The tested setup is Windows 11 build 26200 AMD64, bundled CPython 3.12.14. Python ≥3.10 is package metadata, not a verified matrix for the pinned dependency versions. No CI or additional OS/Python combination was run in this refresh.

## Inputs and commands

The [synthetic configuration](../configs/synthetic_study.json) records seed 0. The study creates all needed inputs under `SYNTHETIC_inputs` in the selected output directory. For separate input inspection:

```sh
python -m cvd_cbd study generate --config configs/synthetic_study.json --root outputs/generated_inputs
python -m cvd_cbd study validate --config configs/synthetic_study.json --root outputs/generated_inputs
```

Then use the [14-stage CLI procedure](CLI.md) on those inputs, or the [single study demo and figure commands](EXAMPLE.md#rebuild), which generate their own inputs. Do not reuse an existing run output directory. Basic product tests create independent temporary inputs:

```sh
python -m pytest tests -q
python scripts/check_repository.py
```

## Controlled environment

The verification runner clears `PYTHONPATH` and `PYTHONHOME`, sets `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUTF8=1`, `MPLBACKEND=Agg`, and sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` to `1`. Matplotlib cache, logs and temporary files are kept outside the checkout. A normal wheel installation and isolated import outside the repository check that the installed package is usable without the source directory on the import path.

For an exact figure rebuild, use the recorded Matplotlib, fonts and environment as well as the random seed. The renderer reads CSV with round-trip float parsing. It uses DejaVu Sans, fixed dimensions/colors and a static metadata field, with no generation timestamp. Re-rendering the same run checks figure-byte determinism only; it is not a second full study and must not be counted as one.

## What equality means

- Source CSV hashes identify exact files from the illustrated run. They do not authenticate scientific measurements.
- Matching metrics or pixels in the same setup does not demonstrate OS-independent output. Library versions, fonts, native math kernels and float serialization can change bytes or numerical results.
- Historical repetition found 360/360 output files identical, and a separate before/after comparison found 31/31 CSVs identical. Historical Windows/Linux comparison found 0/31 byte matches and prediction differences up to about 0.032 nm. These are earlier checks, not newly executed claims.

Use [run verification and comparison tools](../verification/README.md) to state the scope of any new comparison. The [historical audit procedure](EXTERNAL_AUDITS.md) requires separately authorized archives; those archives are not required for the repository's 360 product tests. The current commands and outcomes are recorded in [validation](VALIDATION.md).
