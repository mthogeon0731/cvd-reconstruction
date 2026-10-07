# Publication validation — 2026-10-07

This report is for the 0.2.2 publication copy. The pre-push clean-copy verification below completed successfully. The remote fresh-clone result is delivered separately with the exact commit SHA after pushing; it is not pre-claimed by this pre-push report. Historical counts are in [external audits](EXTERNAL_AUDITS.md).

Environment under test: Windows 11 build 26200, AMD64, bundled CPython 3.12.14. Fresh virtual environments use no system site packages. Runtime/build pins and their licenses are listed in [dependency inventory](DEPENDENCIES.md). A separate Python installer distribution, Linux, macOS, other Python versions, GitHub Actions and actual experimental performance are NOT RUN for this preparation.

The source is copied using an explicit file allowlist into an otherwise empty validation folder. Old output fixtures, virtual environments and audit folders are absent. Installation is a regular wheel installation, not editable. An isolated Python process outside the checkout checks that imports resolve to the new environment's installed package. `PYTHONPATH`/`PYTHONHOME` are cleared; `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUTF8=1`, `MPLBACKEND=Agg` and one thread for OMP/OpenBLAS/MKL/NumExpr are set. Matplotlib cache and raw logs are local, outside the repository.

Initial restricted-network dependency installation could not connect. It was retried with network access using the same pins. A first local wheel build could not access the user pip cache; `--no-cache-dir` resolved this environment restriction. These are recorded setup failures, not suppressed product test failures. The original logs remain locally preserved.


## Completed pre-push results

| Check | Result |
|---|---|
| Pinned dependency install in fresh environment | PASS; all 26 runtime/build pins match |
| Regular package install and pip dependency check | PASS |
| Isolated import outside checkout | PASS; installed site-packages, version 0.2.2 |
| Minimal physics example | PASS; synthetic endpoint SC = 0.8732713697525131 |
| Product collection and execution | 360 collected; 360 passed, 0 failed/errors/skipped |
| Fresh synthetic generation and input validation | PASS |
| Full synthetic demo and model/provenance verification | PASS_SOFTWARE_PIPELINE / PASS_SOFTWARE_ONLY |
| Documented CLI procedure | All 14 stage commands exit 0, in documented order |
| Product test assertions and functions | Preserved against the latest source |

The exact executed Python arguments, exit codes, durations, runtime-file digest and environment are recorded in [machine-readable results](validation_pre_push.json). `<REPO>` and `<LOGS>` replace local absolute directory prefixes only. Raw local logs and JUnit remain outside Git because they contain environment paths. Documentation links are checked separately with `python scripts/check_repository.py` immediately before committing.

Commands after environment creation included:

```sh
python -m pip --isolated install --disable-pip-version-check --retries 1 --timeout 25 -r requirements-build-lock.txt -r requirements-lock.txt
python -m pip --isolated install --no-cache-dir --no-index --no-build-isolation --no-deps .
python -m pip check
python -m pytest tests --collect-only -q -p no:cacheprovider
python -m pytest tests -q -ra -p no:cacheprovider -o junit_family=xunit1 --junitxml=<LOGS>/product-tests.xml --basetemp=<LOGS>/test-temp
python -m cvd_cbd study generate --config configs/synthetic_study.json --root outputs/generated_inputs
python -m cvd_cbd study validate --config configs/synthetic_study.json --root outputs/generated_inputs
python -m cvd_cbd study demo --config configs/synthetic_study.json --out outputs/demo
python verification/verify_run.py --run outputs/demo --out outputs/demo_verification.json
```

The isolated minimal-example and import commands and each of the 14 stage commands appear in the JSON record. No dependency on the prior wheelhouse or prior environment was used: normal package-index installation populated a new environment. The ordinary pip download cache may be used by pip; it is not a project fixture. A second environment is used for the remote-clone verification. A successful installation does not validate other OS/Python combinations or real experimental performance.

The original audit's historical **144 PASS / 1 mutation compatibility FAIL** remains recorded in [external audits](EXTERNAL_AUDITS.md). Neither that suite, its 145-case adapter nor the separate 89-case suite was rerun for this publication preparation. Their omission from the basic product suite is explicit; they are not relabeled as newly passed. No GitHub Actions workflow was run or success badge added.
