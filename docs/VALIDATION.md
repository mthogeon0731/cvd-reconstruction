# Validation records

## Portfolio refresh — 2026-10-07

This is the latest pre-push verification for the README, figures and approved MIT metadata update. Production modules and the 360 product tests are unchanged from baseline commit `1b8dc7f22232736e9d88612767f1e919d616934f`; version remains 0.2.2. [Exact new commands, durations and source/test hashes](validation_portfolio.json).

| New check | Actual result |
|---|---|
| Clean file-only copy and new environment | PASS; no old archives/fixtures, no system site packages |
| Public-index install of 26 build/runtime pins, no pip cache | PASS |
| Regular wheel install, pip check, isolated import outside checkout | PASS; version 0.2.2, SPDX MIT and exact LICENSE installed |
| Minimal example | PASS; synthetic SC = 0.8732713697525131 |
| Product tests | **360 collected / 360 passed; 0 failed, errors or skipped** |
| Separate synthetic input generation/validation | PASS |
| Full synthetic study and verification | PASS_SOFTWARE_PIPELINE / PASS_SOFTWARE_ONLY |
| Figure reproduction | Initial new run and clean-copy new run yield identical plotted source/summary/PNG hashes; repeated rendering of one run is also byte-identical |
| English/Korean README | GitHub Markdown API render, browser preview and image loading checked; local file/heading links checked separately before commit |
| Package/test integrity | Production modules, configurations, pins and product test files unchanged |

Environment: Windows 11 build 26200 AMD64 / bundled CPython 3.12.14. The [reproduction controls](REPRODUCIBILITY.md#controlled-environment) apply. Figure equality covers only the plotted inputs and summary/PNG in this environment, not every output file or another OS. The browser had no GitHub sign-in: the visual check used GitHub-rendered HTML and local assets with approximate GitHub CSS. The actual hosted private-page appearance was not visually confirmed.

The updated Quickstart installation, minimal example, full study, output verifier, figure renderer and product test commands were executed. The full 14-stage CLI replay, separate 14-case smoke check, external audits, other OS/Python combinations, GitHub Actions and real OM/FE-SEM validation were **not rerun** for this refresh. Earlier results below remain a separate record. Post-push fresh-clone installation, minimal example and product tests are reported to the owner with the final remote commit; they are not claimed in advance by this pre-push file.

The exact source files and new image metadata were reviewed for credentials, private paths, internal URLs and unintended personal details. The only binary asset is the small synthetic PNG; its metadata contains the public generator name and resolution, no EXIF. The sole email-pattern code match is matrix multiplication, manually classified as a false positive. Existing approved Git identity is retained. Rights rely on the owner's representation and explicit MIT approval; no independent patent/contract clearance or exhaustive secret-detection guarantee is asserted. [Rights and scope](RIGHTS.md), [presentation references](PORTFOLIO_REVIEW.md).

---

## Initial publication preparation — earlier 2026-10-07 run

This report is for the 0.2.2 publication copy. The pre-push clean-copy verification below completed successfully. The remote fresh-clone result is delivered separately with the exact commit SHA after pushing; it is not pre-claimed by this pre-push report. Historical counts are in [external audits](EXTERNAL_AUDITS.md).

Environment under test: Windows 11 build 26200, AMD64, bundled CPython 3.12.14. Fresh virtual environments use no system site packages. Runtime/build pins and their licenses are listed in [dependency inventory](DEPENDENCIES.md). A separate Python installer distribution, Linux, macOS, other Python versions, GitHub Actions and actual experimental performance are NOT RUN for this preparation.

The source is copied using an explicit file allowlist into an otherwise empty validation folder. Old output fixtures, virtual environments and audit folders are absent. Installation is a regular wheel installation, not editable. An isolated Python process outside the checkout checks that imports resolve to the new environment's installed package. `PYTHONPATH`/`PYTHONHOME` are cleared; `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUTF8=1`, `MPLBACKEND=Agg` and one thread for OMP/OpenBLAS/MKL/NumExpr are set. Matplotlib cache and raw logs are local, outside the repository.

Initial restricted-network dependency installation could not connect. It was retried with network access using the same pins. A first local wheel build could not access the user pip cache; `--no-cache-dir` resolved this environment restriction. These are recorded setup failures, not suppressed product test failures. The original logs remain locally preserved.


### Earlier completed pre-push results

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
