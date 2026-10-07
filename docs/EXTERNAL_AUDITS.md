# Historical results and external audit procedure

The earlier 2026-10-07 follow-up reported **505 passed** for `tests` plus `verification/audit_tests`, **89 passed** for a separately supplied independent suite and **14 passed** for the then-current README stages. These are historical results. The 505 cases combine 360 product cases (including 49 JSON regressions) and 145 adapted audit cases. They are not 505 standalone product tests, and the 89 cases should not be summed into a claim of unique independent coverage.

The untouched original audit reported **144 PASS / 1 FAIL**, not a clean pass. The failure `test_unit_fault_in_audit_copy_is_detected` attempted to replace `scale=float(row['um_per_px'])`, which no longer occurs in the remediated metrology code. The mutation therefore did nothing. A separate historical adapter targets the `scalar_pixel_scale` call and asserts exactly one replacement. Original and adapter are distinct; the original failure is not suppressed, converted to xfail or erased by the adapter's success.

Historical same-environment repetition (360/360 output files identical) and before/after CSV equality (31/31) are also previous results. The earlier Windows/Linux comparison had 0/31 byte-identical CSVs and a reported prediction difference up to approximately 0.032 nm. CSV parsing precision affected the earlier R². None of these facts indicates improved measured performance. This publication preparation does not claim a new Linux/macOS or historical cross-platform run.

## What is external

The original/adapted audit tests consult archival source layouts and some pre-existing feature/output tables. They and the independent verifier are preserved locally outside this repository, alongside their unmodified logs, hashes and reports. They are excluded from the clone-required product suite because archival fixture provenance and old artifacts are outside the distributable file set. No product test is removed or weakened to achieve a pass. Current results appear only in [validation](VALIDATION.md).

To replay the original audit, obtain the separately retained audit package and its authorized fixtures from the owner. Keep its original layout `audit_tests/`, `work/project/`, `baseline/cvd_cbd_rebuild/` and `evidence/`. Place a separate copy of the reviewed product source at `work/project/`; retain the supplied baseline data in the baseline directory. Install the build/runtime pins into a new environment and install that product copy, then run from the audit root:

```sh
python -m pytest audit_tests -q -ra
```

Record the test file hashes and exact source commit, count all PASS/FAIL/ERROR/SKIP outcomes and retain the mutation failure when reproducing the unchanged original suite. Do not replace the mutation assertion just to report success. The historical adapter instead expects its delivered `verification/audit_tests` layout and historical outputs; use its retained diff to identify changes, and report it separately.

The separate independent 89-case suite requires its original `test_independent.py`, a freshly generated feature table and its calibration table. With this product installed, run from the repository root and set `IV_FEATURES` to the absolute path of `outputs/demo/SYNTHETIC_results/features/features.csv` and `IV_CALIBRATION` to the absolute path of `outputs/demo/SYNTHETIC_results/calibration.csv`. Missing `IV_FEATURES` skips tests; that is not a full pass. Then run:

```sh
python -m pytest /path/to/authorized/test_independent.py -q -ra
```

The path above is a placeholder for separately obtained test material, not a file shipped in this repository. A generic audit replay is not part of basic installation. Do not describe any of these suites as newly passed unless they were actually executed against the reported commit with their specified inputs.

## Historical reporting utility

`scripts/build_validation_report.py` is retained because its strict-JSON behavior has a product regression test. It is an archival report builder and is **not** the builder of the current public validation report. Successful execution requires the separately retained `outputs/SYNTHETIC_delivery`, legacy outputs, validation logs and environment records. Running it without that authorized archive is unsupported. It must not be used to turn old results into new claims. Its product regression creates only an invalid temporary JSON input and checks rejection before any report rewrite.
