# Changes

## Portfolio documentation refresh — 2026-10-07

- Retain 0.2.2: production behavior, input contracts and the 360 product tests are unchanged.
- Reorganize English/Korean README using the actual formulation-bo and dcv-vision presentation styles, with linked evidence and explicit synthetic-only interpretation.
- Add an authored pipeline SVG and a reproducible two-panel figure from a new seed-0 study. Keep models, TIFFs and large outputs ignored.
- Distinguish cross-fitted ML SC from synthetic label-derived SC, and historical audits from current verification.
- Apply standard MIT with Copyright (c) 2026 Hogeon Kim after explicit owner approval. Add README/repository metadata and SPDX license fields; require setuptools >=77 (tested pin 80.9.0).
- Keep the existing repository private; public visibility needs separate final-commit approval.

## 0.2.2 — 2026-10-07

- Prepare a minimal repository from the latest 0.2.1 JSON-fix source, preserving original local artifacts outside Git.
- Replace historical-output dependencies in product JSON regression setup with fresh synthetic generation. Preserve all test assertions, expected failures, parameterized cases and product physics/ML behavior.
- Include only source, requirements, empty schemas, generator and verification utilities. Exclude existing images, output bundles, models and external audits requiring archival fixtures.
- Add standalone installation, a minimal physics example, current validation, corrected internal links, public provenance and explicit historical audit limitations.
- Set package/runtime/schema documentation to 0.2.2 to distinguish this publication preparation; the input contract itself is unchanged.
- At initial preparation, leave the project license undecided and repository visibility private; MIT is selected in the subsequent approved refresh above.

## Prior 0.2.1 work — historical context

Six-finding remediation strengthened CSV/header handling, scalar pixel calibration, input provenance, uncertainty and metric contracts. A subsequent JSON follow-up rejects duplicate decoded keys at any object depth, including configuration, model metadata, uncertainty and JSON in observation-window cells. Workflow input validation occurs before output creation. These changes precede this repository preparation. The older 505/89/14 results and the unchanged original audit's 144 PASS / 1 FAIL are described in [external audits](docs/EXTERNAL_AUDITS.md).
