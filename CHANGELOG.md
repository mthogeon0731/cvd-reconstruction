# Changes

## 0.2.2 — 2026-10-07

- Prepare a minimal repository from the latest 0.2.1 JSON-fix source, preserving original local artifacts outside Git.
- Replace historical-output dependencies in product JSON regression setup with fresh synthetic generation. Preserve all test assertions, expected failures, parameterized cases and product physics/ML behavior.
- Include only source, requirements, empty schemas, generator and verification utilities. Exclude existing images, output bundles, models and external audits requiring archival fixtures.
- Add standalone installation, a minimal physics example, current validation, corrected internal links, public provenance and explicit historical audit limitations.
- Set package/runtime/schema documentation to 0.2.2 to distinguish this publication preparation; the input contract itself is unchanged.
- Leave the project license undecided and repository visibility private.

## Prior 0.2.1 work — historical context

Six-finding remediation strengthened CSV/header handling, scalar pixel calibration, input provenance, uncertainty and metric contracts. A subsequent JSON follow-up rejects duplicate decoded keys at any object depth, including configuration, model metadata, uncertainty and JSON in observation-window cells. Workflow input validation occurs before output creation. These changes precede this repository preparation. The older 505/89/14 results and the unchanged original audit's 144 PASS / 1 FAIL are described in [external audits](docs/EXTERNAL_AUDITS.md).
