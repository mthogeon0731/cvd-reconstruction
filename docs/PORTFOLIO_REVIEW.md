# Portfolio presentation review — 2026-10-07

The target started at private main commit `1b8dc7f22232736e9d88612767f1e919d616934f`, version 0.2.2. This is a documentation, visual-example and license/packaging update; production modules and product test functions remain unchanged. Version 0.2.2 is retained consistently because the algorithms and input contracts have not changed.

## Actual reference repositories

| Read-only reference | Inspected revision | Style carried over |
|---|---|---|
| [formulation-bo](https://github.com/mthogeon0731/formulation-bo/tree/d6d25c7bd6c087a5dac5d8f6a1dcfffc8f196e6f) | `d6d25c7` on master | Short project title, bold purpose, prominent real demo figure, direct installation/example, candid limits |
| [dcv-vision](https://github.com/mthogeon0731/dcv-vision/tree/ba404c6613ebdd3ed0539a5b6f7072e56fd65ba2) | `ba404c6` on master | Synthetic caption, compact anchor navigation, Quickstart, output tables and linked technical detail |

Both actual READMEs, root file structures and LICENSE files were read. Neither reference README used badges. This repository adds only one local static badge saying synthetic-only validation, with no CI/platform claim. No reference image or prose is copied. Neither reference repository was modified. The scientific section order is adapted to this project's physical, imaging and ML paths.

The English and Korean pages use the same order: Overview, Pipeline, Quickstart, Example Results, Data & Outputs, Evaluation, Reproducibility, Limitations, Repository Structure, References, License. The pipeline SVG follows `workflow.py`; the compact result plot reads a newly executed synthetic run. Detailed evidence remains in docs, away from the opening screen.

## License and distribution metadata

MIT and its holder were separately approved by the owner. README metadata, repository URLs and SPDX MIT are set in pyproject.toml. The build requirement is setuptools ≥77 because that version introduced SPDX `project.license` and `project.license-files`; the tested build pin is 80.9.0 ([setuptools documentation](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)). No PyPI upload or release is part of this change. [Rights and limits](RIGHTS.md).

The review requires the final remote commit and private visibility to be reported to the owner. Public visibility remains a separate approval.
