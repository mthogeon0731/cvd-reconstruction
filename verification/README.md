# Verification utilities

Product tests are in [tests](../tests) and run with `python -m pytest tests -q` from the repository root. They create their own synthetic inputs.

`verify_run.py` checks a newly generated default synthetic demo, its model predictions and actual input hashes. It has fixed counts for the default configuration; it is not a generic validator for arbitrary studies. It deserializes the model and must only be used on a demo you just generated from trusted source:

```sh
python verification/verify_run.py --run outputs/demo --out outputs/demo_verification.json
```

`compare_runs.py` compares two locally generated result directories with the declared tolerances in [comparison_policy.json](reports/comparison_policy.json). It reports byte and numerical comparisons separately; differences are findings, not automatically process errors:

```sh
python verification/compare_runs.py --before outputs/first/SYNTHETIC_results --after outputs/second/SYNTHETIC_results --out outputs/comparison
```

Generate the `first` and `second` demos locally before using the comparison example. Historical external audits are not bundled; [their procedure and unresolved mutation failure](../docs/EXTERNAL_AUDITS.md) are kept separate from [current validation](../docs/VALIDATION.md).
