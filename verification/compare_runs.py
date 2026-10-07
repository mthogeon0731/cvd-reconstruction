"""Independent positional output comparison; shares only the strict JSON reader.

Usage: python compare_runs.py --before OLD/SYNTHETIC_results
       --after NEW/SYNTHETIC_results --out NEW_COMPARISON_DIRECTORY

Numerical tolerances are loaded from the predeclared comparison_policy.json.
Differences are findings, not command failures; unreadable inputs/errors exit 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from cvd_cbd.json_utils import read_json, loads_json, DuplicateJSONKeyError


EXPECTED_CSV = (
    'calibration.csv', 'growth.csv', 'manifest.csv', 'predictions_final.csv',
    'sc_points_cross_fitted.csv', 'sc_points_fesem.csv', 'sc_points_final.csv',
    'trench_sc_cross_fitted.csv', 'trench_sc_fesem.csv', 'trench_sc_final.csv',
    'uncertainty_draws.csv', 'design/condition_AR.csv', 'design/fdm_convergence.csv',
    'design/mc_trials.csv', 'design/new_dead_end_1pct.csv', 'design/sensitivity.csv',
    'evaluation/all_model_oof.csv', 'evaluation/candidates.csv',
    'evaluation/inner_splits.csv', 'evaluation/outer_splits.csv',
    'evaluation/selected_oof.csv', 'features/features.csv', 'features/image_qc.csv',
    'features/tiles.csv', 'final_model/inner_splits.csv', 'final_model/selection.csv',
    'process_window/process_grid.csv', 'similitude_cross_fitted/points.csv',
    'similitude_cross_fitted/trench_theory.csv', 'similitude_fesem/points.csv',
    'similitude_fesem/trench_theory.csv',
)
EXACT_DISCRETE = {
    'chip', 'slot', 'rep', 'cycles', 'faces', 'bit_depth', 'outer_fold',
    'inner_fold', 'fold', 'trial', 'draw', 'seed', 'nodes', 'df', 'n', 'tiles',
    'rejected_tiles', 'channel_y0', 'channel_y1', 'x0', 'x1', 'y0', 'y1',
    'roi_x0', 'roi_x1', 'roi_y0', 'roi_y1', 'x_anchor_px',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def exact_column(name):
    return (name in EXACT_DISCRETE or name.endswith(('_id', '_ids', '_sha256', '_pixels'))
            or name.startswith('n_') or name.endswith('_count'))


def provenance_column(name):
    return name.endswith('_sha256') or name.startswith('label_evidence_')


def load_csv(path):
    # Validate raw headers independently, before pandas can mangle duplicates.
    with path.open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f'Empty/duplicate output CSV header: {path}')
    typed = pd.read_csv(path, float_precision='round_trip', encoding='utf-8-sig')
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if list(typed) != header or list(raw) != header or len(typed) != len(raw):
        raise ValueError(f'Unexpected CSV parse/header shape: {path}')
    return typed, raw


def json_compare(before, after, rtol, atol, exact_numbers=False):
    """Ignore object key order only; keep array order and all metadata keys."""
    if isinstance(before, bool) or isinstance(after, bool):
        return type(before) is type(after) and before == after
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if not math.isfinite(before) or not math.isfinite(after):
            return before == after
        return before == after if exact_numbers else abs(after-before) <= atol + rtol*abs(before)
    if isinstance(before, dict) and isinstance(after, dict):
        return before.keys() == after.keys() and all(
            json_compare(before[key], after[key], rtol, atol, exact_numbers) for key in before)
    if isinstance(before, list) and isinstance(after, list):
        return len(before) == len(after) and all(
            json_compare(left, right, rtol, atol, exact_numbers) for left, right in zip(before, after))
    return type(before) is type(after) and before == after


def nested_numeric_deltas(before, after):
    """Yield paired numeric scalar errors without discarding positional order."""
    if isinstance(before, bool) or isinstance(after, bool):
        return
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if math.isfinite(before) and math.isfinite(after):
            delta = abs(after-before)
            yield delta, delta/abs(before) if before else (0. if delta == 0 else math.inf)
    elif isinstance(before, dict) and isinstance(after, dict):
        for key in before:
            if key in after:
                yield from nested_numeric_deltas(before[key], after[key])
    elif isinstance(before, list) and isinstance(after, list):
        for left, right in zip(before, after):
            yield from nested_numeric_deltas(left, right)


def errors_summary(pairs):
    pairs = list(pairs)
    if not pairs:
        return dict(max_abs_error=None, max_rel_error=None, unbounded_relative_error_count=0)
    relative = max(x[1] for x in pairs)
    return dict(max_abs_error=float(max(x[0] for x in pairs)),
                max_rel_error='infinity' if math.isinf(relative) else float(relative),
                unbounded_relative_error_count=sum(math.isinf(x[1]) for x in pairs))


def compare_column(name, left, right, raw_left, raw_right, rtol, atol):
    a = raw_left.to_numpy(); b = raw_right.to_numpy()
    missing_a = a == ''; missing_b = b == ''
    same_missing = missing_a == missing_b
    equal = a == b
    record = dict(column=name, before_dtype=str(left.dtype), after_dtype=str(right.dtype),
                  missing_before=int(missing_a.sum()), missing_after=int(missing_b.sum()),
                  missing_mask_equal=bool(same_missing.all()),
                  exact_text_equal=bool(equal.all()), mode='exact_text',
                  structured_normalized_rows=0, structured_key_changes=0)
    paired = ~(missing_a | missing_b)
    numeric = pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right)
    boolean = pd.api.types.is_bool_dtype(left) or pd.api.types.is_bool_dtype(right)
    discrete = exact_column(name)
    deltas = []
    if numeric and not boolean:
        av = left.to_numpy(dtype=float); bv = right.to_numpy(dtype=float)
        finite = paired & np.isfinite(av) & np.isfinite(bv)
        record['mode'] = 'exact_discrete' if discrete else 'numeric_tolerance'
        if not discrete:
            equal[paired] = np.isclose(bv[paired], av[paired], rtol=rtol, atol=atol, equal_nan=False)
        # Literal nonfinite values must match exactly, including infinity sign.
        nonfinite = paired & ~finite
        equal[nonfinite] = a[nonfinite] == b[nonfinite]
        record['nonfinite_nonmissing_before'] = int((paired & ~np.isfinite(av)).sum())
        record['nonfinite_nonmissing_after'] = int((paired & ~np.isfinite(bv)).sum())
        for old, new in zip(av[finite], bv[finite]):
            delta = abs(float(new)-float(old))
            deltas.append((delta, delta/abs(float(old)) if old else (0. if delta == 0 else math.inf)))
    elif boolean:
        record['mode'] = 'exact_boolean'
    else:
        for i in np.flatnonzero(paired):
            old, new = a[i], b[i]
            if old.startswith(('[', '{')) and new.startswith(('[', '{')):
                try:
                    old_json, new_json = loads_json(old,source='comparison CSV, before cell'), loads_json(new,source='comparison CSV, after cell')
                except DuplicateJSONKeyError:
                    raise
                except (ValueError, TypeError):
                    continue
                record['mode'] = 'structured_json_exact_ids' if discrete else 'structured_json'
                matches = json_compare(old_json, new_json, rtol, atol, exact_numbers=discrete)
                if isinstance(old_json, dict) and isinstance(new_json, dict) and old_json.keys() != new_json.keys():
                    record['structured_key_changes'] += 1
                if not equal[i] and matches:
                    record['structured_normalized_rows'] += 1
                equal[i] = matches
                deltas.extend(nested_numeric_deltas(old_json, new_json))
    equal &= same_missing
    record.update(errors_summary(deltas))
    record['within_policy'] = bool(equal.all())
    record['different_rows'] = int((~equal).sum())
    record['example_differences'] = [dict(csv_row=int(i)+2, before=a[i][:240], after=b[i][:240])
                                     for i in np.flatnonzero(~equal)[:3]]
    return record


def compare_csv(before, after, rel, rtol, atol):
    left_path, right_path = before/rel, after/rel
    item = dict(artifact=rel, before_exists=left_path.is_file(), after_exists=right_path.is_file())
    if not item['before_exists'] or not item['after_exists']:
        item.update(status='MISSING_ARTIFACT', common_columns_within_policy=False, columns=[])
        return item
    left, a = load_csv(left_path); right, b = load_csv(right_path)
    ahash, bhash = sha256(left_path), sha256(right_path)
    added = [c for c in b if c not in a]; removed = [c for c in a if c not in b]
    common = [c for c in a if c in b]
    common_order_equal = common == [c for c in b if c in a]
    item.update(before_sha256=ahash, after_sha256=bhash, byte_equal=ahash == bhash,
                newline_normalized_equal=left_path.read_bytes().replace(b'\r\n', b'\n') == right_path.read_bytes().replace(b'\r\n', b'\n'),
                before_shape=list(a.shape), after_shape=list(b.shape), row_count_equal=len(a) == len(b),
                column_order_equal=list(a) == list(b), common_column_order_equal=common_order_equal,
                added_columns=added, removed_columns=removed,
                added_provenance_columns=[c for c in added if provenance_column(c)],
                added_other_columns=[c for c in added if not provenance_column(c)],
                schema_equal=list(a) == list(b), columns=[])
    if len(a) != len(b):
        item.update(status='ROW_COUNT_CHANGED', common_columns_within_policy=False)
        return item
    for column in common:
        item['columns'].append(compare_column(column, left[column], right[column], a[column], b[column], rtol, atol))
    item['common_columns_within_policy'] = all(c['within_policy'] for c in item['columns'])
    item['different_common_columns'] = [c['column'] for c in item['columns'] if not c['within_policy']]
    if not item['common_columns_within_policy']:
        item['status'] = 'VALUE_CHANGED_WITH_SCHEMA_CHANGE' if not item['schema_equal'] else 'VALUE_CHANGED'
    elif added or removed:
        item['status'] = 'SCHEMA_CHANGED_COMMON_VALUES_WITHIN_POLICY'
    elif not common_order_equal:
        item['status'] = 'COLUMN_ORDER_CHANGED'
    else:
        item['status'] = 'BYTE_IDENTICAL' if item['byte_equal'] else 'VALUES_WITHIN_POLICY_BYTES_DIFFER'
    item['complete_artifact_within_policy'] = item['schema_equal'] and item['common_columns_within_policy']
    return item


def compare_tiffs(before, after):
    from PIL import Image
    left_root = before.parent/'SYNTHETIC_inputs'; right_root = after.parent/'SYNTHETIC_inputs'
    if not left_root.is_dir() or not right_root.is_dir():
        return dict(status='NOT RUN', reason='Both sibling SYNTHETIC_inputs directories required', files=[])
    def inventory(root):
        return {p.relative_to(root).as_posix() for p in root.rglob('*')
                if p.is_file() and p.suffix.lower() in ('.tif', '.tiff')}
    files = []
    for rel in sorted(inventory(left_root) | inventory(right_root)):
        p, q = left_root/rel, right_root/rel
        record = dict(artifact=rel, before_exists=p.is_file(), after_exists=q.is_file())
        if p.is_file() and q.is_file():
            with Image.open(p) as image:
                if getattr(image, 'n_frames', 1) != 1: raise ValueError(f'Multi-frame TIFF unsupported: {p}')
                a = np.array(image)
            with Image.open(q) as image:
                if getattr(image, 'n_frames', 1) != 1: raise ValueError(f'Multi-frame TIFF unsupported: {q}')
                b = np.array(image)
            record.update(before_sha256=sha256(p), after_sha256=sha256(q),
                          before_shape=list(a.shape), after_shape=list(b.shape),
                          before_dtype=str(a.dtype), after_dtype=str(b.dtype),
                          pixels_equal=a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b)))
            record['byte_equal'] = record['before_sha256'] == record['after_sha256']
        files.append(record)
    return dict(status='COMPARED', before_root=str(left_root), after_root=str(right_root),
                count=len(files), byte_equal_count=sum(x.get('byte_equal', False) for x in files),
                pixels_equal_count=sum(x.get('pixels_equal', False) for x in files), files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--skip-tiff', action='store_true')
    args = parser.parse_args()
    before, after, out = args.before.resolve(), args.after.resolve(), args.out.resolve()
    policy_path = Path(__file__).resolve().parent/'reports/comparison_policy.json'
    policy = read_json(policy_path,encoding='utf-8')
    # No command-line override or data-dependent tolerance selection is allowed.
    rtol, atol = policy['rtol'], policy['atol']
    if (rtol, atol) != (1e-8, 1e-10):
        parser.error('Preset comparison tolerance changed; expected rtol=1e-8 and atol=1e-10')
    if not before.is_dir() or not after.is_dir():
        parser.error('Both comparison roots must be existing SYNTHETIC_results directories')
    if out == before or out == after or out.is_relative_to(before) or out.is_relative_to(after):
        parser.error('Comparison output must be outside both input roots')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Comparison output must be new or empty')
    def extra_csv(root):
        return sorted(p.relative_to(root).as_posix() for p in root.rglob('*.csv')
                      if p.relative_to(root).as_posix() not in EXPECTED_CSV)
    comparisons = [compare_csv(before, after, rel, rtol, atol) for rel in EXPECTED_CSV]
    tiffs = dict(status='NOT RUN', reason='--skip-tiff specified', files=[]) if args.skip_tiff else compare_tiffs(before, after)
    summary = dict(expected_csv_count=len(EXPECTED_CSV), compared_csv_count=sum(x['before_exists'] and x['after_exists'] for x in comparisons),
                   byte_equal_csv_count=sum(x.get('byte_equal', False) for x in comparisons),
                   common_values_within_policy_csv_count=sum(x['common_columns_within_policy'] for x in comparisons),
                   complete_artifact_within_policy_csv_count=sum(x.get('complete_artifact_within_policy', False) for x in comparisons),
                   schema_changed_csv_count=sum(not x.get('schema_equal', False) for x in comparisons if x['before_exists'] and x['after_exists']),
                   changed_common_value_csv_count=sum(not x['common_columns_within_policy'] for x in comparisons if x['before_exists'] and x['after_exists']),
                   missing_artifact_csv_count=sum(x['status'] == 'MISSING_ARTIFACT' for x in comparisons))
    result = dict(before=str(before), after=str(after), policy=policy, policy_sha256=sha256(policy_path),
                  comparison_rules=dict(rows='Positional comparison only; never sorted', identifiers='Exact text including leading zeroes',
                    booleans='Exact; no numeric tolerance', missing='CSV empty-cell masks must match exactly',
                    numbers='abs(after-before) <= atol + rtol*abs(before)',
                    relative_error='abs(after-before)/abs(before); nonzero change from zero is infinity',
                    structured_json='Dictionary key order ignored and reported; all keys, list order and identifier values retained',
                    paths='Exact; path separator changes are not silently normalized',
                    schema='Added/removed/reordered columns are reported separately and cannot count as unchanged complete artifacts',
                    scope='Explicit 31 study CSV paths only; unrelated/independent CSVs listed as excluded'),
                  environment=dict(python=platform.python_version(), platform=platform.platform(),
                    numpy=importlib.metadata.version('numpy'), pandas=importlib.metadata.version('pandas')),
                  excluded_csv_before=extra_csv(before), excluded_csv_after=extra_csv(after),
                  summary=summary, csv_artifacts=comparisons, tiffs=tiffs)
    out.mkdir(parents=True, exist_ok=True)
    json_dump(out/'comparison.json', result)
    json_dump(out/'summary.json', summary)
    artifact_rows = []
    column_rows = []
    for item in comparisons:
        row = {k: v for k, v in item.items() if k != 'columns'}
        artifact_rows.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        for column in item['columns']:
            column_rows.append(dict(artifact=item['artifact'], **{
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in column.items()}))
    pd.DataFrame(artifact_rows).to_csv(out/'artifact_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(column_rows).to_csv(out/'column_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(tiffs['files']).to_csv(out/'tiff_comparison.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'Comparison error: {exc}', file=sys.stderr)
        raise SystemExit(2)
