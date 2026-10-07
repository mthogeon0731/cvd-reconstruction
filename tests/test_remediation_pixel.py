"""ML-03 regression contracts, including direct/legacy/saved-model entry points."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.linear_model import LinearRegression

from cvd_cbd.input_contracts import (
    validate_pixel_scale, scalar_pixel_scale, reject_axis_pixel_calibration,
)
from cvd_cbd.imaging import ImagingConfig, FEATURE_COLUMNS, extract_manifest
from cvd_cbd.metrology import tile_boxes
from cvd_cbd.study_config import StudyConfig
from cvd_cbd.config import Config
from cvd_cbd.thickness import (
    TrainingConfig, fit_grouped_thickness, predict_thickness, synthetic_feature_fixture,
    _metrics, _validate_table,
)


INVALID = [0, -.5, np.nan, np.inf, -np.inf, True, np.bool_(True), None, '', '0.5 um/px', '500 nm/px', '1_000']


@pytest.mark.parametrize('value', INVALID)
def test_pixel_validator_rejects_invalid_without_coercing_boolean(value):
    with pytest.raises(ValueError, match='pixel|um/px'):
        validate_pixel_scale(value)


def test_pixel_numbers_have_one_explicit_unit_and_preserve_input_shape():
    actual = validate_pixel_scale([['.5', '+5e-1'], [' 0.5 ', 0.5]])
    np.testing.assert_array_equal(actual, np.full((2, 2), .5))
    assert scalar_pixel_scale('5e-1') == .5
    for vector in ([.5, True], [.5, np.bool_(False)]):
        with pytest.raises(ValueError, match='boolean'):
            validate_pixel_scale(vector)
    for unsupported in ([.5, .6], {'x': .5, 'y': .6}, [[.5]]):
        with pytest.raises(ValueError):
            scalar_pixel_scale(unsupported)


@pytest.mark.parametrize('kind', ['flat', 'trench'])
@pytest.mark.parametrize('value', INVALID)
def test_direct_tile_geometry_checks_scale_before_pixel_arithmetic(kind, value):
    row = dict(kind=kind, um_per_px=value, roi_mode='manual', z_anchor_um=50.,
               x_anchor_px=128., L_um=270., roi_x0=0, roi_x1=256, roi_y0=0, roi_y1=128)
    with pytest.raises(ValueError, match='pixel|um/px'):
        tile_boxes((128, 256), row, StudyConfig(), (34, 94))


def test_numeric_string_scale_preserves_known_micrometre_window():
    row = dict(kind='trench', um_per_px='5e-1', roi_mode='auto',
               z_anchor_um=50., x_anchor_px=128., L_um=270.)
    boxes, roi = tile_boxes((128, 256), row, StudyConfig(), (34, 94))
    assert boxes == [(96, 48, 128, 80), (128, 48, 160, 80)]
    assert roi == (96, 38, 160, 90)
    intervals = [(50 + (a - 128) * .5, 50 + (c - 128) * .5) for a, _, c, _ in boxes]
    assert intervals == [(34., 50.), (50., 66.)]


@pytest.mark.parametrize('name', ['um_per_px_x', 'um_per_px_y', 'pixel_size_x_um', 'pixel_size_um_y', 'pixel_width_um'])
def test_axis_specific_calibration_never_silently_ignored(name):
    with pytest.raises(ValueError, match='axis-specific'):
        reject_axis_pixel_calibration(['um_per_px', name])
    row = dict(kind='flat', um_per_px=.5, roi_mode='auto', **{name: .7})
    with pytest.raises(ValueError, match='axis-specific'):
        tile_boxes((64, 64), row, StudyConfig())


@pytest.fixture
def legacy_manifest(tmp_path):
    Image.fromarray(np.full((64, 64), 128, np.uint8)).save(tmp_path / 'image.png')
    return pd.DataFrame([dict(sample_id='s', image_id='i', condition_id='c', image_path='image.png',
        roi_x=0, roi_y=0, roi_width=64, roi_height=64, pixel_size_um=.5, data_origin='synthetic_demo')])


@pytest.mark.parametrize('value', [0, -.5, np.nan, np.inf, True, '500 nm/px'])
def test_legacy_manifest_csv_scale_rejected(tmp_path, legacy_manifest, value):
    legacy_manifest['pixel_size_um'] = value
    path = tmp_path / 'manifest.csv'
    legacy_manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match='pixel|um/px'):
        extract_manifest(path, config=ImagingConfig(tile_size=32, expected_width=None, expected_height=None))


def test_legacy_csv_numeric_format_and_axis_contract(tmp_path, legacy_manifest):
    path = tmp_path / 'manifest.csv'
    legacy_manifest['pixel_size_um'] = '5e-1'
    legacy_manifest.to_csv(path, index=False)
    actual = extract_manifest(path, config=ImagingConfig(tile_size=32, expected_width=None, expected_height=None))
    assert actual.pixel_size_um.iloc[0] == .5 and actual.n_tiles.iloc[0] == 4
    legacy_manifest['pixel_size_x_um'] = .7
    legacy_manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match='axis-specific'):
        extract_manifest(path)


@pytest.mark.parametrize('value', [0, -.5, np.nan, np.inf, True, '.5 um/px'])
def test_legacy_public_training_rejects_bad_scale_before_creating_model(tmp_path, value):
    data = synthetic_feature_fixture(6, 2)
    data['pixel_size_um'] = value
    out = tmp_path / 'fit'
    with pytest.raises(ValueError, match='pixel|um/px'):
        fit_grouped_thickness(data, out, config=TrainingConfig(allow_synthetic=True))
    assert not out.exists()


def test_legacy_training_normalizes_strings_and_rejects_axes():
    data = synthetic_feature_fixture(6, 2)
    data['pixel_size_um'] = '5e-1'
    _validate_table(data, FEATURE_COLUMNS, True)
    np.testing.assert_array_equal(data.pixel_size_um, np.full(len(data), .5))
    data['pixel_size_y_um'] = .6
    with pytest.raises(ValueError, match='axis-specific'):
        _validate_table(data, FEATURE_COLUMNS, True)


@pytest.fixture
def own_legacy_model(tmp_path):
    # This test creates its own estimator and artifact; no supplied pickle is read.
    X = np.array([[0., 1.], [1., 2.], [2., 3.]])
    model = LinearRegression().fit(X, [1., 3., 5.])
    bundle = dict(pipeline=model, feature_columns=['feature_a', 'feature_b'],
                  pixel_size_um=.5, feature_min=X.min(axis=0), feature_max=X.max(axis=0),
                  report={'data_origin': 'synthetic_demo'})
    path = tmp_path / 'own.joblib'
    joblib.dump(bundle, path)
    data = pd.DataFrame(dict(sample_id=['a', 'b'], pixel_size_um=[.5, .5], feature_a=[0., 1.], feature_b=[1., 2.]))
    return path, bundle, data


@pytest.mark.parametrize('where', ['input', 'saved_model'])
@pytest.mark.parametrize('value', [0, -.5, np.nan, np.inf, True, '500 nm/px'])
def test_legacy_reloaded_prediction_validates_both_calibrations(own_legacy_model, where, value):
    path, bundle, data = own_legacy_model
    if where == 'input':
        data['pixel_size_um'] = value
    else:
        bundle['pixel_size_um'] = value
        joblib.dump(bundle, path)
    with pytest.raises(ValueError, match='pixel|um/px'):
        predict_thickness(path, data, trust_model=True)


def test_legacy_reloaded_prediction_string_scale_and_axis_rejection(own_legacy_model):
    path, bundle, data = own_legacy_model
    expected = predict_thickness(path, data, trust_model=True)
    data['pixel_size_um'] = '5e-1'
    bundle['pixel_size_um'] = '5e-1'
    joblib.dump(bundle, path)
    actual = predict_thickness(path, data, trust_model=True)
    pd.testing.assert_frame_equal(actual, expected)
    bundle['pixel_size_x_um'] = .7
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match='axis-specific'):
        predict_thickness(path, data, trust_model=True)


def test_saved_configs_do_not_accept_unsupported_pixel_override(tmp_path):
    # Pixel calibration belongs to each image/model, not global geometry settings.
    study = tmp_path / 'study.json'
    study.write_text(json.dumps({'um_per_px': .5}))
    with pytest.raises(ValueError, match='Unknown config keys'):
        StudyConfig.load(study)
    reference = Path(__file__).resolve().parents[1] / 'configs/reference_demo.json'
    spec = json.loads(reference.read_text())
    spec['pixel_size_x_um'] = .5
    legacy = tmp_path / 'legacy.json'
    legacy.write_text(json.dumps(spec))
    with pytest.raises(TypeError, match='pixel_size_x_um'):
        Config.load(legacy)


@pytest.mark.parametrize('spaced', [False, True])
def test_legacy_csv_duplicate_scale_header_rejected_before_mangling(tmp_path, legacy_manifest, spaced):
    duplicate = legacy_manifest.copy()
    name = ' pixel_size_um ' if spaced else 'pixel_size_um'
    extra = pd.DataFrame({name: [50.]})
    path = tmp_path / 'duplicate.csv'
    pd.concat([duplicate, extra], axis=1).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate CSV header.*pixel_size_um'):
        extract_manifest(path)
    data = synthetic_feature_fixture(6, 2)
    extra = pd.DataFrame({name: [50.] * len(data)})
    ambiguous = pd.concat([data, extra], axis=1)
    with pytest.raises(ValueError, match='Duplicate column names.*pixel_size_um'):
        _validate_table(ambiguous, FEATURE_COLUMNS, True)
    ambiguous.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate CSV header.*pixel_size_um'):
        fit_grouped_thickness(path, tmp_path / 'fit', config=TrainingConfig(allow_synthetic=True))


def test_legacy_metric_shape_contract_prevents_pairwise_broadcasting():
    value = _metrics([1., 2.], [[1.], [2.]], ['a', 'b'], True)
    assert value['rmse_nm'] == 0 and value['r2'] == 1
    with pytest.raises(ValueError):
        _metrics([1., 2.], [[1., 2.]], ['a', 'b'], False)
    with pytest.raises(ValueError):
        _metrics([1., 2.], [1., 2.], ['a'], False)
    assert _metrics([], [], [], False)['r2'] is None
    assert _metrics([1., 1.], [1., 2.], ['a', 'b'], True)['r2'] is None
    assert _metrics([0., 0.], [1., 2.], ['a', 'b'], True)['mean_absolute_percentage_error'] is None
