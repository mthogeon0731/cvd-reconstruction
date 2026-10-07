"""Synthetic unit/smoke checks, not validation of experimental performance."""
import json
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import numpy as np
import pandas as pd

from cvd_cbd.imaging import ImagingConfig, FEATURE_COLUMNS, clahe, extract_manifest, extract_region_features
from cvd_cbd.thickness import TrainingConfig, condition_weights, fit_grouped_thickness, predict_thickness, synthetic_feature_fixture


class ImagingTests(unittest.TestCase):
    def test_clahe_constant_and_bounded(self):
        constant = np.full((64, 64), 0.35)
        np.testing.assert_allclose(clahe(constant), constant)
        gradient = np.tile(np.linspace(0, 1, 64), (64, 1))
        enhanced = clahe(gradient)
        self.assertEqual(enhanced.shape, gradient.shape)
        self.assertTrue(np.isfinite(enhanced).all())
        self.assertGreaterEqual(enhanced.min(), 0)
        self.assertLessEqual(enhanced.max(), 1 + 1e-12)

    def test_tiles_aggregate_one_region(self):
        image = np.full((128, 128), 128, dtype=np.uint8)
        output = extract_region_features(image, config=ImagingConfig(tile_size=64))
        self.assertEqual(output['n_tiles'], 4)
        self.assertEqual(output['n_valid_pixels'], 128 * 128)
        self.assertEqual(set(FEATURE_COLUMNS), {k for k in output if k.startswith('feature_')})
        self.assertAlmostEqual(output['feature_raw_intensity_mean_mean'], 128/255)
        self.assertAlmostEqual(output['feature_edge_gradient_mean_mean'], 0)

    def test_invalid_mask_and_image(self):
        image = np.ones((64, 64))
        with self.assertRaises(ValueError):
            extract_region_features(image, np.zeros_like(image, dtype=bool))
        with self.assertRaises(ValueError):
            extract_region_features(image * np.nan)
        with self.assertRaises(ValueError):
            extract_region_features(image, np.ones((8, 8)))

    def test_manifest_keeps_roi_target_once_and_rejects_duplicate_measurement(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.tile(np.arange(128, dtype=np.uint8), (128, 1))).save(root / 'synthetic.png')
            row = dict(sample_id='s1', image_id='i1', condition_id='c1', image_path='synthetic.png',
                       roi_x=0, roi_y=0, roi_width=128, roi_height=128, pixel_size_um=.5,
                       thickness_nm=50, measurement_id='SYNTHETIC-m1', match_basis='registered_roi', data_origin='synthetic_demo')
            manifest = root / 'manifest.csv'
            pd.DataFrame([row]).to_csv(manifest, index=False)
            config = ImagingConfig(tile_size=64, expected_width=None, expected_height=None)
            features = extract_manifest(manifest, root / 'features.csv', config)
            self.assertEqual(len(features), 1)
            self.assertEqual(features.n_tiles.iloc[0], 4)
            self.assertEqual(features.thickness_nm.iloc[0], 50)
            with self.assertRaises(FileExistsError):
                extract_manifest(manifest, root / 'features.csv', config)
            with self.assertRaisesRegex(ValueError, 'input manifest'):
                extract_manifest(manifest, manifest, config)
            pd.DataFrame([row, dict(row, sample_id='s2')]).to_csv(manifest, index=False)
            with self.assertRaisesRegex(ValueError, 'reused'):
                extract_manifest(manifest, config=config)


class GroupedThicknessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.data = synthetic_feature_fixture(n_conditions=10, regions_per_condition=2)
        cls.config = TrainingConfig(seed=71, allow_synthetic=True)
        cls.report = fit_grouped_thickness(cls.data, cls.path, config=cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_group_isolation_and_artifacts(self):
        self.assertFalse(set(self.report['train_conditions']) & set(self.report['test_conditions']))
        split = pd.read_csv(self.path / 'split_assignments.csv')
        self.assertTrue((split.groupby('condition_id').split.nunique() == 1).all())
        train = split[split.split.eq('train')]
        self.assertTrue((train.groupby('condition_id').cv_validation_fold.nunique() == 1).all())
        self.assertTrue((split.groupby('image_id').split.nunique() == 1).all())
        comparison = pd.read_csv(self.path / 'cv_results.csv')
        self.assertEqual(set(comparison.model), {'KRR', 'RF', 'SVR'})
        self.assertEqual(self.report['data_origin'], 'synthetic_demo')
        self.assertEqual(self.report['model_fitted_on'], 'training conditions only')
        self.assertTrue((self.path / 'selected_model.joblib').exists())
        self.assertEqual(json.loads((self.path / 'metrics.json').read_text())['selected_model'], self.report['selected_model'])

    def test_refuses_existing_training_and_prediction_outputs(self):
        with self.assertRaises(FileExistsError):
            fit_grouped_thickness(self.data, self.path, config=self.config)
        with self.assertRaises(FileExistsError):
            predict_thickness(self.path / 'selected_model.joblib', self.data,
                              self.path / 'metrics.json', trust_model=True)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            predict_thickness(self.path / 'selected_model.joblib', self.data,
                              self.path / 'selected_model.joblib', trust_model=True)

    def test_model_not_loaded_without_explicit_trust(self):
        with patch('cvd_cbd.thickness.joblib.load') as loader:
            with self.assertRaisesRegex(ValueError, 'arbitrary code'):
                predict_thickness(self.path / 'selected_model.joblib', self.data)
            loader.assert_not_called()

    def test_weights_balance_conditions(self):
        groups = np.array(['a', 'a', 'a', 'b'])
        weights = condition_weights(groups)
        self.assertAlmostEqual(weights[groups == 'a'].sum(), weights[groups == 'b'].sum())

    def test_predict_and_extrapolation_flag(self):
        rows = self.data.iloc[:2].copy()
        rows.loc[rows.index[0], FEATURE_COLUMNS[0]] = 100
        predicted = predict_thickness(self.path / 'selected_model.joblib', rows, trust_model=True)
        self.assertEqual(len(predicted), 2)
        self.assertTrue(predicted.outside_training_feature_range.iloc[0])
        self.assertTrue(np.isfinite(predicted.predicted_thickness_nm).all())

    def test_missing_labels_and_too_few_groups_rejected(self):
        data = self.data.copy()
        data.loc[0, 'thickness_nm'] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            fit_grouped_thickness(data, self.path, config=self.config)
        with self.assertRaisesRegex(ValueError, 'distinct conditions'):
            fit_grouped_thickness(synthetic_feature_fixture(4), self.path, config=self.config)
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            fit_grouped_thickness(self.data, self.path)

    def test_duplicate_measurement_and_image_leakage_rejected(self):
        data = self.data.copy()
        data.loc[1, 'measurement_id'] = data.measurement_id.iloc[0]
        with self.assertRaisesRegex(ValueError, 'unique'):
            fit_grouped_thickness(data, self.path, config=self.config)
        data = self.data.copy()
        data.loc[2, 'image_id'] = data.image_id.iloc[0]
        with self.assertRaisesRegex(ValueError, 'original image'):
            fit_grouped_thickness(data, self.path, config=self.config)

    def test_scale_mismatch_and_feature_leakage_rejected(self):
        data = self.data.copy()
        data.loc[0, 'pixel_size_um'] = 1.0
        with self.assertRaisesRegex(ValueError, 'calibration'):
            fit_grouped_thickness(data, self.path, config=self.config)
        with self.assertRaisesRegex(ValueError, 'predictors'):
            fit_grouped_thickness(self.data, self.path, feature_columns=['thickness_nm'], config=self.config)


if __name__ == '__main__':
    unittest.main()
