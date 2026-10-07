import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from cvd_cbd.config import Config, FEATURES
from cvd_cbd.data import generate_synthetic
from cvd_cbd.surrogate import train_surrogate, predict_surrogate
from cvd_cbd.design import select_screening_design
from cvd_cbd.monte_carlo import noise_stress_test
from cvd_cbd.demo import run_demo

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(Config.load(ROOT / "configs/reference_demo.json"), n_samples=180, n_estimators=12)
        cls.data = generate_synthetic(cls.cfg)
        cls.model, cls.metrics, cls.importance, cls.predictions, cls.split = train_surrogate(cls.data, cls.cfg)

    def test_generation_reproducible(self):
        pd.testing.assert_frame_equal(self.data, generate_synthetic(self.cfg))
        self.assertTrue(self.data.sc_proxy.between(0, 1).all())
        self.assertTrue(np.all(np.isfinite(self.data.da)))

    def test_splits_are_disjoint(self):
        self.assertEqual(len(self.split), self.cfg.n_samples)
        self.assertEqual(sum(self.split == "test"), 36)
        self.assertEqual(sum(self.split == "validation"), 36)
        self.assertEqual(sum(self.split == "train"), 108)
        self.assertFalse(set(self.data.loc[self.split == "train", "sample_id"]) & set(self.predictions.sample_id))

    def test_domain_rejection(self):
        x = self.data[FEATURES].iloc[:1].copy()
        x.loc[x.index[0], "aspect_ratio"] = 49
        with self.assertRaises(ValueError): predict_surrogate(self.model, x, self.cfg)
        x.loc[x.index[0], "aspect_ratio"] = np.nan
        with self.assertRaises(ValueError): predict_surrogate(self.model, x, self.cfg)

    def test_unique_reproducible_design(self):
        a, count = select_screening_design(self.model, self.cfg, self.importance)
        b, _ = select_screening_design(self.model, self.cfg, self.importance)
        self.assertEqual(count, 243)
        self.assertEqual(len(a), 27)
        self.assertEqual(len(a.drop_duplicates(FEATURES)), 27)
        pd.testing.assert_frame_equal(a, b)

    def test_noise_zero_and_reproducibility(self):
        truth = np.linspace(.3, .9, 27)
        a, _ = noise_stress_test(truth, repeats=40, noise_sd_fraction=0)
        np.testing.assert_allclose(a.fixed_theory_r2, 1)
        np.testing.assert_allclose(a.ols_in_sample_r2, 1)
        b, _ = noise_stress_test(truth, repeats=40)
        c, _ = noise_stress_test(truth, repeats=40)
        pd.testing.assert_frame_equal(b, c)
        with self.assertRaises(ValueError): noise_stress_test(np.ones(27))

    def test_zero_reaction_demo_reports_undefined_stress_test(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "zero"
            run_demo(replace(self.cfg, surface_rate_ref_m_s=0), out, plots=False)
            summary = json.loads((out / "monte_carlo_summary.json").read_text())
            self.assertEqual(summary["status"], "not_applicable_constant_sc_design")

    def test_end_to_end_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "result"
            result = run_demo(self.cfg, out, plots=False, monte_carlo_repeats=40)
            self.assertEqual(result["data_count"], 180)
            self.assertTrue((out / "synthetic_surrogate.joblib").is_file())
            self.assertTrue((out / "monte_carlo_summary.json").is_file())
            with self.assertRaises(ValueError): run_demo(self.cfg, out, plots=False)

if __name__ == "__main__": unittest.main()
