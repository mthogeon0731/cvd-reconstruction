import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from cvd_cbd.config import Config
from cvd_cbd.physics import coefficients, analytic_profile, fdm_profile, illustrative_da_at_sc

ROOT = Path(__file__).resolve().parents[1]

class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.load(ROOT / "configs/reference_demo.json")

    def test_units_reference(self):
        c = coefficients(3, 1, 65, self.cfg)
        self.assertAlmostEqual(float(c["depth_m"]), 90e-6)
        self.assertAlmostEqual(float(c["diffusivity_m2_s"]), 1e-9)
        self.assertAlmostEqual(float(c["surface_rate_m_s"]), 1e-7)
        expected = 2e-7 / 30e-6 * (90e-6) ** 2 / 1e-9
        self.assertAlmostEqual(float(c["da"]), expected)

    def test_zero_sink_and_robin_limit(self):
        x = np.linspace(0, 1, 31)
        np.testing.assert_allclose(analytic_profile(x, 0), 1)
        np.testing.assert_allclose(analytic_profile(x, 0, 2), (3 - 2 * x) / 3)
        _, numerical = fdm_profile(0, 2, 31)
        np.testing.assert_allclose(numerical, (3 - 2 * x) / 3, atol=1e-12)

    def test_sech_and_large_phi_stable(self):
        for da in (0, .216, 1, 9, 100):
            self.assertAlmostEqual(float(analytic_profile(1, da)), 1 / np.cosh(np.sqrt(da)))
        result = analytic_profile(np.linspace(0, 1, 20), 1e6)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(result[0], 1)

    def test_extreme_robin_no_cancellation(self):
        actual = float(analytic_profile(1, 1e-18, 1e8))
        self.assertAlmostEqual(actual / (1 / (1 + 1e8)), 1.0, places=12)

    def test_robin_reduces_bottom_concentration(self):
        self.assertLess(analytic_profile(1, 1, .2), analytic_profile(1, 1, 0))

    def test_fdm_second_order_convergence(self):
        for da, bi in ((.2, 0), (4, 0), (4, .5)):
            errors = []
            for n in (51, 101, 201):
                x, u = fdm_profile(da, bi, n)
                errors.append(np.max(np.abs(u - analytic_profile(x, da, bi))))
            self.assertLess(errors[-1], 2e-5)
            self.assertGreater(errors[0] / errors[1], 3.8)
            self.assertGreater(errors[1] / errors[2], 3.8)

    def test_monotonicity_and_similarity(self):
        c = coefficients(np.array([1, 2, 3]), 1, 65, self.cfg)
        self.assertTrue(np.all(np.diff(analytic_profile(1, c["da"])) < 0))
        c1 = coefficients(2, 4, 65, self.cfg)
        c2 = coefficients(4, 1, 65, self.cfg)
        self.assertAlmostEqual(float(c1["da"]), float(c2["da"]))

    def test_sc_cutoff_is_algebraic(self):
        da = illustrative_da_at_sc(.9)
        self.assertAlmostEqual(float(analytic_profile(1, da)), .9)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): coefficients(1, 0, 65, self.cfg)
        with self.assertRaises(ValueError): coefficients(1, 1, -273.15, self.cfg)
        with self.assertRaises(ValueError): fdm_profile(-1)
        with self.assertRaises(ValueError): analytic_profile(1.1, 1)
        with self.assertRaises(ValueError): replace(self.cfg, diffusivity_ref_m2_s=-1)
        with self.assertRaises(ValueError): replace(self.cfg, n_samples=2)

if __name__ == "__main__": unittest.main()
