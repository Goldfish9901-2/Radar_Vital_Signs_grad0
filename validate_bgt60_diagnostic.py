"""CPU checks for range calibration, DC removal and phase diagnostics."""
import unittest
import numpy as np
from src.data.diagnose_bgt60_adc import distance_axis, range_profiles, phase_diagnostics, spectral_metrics


class DiagnosticsChecks(unittest.TestCase):
    def test_calibration_requires_slope_and_provenance(self):
        self.assertIsNone(distance_axis(512, None))
        calibration = dict(sample_rate_hz=2e6, chirp_slope_hz_per_s=2e13, source="synthetic")
        axis = distance_axis(512, calibration)
        self.assertAlmostEqual(axis[1], 299792458*2e6/(2*2e13*512))
        with self.assertRaises(ValueError):
            distance_axis(512, {**calibration, "source": ""})
        with self.assertRaises(ValueError):
            distance_axis(512, {**calibration, "chirp_slope_hz_per_s": -1})

    def test_fast_time_dc_removes_bias_and_preserves_range_tone(self):
        adc = np.broadcast_to(100+np.cos(2*np.pi*12*np.arange(128)/128), (5, 3, 4, 128))
        profiles = range_profiles(adc)
        self.assertEqual(profiles["none"].argmax(), 0)
        self.assertEqual(profiles["mean"].argmax(), 12)
        self.assertLess(profiles["mean"][0], 1e-20)
        np.testing.assert_allclose(profiles["mean"][12], profiles["none"][12], rtol=1e-12)

    def test_zero_features_do_not_generate_fft_hr(self):
        metrics, probe = phase_diagnostics(np.ones((256, 1, 1, 1), dtype=np.complex64))
        self.assertFalse(metrics["feature_nonzero"])
        self.assertIsNone(metrics["fft_hr_bpm"])
        self.assertEqual(metrics["phase_variance"], 0)
        self.assertEqual(np.count_nonzero(probe), 0)

    def test_ideal_periodic_phase_retains_hr_peak(self):
        phase = .2*np.sin(2*np.pi*1.25*np.arange(256)/20)
        cube = np.exp(1j*phase)[:, None, None, None].astype(np.complex64)
        metrics, _ = phase_diagnostics(cube)
        self.assertTrue(metrics["feature_nonzero"])
        self.assertAlmostEqual(metrics["fft_hr_bpm"], 75)
        self.assertGreater(metrics["hr_band_energy"], 0)
        self.assertIsNone(spectral_metrics(np.zeros(256))["fft_hr_bpm"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
