"""Synthetic checks for direct phase and timestamp audit."""
import unittest
import numpy as np
from src.data.diagnose_bgt60_phase import direct_traces, alignment_audit, peak_candidates
from src.data.diagnose_bgt60_adc import spectral_metrics


class Checks(unittest.TestCase):
    def test_phase_frequency(self):
        t = np.arange(384)/30
        z = np.exp(1j*.1*np.sin(2*np.pi*1.25*t))
        for method in ['unwrap', 'edacm']:
            self.assertAlmostEqual(spectral_metrics(direct_traces(z)[method], 30)['fft_hr_bpm'], 75)

    def test_roi_excludes_dc(self):
        power = np.ones(32)
        power[0], power[1], power[12] = 1e9, 1e8, 100
        self.assertEqual(peak_candidates(power)[0], 12)

    def test_timestamps_preserve_reference_origin(self):
        ref = np.array([[10., 70., 12.], [11., 71., 12.]])
        audit = alignment_audit(ref, 18000, 30)
        self.assertEqual(audit['reference_first_time'], 10)
        self.assertFalse(audit['start_offset_verified'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
