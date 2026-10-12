import unittest
import numpy as np
import json
from analyze_bgt60_spectral_attribution import analyze_phase,harmonic_mask,DF


class Checks(unittest.TestCase):
    def test_known_respiration_and_harmonic(self):
        t=np.arange(384)/30
        phase=2*np.sin(2*np.pi*.3125*t)+.2*np.sin(2*np.pi*.9375*t)+.05*np.sin(2*np.pi*1.25*t)
        row,_=analyze_phase(phase)
        self.assertEqual(row['radar_rr_hz'],.3125)
        self.assertEqual(row['prediction_bpm'],56.25)
        self.assertTrue(row['prediction_near_harmonic'])
        self.assertFalse(row['prediction_near_shifted_comb'])
        json.dumps(row,allow_nan=False)

    def test_tolerance_boundary(self):
        mask=harmonic_mask(np.array([.9375+DF,.9375+DF+.0001]),.3125)
        self.assertEqual(mask.tolist(),[True,False])

    def test_zero_signal_invalid(self):
        row,_=analyze_phase(np.zeros(384))
        self.assertIsNone(row['prediction_bpm'])
        self.assertIsNone(row['radar_rr_hz'])


if __name__=='__main__':unittest.main(verbosity=2)
