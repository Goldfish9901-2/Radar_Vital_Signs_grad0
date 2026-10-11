import unittest
import numpy as np
from src.data.evaluate_bgt60_selection import select_channels, fft_hr


class Checks(unittest.TestCase):
    def test_quality_rejects_amplitude_nulls(self):
        z=np.ones((384,3,5),complex)
        z[:,0,0]=10
        z[:10,0,0]=1e-9
        choices,_=select_channels(z,np.array([0]))
        self.assertEqual(choices['energy'],[0])
        self.assertNotEqual(choices['quality'],[0])

    def test_causal_and_tie_stable(self):
        z=np.ones((768,3,5),complex)
        short,_=select_channels(z,np.array([0,192]))
        long,_=select_channels(z,np.array([0,192,384]))
        self.assertEqual(short['quality'],long['quality'][:2])
        self.assertEqual(long['quality'],[0,0,0])

    def test_fft_matches_known_phase(self):
        t=np.arange(384)/30
        self.assertEqual(fft_hr(np.exp(1j*.1*np.sin(2*np.pi*1.25*t))),75)
        self.assertIsNone(fft_hr(np.ones(384,complex)))


if __name__=='__main__':unittest.main(verbosity=2)
