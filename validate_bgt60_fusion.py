import unittest
import numpy as np
from scipy.signal import detrend
from src.data.evaluate_bgt60_fusion import fuse_phase,hr_correlation_phase
from src.data.evaluate_bgt60_selection import fft_hr,phase_fft_hr


class Checks(unittest.TestCase):
    def test_projection_keeps_known_hr(self):
        t=np.arange(384)/30
        phase=(10*np.sin(2*np.pi*.3125*t)+np.sin(2*np.pi*1.25*t))[:,None]
        self.assertEqual(phase_fft_hr(hr_correlation_phase(phase)[:,0]),75)
    def test_hr_correlation_rejects_shared_low_frequency(self):
        t=np.arange(384)/30
        low=10*np.sin(2*np.pi*.3125*t)
        x=np.stack([low+np.sin(2*np.pi*1.25*t),low+np.sin(2*np.pi*2.03125*t)],axis=1)
        _,full=fuse_phase(x,0)
        fused,hr=fuse_phase(x,0,hr_correlation_phase(x))
        self.assertEqual(full['contributors'],[0,1])
        self.assertEqual(hr['contributors'],[0])
        np.testing.assert_array_equal(fused,x[:,0])
    def test_positive_fusion_preserves_frequency(self):
        t=np.arange(384)/30
        p=detrend(.1*np.sin(2*np.pi*1.25*t))
        fused,detail=fuse_phase(np.stack([p,2*p,-p],axis=1),0)
        self.assertEqual(detail['contributors'],[0,1])
        self.assertAlmostEqual(sum(detail['weights']),1)
        self.assertEqual(phase_fft_hr(fused),75)

    def test_no_correlation_falls_back(self):
        t=np.arange(384)/30
        phase=detrend(np.stack([np.sin(2*np.pi*1.25*t),np.sin(2*np.pi*2*t)],axis=1),axis=0)
        fused,detail=fuse_phase(phase,0)
        np.testing.assert_array_equal(fused,phase[:,0])
        self.assertEqual(detail['contributors'],[0])

    def test_constant_phase_is_invalid(self):
        fused,_=fuse_phase(np.zeros((384,5)),2)
        self.assertIsNone(phase_fft_hr(fused))

    def test_shared_fft_matches_single(self):
        t=np.arange(384)/30
        z=np.exp(1j*(3+.1*np.sin(2*np.pi*1.25*t)))
        self.assertEqual(fft_hr(z),phase_fft_hr(detrend(np.unwrap(np.angle(z)))))


if __name__=='__main__':unittest.main(verbosity=2)
