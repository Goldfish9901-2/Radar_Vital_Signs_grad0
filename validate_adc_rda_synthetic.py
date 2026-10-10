"""CPU-only synthetic validation of the production ADC conversion functions."""
import ast
from pathlib import Path
from typing import Optional, Tuple
import unittest
import numpy as np

# Isolate production functions from optional dataset-loader dependencies.
source = Path(__file__).parent / "src/data/loaders/export_all_datasets.py"
names = {"_select_even_indices", "_crop_or_pad_last_axis", "_range_window_indices",
         "_build_rda_cube_from_frame", "convert_adc_cube_to_rda"}
tree = ast.parse(source.read_text(encoding="utf-8"))
module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name in names], type_ignores=[])
env = dict(np=np, Optional=Optional, Tuple=Tuple,
           TARGET_DOPPLER=8, TARGET_ANGLE=16, TARGET_RANGE=8)
exec(compile(module, str(source), "exec"), env)
convert = env["convert_adc_cube_to_rda"]


def adc(frames=1, chirps=64, samples=128, doppler=0, phase=None):
    fast = np.exp(2j*np.pi*12*np.arange(samples)/samples)
    slow = np.exp(2j*np.pi*doppler*np.arange(chirps)/chirps)
    if phase is None:
        phase = np.zeros(frames)
    return (np.exp(1j*phase)[:, None, None, None]
            * slow[None, None, :, None] * fast[None, None, None, :])


class SyntheticADCChecks(unittest.TestCase):
    def test_legacy_matches_previous_formula(self):
        rng = np.random.default_rng(4)
        frame = rng.normal(size=(3, 64, 128)) + 1j*rng.normal(size=(3, 64, 128))
        x = frame.astype(np.complex64)
        x = x - x.mean(axis=1, keepdims=True)
        x = np.fft.fft(x, axis=-1)[..., :64]
        x = np.fft.fftshift(np.fft.fft(x, n=8, axis=1), axes=1)
        x = np.fft.fftshift(np.fft.fft(x, n=16, axis=0), axes=0)
        actual = env["_build_rda_cube_from_frame"](
            frame, np.arange(64), np.arange(128), 8, 16)
        np.testing.assert_array_equal(actual, x.transpose(1, 0, 2))

    def test_range_and_doppler_peak(self):
        cube, bins, center = convert(adc(doppler=2), doppler_mode="full_fft_crop",
                                     clutter_mode="none")
        self.assertEqual(center, 12)
        peak = np.unravel_index(np.abs(cube[0]).argmax(), cube.shape[1:])
        self.assertEqual(peak[0], 6)
        self.assertEqual(bins[peak[2]], 12)

    def test_late_chirps_used(self):
        signal = adc()
        signal[:, :, :8] = 0
        old, _, _ = convert(signal, clutter_mode="none")
        full, _, _ = convert(signal, doppler_mode="full_fft_crop", clutter_mode="none")
        self.assertEqual(float(np.abs(old).max()), 0)
        self.assertGreater(float(np.abs(full).max()), 1)

    def test_frame_phase_and_heart_peak(self):
        fs, n, hr_hz = 20., 400, 1.2
        phase = .2*np.sin(2*np.pi*hr_hz*np.arange(n)/fs)
        cube, bins, _ = convert(adc(frames=n, phase=phase),
                                doppler_mode="full_fft_crop", clutter_mode="none")
        recovered = np.unwrap(np.angle(cube[:, 4, 8, np.flatnonzero(bins == 12)[0]]))
        np.testing.assert_allclose(recovered, phase, atol=1e-6)
        power = np.abs(np.fft.rfft(recovered-recovered.mean()))
        self.assertAlmostEqual(np.fft.rfftfreq(n, 1/fs)[power.argmax()]*60, 72.)
        removed, _, _ = convert(adc(frames=n, phase=phase),
                                doppler_mode="full_fft_crop", clutter_mode="chirp_mean")
        self.assertLess(float(np.abs(removed).max()), float(np.abs(cube).max())*1e-5)

    def test_odd_and_zero_padded_doppler(self):
        for chirps, target in [(9, 5), (3, 8)]:
            cube, _, _ = convert(adc(chirps=chirps), target_doppler=target,
                                 doppler_mode="full_fft_crop", clutter_mode="none")
            self.assertEqual(cube.shape, (1, target, 16, 8))
            self.assertEqual(np.abs(cube[0]).sum(axis=(1, 2)).argmax(), target//2)

    def test_invalid_arguments(self):
        for kwargs in [dict(doppler_mode="typo"), dict(clutter_mode="typo"),
                       dict(sampling_mode="typo"), dict(range_center_bin=-1),
                       dict(fast_time_dc="typo"),
                       dict(range_center_bin=64), dict(range_center_bin=1.5),
                       dict(target_doppler=0)]:
            with self.assertRaises(ValueError):
                convert(adc(), **kwargs)
        with self.assertRaises(ValueError):
            convert(np.empty((0, 1, 64, 128)))

    def test_fixed_roi_does_not_follow_stronger_reflector(self):
        signal = adc()
        stronger = 10*np.exp(2j*np.pi*25*np.arange(128)/128)
        signal = signal + stronger[None, None, None, :]
        _, automatic, center = convert(signal, clutter_mode="none")
        self.assertEqual(center, 25)
        cube, fixed, center = convert(signal, clutter_mode="none", range_center_bin=12)
        self.assertEqual(center, 12)
        self.assertNotIn(12, automatic)
        self.assertIn(12, fixed)
        self.assertGreater(np.abs(cube).max(), 1)

    def test_native_sampling_is_independent_of_doppler_policy(self):
        signal = adc(chirps=128, samples=512)
        signal[:, :, :8] = 0
        legacy, _, _ = convert(signal, clutter_mode="none", sampling_mode="native")
        full, _, center = convert(signal, clutter_mode="none", sampling_mode="native",
                                  doppler_mode="full_fft_crop")
        self.assertEqual(np.abs(legacy).max(), 0)
        self.assertEqual(center, 12)
        self.assertGreater(np.abs(full).max(), 1)

    def test_angle_peak_for_uniform_half_wavelength_array(self):
        # Spatial frequency 1/4 cycle per element -> shifted 16-bin FFT index 12.
        signal = adc()*np.exp(2j*np.pi*.25*np.arange(4))[None, :, None, None]
        cube, _, _ = convert(signal, doppler_mode="full_fft_crop", clutter_mode="none")
        self.assertEqual(np.abs(cube[0]).sum(axis=(0, 2)).argmax(), 12)

    def test_fast_time_dc_is_separate_from_chirp_mean(self):
        signal = adc() + 100
        contaminated, _, center = convert(signal, clutter_mode="none", sampling_mode="native")
        self.assertEqual(center, 0)
        clean, bins, center = convert(signal, clutter_mode="none", sampling_mode="native",
                                      fast_time_dc="mean")
        self.assertEqual(center, 12)
        peak = np.abs(clean[0]).sum(axis=(0, 1)).argmax()
        self.assertEqual(bins[peak], 12)
        self.assertGreater(np.abs(clean).max(), 1)
        removed, _, _ = convert(signal, clutter_mode="chirp_mean", fast_time_dc="mean")
        self.assertLess(np.abs(removed).max(), np.abs(clean).max()*1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
