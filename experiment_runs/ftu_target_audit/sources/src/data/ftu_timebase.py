"""Explicit physical FTU time mapping; offset must be supplied, never inferred from HR."""
import numpy as np


def frame_times(n_frames, sampling_rate_hz, *, radar_start_on_reference_clock_s):
    if not isinstance(n_frames,(int,np.integer)) or n_frames<1:
        raise ValueError('n_frames must be a positive integer')
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz<=0 or not np.isfinite(radar_start_on_reference_clock_s):
        raise ValueError('Require finite positive rate and explicit start offset')
    return float(radar_start_on_reference_clock_s)+np.arange(n_frames,dtype=np.float64)/sampling_rate_hz


def labels_at_frames(reference_time, heart_rate, radar_time):
    t=np.asarray(reference_time,dtype=np.float64); y=np.asarray(heart_rate,dtype=np.float64)
    rt=np.asarray(radar_time,dtype=np.float64)
    if t.ndim!=1 or y.shape!=t.shape or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError('Require matching finite reference timestamps and HR')
    if np.any(np.diff(t)<=0): raise ValueError('Reference timestamps must be strictly increasing')
    if rt.ndim!=1 or not np.isfinite(rt).all() or np.any(np.diff(rt)<=0):
        raise ValueError('Radar timestamps must be finite and strictly increasing')
    return np.interp(rt,t,y,left=np.nan,right=np.nan)
