"""Replace only the defective ellipse solver; reuse the archived full protocol."""
from pathlib import Path
import json
import run_synchronized_real_pilot as original
from synchronized_ellipse_fixed import ellipse_phase

original.ellipse_phase = ellipse_phase
original.OUT = Path('/kaggle/working/sync_real_fixed')
original.main()
