"""CPU-only native range/phase audit on Kaggle, without VMD or fitting."""
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    work = Path('/kaggle/working')
    if not work.is_dir() or not Path('/kaggle/input').is_dir():
        raise RuntimeError('Kaggle only')
    archive = next(Path('/kaggle/input').rglob('radar_vital_signs_code.zip'))
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(work / 'code')
    code = work / 'code/Radar_Vital_Signs_grad0'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
    subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system',
                    'numpy', 'scipy', 'pandas', 'h5py'], check=True)
    subprocess.run([sys.executable, 'validate_bgt60_phase.py'], cwd=code, check=True)
    subprocess.run([sys.executable, '-m', 'src.data.diagnose_bgt60_phase',
                    '--input-root', '/kaggle/input', '--output-dir', work / 'phase_audit'], cwd=code, check=True)


if __name__ == '__main__':
    main()
