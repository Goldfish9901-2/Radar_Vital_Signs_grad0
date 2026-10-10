"""CPU-only native range/phase audit on Kaggle, without VMD or fitting."""
from pathlib import Path
import subprocess
import sys
import zipfile


def resolve_code(input_root, work):
    # Kaggle can unpack uploaded ZIP datasets into a directory tree.
    archives = list(input_root.rglob('radar_vital_signs_code.zip'))
    if archives:
        with zipfile.ZipFile(archives[0]) as bundle:
            bundle.extractall(work / 'code')
        candidates = [work / 'code/Radar_Vital_Signs_grad0']
    else:
        candidates = [p.parent for p in input_root.rglob('validate_bgt60_phase.py')]
    for code in candidates:
        if (code / 'src/data/diagnose_bgt60_phase.py').is_file():
            return code
    raise RuntimeError('Mounted code dataset lacks direct-phase diagnostic files; check dataset_sources and version')


def main():
    work = Path('/kaggle/working')
    if not work.is_dir() or not Path('/kaggle/input').is_dir():
        raise RuntimeError('Kaggle only')
    code = resolve_code(Path('/kaggle/input'), work)
    print('Resolved code root:', code, flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
    subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--system',
                    'numpy', 'scipy', 'pandas', 'h5py'], check=True)
    subprocess.run([sys.executable, 'validate_bgt60_phase.py'], cwd=code, check=True)
    subprocess.run([sys.executable, '-m', 'src.data.diagnose_bgt60_phase',
                    '--input-root', '/kaggle/input', '--output-dir', work / 'phase_audit'], cwd=code, check=True)


if __name__ == '__main__':
    main()
