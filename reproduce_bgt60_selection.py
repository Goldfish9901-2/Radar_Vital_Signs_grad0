"""Reproduce the frozen selection experiment locally; no model training."""
import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time


def compare(actual, expected, path='root'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys()==expected.keys(), path
        for key in expected:compare(actual[key],expected[key],f'{path}.{key}')
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected), path
        for index,(a,e) in enumerate(zip(actual,expected)):compare(a,e,f'{path}[{index}]')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert isinstance(actual,(int,float)) and math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-8),path
    else:
        assert actual==expected,path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=Path('tmp/bgt60_p03_pair_v2_results/pair_audit'))
    parser.add_argument('--reference-dir',type=Path,default=Path('tmp/bgt60_p03_selection_results/selection_audit'))
    parser.add_argument('--output-dir',type=Path,default=Path('tmp/bgt60_p03_selection_local'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    started=time.perf_counter()
    subprocess.run([sys.executable,str(root/'validate_bgt60_selection.py')],cwd=root,check=True)
    test_seconds=time.perf_counter()-started
    calc_start=time.perf_counter()
    subprocess.run([sys.executable,'-m','src.data.evaluate_bgt60_selection',
                    '--input-root',str(args.input_dir.resolve()),'--output-dir',str(args.output_dir.resolve())],cwd=root,check=True)
    calc_seconds=time.perf_counter()-calc_start
    checked=[]
    for name in ['protocol.json','input_provenance.json','summary.json','prespecified_controls.json',
                 'p03_0.3m_selections.json','p03_0.6m_selections.json']:
        actual=json.loads((args.output_dir/name).read_text())
        expected=json.loads((args.reference_dir/name).read_text())
        compare(actual,expected,name)
        checked.append(name)
    import numpy, scipy
    result={'python':platform.python_version(),'platform':platform.system(),
            'numpy':numpy.__version__,'scipy':scipy.__version__,
            'tests_seconds':test_seconds,'compute_seconds':calc_seconds,'total_seconds':time.perf_counter()-started,
            'files_matched':checked,'float_tolerance':1e-8,'training':False,
            'local_policy':'known <=5 minute analyses stay local with uv'}
    (args.output_dir/'reproduction_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
