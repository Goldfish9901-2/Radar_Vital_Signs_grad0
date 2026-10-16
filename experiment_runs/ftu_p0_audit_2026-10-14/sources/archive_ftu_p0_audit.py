"""Freeze P0 evidence and source bytes without touching the earlier audit archive."""
import json
from pathlib import Path
import shutil
from src.data.audit_ftu_p0 import digest, write_json


def main():
    out=Path('experiment_runs/ftu_p0_audit_2026-10-14')
    remote=Path('tmp/ftu_p0_result')
    stage=Path('tmp/ftu_p0_audit_upload')
    result=json.loads((out/'remote_summary.json').read_text())
    assert digest('src/data/audit_ftu_p0.py')==result['source_sha256']
    assert digest('summarize_ftu_p0_audit.py')==json.loads((out/'validation.json').read_text())['source_sha256']
    assert digest('validate_ftu_p0_audit.py')==json.loads((out/'numerical_validation.json').read_text())['source_sha256']
    shipped=json.loads((stage/'source_manifest.json').read_text())
    for rel,expected in shipped.items():
        assert digest(rel)==expected, rel
    sources=list(shipped)+['summarize_ftu_p0_audit.py','validate_ftu_p0_audit.py',
                          'kaggle/prepare_ftu_p0_audit.py','archive_ftu_p0_audit.py']
    for rel in sources:
        target=out/'sources'/rel; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(rel,target)
    for source,name in [(stage/'source_manifest.json','kaggle_source_manifest.json'),
                        (stage/'kernel-metadata.json','kernel-metadata.json'),
                        (remote/'kernel.log','kernel.log')]:
        shutil.copyfile(source,out/name)
    write_json(out/'gates.json',dict(mode_independence='FAIL',time_alignment='UNVERIFIED',
        retraining_allowed=False,training_launched=False,
        mode_reason='All 3520 exported windows have seven identical channels and all-one weights; numerical clipping reproduced.',
        time_reason='Endpoint-stretched labels; no independently established cross-device synchronization; 28 model records lack raw configuration logs.'))
    manifest={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*'))
              if p.is_file() and p.name!='artifact_manifest.json'}
    write_json(out/'artifact_manifest.json',manifest)
    for rel,expected in manifest.items(): assert digest(out/rel)==expected
    print(f'Archived and hash-verified {len(manifest)} artifacts; P1 remains disabled.')


if __name__=='__main__': main()
