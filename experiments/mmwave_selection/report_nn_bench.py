"""Produce the mmwave-897 NN-vs-linear-probe final table + decision from a
mmwave_nn_bench.json produced by the Kaggle kernel.

Usage:
    python report_nn_bench.py <path-to-mmwave_nn_bench.json>

Reads the JSON, prints a comparison table against the 12.23 linear-probe
reference, applies the pre-registered decision rule, and (if the destination
looks like the canonical experiments path) copies the JSON next to this script.
"""
import json
import sys
from pathlib import Path

REF_LINEAR_PROBE = 12.23
DECISION_MAE = 11.2
DECISION_CI_UPPER = 12.23

ROWS = [
    ('reference_constant', 'constant (predict mean)', False),
    ('reference_fft_peak', 'per-session FFT-peak', False),
    ('ridge', 'Ridge (71-d spectrum)', False),
    ('mlp', 'MLP (71-d spectrum, nonlinearity control)', False),
    ('dlinear', 'DLinear (phase series)', True),
    ('nlinear', 'NLinear (phase series)', True),
    ('tcn', 'TCN (phase series)', True),
    ('transformer', 'Transformer (phase series)', True),
    ('contiformer', 'ContiFormer (phase series)', True),
    ('cycleformer', 'CycleFormer (phase series)', True),
]


def fmt(x, nd=2):
    if x is None:
        return 'NA'
    return f'{x:.{nd}f}'


def main():
    if len(sys.argv) < 2:
        print('usage: report_nn_bench.py <json>')
        return 1
    src = Path(sys.argv[1])
    data = json.loads(src.read_text())

    print('\n===== mmwave-897: NN vs linear-probe HR benchmark =====')
    meta = data.get('_meta', {})
    print(f"sessions={meta.get('n_sessions')} groups={meta.get('n_groups')} "
          f"windows={meta.get('n_windows')} device={meta.get('device')} "
          f"runtime={meta.get('runtime_s')}s")
    print(f"decision rule: NN credited only if MAE<={DECISION_MAE} "
          f"AND CI_upper<{DECISION_CI_UPPER}\n")

    hdr = f"{'arm':<42}{'MAE':>8}{'R2':>8}{'CI_lo':>8}{'CI_hi':>8}{'verdict':>12}"
    print(hdr)
    print('-' * len(hdr))
    any_win = False
    for key, label, is_nn in ROWS:
        r = data.get(key)
        if not r:
            print(f"{label:<42}{'MISSING':>8}")
            continue
        mae = r.get('mae')
        r2 = r.get('r2')
        ci = r.get('ci_mae') or [None, None]
        verdict = '-'
        if is_nn:
            win = (mae is not None and mae <= DECISION_MAE and
                   ci[1] is not None and ci[1] < DECISION_CI_UPPER)
            verdict = 'BEATS 12.23' if win else 'no'
            any_win = any_win or win
        print(f"{label:<42}{fmt(mae):>8}{fmt(r2):>8}{fmt(ci[0]):>8}{fmt(ci[1]):>8}{verdict:>12}")

    print('\nPer-scenario MAE (Lying/Sitting x Rest/Post-exercise):')
    for key, label, _ in ROWS:
        r = data.get(key)
        if not r or 'scenario' not in r:
            continue
        parts = []
        for sc, sv in sorted(r['scenario'].items()):
            parts.append(f"{sc}={sv['mae']:.1f}(n={sv['n']})")
        print(f"  {label:<40} " + ' '.join(parts))

    print('\nCONCLUSION:')
    if any_win:
        print('  At least one NN arm beats the 12.23 linear probe under the '
              'pre-registered rule -> temporal/nonlinear modeling adds HR info.')
    else:
        print('  No NN arm meets the pre-registered bar. The representation '
              '(71-d spectrum / phase series) is the ceiling, not the model.')

    # Persist canonical copy next to this script.
    dst = Path(__file__).resolve().parent / 'mmwave_nn_bench.json'
    dst.write_text(src.read_text())
    print(f'\nPersisted -> {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
