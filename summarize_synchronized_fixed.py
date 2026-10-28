"""Archive fixed outputs, validate invariants, and report calibration failures."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('experiment_runs/synchronized_real_pilot_fixed')
OLD = Path('experiment_runs/synchronized_real_pilot')
DATA = Path('tmp/sync_fixed_results/sync_real_fixed')


def table(frame, columns):
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append('| '+' | '.join('NA' if pd.isna(v) else (f'{v:.3e}' if abs(v)>1e6 or 0<abs(v)<1e-3 else f'{v:.4f}') if isinstance(v, float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    began = time.perf_counter()
    for path in DATA.iterdir():
        if path.suffix in ['.csv', '.json']: shutil.copyfile(path, ROOT/path.name)
        elif path.name.endswith('_phase20.npz'):
            (ROOT/'signals').mkdir(exist_ok=True); shutil.copyfile(path, ROOT/'signals'/path.name)
    shutil.copyfile(OLD/'protocol.json', ROOT/'protocol.json')
    sources = ['synchronized_ellipse_fixed.py', 'validate_synchronized_ellipse_fixed.py', 'run_synchronized_real_pilot_fixed.py',
               'run_synchronized_real_pilot.py', 'verify_synchronized_real_pilot.py', 'summarize_synchronized_fixed.py',
               'prepare_synchronized_fixed.py', 'build_ecg_only_review.py', 'freeze_ecg_only_decisions.py', 'kaggle/prepare_synchronized_fixed.py']
    for name in sources:
        target = ROOT/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
    (ROOT/'source_hashes.json').write_text(json.dumps({name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}, indent=2))
    # Independent existing verifier recomputes cycle-integral reference, periodogram and closed-form metrics.
    import verify_synchronized_real_pilot as verifier
    verifier.ROOT, verifier.DATA = ROOT, DATA
    verifier.main()
    repair = json.loads((ROOT/'repair_protocol.json').read_text())
    for name, digest in repair['source_hashes'].items(): assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    old_manifest = json.loads((OLD/'artifact_manifest.json').read_text())
    assert all(hashlib.sha256((OLD/name).read_bytes()).hexdigest() == digest for name, digest in old_manifest.items())
    old_acquisition = {x['name']: x for x in json.loads((OLD/'acquisition.json').read_text())}
    new_acquisition = json.loads((ROOT/'acquisition.json').read_text())
    assert len(new_acquisition) == 6 and all(x['sha256'] == old_acquisition[x['name']]['sha256'] for x in new_acquisition)
    for path in DATA.glob('*_beats.csv'):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256((OLD/path.name).read_bytes()).hexdigest()
    downloaded_repair = Path('tmp/sync_fixed_results/frozen_fixed_code/repair_protocol.json')
    # Launcher serializes text on Windows then writes on Linux: normalize newline transport only.
    assert downloaded_repair.read_text() == (ROOT/'repair_protocol.json').read_text()
    frame = pd.read_csv(ROOT/'window_predictions.csv')
    old_frame = pd.read_csv(OLD/'window_predictions.csv')
    merged = frame.merge(old_frame, on=['subject','scenario','start_s','end_s','center_s','reference_offset_s'], suffixes=('_fixed','_old'), validate='one_to_one')
    assert len(merged) == len(frame)
    assert np.allclose(merged.reference_fixed, merged.reference_old, atol=1e-12, equal_nan=True)
    assert np.all(merged.radar_valid_fixed == merged.radar_valid_old)
    reviews = json.loads((ROOT/'ecg_review_decisions.json').read_text())['records']
    maximum_review_error = 0.
    for review in reviews:
        stem = review['record']; pid = stem.split('_')[0]; scenario = 'Resting' if 'Resting' in stem else 'Valsalva'
        beats = pd.read_csv(ROOT/'reviewed_beats'/f'{stem}.csv').beat_time_s.to_numpy()
        old_beats = pd.read_csv(OLD/f'{stem}_lead{review["primary_lead"]}_beats.csv').beat_time_s.to_numpy()
        old_beats = old_beats[(old_beats >= review['review_start_s'])&(old_beats < review['review_end_s'])]
        assert np.array_equal(beats, old_beats)
        for row in frame[(frame.subject == pid)&(frame.scenario == scenario)].itertuples():
            a, b = row.start_s+row.reference_offset_s, row.end_s+row.reference_offset_s
            assert a >= beats[0] and b <= beats[-1]
            hr = 60*(np.interp(b, beats, np.arange(len(beats)))-np.interp(a, beats, np.arange(len(beats))))/(b-a)
            maximum_review_error = max(maximum_review_error, abs(hr-row.reference))
            assert abs(hr-row.reference) < 1e-9
    metrics = pd.read_csv(ROOT/'record_metrics.csv')
    old_metrics = pd.read_csv(OLD/'record_metrics.csv')
    primary = metrics[metrics.reference_offset_s == 0].copy()
    comparisons = primary.merge(old_metrics[old_metrics.reference_offset_s == 0][['subject','scenario','mae']], on=['subject','scenario'], suffixes=('_fixed','_old'))
    comparisons.to_csv(ROOT/'paired_primary_metrics.csv', index=False)
    records = json.loads((ROOT/'records.json').read_text()); errors = json.loads((ROOT/'errors.json').read_text())
    old_records = json.loads((OLD/'records.json').read_text())
    coverage = []
    for member in repair['members']:
        stem = Path(member['name']).stem; pid = stem.split('_')[0]; scenario = 'Resting' if 'Resting' in stem else 'Valsalva'
        original = next(r for r in old_records if r['subject'] == pid and r['scenario'] == scenario)
        current = next((r for r in records if r['subject'] == pid and r['scenario'] == scenario), None)
        if current is None:
            assert not (DATA/f'{stem}_phase20.npz').exists()
            assert not ((frame.subject == pid)&(frame.scenario == scenario)).any()
            assert any(e['record'] == stem+'.mat' for e in errors)
        coverage.append({'record': stem, 'calibration': 'valid' if current else 'failed; no HR',
                         'old_center_norm': float(np.linalg.norm(original['ellipse_center'])),
                         'old_radial_rmse': original['calibration_radial_rmse'],
                         'old_mapped_spread_upper_bound': float(np.sqrt(2)*np.linalg.norm(original['ellipse_transform'])),
                         'fixed_radial_rmse': current['calibration_radial_rmse'] if current else np.nan,
                         'fixed_mapped_largest_std': current['mapped_centered_singular_values'][0] if current else np.nan})
    coverage = pd.DataFrame(coverage); coverage.to_csv(ROOT/'calibration_coverage.csv', index=False)
    independent = {'elapsed_seconds': time.perf_counter()-began, 'same_raw_file_hashes': 6,
                   'same_reference_and_windows_rows': len(merged), 'maximum_reviewed_reference_difference_bpm': maximum_review_error,
                   'original_artifact_hashes_preserved': len(old_manifest), 'no_hr_for_failed_calibrations': len(errors),
                   'reviewed_records': len(reviews), 'peak_edits': 0, 'source_hashes_match_frozen_repair': True,
                   'repair_protocol_text_identical_after_newline_transport': True,
                   'downloaded_repair_byte_sha256': hashlib.sha256(downloaded_repair.read_bytes()).hexdigest(),
                   'frozen_repair_byte_sha256': hashlib.sha256((ROOT/'repair_protocol.json').read_bytes()).hexdigest()}
    (ROOT/'repair_invariant_validation.json').write_text(json.dumps(independent, indent=2))
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for index, pid in enumerate(['GDN0002','GDN0006','GDN0007']):
        for column, scenario in enumerate(['Resting','Valsalva']):
            ax = axes[index, column]
            old = old_frame[(old_frame.subject == pid)&(old_frame.scenario == scenario)&(old_frame.reference_offset_s == 0)]
            g = frame[(frame.subject == pid)&(frame.scenario == scenario)&(frame.reference_offset_s == 0)]
            t = old.center_s-old.start_s.min()
            ax.plot(t, old.reference, color='black', label='Reviewed primary ECG')
            ax.plot(t, old.prediction, color='gray', linestyle=':', alpha=.65, label='Archived defective solver')
            if len(g): ax.plot(g.center_s-old.start_s.min(), g.prediction, label='Fixed solver FFT', color='tab:blue')
            else: ax.text(.5, .82, 'Calibration failed: no corrected HR', ha='center', transform=ax.transAxes, color='tab:red')
            ax.set_title(pid+' / '+scenario); ax.set_ylabel('HR (bpm)'); ax.set_xlabel('Window center relative to scoring start (s)'); ax.grid(alpha=.2)
    fig.suptitle('Same six records: numeric repair; failed calibrations produce no HR', y=.995)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.965), ncol=3)
    fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(ROOT/'paired_trajectories.png', dpi=150); plt.close(fig)
    execution = json.loads((ROOT/'execution.json').read_text())
    analytic = json.loads((ROOT/'analytic_validation.json').read_text())
    report = f'''# 椭圆校准数值修复：同六条复跑与ECG-only审查

归档完成日期2026-10-28。修复冻结`0e745ca`；ECG-only结论冻结`47153ed`，此后才下载/查看修正版雷达结果。原协议、结果和54项产物哈希全部保留。其余27名受试者未访问，无训练。

**原实现存在可复现的数值错误。此前五条记录的退化校准输出不能作为有效前端性能证据。修复后2条有效、4条按原残差上限校准失败，不输出HR。** 有效的GDN0002恢复段MAE由11.73降至1.37bpm，但尚无充分的常数/趋势外动态收益；不能用这个低MAE宣称跟踪成功。

## 数值错误与解析验证

原代码只保留广义特征向量，丢弃特征值；奇异约束矩阵产生的无穷特征值候选因此进入椭圆评选。其中心可达1e15–1e16，变换几乎为零，将整个校准点云压到单位圆上一点，径向误差却接近机器零。错误由实现造成，不应因原协议冻结而保留。

修复保留广义特征值齐次alpha/beta，排除非有限/近零beta、复特征对、近零二次项、非正定或退化level；检查特征对后向误差，并补充映射点云中心化奇异值、圆方差、相角跨度检查。原0.2径向残差上限、轴比100上限及最初30秒校准不变。不放宽失败条件，不改拟合目标或用HR选解。

解析测试本地uv环境{analytic['elapsed_seconds']:.4f}秒：完整椭圆相位RMSE1.36e-14rad，部分弧5.68e-15rad；固定带噪椭圆中原实现相位RMSE254.626rad、径向残差1.11e-16，修复后相位RMSE0.002086rad。恒点、直线、非有限输入和“单位圆上单点”全部拒绝。最后一例径向误差正好为零，直接证明只查径向残差不足。所有门槛在修复后真实评分前冻结，是数值支持护栏，不保证椭圆可辨识或生理正确。

## 同六条的校准覆盖

{table(coverage, ['record','calibration','old_center_norm','old_radial_rmse','old_mapped_spread_upper_bound','fixed_radial_rmse','fixed_mapped_largest_std'])}

旧映射spread上界为sqrt(2)*变换Frobenius范数：校准输入两轴经逐轴标准化后总方差为2，因此可在不重读旧原始I/Q的情况下界定映射点云变化。五个退化输出的上界接近数值零。四条失败均为**有限候选的径向残差超过原0.2上限**，无穷候选已排除；失败不等于原雷达没有心跳。校准覆盖2/6只描述工程可用性，不是通过记录数配额。

## ECG-only技术质量裁定

全部主导联保持原映射：GDN0002/0007用ECG2，GDN0006用ECG1。只用原始主导联、自动峰、固定时间定义制作24页材料，逐10秒条带检查全部184秒/记录，共1104秒，并检查边界。

六条均未发现明确的主导联漏检/误检，未增删峰、未换导联。GDN0002静息188–208秒有基线漂移/瞬态，但窄QRS仍可辨且与标记对应；GDN0007窄QRS与后续宽T波可分。全部页、逐记录说明和reviewed_beats随报告归档。

**这是助手完成的研究级技术目视审查，不是独立临床专家裁定。** 修正版雷达输出在审查结论提交前未打开，但审查者此前看过缺陷版预测，不能声称对旧研究回溯完全盲法，也不声称得到逐搏临床金标准。上一轮双导联不一致只说明需要查参考，不能单独否定清晰的主导联。本轮因此接受指定主导联用于有限窗口平均HR分析，不判定副导联所有异常的机制。

保留原自动峰对照；审查未改峰，reviewed与automatic在共同支持上目标差最大{maximum_review_error:.3g}bpm。审查接受并非重新运行另一个检测器来追求更好雷达分数。

## 冻结评分的有效记录（0秒）

{table(comparisons, ['subject','scenario','windows_scored','mae_old','mae_fixed','mean_oracle_mae','median_oracle_mae','trend_oracle_mae','change_rmse','constant_change_rmse','detrended_rmse','trend_detrended_rmse','detrended_r','detrended_r2'])}

恢复段虽比均值oracle1.415bpm略好（MAE1.367），仍差于中位数1.338、时间趋势1.170。去趋势相关性0.674为正，但残差RMSE3.068高于趋势零残差基线1.577，R²=-2.785；窗间变化RMSE3.170也高于常数0.447。说明相关方向与幅度准确性不能混为一谈。静息段13.905bpm与旧版相同，因该记录原先选择的是有效有限解。

其余四条无有效校准，不填常数、不沿用旧预测、不计算伪造HR误差。有效两条均来自同一受试者，不能用条件MAE当六记录/三受试者整体性能或跨受试者证据。78个重叠窗口不是78个独立样本；不计算窗口级显著性。

![修复前后与全部失败记录](paired_trajectories.png)

## 固定时间偏移

{table(metrics, ['subject','scenario','reference_offset_s','mae','detrended_rmse','detrended_r2'])}

仍为0秒主结果、±0.5秒全部报告，不选最优偏移。这不是测得的同步误差界；公开Gold-code同步方法仍有依据，但删除的原同步序列使逐记录剩余误差无法独立约束。ECG质量审查不解决这个问题。

## 核验与决策

Kaggle CPU下载+复跑{execution['elapsed_seconds']:.3f}秒；本地独立不变量/评分核验{independent['elapsed_seconds']:.3f}秒。六个原文件SHA256与旧版一致；有效记录窗口/偏移/主导联参考逐行一致；独立cycle积分与periodogram复算468个读数，失败记录无HR输出。原54个产物哈希均未改变。执行源码与修复冻结哈希一致。

本轮修复与研究级参考审查已完成，**暂不进入其余27名受试者验证**：当前固定校准只支持2/6记录，有效记录尚未恢复幅度正确的动态；数值修复不等于前端已具备稳定适用性。下一步若研究校准对真实I/Q的适用条件，继续限于这六条、先冻结诊断，不降低0.2上限追求覆盖，不同时搜索读数/标签/峰值规则。进入独立受试者前还需可追溯的参考质量流程与同步误差预算。
'''
    (ROOT/'REPORT.md').write_text(report, encoding='utf-8')
    (ROOT/'decision.json').write_text(json.dumps({'calibration_valid': 2, 'calibration_failed_no_hr': 4,
        'technical_ecg_review_complete': True, 'expert_clinical_adjudication': False,
        'peak_edits': 0, 'old_outputs_preserved': True, 'remaining27_accessed': False,
        'enter_remaining27_validation': False, 'tracking_success_claim': False,
        'reason': 'Limited calibration coverage and no amplitude-correct dynamic recovery; clock error budget unresolved'}, indent=2))
    manifest = {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json'}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(comparisons[['subject','scenario','mae_old','mae_fixed','detrended_r2']].to_string(index=False))


if __name__ == '__main__': main()
