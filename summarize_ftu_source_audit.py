"""Archive and independently verify the downloaded frozen-model audit artifacts."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.audit_predictions import sha256
from src.data.audit_ftu_continuous import power_windows

start = time.perf_counter()
root = Path('experiment_runs/ftu_source_audit_2026-10-13')
source = Path('tmp/ftu_audit_discovery/e2a_output')
evidence = root/'e2a_evidence'
evidence.mkdir(parents=True,exist_ok=True)
shutil.copy2('tmp/ftu_audit_discovery/e2a/nn-bench-e2a-hr-adavmd-proposed-ftu.py',
             evidence/'source_notebook.py')
for rel in ('preds_cycleformer_FTU_test.csv','training_exports/manifest.csv',
            'training_exports/build_config.json','model_outputs/cycleformer/run_config.json',
            'model_outputs/cycleformer/eval_test_FTU.json'):
    dst = evidence/rel; dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/rel,dst)
controls_dir = root/'e2a_controls'
shutil.copytree('tmp/ftu_source_audit_result/ftu_audit_controls',controls_dir,dirs_exist_ok=True)
shutil.copy2('tmp/ftu_source_audit_upload/source_manifest.json',controls_dir/'source_manifest.json')
shutil.copy2('tmp/ftu_source_audit_upload/kernel-metadata.json',controls_dir/'kernel-metadata.json')
shutil.copy2('tmp/ftu_source_audit_result/ftu-source-signal-use-audit.log',controls_dir/'kernel.log')
pred = pd.read_csv(controls_dir/'predictions.csv')
scores = pd.read_csv(controls_dir/'controls.csv')
meta = json.loads((controls_dir/'metadata.json').read_text())
audit = json.loads((root/'e2a_cycleformer/summary.json').read_text())
assert len(pred) == 704*10*3
assert not pred.duplicated(['seed','condition','dataset','group_key','sample_tag','window_start']).any()
max_mae_error = 0.; max_donor_error = 0.
for (seed, condition), part in pred.groupby(['seed','condition']):
    assert len(part) == 704
    recomputed = np.mean(abs(part.pred_bpm-part.label_bpm))
    row = scores.loc[(scores.seed == seed) & (scores.condition == condition) & (scores.group == 'ALL')].iloc[0]
    max_mae_error = max(max_mae_error,abs(recomputed-row.mae))
    assert np.isclose((abs(part.pred_bpm-part.label_bpm)-
                       abs(part.reference_pred_bpm-part.label_bpm)).mean(), row.delta_mae,atol=1e-10)
    if condition.startswith('replace'):
        assert (part.group_key != part.donor_group).all()
    if condition == 'replace_both':
        original = pred.loc[(pred.seed == seed) & (pred.condition == 'original')]
        expected = original.pred_bpm.to_numpy()[part.donor_index.to_numpy()]
        max_donor_error = max(max_donor_error,float(max(abs(expected-part.pred_bpm))))
assert max_mae_error < 1e-10 and max_donor_error < 1e-3
overall = scores.loc[scores.group == 'ALL']
summary = overall.groupby(['condition','reference'],sort=False).agg(
    mae=('mae','mean'),delta_mae=('delta_mae','mean'),delta_min=('delta_mae','min'),
    delta_max=('delta_mae','max'),prediction_change=('mean_abs_prediction_change','mean')).reset_index()
summary.to_csv(controls_dir/'controls_summary.csv',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plot = summary.loc[~summary.condition.isin(['original','reencoded'])]
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(plot.condition,plot.delta_mae,color=['#b14c42' if v > 0 else '#398080' for v in plot.delta_mae])
ax.errorbar(plot.delta_mae,plot.condition,
            xerr=np.array([plot.delta_mae-plot.delta_min,plot.delta_max-plot.delta_mae]),
            fmt='none',color='black',capsize=3)
ax.axvline(0,color='gray',lw=1)
ax.set_xlabel('Paired delta MAE (bpm); bars: seed mean, whiskers: seed range')
fig.tight_layout();fig.savefig(controls_dir/'delta_mae.png',dpi=160);plt.close(fig)
validation = dict(elapsed_sec=time.perf_counter()-start,rows=len(pred),
    max_recomputed_mae_error=max_mae_error,max_paired_donor_prediction_error=max_donor_error,
    same_test_windows=True,unique_rows=True,cross_subject_donors=True,
    inference_cpu_seconds=meta['elapsed_sec'],local_prediction_audit_seconds=audit['elapsed_sec'])
(root/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
o = audit['overall']
text = '''# FTU 源模型审计（2026-10-13）

已完成找回的 E2a CycleFormer 的预测审计和冻结模型特征破坏对照；没有训练。
它显示明显的预测收缩，未显示有效组内 HR 跟踪。此结论只适用于这份 checkpoint。

## 身份和未完成项

- 原始目标：历史 CycleFormer / v3 small / Transformer，FTU MAE 分别为
  4.989 / 3.425 / 4.034。尚未定位这三个 checkpoint，不能声称已复算。
- 已审计补充模型：[E2a notebook](https://www.kaggle.com/code/goldfish9901/nn-bench-e2a-hr-adavmd-proposed-ftu)。
  FTU-only，训练 2112 窗口、测试 704 窗口（p01/p08 各 352）。
- [冻结推理 notebook](https://www.kaggle.com/code/goldfish9901/ftu-source-signal-use-audit)，v1，私有 CPU。
- 公有/私有列表及 notebook 输出的查找清单在 `discovery/`。
  仅有 RDA 导出的 dataset 不是模型 checkpoint。检索错误明确记录，不能视作无产物。

## 预测审计

| 指标 | 结果 |
|---|---:|
'''
for name,value in [('模型 MAE',o['mae']),('训练均值基线 MAE',o['train_mean_mae']),
                   ('相对均值基线改善（正值更好）',o['gain_vs_train_mean']),
                   ('真实 HR 标准差',o['label_std']),('预测 HR 标准差',o['pred_std']),
                   ('R_var',o['r_var']),('整体 Pearson r',o['pearson_r']),
                   ('受试者去均值后 Pearson r',audit['centered_within_subject']['pearson_r'])]:
    text += f'| {name} | {value:.6f} |\n'
text += '''
p01/p08 的组内 r 分别为 −0.0093 / −0.0102，R_var 为 0.169 / 0.090。
两组预测均值约 73 bpm。均值与中位数均只由训练集计算；训练中位数基线 MAE
为 5.979，模型虽优于它，却不如训练均值。不能只挑有利的常数基线。

![预测与误差分布](e2a_cycleformer/errors.png)

- [p01 全部记录轨迹](e2a_cycleformer/trajectories_FTU_participant_1.png)
- [p08 全部记录轨迹](e2a_cycleformer/trajectories_FTU_participant_8.png)
- [每受试者统计](e2a_cycleformer/by_subject.csv)；[每记录统计](e2a_cycleformer/by_record.csv)

## 特征破坏

三种子 11/29/47；每次完整 704 窗口、同一 checkpoint、同一标签。
ΔMAE = 破坏条件 MAE − 配对参考 MAE。所有同步重算条件对照 reencoded。
种子范围不是总体置信区间，测试只有两个受试者。

| 条件 | 参考 | MAE | ΔMAE 均值 | 种子最小 / 最大 | 预测绝对变化 |
|---|---|---:|---:|---:|---:|
'''
for row in summary.itertuples(index=False):
    text += (f'| {row.condition} | {row.reference} | {row.mae:.4f} | {row.delta_mae:+.4f} | '
             f'{row.delta_min:+.4f} / {row.delta_max:+.4f} | {row.prediction_change:.4f} |\n')
text += '''
![配对 MAE 变化](e2a_controls/delta_mae.png)

双分支替换成其他受试者后 ΔMAE 仅约 +0.014 bpm；单独打乱时间甚至改善 MAE。
频率 bin 置乱使 MAE 增加约 0.458 bpm，说明输出仍对频率布局敏感，但这不能证明
学到了真实心跳；原始预测本身没有组内相关，且破坏频谱可能只是引入 OOD 偏移。
循环平移是平移不变性检查，不能把其无影响当作不使用周期信息的证明。

## 校验、计时与下一步

'''
text += f'- Kaggle CPU 冻结推理：{meta["elapsed_sec"]:.2f} 秒。因本地缺窗口及 checkpoint，使用 Kaggle。\n'
text += f'- 本地 uv 预测审计和绘图：{audit["elapsed_sec"]:.2f} 秒；本地结果复核：{validation["elapsed_sec"]:.2f} 秒。\n'
text += f'- checkpoint SHA256：`{meta["checkpoint_sha256"]}`。\n'
text += f'- manifest SHA256：`{meta["manifest_sha256"]}`。\n'
text += f'- 210 个分组统计行 / 21120 个逐窗口条件预测；逐窗口重算 MAE 最大差 {max_mae_error:.3g}。\n'
text += f'- 双分支替换预测与原模型对 donor 的原始预测最大差 {max_donor_error:.3g} bpm。\n'
text += '''- 原始 CPU 推理已逐窗口匹配历史 GPU 导出（绝对容差 0.001 bpm），严格载入权重；
  三次原始推理完全相同。检查 manifest 哈希、标签一致性及 train/test 组隔离。
- 暂不据此重训或启动 PhysDrive Source-Free。继续找回三个历史 checkpoint；
  连续记录的 RDA→EDACM→AdaVMD 图尚未执行，不能用标准化窗口图冒充。
  下一步解释样本已按好/差/HR 变化最大写入 `e2a_cycleformer/summary.json`。
'''
discovery = json.loads((root/'discovery/notebooks/summary.json').read_text())
found = [x for x in discovery['notebooks'] if 'error' not in x]
failed = [x for x in discovery['notebooks'] if 'error' in x]
text += '\n## Notebook 输出检索更新（2026-10-14）\n\n'
text += (f'当前账号 {len(found)} 个相关 Notebook 的最新输出清单已完成分页核对；'
         f'本次汇总未完成项 {len(failed)}。包括混合域 T0、FTU E2a、早期 PhysDrive '
         'pipeline/ablation、RDA pool、BGT60 和 mmwave 实验。'
         '这些最新输出中未发现 v3 small、FTU Transformer 或历史 4.989 的可验证 checkpoint。'
         '这不等于检查了所有旧版本输出、已删除 Notebook 或其他账号。\n')
text += ('原始三模型比较仍缺权重和对应数据版本，不能用 E2a 数字覆盖历史结果。'
         '公有/私有 dataset 文件索引与每个 Notebook 的 checkpoint 命中路径保存在 discovery。\n')
continuous_source = Path('tmp/ftu_continuous_audit_result/ftu_continuous')
if (continuous_source/'summary.json').exists():
    continuous = root/'e2a_continuous'
    shutil.copytree(continuous_source,continuous,dirs_exist_ok=True)
    for name in ('source_manifest.json','kernel-metadata.json'):
        shutil.copy2(Path('tmp/ftu_continuous_audit_upload')/name,continuous/name)
    shutil.copy2('tmp/ftu_continuous_audit_result/ftu-continuous-stage-audit.log',continuous/'kernel.log')
    cmeta = json.loads((continuous/'summary.json').read_text())
    assert cmeta['prediction_sha256'] == sha256(evidence/'preds_cycleformer_FTU_test.csv')
    stage_metrics = pd.read_csv(continuous/'stage_metrics.csv')
    assert len(stage_metrics) == 96
    assert np.isfinite(stage_metrics.select_dtypes(include='number')).all().all()
    assert (stage_metrics.non_dc_total_power > 1e-12).all()
    assert (stage_metrics.band_total_power > 1e-12).all()
    assert all(r['max_window_label_error_bpm'] < 1e-4 for r in cmeta['records'])
    validation_started = time.perf_counter()
    max_power_error = 0.
    for record in cmeta['records']:
        with np.load(continuous/(record['category']+'_signals.npz'),allow_pickle=False) as d:
            for index,name in enumerate(['raw_unwrap','edacm_single','edacm_fused','vmd_modes']):
                power,f,starts = power_windows(d[name])
                max_power_error = max(max_power_error,float(np.max(abs(power.sum(axis=0)-d['stage_power'][index]))))
                assert np.allclose(power.sum(axis=0),d['stage_power'][index],atol=1e-8,rtol=1e-5)
                assert np.array_equal(starts,d['window_starts'])
                assert np.array_equal(f,d['frequencies_hz'])
    continuous_validation = dict(records=3,stage_metric_rows=96,max_power_recalculation_error=max_power_error,
        all_labels_reproduced=True,prediction_file_hash_matches=True,
        elapsed_sec=time.perf_counter()-validation_started)
    (continuous/'validation.json').write_text(json.dumps(continuous_validation,indent=2),encoding='utf-8')
    csummary = stage_metrics.groupby(['category','sample_tag','stage'],sort=False).agg(
        ridge_mae=('ridge_abs_error','mean'),hr_band_fraction=('hr_band_fraction','mean'),
        ref_bin_fraction=('reference_nearest_bin_fraction','mean')).reset_index()
    csummary.to_csv(continuous/'stage_summary.csv',index=False)
    text = text.replace('连续记录的 RDA→EDACM→AdaVMD 图尚未执行，不能用标准化窗口图冒充。',
                        '连续记录的 RDA→EDACM→AdaVMD 图已完成，见下方补充诊断。')
    text += '\n## 三条连续记录的信号诊断（2026-10-14）\n\n'
    text += (f'Kaggle CPU 运行 {cmeta["elapsed_sec"]:.2f} 秒；每条处理完整 1200 帧。'
             '使用 E2a 归档的 EDACM/HR-AdaVMD 代码，叠加同一 checkpoint 的原始逐窗口预测。'
             '12.8 秒窗、6.4 秒步长，无补零，分辨率 4.6875 bpm。\n\n')
    for r in cmeta['records']:
        text += (f'- {r["category"]}: `{r["sample_tag"]}`；'
                 f'[阶段时频图](e2a_continuous/{r["category"]}_stages.png)、'
                 f'[Range–Frequency 图](e2a_continuous/{r["category"]}_range_frequency.png)。'
                 f'参考标签复算最大差 {r["max_window_label_error_bpm"]:.3g} bpm。\n')
    text += '\n| 样本类别 | 阶段 | 主频脊线 MAE | HR 频带功率占比 | 参考最近 bin 功率占比 |\n|---|---|---:|---:|---:|\n'
    for row in csummary.itertuples(index=False):
        text += f'| {row.category} | {row.stage} | {row.ridge_mae:.3f} | {row.hr_band_fraction:.4f} | {row.ref_bin_fraction:.4f} |\n'
    text += ('\n功率占比以非 DC 总功率为分母；AdaVMD 使用七通道功率之和，不与原相位比较幅度增益。'
             'whole-record 的目标选择和分解与模型逐窗口的预处理不同，因此这些图用于解释，'
             '不证明哪一阶段导致模型塌缩。绿色横线标出选择器选中的距离索引，Range–Frequency '
             '固定最优单元的 Doppler/angle，展示的是裁剪后的 8 个距离索引。\n\n')
    text += ('时间映射沿用历史导出，并已逐窗口复现标签；它将参考起止时间线性映射到雷达帧，'
             '未独立验证雷达与参考时钟。下面列出其时间跨度，不能将标签复现当作同步正确的证明。\n\n')
    for r in cmeta['records']:
        text += (f'- `{r["sample_tag"]}`：参考映射跨度 {r["historical_frame_time_span"]:.4f} s；'
                 f'20 Hz 雷达首尾帧跨度 {r["nominal_radar_span"]:.4f} s。\n')
    text += '\n这些样本按测试表现选出，只用于探索解释；不能由三条记录证明 FTU 原始雷达没有心跳信息。\n'
    text += '''
### 这三条记录支持什么

- “预测最好”记录的参考 HR 本身接近 73 bpm 且变化很小，模型输出也接近 73。
  低误差可与近常数输出并存，不是心跳跟踪的充分证据。
- 该记录在原 RDA 相位/同单元 EDACM/融合 EDACM 的主频脊线 MAE 均约 6.194 bpm；
  AdaVMD 后为 13.359。融合 EDACM 的 HR 频带功率占比从 0.293 提高到 0.674，
  但脊线更偏离参考，说明频带强化不能单独作为表征改进指标。
- HR 变化最大的记录中，参考窗口 HR 从约 89 降到约 70 bpm，预测仍约 73。
  AdaVMD 的脊线 MAE 为 13.500 bpm，未展示稳定的参考 HR 跟随。
- 三条记录都复现了原始标签，但历史参考跨度为 59 / 61 / 61 秒，而 1200 帧、
  20 Hz 的首尾帧跨度为 59.95 秒。先审计同步和标签生成，不能靠拟合时延纠正测试误差。

下一轮优先验证真实时间轴、标签生成和可跟踪频谱；本轮未启动重训或跨域适配。
整记录 VMD 与原模型的逐窗口 VMD 不同，以上观察尚不能直接归因模型塌缩于 VMD。
'''
text = text.replace('210 个分组统计行',f'{len(scores)} 个分组统计行')
(root/'REPORT.md').write_text(text,encoding='utf-8')
manifest = {str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()
            and p.name != 'artifact_manifest.json'}
(root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
print(json.dumps(validation))
