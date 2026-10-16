"""Validate Kaggle exports against archived raw Polar CSVs and archive evidence."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
from src.data.ftu_timebase import frame_times, labels_at_frames
from src.features.representations import frequency_features

started=time.perf_counter()
root=Path('experiment_runs/ftu_timebase_edacm_v2')
root.mkdir(parents=True,exist_ok=True)
download=Path('tmp/ftu_edacm_v2_result')
rawroot=Path('tmp/ftu_p0_result/ftu_p0')
refs={r['source']:r for r in json.loads((rawroot/'reference_times.json').read_text())}
timing=pd.read_csv('experiment_runs/ftu_p0_audit_2026-10-14/raw_timing_audit.csv')
references={}
for r in timing[timing.saved_window_count.notna()].itertuples():
    info=refs[r.hr_source]; path=rawroot/'raw_time_evidence'/info['archived_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
    df=pd.read_csv(path,skiprows=3,header=None)
    references[r.sample_tag]=(pd.to_timedelta(df.iloc[:,1]).dt.total_seconds().to_numpy(),
                              pd.to_numeric(df.iloc[:,2]).to_numpy())
archive=download/'results.zip'
with zipfile.ZipFile(archive) as z:
    manifest=pd.read_csv(z.open('training_exports_ftu_edacm_v2/manifest.csv'))
    validation=json.loads(z.read('validation.json'))
    assert validation['windows']==len(manifest)==3520
    errors=[]; frequency_errors=[]
    for row in manifest.itertuples():
        rel='training_exports_ftu_edacm_v2/windows/'+row.window_path.split('/windows/',1)[1]
        with np.load(io.BytesIO(z.read(rel))) as arrays:
            xt,xf=arrays['x_time'],arrays['x_freq']
            assert xt.shape==(1,256) and xf.shape==(1,129)
            assert np.isfinite(xt).all() and np.isfinite(xf).all()
            recomputed=frequency_features(xt)[0]
            frequency_errors.append(float(np.max(abs(xf-recomputed))))
            # Cross-platform float32 FFT/normalization can differ by a few ULPs.
            # Keep the strict 1e-6 mismatch count visible; reject errors >1e-5.
            np.testing.assert_allclose(xf,recomputed,atol=1e-5,rtol=0)
            rt,hr=references[row.sample_tag]
            times=frame_times(1200,20,radar_start_on_reference_clock_s=0)[row.window_start:row.window_end]
            expected=labels_at_frames(rt,hr,times)
            np.testing.assert_allclose(arrays['label_series_heart_rate'],expected,atol=2e-5)
            errors.append(abs(float(arrays['label_heart_rate'])-float(np.nanmean(expected))))
    assert max(errors)<2e-5
    old=pd.read_csv('experiment_runs/ftu_source_audit_2026-10-13/e2a_evidence/training_exports/manifest.csv')
    check=manifest.merge(old,on=['sample_tag','window_start'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==3520 and (check.split_new==check.split_old).all()
    assert manifest.groupby('group_key').split.nunique().max()==1
    for name in z.namelist():
        if '/windows/' in name or name.endswith('/') or name.startswith('input_manifest/'):
            continue
        dest=(root/name).resolve()
        assert dest.is_relative_to(root.resolve())
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
shutil.copyfile(archive,root/'full_artifacts.zip')
shutil.copyfile(download/'kernel.log',root/'kernel.log')
shutil.copyfile('tmp/ftu_edacm_local_validation.json',root/'local_integration_validation.json')
upload=Path('tmp/ftu_edacm_upload')
for name in ['kernel-metadata.json','source_manifest.json']:
    shutil.copyfile(upload/name,root/name)
sources=json.loads((upload/'source_manifest.json').read_text())
for name,digest in sources.items():
    data=Path(name).read_bytes()
    assert hashlib.sha256(data).hexdigest()==digest
    dest=root/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
for name in ['src/training/train_model.py','validate_ftu_edacm.py','kaggle/prepare_ftu_edacm.py','archive_ftu_edacm.py']:
    dest=root/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
result=dict(windows=3520,raw_csv_label_max_abs_error=max(errors),split_matches_history=True,
            frequency_recomputed=True,frequency_max_abs_error=max(frequency_errors),
            frequency_windows_above_1e_minus6=sum(e>1e-6 for e in frequency_errors),
            frequency_acceptance_atol=1e-5,local_numpy_version=np.__version__,
            elapsed_sec=time.perf_counter()-started)
(root/'independent_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
df=pd.read_csv(root/'signal_baselines.csv')
rm=pd.read_csv(root/'record_metrics.csv')
summary=pd.read_csv(root/'baseline_summary.csv')
test=summary[(summary.scope=='split')&(summary.group=='test')]
lines=['# FTU 新时间轴与 EDACM-only 实验', '',
    '完成 PLAN.md 的本轮四项任务；未训练模型。同步仍是显式假设，不能称为精确同步验证。', '',
    '## 数据与实现', '',
    '- 正式构建器使用 20 Hz、起始偏移 0 秒；保留参考时间戳，范围外标签为 NaN。',
    '- 440 条记录、3,520 个窗口；窗口 256 帧，步长 128 帧。冻结历史 E2a 受试者划分。',
    '- 每个窗口导出 x_time=(1,256)、x_freq=(1,129)，来自同一 EDACM 信号，无 VMD、无通道复制。',
    '- TCN / Transformer / CycleFormer 的单通道工厂前向检查通过；训练入口按输入推断通道数。',
    '- 固定偏移 0、±1、±2 秒的共同有效集合包含 3,520 个窗口，门槛为 80% 标签覆盖。',
    '', '## 测试集信号诊断（704 个窗口；不是神经网络 MAE）', '',
    '| 方法 | MAE (bpm) |','|---|---:|']
for r in test.itertuples():lines.append(f'| {r.method} | {r.mae:.4f} |')
lines += ['',f"前三谱峰在参考 HR ±5 bpm 内的命中率：{df[df.split=='test'].top3_hit.mean():.2%}；"
    f"循环错位四个窗口后为 {df[df.split=='test'].shifted_top3_hit.mean():.2%}。",
    '差距很小，尚不足以支持可靠动态跟踪；循环错位对近恒定 HR 是弱对照，不能单独证明没有心跳信息。',
    'FFT 使用窗口内选点与 EDACM；STFT 使用连续记录选点与 EDACM，两者并非仅变换算法不同。',
    '每条记录仅八个重叠窗口，相关系数需谨慎解释；未把窗口当成独立受试者来作显著性推断。',
    f"全部记录内相关系数中位数：FFT={rm[rm.method=='fft'].pearson_r.median():.4f}，"
    f"STFT={rm[rm.method=='stft'].pearson_r.median():.4f}（仅统计相关系数有定义的记录）。",
    '', '## 校验与产物', '',
    f"独立使用历史归档的原始 Polar CSV 复算全部标签，最大误差 {max(errors):.8f} bpm。",
    f"跨环境频域特征复算最大绝对误差 {max(frequency_errors):.8g}；"
    f"{sum(e>1e-6 for e in frequency_errors)} 个窗口超过 1e-6，全部在 float32 校验容差 1e-5 内。原环境校验使用 1e-6。",
    f"Kaggle 计算用时 {validation['elapsed_sec']:.2f} 秒；本地独立校验 {result['elapsed_sec']:.2f} 秒。",
    '',
    '- full_artifacts.zip：完整新版窗口、manifest、配置、统计与十名受试者各一条预先固定记录的时频图。',
    '- label_alignment.csv / offset_baselines.csv：固定偏移、共同窗口集合及对应误差。',
    '- signal_baselines.csv / record_metrics.csv / baseline_summary.csv：逐窗口、逐记录、逐受试者结果。',
    '- protocol.json / validation.json / independent_validation.json / source_manifest.json：协议及证据。',
    '- 解压后训练入口使用 training_exports_ftu_edacm_v2；加载器支持迁移后的窗口路径。',
    '', '## 判断', '',
    '数据导出与单通道接口已具备后续最小监督学习的工程条件，但本轮频谱诊断没有提供可靠动态跟踪证据。',
    '暂不扩大到 CycleFormer、VMD 或跨域实验；后续若做 TCN，应明确定位为检验可学习性的探索实验。',
    '旧的 proposed / HR-AdaVMD 数值问题仍未修复；新版 EDACM-only 完全绕过该路径。',
    '', '[Kaggle CPU 实验 v2](https://www.kaggle.com/code/goldfish9901/ftu-timebase-edacm-v2)', '']
(root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
(root/'artifact_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps(result))
