"""Independent six-record validation, paired correspondence metrics, and report."""
import ast,base64,hashlib,io,json,shutil,subprocess,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_ftu_adc_controlled import load_post,process,readout
from audit_ftu_adc_phase import digest

root=Path('experiment_runs/ftu_adc_controlled');started=time.perf_counter()
def correlation(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else np.nan
def metrics(g):
    if len(g)==0:return dict(n=0,mae=np.nan,centered_rmse=np.nan,centered_r=np.nan,centered_std_ratio=np.nan,centered_r2=np.nan,near_reference_power_fraction=np.nan)
    y=g.label_bpm.to_numpy();p=g.peak_bpm.to_numpy();cy=y-g.groupby('sample_tag').label_bpm.transform('mean').to_numpy();cp=p-g.groupby('sample_tag').peak_bpm.transform('mean').to_numpy()
    return dict(n=len(g),mae=float(abs(y-p).mean()),centered_rmse=float(np.sqrt(np.mean((cp-cy)**2))),centered_r=correlation(cy,cp),
        centered_std_ratio=float(cp.std()/cy.std()) if cy.std()>1e-10 else np.nan,
        reference_centered_std=float(cy.std()),prediction_centered_std=float(cp.std()),
        centered_r2=float(1-np.sum((cp-cy)**2)/np.sum(cy*cy)) if np.sum(cy*cy)>1e-10 else np.nan,
        near_reference_power_fraction=float(g.reference_near_power_fraction.mean()))
def md(df):
    def fmt(x):return 'undefined' if pd.isna(x) else f'{x:.7g}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

protocol=json.loads((root/'protocol.json').read_text());selected=pd.read_csv(root/'selected_records.csv');ro=pd.read_csv(root/'readouts.csv');quality=pd.read_csv(root/'signal_quality.csv')
assert len(selected)==6 and set(ro.sample_tag)==set(selected.sample_tag)
assert ro[ro.scored].groupby(['variant','kind']).size().eq(36).all()
prior=json.loads(subprocess.check_output(['git','show','53a1be7:experiment_runs/ftu_adc_controlled/protocol.json']));assert protocol==prior
# Verify the exact frozen files embedded in the executed Kaggle runner.
runner=Path('tmp/ftu_adc_controlled_upload/run.py');tree=ast.parse(runner.read_text())
bundle=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle))) as archive:
    for name in archive.namelist():assert archive.read(name)==(root/name).read_bytes(),name
post=load_post(root/'sources/edacm.py');verification=[];floor_masks={}
for tag in selected.sample_tag:
    with np.load(root/'signals'/f'{tag}.npz') as z:
        energy=z['calibration_energy'];profile=energy.mean(axis=0);profile[:2]=-np.inf;center=int(profile.argmax())
        score=energy[:,z['roi']].copy();score[:,z['roi']<2]=-np.inf
        flat=np.argsort(-score.ravel(),kind='stable')[:3];aa,rr=np.unravel_index(flat,score.shape);expected=np.stack([aa,z['roi'][rr]],axis=1)
        np.testing.assert_array_equal(expected,z['targets'])
        weights=np.array([energy[a,r] for a,r in expected],dtype=np.float32);weights/=weights.sum();np.testing.assert_array_equal(weights,z['weights'])
        mapped=np.interp(np.arange(1200)/20,z['reference_time'],z['reference_hr'],left=np.nan,right=np.nan)
        np.testing.assert_allclose(mapped,z['mapped_reference_hr'],rtol=0,atol=0)
        for variant in ['legacy_mean','full_mean','legacy_none','full_none']:
            masks=[]
            for j,s in enumerate(z['starts']):
                s=int(s);seg=z[variant+'_z'][:,s:s+256];base=z['full_none_z'][:,s:s+256]*(.125 if variant.startswith('legacy') else 1.)
                actual=np.sqrt(np.mean(abs(seg.astype(complex))**2,axis=1));baseline=np.sqrt(np.mean(abs(base.astype(complex))**2,axis=1))
                valid=np.isfinite(seg).all(axis=1)&(abs(seg)>0).all(axis=1)&(actual>baseline*1e-6);masks.append(valid)
                delta,cum,phase,fused,signal=process(seg,weights,post,valid)
                for key,x in [('delta',delta),('cumulative',cum),('phase',phase),('fused',fused),('signal',signal)]:np.testing.assert_array_equal(x,z[variant+'_'+key][j])
                label=float(np.nanmean(mapped[s:s+256]))
                for kind,trace,ok in [('fused',signal,valid.all())]+[(f'single{k}',phase[k],valid[k]) for k in range(3)]:
                    g=ro[(ro.sample_tag==tag)&(ro.variant==variant)&(ro.kind==kind)&(ro.window_start==s)].iloc[0]
                    assert bool(g.valid_phase)==bool(ok);np.testing.assert_allclose(label,g.label_bpm,atol=1e-12)
                    if ok:
                        h=readout(trace,label);np.testing.assert_allclose(g.peak_bpm,h['peak_bpm'],atol=0,rtol=0)
                        np.testing.assert_allclose(g.reference_near_power_fraction,h['reference_near_power_fraction'],atol=2e-12)
            floor_masks[tag+'_'+variant]=np.array(masks)
        verification.append(dict(sample_tag=tag,selection_recomputed=True,all_saved_arrays_peaks_labels_verified=True))
np.savez_compressed(root/'reliability_masks.npz',**floor_masks)
scored=ro[ro.scored];stats=[];records=[]
for (variant,kind),g in scored.groupby(['variant','kind']):
    valid=g[g.valid_phase];stats.append(dict(variant=variant,kind=kind,scope='all_six',coverage=len(valid)/36,**metrics(valid)))
    h=valid[~valid.sample_tag.str.startswith('p03_')];stats.append(dict(variant=variant,kind=kind,scope='five_with_config_logs',coverage=len(h)/30,**metrics(h)))
    for tag,h in g.groupby('sample_tag'):records.append(dict(variant=variant,kind=kind,sample_tag=tag,coverage=float(h.valid_phase.mean()),**metrics(h[h.valid_phase])))
stats=pd.DataFrame(stats);records=pd.DataFrame(records);stats.to_csv(root/'metrics.csv',index=False);records.to_csv(root/'record_metrics.csv',index=False)
pairs=[];pooledpairs=[]
for kind in ['fused','single0','single1','single2']:
    for candidate in ['legacy_none','full_none']:
        a=scored[(scored.variant==candidate)&(scored.kind==kind)&scored.valid_phase]
        b=scored[(scored.variant=='legacy_mean')&(scored.kind==kind)&scored.valid_phase]
        joined=a.merge(b,on=['sample_tag','window_start'],suffixes=('_candidate','_control'))
        for tag in selected.sample_tag:
            g=joined[joined.sample_tag==tag];ma=metrics(pd.DataFrame(dict(sample_tag=g.sample_tag,label_bpm=g.label_bpm_candidate,peak_bpm=g.peak_bpm_candidate,reference_near_power_fraction=g.reference_near_power_fraction_candidate)))
            mb=metrics(pd.DataFrame(dict(sample_tag=g.sample_tag,label_bpm=g.label_bpm_control,peak_bpm=g.peak_bpm_control,reference_near_power_fraction=g.reference_near_power_fraction_control)))
            pairs.append(dict(candidate=candidate,kind=kind,sample_tag=tag,n_common=len(g),
                delta_mae=ma['mae']-mb['mae'],delta_centered_rmse=ma['centered_rmse']-mb['centered_rmse'],
                delta_near_reference_power=ma['near_reference_power_fraction']-mb['near_reference_power_fraction']))
        for group in ['candidate','control']:
            g=pd.DataFrame(dict(sample_tag=joined.sample_tag,label_bpm=joined['label_bpm_'+group],peak_bpm=joined['peak_bpm_'+group],reference_near_power_fraction=joined['reference_near_power_fraction_'+group]))
            pooledpairs.append(dict(candidate=candidate,kind=kind,role=group,**metrics(g)))
pairs=pd.DataFrame(pairs);pooledpairs=pd.DataFrame(pooledpairs);pairs.to_csv(root/'paired_record_changes.csv',index=False);pooledpairs.to_csv(root/'paired_pooled_metrics.csv',index=False)
decision=[]
both_candidates_full_coverage=bool(stats[(stats.kind=='fused')&(stats.scope=='all_six')&stats.variant.isin(['legacy_none','full_none'])].coverage.eq(1).all())
for candidate in ['legacy_none','full_none']:
    g=pairs[(pairs.candidate==candidate)&(pairs.kind=='fused')];good=(g.n_common>=4)&(g.delta_mae<0)&(g.delta_centered_rmse<0)&(g.delta_near_reference_power>0)
    p=pooledpairs[(pooledpairs.candidate==candidate)&(pooledpairs.kind=='fused')].set_index('role')
    decision.append(dict(candidate=candidate,records_jointly_improved=int(good.sum()),pooled_centered_r=float(p.loc['candidate','centered_r']),
        pooled_rmse_improved=bool(p.loc['candidate','centered_rmse']<p.loc['control','centered_rmse']),
        both_candidates_full_coverage=both_candidates_full_coverage,
        expansion_criterion_met=bool(both_candidates_full_coverage and good.sum()>=5 and p.loc['candidate','centered_r']>0 and p.loc['candidate','centered_rmse']<p.loc['control','centered_rmse'])))
decision=pd.DataFrame(decision);decision.to_csv(root/'expansion_decision.csv',index=False)
(root/'figures').mkdir(exist_ok=True)
for tag in selected.sample_tag:
    g=scored[(scored.sample_tag==tag)&(scored.kind=='fused')]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    with np.load(root/'signals'/f'{tag}.npz') as z:
        axes[0].plot((g[g.variant=='full_none'].window_start+128)/20,g[g.variant=='full_none'].label_bpm,'k--',label='reference (assumed time)')
        for variant in ['legacy_mean','full_mean','legacy_none','full_none']:
            h=g[g.variant==variant];axes[0].plot((h.window_start+128)/20,h.peak_bpm,'.-',label=variant)
            axes[1].plot(np.arange(1200)/20,abs(z[variant+'_z'][0]),label=variant)
            trace=z[variant+'_fused'][2];axes[2].plot(np.arange(256)/20,trace,label=variant)
        axes[0].set(title='Fixed peak readout; scored windows',xlabel='Window center(s)',ylabel='bpm')
        axes[1].set(title='Frozen target0 complex amplitude',xlabel='Frame time(s)',ylabel='Raw ADC FFT amplitude',yscale='log')
        axes[2].set(title='First scored window, pre-z fused phase',xlabel='Window time(s)',ylabel='Detrended phase(rad)')
        for ax in axes:ax.legend(fontsize=6)
    fig.suptitle(tag);fig.tight_layout();fig.savefig(root/'figures'/f'{tag}.png',dpi=130);plt.close(fig)
fused=stats[(stats.kind=='fused')&(stats.scope=='all_six')];five=stats[(stats.kind=='fused')&(stats.scope=='five_with_config_logs')];syn=pd.read_csv(root/'synthetic_metrics.csv');timing=pd.read_csv(root/'previous_timing_correction.csv');precision=pd.read_csv(root/'previous_precision_recheck.csv')
execution=json.loads((root/'execution.json').read_text());sgate=json.loads((root/'synthetic_gate.json').read_text());raw=pd.read_csv(root/'raw_verification.csv')
recovery=json.loads((root/'official_recovery_execution.json').read_text())
expanded=bool(decision.expansion_criterion_met.any())
corrections=f'''# Corrections to b6bd9ba ADC phase report

1. 完整 FFT 的较大 chirp0 相位误差主要来自配置决定的积分时刻，不应解读为更多相位破坏。前8个/全部64个历史 chirp 索引的采样时刻质心分别是0.399/3.591891毫秒；没有用 HR 拟合延时。

{md(timing)}

2. 旧报告的“最大相对L2差0.1564”实际含 max(参考范数,1) 分母下限。此数来自近乎零的 full_mean 连续微动条件，不是有效信号15.64%的误差。原保存求和顺序中，该条件参考范数约0.161567，差值范数约0.156401，理论无clutter增益归一化RMS差约1.22e-10。修订表逐条件列绝对误差、参考范数以及理论相干增益归一化误差；低幅度条件的真实相对误差标为undefined。complex128 在重新求均值及求和全过程复核，而非只在最后转换类型。

{md(precision[(precision.reference_precision=='complex128')][['case','variant','max_abs_error','reference_l2','relative_l2','rms_error_over_theoretical_gain']])}

有效无均值消除信号相对complex128参考的相对L2误差约1.7–2.2e-8；零输出/残差条件不计算信号相对精度。全逐条件表 previous_precision_recheck.csv 同时保留原保存求和顺序，明确标识旧分母下限统计量。

3. 新协议53a1be7只要求候选通过；旧均值消除臂是预期失败机制对照，失败不再阻止真实实验。首次合成因共享读出器硬编码256点而停止，b4db09c修正长度接口后重跑，未改协议/门槛/样本。旧b6bd9ba协议的停止决定仍被保留，没有事后称旧门控已通过。
'''
(root/'CORRECTIONS.md').write_text(corrections,encoding='utf-8')
report=f'''# FTU six-record ADC phase-controlled 2×2

## Decision

真实六记录对照已完成。按协议的联合参考对应规则，{'至少一个候选达到扩展判据，可审阅后进入54条固定扩展；本轮仅报告六条结果。' if expanded else '两个候选均未达到扩展判据；本轮不扩展54条、不重训。'} 结论限于六条固定哈希记录、六名原训练受试者的诊断；每人只有一条记录，不能当作跨受试者泛化或稳定的受试者总体证据。

{md(decision)}

## Controlled real results

每臂原定36个评分窗口，全部标签/窗口相同。以下为融合信号；coverage是超过预定数值幅度门槛且相位有定义的窗口比例，不代表生理信号可信度或准确率。低幅度对照标undefined，不给出虚构HR；各候选与legacy_mean的效果比较使用配对共同有效窗口，见 paired_record_changes.csv / paired_pooled_metrics.csv。

{md(fused[['variant','coverage','n','mae','centered_rmse','centered_r','reference_centered_std','prediction_centered_std','centered_std_ratio','centered_r2','near_reference_power_fraction']])}

候选去均值RMSE虽然降低，预测波动仍约参考的9倍，且相关性为负；旧对照正相关也伴随约14倍波动和强负R²，不能据正相关宣称恢复跟踪。full_mean为DC均值消除的预期机制对照，全部窗口低于数值幅度门槛而弃权；不是把有限浮点残差的相角解释为生理相位，也不是对全部Doppler输出的信息否定。

{md(pairs[pairs.kind=='fused'].drop(columns=['kind']))}

排除缺设备日志的p03后，候选MAE及参考附近功率反而更差；总体功率的小幅增加主要由p03贡献，没有一致收益。

{md(five[['variant','n','mae','centered_rmse','centered_r','near_reference_power_fraction']])}

跨记录联合改善要求：同一候选至少5/6条记录在≥4个共同有效窗口上，同时降低MAE、降低去均值RMSE、提高正确参考附近功率，且合并去均值相关性>0、RMSE优于legacy_mean。单纯幅度增大、曲线平滑或有峰不能通过。R²只作合并诊断，不平均逐记录R²。参考动态波动小、256点峰值分辨率4.6875bpm、窗口重叠，均限制动态评价；未做独立窗口显著性检验。

![冻结名单首条记录；其余五条同目录](figures/p02_Orientation_80_cm__left_side_r4.png)

## Fixed region and pipeline

原六条选样不变；各记录最初256帧校准，不评分与其重叠的窗口。共同物理Doppler为DC（两FFT shifted index4），使用相同历史64chirp索引、250ADC、4RX；非零Doppler下标不互当物理频率。

按full_none校准段DC能量确定固定8-bin range ROI（排除原始range0/1），在ROI中固定三个幅度最大的beam/range及归一化幅度权重。四臂共享这些身份和权重，全程未调用EDACM/HR打分器。P2→float32累积→原函数线性去趋势→相位加权→原函数z-score→Hann功率→固定0.5–2.75Hz最大bin；单目标和融合均保存。该校准规则未证明选中了胸壁，range/beam在此仍是索引；TX同时启用的实际波束解释仍有限。

保存了共同ROI全部beam的标准化前复数序列、三个固定目标复数序列、原始增量/累积/去趋势相位/融合信号、参考时间轴和标签。幅度可由保存的复数逐值恢复；reliability_masks.npz 保存每窗口每目标有效标记。低幅度保留原始复数/数值增量，但下游物理相位、融合和评分以NaN显式标记。时间沿用frame/20+0s，未搜索延时。此次校准与旧54记录audit不同，所以不能把其MAE与旧报告直接作单因素配对。

## Independent layout check

依据 [TI SWRA581B](https://www.ti.com/lit/an/swra581b/swra581b.pdf) 第6节图11、§9.2（pp10,16–17）独立翻译MATLAB读取顺序：双lane、每chirp按RX0→RX3块排列、两I后两Q，16-bit有符号。与现有loader不是同一份实现；本轮在全部六条真实bin、全部1200帧逐块严格核对，均逐值一致。新读取器还在所有合成量化ADC上与生产读取器一致。

已有日志支持AR1642、DCA1000、四RX、双lane；官方README也确认设备为AWR1642BOOST，但p03缺少单记录设备日志，继续标记为配置未核实，未用其他人的日志顶替。原始文件均需匹配1200×128×4×250×4字节。厂商说明要求去包头、按序的ADC负载；文件大小和现存capture日志提供一致性证据，仍不是硬件已知信号注入的绝对证明。逐记录日志存在性、零填充/乱序记录见 raw_verification.csv，完整原文在evidence/。排除p03的五记录指标另列 metrics.csv 的 five_with_config_logs，供检查结论是否依赖缺日志记录。

初次Kaggle执行因五个镜像分片缺少原定p03 Distance ADC而停止（日志kaggle_v1_missing_input.txt），没有换记录。官方4TU API和原始README已归档；通过原始非solid RAR的Quick Open目录和CRC校验，定位同一记录并只下载425660324字节压缩区段，libarchive校验解压为614400000字节，在第二次执行补回。输入恢复补充记录为e776469；选样、校准规则、后处理、候选门槛及六记录扩展规则均未变。恢复原始哈希/耗时见official_recovery_execution.json，目录证据在official_recovery/。官方数据来源：[4TU DOI](https://doi.org/10.4121/48acba04-96bc-4131-b52f-9e18458ad92b)。

## Synthetic candidate gate and corrections

协议53a1be7在新结果生成前冻结；两个不减均值候选通过基线及三组非整频弱微动检查，旧臂仅作机制对照。

{md(syn[['case','variant','role','phase_amplitude','rmse_chirp0_rad','rmse_mean_time_rad','rmse_coherent_phase_rad','peak_bpm','pass']])}

按配置质心修正后，基线两候选的相位误差约1.4e-5/1.8e-5rad；不能说完整FFT更破坏相位。旧报告0.1564是带分母下限的统计量，修正和complex128逐条件复核见 [CORRECTIONS.md](CORRECTIONS.md)。此报告纠正旧措辞，保留历史版本和原始结果。

## Execution and verification

本地uv合成耗时{sgate['elapsed_sec']:.3f}秒；缺少本地原始数据，真实六记录仅在Kaggle CPU运行，处理耗时{execution['elapsed_sec']:.3f}秒，另有官方p03范围下载及校验解压耗时{recovery['elapsed_sec']:.3f}秒；没有训练。逐块TI-port与生产loader一致、选点权重重算、全部保存后处理数组/峰值/标签重算、原始RDA哈希与上轮清单核对均通过；本地复核耗时记录在validation.json。运行配置、输入哈希、逐窗口表、六条完整轨迹图和源码均归档。
'''
(root/'REPORT.md').write_text(report,encoding='utf-8')
plan=Path('PLAN.md');text=plan.read_text(encoding='utf-8');marker='## 2026-10-19 六记录ADC候选与机制对照'
if marker not in text:plan.write_text(text+'\n\n'+marker+'\n\n协议53a1be7区分候选门控与已知失败对照；独立TI读取路径与生产loader在六条原始记录全部帧一致。两个不减均值候选通过弱微动非整频合成门控，完成共同校准空间区域下真实2×2。'+('至少一个候选达到六记录扩展判据；尚未运行54条。' if expanded else '候选未达到跨记录联合参考对应收益门槛，不扩展54条、不重训。')+'积分时刻与0.1564口径已在新报告明确修正。详见 [报告](experiment_runs/ftu_adc_controlled/REPORT.md)。\n',encoding='utf-8')
for script in ['summarize_ftu_adc_controlled.py','correct_ftu_adc_reporting.py','bundle_ftu_adc_controlled.py','fetch_ftu_adc_controlled.py']:shutil.copyfile(script,root/'sources'/script)
shutil.copyfile(runner,root/'sources/kaggle_runner.py')
(root/'validation.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-started,records=6,protocol_unchanged=True,
    frozen_bundle_verified=True,selection_weights_recomputed=True,all_postprocessing_arrays_exact=True,labels_peaks_recomputed=True,
    model_training=False,expansion_criterion_met=expanded),indent=2))
(root/'artifact_manifest.json').write_text(json.dumps({p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},indent=2))
print(fused.to_string(index=False));print(pairs[pairs.kind=='fused'].to_string(index=False));print(decision.to_string(index=False))
