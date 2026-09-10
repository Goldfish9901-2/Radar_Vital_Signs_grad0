# mmwave-897-2026 数据集探索总结（2026-09-01）

外部参照数据集：Nature Sci Data 13:897 (2026)，DOI 10.5281/zenodo.16760683
TI 60.25GHz 雷达（IWR6843），2TX×4RX，8 虚拟天线，64 range bins，10Hz，零多普勒 rFFT
110 人 × 4 场景（Lying/Sitting × Rest/Post-exercise），Movesense ECG 250Hz 参照。

所有实验代码：`src/data/mmwave_{baseline,diagnose,fusion,vote,split_half,probe,ibi}.py`
所有结果：`experiments/mmwave_selection/*.json`（440 sessions 全量）

---

## 1. 核心结论

| 结论 | 证据 |
| --- | --- |
| **单 bin selection 无提升空间** | 13 种 selection + 4 融合 + 4 投票全部落在 15-18 BPM（oracle 除外） |
| **oracle 1.64 是选择偏差，不是信息上限** | split-half：前一半选 bin、后一半验证 → MAE 1.64 → **18.3**（恶化 11 倍）；good bins 时间 Jaccard 仅 0.08 |
| **bin 2 (0.63m) = 全局能量最大**，且是唯一"物理合理"的近期 bin（雷达距胸 0.5m） | energy = 中位数 bin 的 2 万倍；89.8% session 全局最大 |
| **整段 FFT-peak 被 HR 漂移伤害** | 分段 FFT（10s 窗）全量 MAE 13.6 vs 15.7；Lying/Rest <5bpm 命中 62%→73% |
| **信号质量人/场景异质性极大** | Lying/Rest 73% session <5bpm；Sitting/Post 仅 17% |
| **全 cube 频谱线性不可读 HR** | Ridge subject-level CV：single-bin R²=-16，4402 维矩阵 R²=-65 |

## 2. 各方法全量 MAE（bpm，440 sessions）

| 方法 | MAE | median | <5bpm% | 备注 |
| --- | ---: | ---: | ---: | --- |
| oracle（全段扫描） | 1.64 | 0.29 | 90.7 | 选择偏差，勿当作上限 |
| oracle split-half | 18.3 | — | 13.0 | 真实"选 bin"水平的估计 |
| fft_seg_med（bin2） | 13.6 | 7.4 | 42 | 全量最优真实方法 |
| fft_whole（bin2） | 15.7 | 9.5 | 36 | 论文 baseline 同款 |
| ibi_bin2（find_peaks） | 13.7 | 9.7 | 27 | Post 场景略优于 FFT |
| coherent_topk | 15.2 | — | — | 无选择融合中最优 |
| max_energy | 15.7 | 9.5 | 36 | 原 baseline |
| power_weighted | 17.6 | — | — | 加权失败 |
| vote_*（跨 bin 投票） | 15.8-17.0 | — | — | 全部失败 |

分层（fft_seg_med）：Lying/Rest 6.6 · Lying/Post 14.6 · Sitting/Rest 11.8 · Sitting/Post 21.2

## 3. 关键陷阱（对主战场 pipeline 的迁移教训）

1. **"oracle 上限"要警惕选择偏差**：同一批帧既选目标又评估，会把误差压低到不可复现的水平。
   判定方法 = split-half（选择集与评估集分离）。PhysDrive 的 constant-baseline 对比也要用同一逻辑复核。

2. **零多普勒 rFFT + 单 bin 相位的信息有限**：这佐证了 RDA 层需要更强的表示（EDACM/AdaVMD 路线），
   而不是更聪明的"选 bin"。

3. **分段/短窗估计明显优于整段**：HR 漂移（尤其运动后）伤害整段 FFT-peak。
   建议在主 pipeline 的 HR 估计器/评估中引入短窗聚合，这可能是 1-2 BPM 的免费提升。

4. **max-energy 选中近期 DC 耦合 bin 是本数据集常态**：bin 2 (0.63m) 能量 2 万倍于中位数且恰为人体所在。
   若目标距离未知，能量法对强近场反射无判别力——需要距离先验或杂波抑制（MTI/PCA）前置。

5. **Post-exercise 场景（HR 高 + 漂移快）是最大硬骨头**：任何单值估计器（FFT/IBI）都难以 <5bpm。
   需要 HR 轨迹跟踪（短窗滑动 + 平滑/卡尔曼）或放弃该场景的单值指标。

## 4. HR 轨迹级验证（补充，2026-09-01）

5s 滑动窗 IBI 轨迹，雷达 bin2 vs ECG（421 有效 session）：

| 指标 | 值 |
| --- | --- |
| 轨迹 Pearson 均值 | **0.004**（≈0，完全无关） |
| pct_pearson>0.8 | **0.24%**（仅 1/421 session） |
| 轨迹 MAE | 26.5 bpm |
| 末段 15s HR MAE | 23.2 bpm |

全部 4 场景均如此。**决定性结论：心跳信号在时间分辨级别上不存在于
零多普勒 rFFT 相位中**（即使滑动窗也无法跟随 ECG HR 变化）。
这是数据性质（零多普勒平均 + 无角度处理 + 近场杂波）决定的，不是算法问题。

## 5. 最终结论

完整证据链（440 sessions，全部方法）：

| 路线 | 结果 |
| --- | --- |
| selection（13 方法） | 15-18 BPM |
| 融合（4 方法） | 15-18 BPM |
| 跨 bin 投票（4 方法） | 15-18 BPM |
| oracle split-half | 1.64 → **18.3**（选择偏差） |
| 全 cube 线性 probe | R² << 0 |
| 分段 FFT | 13.6（唯一边际改善） |
| IBI 路径 | 13.7（Post 略优） |
| HR 轨迹（滑动窗） | Pearson≈0 |

**mmwave-897 作为外部参照的使命已完成**，核心收获：
1. selection 非 RDA 瓶颈；单 bin 相位+FFT 范式在该数据上上限 ≈13-16 BPM
2. oracle 类"上限"必须用 split-half 校验（选择偏差陷阱）
3. 分段/短窗估计 > 整段估计（HR 漂移），约 2 BPM 免费增益
4. 能量法对近场 DC 耦合 bin 无判别力 → 需杂波抑制前置（MTI/PCA）
5. 零多普勒 rFFT 数据集的心跳信号天花板低——对 PhysDrive 的警示：
   它的完整 ADC/多普勒信号链远强于此，不应因此数据集否定 RDA 工作价值

## 6. 建议的后续动作

- [x] 分段估计器改进：回主 pipeline 验证（evaluate_model.py 引入短窗聚合）
- [x] split-half oracle 校验法：纳入后续全部实验标准流程
- [ ] RDA 算法池主线：clutter suppression（MTI/PCA）作为第一个可迁移实验
- [ ] 呼吸验证（论文称呼吸与加速度计一致）——如需确认慢速信号链可用
