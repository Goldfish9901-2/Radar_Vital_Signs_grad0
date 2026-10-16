# FTU frozen source-model audit

2026-10-14 更新：[P0 模态与时间轴审计](FTU_P0_AUDIT.md)发现 HR-AdaVMD
数值退化。此前连续谱图中的 AdaVMD 频带能量变化不能作为有效模态分解或心跳增强的证据。
历史预测审计数值仍保留，但不能据此排除原始雷达中的心跳信息，或判定 backbone 无效。

暂停 BGT60 局部算法调整。先回答模型是否跟踪组内 HR、是否依赖输入信息；不训练、不适配。

## Artifact identity

历史目标是 2026-06-23 的 CycleFormer (FTU MAE 4.988688)、v3 small
(3.425270)、Transformer (4.034208)。模型名称或架构相同不构成 checkpoint
身份匹配。检查 checkpoint SHA256、run_config、manifest SHA256、训练标签统计、
受试者划分、逐窗口预测及历史 MAE。

当前找回的 `goldfish9901/nn-bench-e2a-hr-adavmd-proposed-ftu` 是另一次运行，
CycleFormer 的 FTU test MAE 为 4.805852。必须作为独立补充证据，不能冒充三个
历史模型的复算。T0 notebook 是混合数据集训练，也不能当作 FTU 源模型。

## Prediction audit

`python -m src.training.audit_predictions --predictions PRED.csv --export-dir EXPORT_DIR
--run-config MODEL_DIR/run_config.json --output OUTPUT_DIR --expected-mae HISTORICAL_MAE`

输入必须有 dataset/group_key/sample_tag/window_start/label_bpm/pred_bpm。
检查与 manifest 的完整测试集一一对应、无重复或非有限数值、标签一致、
训练/测试组不交叉，以及训练均值与 checkpoint 统计一致。

输出总体、每受试者、每连续记录的均值/标准差、R_var、Pearson r、MAE/RMSE、
训练均值和训练中位数基线及其增益；输出记录轨迹图和误差分布图。相关系数对
不足三个点或常数序列返回 null。轨迹只在单条记录内连接，禁止跨场景拼接为时间。
受试者去均值统计只用于诊断，不是使用测试标签校准后的部署成绩。

## Feature controls

`python -m src.training.audit_feature_controls --model-dir MODEL_DIR --export-dir EXPORT_DIR
--expected-predictions PRED.csv --output OUTPUT_DIR`

冻结同一 checkpoint，完整同一测试集，三个预设随机种子：11、29、47。
严格加载 state_dict，先验证原始推理逐窗口复现（绝对容差 0.001 bpm）。

| 条件 | x_time | x_freq | 配对参考 |
|---|---|---|---|
| original | 原始 | 原始 | original |
| reencoded | 原始 | 从标准化后的 x_time 重算 | original |
| time_shuffle_time_only | 时间置乱（所有通道共享顺序） | 原始 | original |
| freq_shuffle_freq_only | 原始 | 频率 bin 置乱（所有通道共享顺序） | original |
| time_shuffle_reencoded | 时间置乱 | 从置乱信号重算 | reencoded |
| circular_shift_time_only | 非零循环平移 | 原始 | original |
| circular_shift_reencoded | 非零循环平移 | 从平移信号重算 | reencoded |
| replace_time_only | 其他测试受试者输入 | 原始 | original |
| replace_freq_only | 原始 | 其他测试受试者输入 | original |
| replace_both | 同一 donor 的双分支 | 同一 donor 的双分支 | original |

donor 只按受试者选择，不看标签；有放回采样。种子、donor 索引、donor 组、
每窗口的配对误差变化均落盘。单分支破坏可能制造不一致/OOD 输入；循环平移
不一定破坏 HR 信息；所有结果仅作信息利用诊断。不能将“敏感”直接等同于
“学到了心跳周期”。

导出 x_time 已做全局 zscore，无法恢复 FFT 前的幅度，因此同步重算的条件必须
对照 reencoded，且单独报告 reencoded 相对 original 的变化。

## Execution and follow-up

已有 CSV 的统计/绘图在本地 uv 环境运行并计时。缺少窗口/checkpoint 的推理在
Kaggle 私有 CPU notebook 运行，不启动训练。`kaggle/prepare_ftu_source_audit.py`
将原 notebook 中的模型定义打包，以避免模型代码版本漂移。

只有当原始预测审计显示跟踪能力后，才推进 PhysDrive Source-Only/Source-Free。
连续 RDA → EDACM → HR-AdaVMD 的信号图使用完整记录及其采样率/标签时间轴；
不能把标准化窗口拼成“连续原始相位”。已输出成功、失败、HR 变化最大记录的
示例 ID，供下一步连续信号解释使用。
