# FTU P1：单种子双分支 TCN

**本轮未观察到可靠的记录内动态心率跟踪；不满足扩展到更多模型或跨域实验的证据要求。**
输出确实随输入变化，但与真实记录内 HR 变化近乎不相关。这个结论只针对本次固定配置、划分和预算，不能证明 FTU 不可学习。

固定种子 42、最多 30 epochs、早停耐心值 8；仅训练一次，按验证集 MAE 选 checkpoint。
实际训练 9 epochs，选择 epoch 1。
最早 epoch 被验证集 MAE 选中；后续训练没有改善该选择指标。未按测试集相关性改选 checkpoint。
输入为已验证的 EDACM-only v2；单通道时域和频域，保持历史 6/2/2 受试者划分。

| 测试条件 | MAE (bpm) | 全局 R_var |
|---|---:|---:|
| TCN | 5.4988 | 0.3272 |
| 跨受试者双分支替换 | 5.5554 | 0.3272 |
| 训练均值 | 4.3315 | 0 |
| 训练中位数 | 5.9851 | 0 |

R_var 定义为标准差比 σ(pred)/σ(HR)，不是方差之比。全局统计不能单独说明动态跟踪。

| 记录内去均值后汇总 | Pearson r | R_var |
|---|---:|---:|
| 原始 | -0.0321 | 0.2574 |
| 替换 | -0.0327 | 0.2574 |

替换 ΔMAE = +0.0567 bpm；受试者与逐记录指标见 tracking_summary.csv / record_metrics.csv。

## 协议和边界

- p01 与 p08 按记录后缀、窗口位置一一交换；不参考 HR、误差或模型输出，两条分支同步替换。
- 原始与负对照使用同一 best.pt；原标签保持不变。替换预测已与对应供体预测核对。
- 替换是双射，因此整体预测分布与全局 R_var 不变；重点比较标签对应关系、MAE 和记录内相关性。
- 保留条件与时间位置的替换是保守诊断，共享条件变化可能使替换后仍有相关性。
- 测试仅两名已参与探索的受试者；每条记录八个重叠窗口。不能据此推断总体泛化或显著性。
- 雷达起始偏移仍假定为 0 秒；结果不是硬件同步验证。
- 固定随机种子，确定性算法采用 warn_only；不承诺跨硬件位级一致。
- AdamW、SmoothL1(beta=0.5)、梯度裁剪 1.0、ReduceLROnPlateau(factor=0.5, patience=4)；未用测试结果调参。

## 产物与复核

- model/best.pt：验证集选择的 checkpoint；final.pt 保存相同最佳权重及测试指标。
- predictions.csv：全部训练/验证/测试原始预测及测试负对照，共 4,224 行。
- plots/：两名测试受试者的全部记录轨迹、最大真实 HR 变化记录及误差图。
- protocol.json、model/run_config.json、source_manifest.json、input_window_hashes.json：完整配置及来源。
- 独立核对 3,520 个输入窗口哈希、划分与标签、checkpoint 哈希和替换映射。
- Kaggle 全流程 66.74 秒；本地统计与校验 4.74 秒。

[Kaggle 实验](https://www.kaggle.com/code/goldfish9901/ftu-edacm-tcn-p1-seed42)

## 各测试受试者的记录内动态指标

| 受试者 | 条件 | 记录去均值 r | 记录去均值 R_var | 逐记录 r 中位数 |
|---|---|---:|---:|---:|
| FTU/participant/1 | original | 0.0139 | 0.3370 | -0.0994 |
| FTU/participant/8 | original | -0.0634 | 0.2157 | 0.0855 |
| FTU/participant/1 | subject_swap | 0.0037 | 0.3340 | 0.0829 |
| FTU/participant/8 | subject_swap | -0.0572 | 0.2177 | -0.0964 |
