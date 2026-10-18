# Corrections to b6bd9ba ADC phase report

1. 完整 FFT 的较大 chirp0 相位误差主要来自配置决定的积分时刻，不应解读为更多相位破坏。前8个/全部64个历史 chirp 索引的采样时刻质心分别是0.399/3.591891毫秒；没有用 HR 拟合延时。

| mode | centroid_delay_ms | phase_rmse_chirp0 | predicted_time_only_rmse | phase_rmse_mean_time |
| --- | --- | --- | --- | --- |
| legacy | 0.399 | 0.0007419203 | 0.000736901 | 1.419961e-05 |
| full_fft_crop | 3.591891 | 0.006633586 | 0.006633153 | 1.811886e-05 |

2. 旧报告的“最大相对L2差0.1564”实际含 max(参考范数,1) 分母下限。此数来自近乎零的 full_mean 连续微动条件，不是有效信号15.64%的误差。原保存求和顺序中，该条件参考范数约0.161567，差值范数约0.156401，理论无clutter增益归一化RMS差约1.22e-10。修订表逐条件列绝对误差、参考范数以及理论相干增益归一化误差；低幅度条件的真实相对误差标为undefined。complex128 在重新求均值及求和全过程复核，而非只在最后转换类型。

| case | variant | max_abs_error | reference_l2 | relative_l2 | rms_error_over_theoretical_gain |
| --- | --- | --- | --- | --- | --- |
| frame_constant | legacy_chirp_mean | 0 | 0 | undefined | 0 |
| frame_constant | legacy_none | 0.2016475 | 1.600007e+08 | 1.855629e-08 | 1.855637e-08 |
| frame_constant | full_fft_crop_chirp_mean | 0 | 0 | undefined | 0 |
| frame_constant | full_fft_crop_none | 1.61318 | 1.280005e+09 | 1.855629e-08 | 1.855637e-08 |
| continuous_motion | legacy_chirp_mean | 0.001777295 | 544376.1 | 3.175872e-08 | 1.080543e-10 |
| continuous_motion | legacy_none | 0.3354289 | 1.600002e+08 | 2.172316e-08 | 2.172319e-08 |
| continuous_motion | full_fft_crop_chirp_mean | 0.001443109 | 1.549379e-10 | undefined | 8.279851e-12 |
| continuous_motion | full_fft_crop_none | 2.072944 | 1.280002e+09 | 1.745886e-08 | 1.745889e-08 |

有效无均值消除信号相对complex128参考的相对L2误差约1.7–2.2e-8；零输出/残差条件不计算信号相对精度。全逐条件表 previous_precision_recheck.csv 同时保留原保存求和顺序，明确标识旧分母下限统计量。

3. 新协议53a1be7只要求候选通过；旧均值消除臂是预期失败机制对照，失败不再阻止真实实验。首次合成因共享读出器硬编码256点而停止，b4db09c修正长度接口后重跑，未改协议/门槛/样本。旧b6bd9ba协议的停止决定仍被保留，没有事后称旧门控已通过。
