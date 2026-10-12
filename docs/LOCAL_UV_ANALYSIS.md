# 本地 uv 分析环境

模型训练继续放在 Kaggle。已有数据、确定计算在五分钟内完成的诊断和统计分析，直接本地运行；预计更长或缺少数据时再使用 Kaggle。判断依据采用同类任务的实测计时，不凭文件小就推断计算便宜。

环境使用 Windows native uv、Python 3.13 和仓库 `.venv`；`uv.lock` 固定 CPU 分析依赖，不包含 PyTorch 训练环境。

```powershell
setup_local_uv.cmd
uv run --locked --no-sync python reproduce_bgt60_selection.py
```

复现脚本需要已下载的两份 p03 复数 NPZ 和 Kaggle 选择实验 JSON。默认输入位于 `tmp/bgt60_p03_pair_v2_results/pair_audit`，对照位于 `tmp/bgt60_p03_selection_results/selection_audit`。可用 `--input-dir`、`--reference-dir` 和 `--output-dir` 指定其他路径。

脚本先运行三项合成检查，再直接调用与 Kaggle 相同的实验模块，校验输入 SHA256、冻结协议、选择轨迹、全部窗口预测、预设对照和汇总指标。浮点比较容差为 `1e-8`；计时与版本信息写入输出目录 `reproduction_validation.json`。本地输出默认位于 `tmp/bgt60_p03_selection_local`，不会覆盖 Kaggle 原始结果。

其他短分析同样使用 `uv run --locked --no-sync python ...`，保留协议、耗时和校验记录。更新锁定依赖后重新执行 `setup_local_uv.cmd`。

当前环境不支持依赖 Python minor-version junction 的 uv 启动器。`setup_local_uv.cmd` 使用真实版本目录中的 Python，通过标准库 `venv --copies` 创建解释器，再由 uv 从 `uv.lock` 导出带哈希的依赖并以 copy 模式同步安装。`pyproject.toml` 默认设置 `link-mode = "copy"`。这里避免直接执行 `uv sync`，因为它可能重建启动器并再次引用失效别名；日常执行使用 `--no-sync` 保留复制式解释器。

脚本默认真实解释器位于 `%APPDATA%\uv\python\cpython-3.13.14-windows-x86_64-none\python.exe`。如果位置不同，显式传入实际路径：`setup_local_uv.cmd "C:\真实目录\python.exe"`。脚本不会删除环境中的数据目录。

## 2026-10-12 本地复现记录

Windows native Python 3.13.14，NumPy 2.5.4，SciPy 1.18.1。p03 两段记录的三种选择策略计算耗时 1.42 秒；含三项合成检查、完整计算和六份 JSON 对比共 3.73 秒。全部对比通过（容差 `1e-8`），未训练模型、未上传 Kaggle。依赖首次下载安装约 1 分钟，安装时间不计入上述实验计算耗时。

此任务已确认适合本地运行。机器负载、输入规模或处理方法改变后，需重新估计耗时，不能将本次两段记录的耗时直接推广到全量原始 ADC 处理。
