现在是改进 **RDA conversion layer**，我建议不要一上来就“找一个论文算法照抄”，而是把这件事设计成一个**可替换的 RDA 前端算法池**：论文负责提供候选模块，你负责把它们组合成几条完整 pipeline，再用后面的 representation validator + backbone benchmark 判断到底有没有价值。

我建议按下面这个结构做。

---

# 1. 先把问题拆成 5 个可替换阶段

你现在的旧流程大致是：

```text
ADC
 │
 ├─ clutter suppression
 │
 ├─ Range FFT
 │
 ├─ Doppler FFT
 │
 ├─ Angle FFT
 │
 ├─ localization / range selection
 │
 ▼
统一 RDA (8 × 16 × 8)
 │
 ▼
EDACM → HR-AdaVMD → representation
```

不要把“新 RDA 算法”理解成一个整体黑盒。

更适合研究的是：

```text
                ┌─ Clutter
ADC ─ Range FFT ┤
                ├─ Doppler
                ├─ Angle / Beamforming
                ├─ Localization
                └─ Range selection
                         │
                         ▼
                    RDA cube
```

其中五类正好对应你刚才说的：

1. **Clutter suppression**
2. **Localization**
3. **Range selection**
4. **Beamforming / angle processing**
5. **Phase extraction**

最后一项严格说已经跨到 RDA → representation 了，但我建议**先放进搜索池**，因为很多 radar vital-signs 论文真正的贡献并不止于 RDA，而是在“从 RDA 找到人体微动相位”。

---

# 2. 不要直接做 5×N 个实验，先建立 Candidate Matrix

建议建：

```text
docs/RDA_ALGORITHMS.md
```

每找到一个论文算法，就记录：

| ID | 类别           | 方法                     | 输入  | 输出            | 是否需标签 | 复杂度      | 可跨数据集 |
| -- | ------------ | ---------------------- | --- | ------------- | ----- | -------- | ----- |
| C0 | clutter      | slow-time mean         | ADC | ADC           | no    | low      | ★★★★★ |
| C1 | clutter      | MTI                    | ADC | ADC           | no    | low      | ★★★★  |
| C2 | clutter      | high-pass              | ADC | ADC           | no    | low      | ★★★★★ |
| L0 | localization | max-energy             | RDA | target bin    | no    | low      | ★★★★★ |
| L1 | localization | spatial consistency    | RDA | target region | no    | medium   | ★★★★  |
| R0 | range        | fixed crop             | RDA | range ROI     | no    | low      | ★★★★★ |
| R1 | range        | energy-based           | RDA | range ROI     | no    | low      | ★★★★  |
| B0 | beamforming  | FFT beamforming        | RX  | angle bins    | no    | low      | ★★★★★ |
| B1 | beamforming  | conventional BF        | RX  | angle         | no    | medium   | ★★★★  |
| P0 | phase        | single-bin phase       | RDA | phase         | no    | very low | ★★★★★ |
| P1 | phase        | multi-bin phase fusion | RDA | phase         | no    | medium   | ★★★★  |

这样以后别人说：

> “我找到一个新的 localization 算法。”

你不是再改整个 pipeline，而是：

```text
L2 = paper-X localization
```

然后插进去。

---

# 3. 第一轮不要追求 SOTA，先找“便宜但物理上合理”的算法

你现在最重要的问题不是：

> 哪个 RDA 算法论文 MAE 最低？

而是：

> **我们目前的 RDA 到底在哪一步丢掉了 HR information？**

所以第一轮候选应该优先：

### A. Clutter

优先找：

```text
mean subtraction
MTI
high-pass filtering
PCA/SVD clutter removal
background subtraction
low-rank + sparse decomposition
```

建议顺序：

```text
C0 mean subtraction
C1 temporal high-pass
C2 MTI
C3 PCA/SVD
C4 RPCA
```

因为这是最容易解释的一组：

```text
static reflection
        ↓
remove low-frequency / stationary component
        ↓
preserve micro-motion
```

---

# 4. Localization 是我最建议你重点挖的

因为你现在的代码实际上：

```text
全样本 range energy
        ↓
找最大能量区域
        ↓
固定 range crop
```

这个假设对于：

> 人在固定实验室里

可能没问题。

但对于：

> PhysDrive / 车载

就非常可疑。

尤其你已经发现：

```text
PhysDrive:
best MAE ≈ 10.75
constant baseline ≈ 10.94
```

这意味着：

> **如果人体所在的 spatial region 根本没找准，后面 CycleFormer/Mamba/Transformer 再聪明也没用。**

所以我会重点搜索：

### Localization

关键词：

```text
mmWave vital signs target localization
mmWave radar human localization vital signs
radar vital sign range angle localization
mmWave chest localization
radar ROI localization vital signs
range angle bin selection vital signs
```

尤其注意这些方法：

```text
energy-based localization
range-angle map localization
range-Doppler localization
range-angle-Doppler localization
spatial consistency
temporal consistency
tracking-based localization
```

---

# 5. Range selection 单独作为一个实验轴

你现在：

```text
range center
   ↓
连续 8 bins
```

这个非常值得挑战。

可以设计：

### R0：当前

```text
global energy center
→ ±4 bins
```

### R1：peak range

```text
argmax_r E(r)
→ ±4 bins
```

### R2：weighted range

```text
r = Σ r E(r) / Σ E(r)
```

### R3：HR-band weighted

不要用总能量：

```text
E(r)
```

而使用：

```text
E_HR(r)
```

即：

```text
range
  ↓
phase / Doppler signal
  ↓
HR-band energy
  ↓
选择最可能包含心跳的 range
```

这一个尤其值得做。

因为：

> **总能量最大 ≠ 心跳信号最大。**

人体衣服、座椅、车内结构、静态反射都可能有很大的 amplitude。

---

# 6. Beamforming 不要只搜索“Angle FFT”

这里有一个容易踩的坑：

你现在：

```text
RX
 ↓
FFT
 ↓
Angle bins
```

实际上只是最简单的 beamforming。

可以搜索：

```text
conventional beamforming
Bartlett beamformer
Capon / MVDR
MUSIC
delay-and-sum
near-field beamforming
virtual array beamforming
```

但我建议**第一阶段只实现**：

```text
FFT beamforming
Bartlett
MVDR
```

因为：

```text
FFT
 ↓
便宜 baseline

Bartlett
 ↓
经典 beamforming

MVDR
 ↓
adaptive spatial filtering
```

已经可以回答：

> “角度维处理 sophistication 有没有用？”

MUSIC 之类先不要急。

---

# 7. Phase extraction 要特别小心

这一层和你现在的：

```text
EDACM
```

已经发生重叠了。

所以不要让两个项目边界混掉。

我建议明确：

```text
RDA layer
──────────────
输出：
complex RDA cube
```

然后：

```text
Representation layer
────────────────────
RDA
 ↓
target selection
 ↓
phase extraction
 ↓
EDACM / alternatives
```

但是论文搜索时仍然可以研究：

```text
phase extraction
phase difference
conjugate multiplication
arctan phase
phase unwrapping
multi-bin phase fusion
```

因为这些东西可能直接告诉你：

> 当前 EDACM 是否其实把一个更简单、更稳定的 phase representation 排除了。

---

# 8. 我会设计三层实验，而不是直接全部跑

## Phase 0：物理 sanity

完全不训练 NN。

对每种 RDA candidate 计算：

```text
range profile
Doppler profile
angle profile
HR-band energy
SNR
phase variance
phase continuity
```

然后生成：

```text
RDA diagnostic report
```

例如：

```text
                proposed    MTI    PCA    MVDR
------------------------------------------------
HR-band energy    0.21      0.34   0.31   0.38
spectral SNR      1.8       2.4    2.1    2.7
phase variance    0.73      0.51   0.55   0.48
...
```

这一步 **CPU-only**。

---

# 9. Phase 1：Representation validator

然后：

```text
RDA candidate
    ↓
EDACM + HR-AdaVMD
    ↓
x_time / x_freq
    ↓
validate_representation.py
```

你已经把这个基础设施做完了。

因此可以直接得到：

```text
representation
    structural
    variance
    spectrum
    HR correlation
    linear probe R²
```

这一步非常重要。

因为可能出现：

```text
RDA-A
HR correlation = 0.01

RDA-B
HR correlation = 0.12
```

那就已经说明：

> B 值得进入 NN 实验。

无需先花 GPU。

---

# 10. Phase 2：只用一个 backbone 做筛选

不要一上来：

```text
4 backbones × 10 RDA algorithms
```

会再次把 GPU 烧掉。

先固定：

```text
CycleFormer
```

或者甚至：

```text
TCN
```

跑：

```text
RDA candidate × dataset
```

尤其：

```text
FTU
BGT60
PhysDrive
```

最后形成：

| RDA            |  FTU | BGT60 | PhysDrive |
| -------------- | ---: | ----: | --------: |
| current        | 3.76 |  3.14 |     10.75 |
| MTI            |    ? |     ? |         ? |
| PCA            |    ? |     ? |         ? |
| MVDR           |    ? |     ? |         ? |
| HR-range       |    ? |     ? |         ? |
| localization-X |    ? |     ? |         ? |

这时候你就能回答：

> RDA 算法到底能不能解决 PhysDrive 的 representation bottleneck？

---

# 11. 最后才做 factorial

如果发现：

```text
RDA-X
PhysDrive 10.75 → 7.8
```

那才值得：

```text
RDA-X
 ×
DLinear
ContiFormer
CycleFormer
Transformer
```

因为这时候你已经知道：

> 新 RDA 确实改变了 information ceiling。

然后才能研究：

```text
representation × backbone
```

而不是：

```text
backbone 在垃圾输入上互相竞争
```

---

# 12. 论文搜索我建议这样分优先级

你要找 2020–2026 论文的话，我会把搜索资源分成：

### Tier 1：直接相关

```text
mmWave radar vital signs
heart rate estimation
respiration estimation
remote physiological measurement
```

### Tier 2：算法组件

```text
mmWave clutter suppression
mmWave target localization
range bin selection
range-angle localization
beamforming vital signs
phase extraction radar vital signs
```

### Tier 3：车载

这个对 PhysDrive 尤其重要：

```text
in-vehicle mmWave vital signs
driver monitoring radar vital signs
automotive radar physiological sensing
mmWave driver heart rate
```

因为你真正要解决的是：

```text
laboratory
   ↓
vehicle
```

而不是一般意义上的 radar signal processing。

---

# 13. 最终代码架构也应该顺着这个实验设计

我建议最终变成：

```text
src/radar/
├── rda.py
├── clutter.py
├── localization.py
├── range_selection.py
├── beamforming.py
└── phase.py
```

然后：

```python
RDAConfig(
    clutter="mti",
    localization="energy",
    range_selection="hr_band",
    beamforming="fft",
)
```

得到：

```text
ADC
 ↓
clutter method
 ↓
RDA
 ↓
localization
 ↓
range selection
 ↓
beamforming
 ↓
complex RDA
```

**而不是**：

```python
convert_adc_cube_to_rda_v2()
convert_adc_cube_to_rda_v3()
convert_adc_cube_to_rda_v4()
```

后者很快会变成一坨实验代码。

---

# 14. 你现在最值得做的第一批

如果让我替你定第一轮，我不会选十几个论文算法，而是：

```text
                当前 baseline
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
      Clutter      Localization   Range
        │            │            │
      mean          energy       global
      MTI           spatial      HR-band
      PCA           temporal     weighted
        │            │            │
        └────────────┼────────────┘
                     ↓
              Beamforming
                FFT / MVDR
                     ↓
                complex RDA
                     ↓
             EDACM + AdaVMD
                     ↓
            representation probe
```

也就是**先用 8～10 个“经典且可解释”的候选构成 baseline pool**，而不是追求 8～10 篇论文的完整复现。

然后：

```text
CPU diagnostic
      ↓
representation validator
      ↓
linear probe
      ↓
筛掉 70% 候选
      ↓
CycleFormer
      ↓
PhysDrive
      ↓
最好的 2～3 个
      ↓
4-backbone ablation
```

这和你现在已经搭好的 NN 实验基础设施是完全接得上的。

**尤其值得强调的是：你现在已经证明“backbone 不是主要矛盾”，所以 RDA 这条线应该把目标定成“提高 representation information ceiling”，而不是单纯追求更漂亮的 RDA cube。** 这样 RDA 工作和你前面已经完成的 `RadarRepresentation → validator → linear probe → backbone ablation` 是一条完整的因果链。
# p.s. 
现在有新的数据集 https://www.nature.com/articles/s41597-026-07172-9 

**第一优先：把数据下载下来，先别训练。**

然后做：

```text
① 解析 radar_rFFTs.zlib
② 解析 radar_timestamps.csv
③ 解析 movesense_ecg.csv
④ 对齐 radar ↔ ECG
⑤ 复现论文自己的 FFT/phase HR baseline
```

论文还提供了 `ExampleCode.ipynb` 和 `helper_fns.py`，可以直接作为数据读取起点。([DOI][1])

然后：

```text
⑥ range-selection ablation
⑦ phase extraction ablation
⑧ representation validator
⑨ subject-level CV
⑩ 最后才接 CycleFormer
```

**而对于你目前负责的 RDA conversion：先不要把它当成“第四个 ADC 数据集”。**它更适合作为一个外部参照，帮助你验证 **range selection / phase extraction / representation**；真正能让你测试“新的 ADC→RDA 算法”的，仍然需要原始 ADC 数据集。

这篇数据集最大的价值，恰恰是它能帮你把 **“RDA 改好了”** 和 **“只是我们现有三个数据集上的 pipeline 碰巧有效”** 区分开。
# 运行要求
* 本地不训练 到kaggle训练
* kaggle cli 在 ~/kaggle uv管理
