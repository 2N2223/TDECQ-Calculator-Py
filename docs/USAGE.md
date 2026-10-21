# TDECQ / TECQ 工具使用说明（V1.3）

本说明覆盖发布包内的命令行（`run_tdecq.py`）与图形界面（`run_gui.py`）。
标准档位（`ieee8023-2022` / `ieee8023dj-draft`）的定义与差异见
[STANDARDS.md](STANDARDS.md)。

## 1. 环境与安装

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.10+**（开发验证用 3.14） | 纯 numpy 实现，无编译依赖 |
| numpy / scipy / matplotlib | `numpy>=1.24`、`scipy>=1.10`、`matplotlib>=3.5` | 唯一第三方依赖 |
| tkinter | Windows/macOS 官方安装包自带；Linux 需 `python3-tk` | 仅 GUI 需要，CLI 不需要 |

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy>=1.24 / scipy>=1.10 / matplotlib>=3.5
```

Linux 缺组件：Debian/Ubuntu `sudo apt install python3-venv python3-pip python3-tk`；
Fedora `sudo dnf install python3-tkinter`；macOS Homebrew Python
`brew install python-tk@3.14`（版本号对齐所用 Python）。

Windows 提示：把 `python` 换成 `py -3`，激活用 `.venv\Scripts\activate`。

## 2. CLI 快速开始

```bash
# 电域 TECQ，码型/网格/单位全部自动（默认档 ieee8023-2022，standard 模式）
python run_tdecq.py --input wave.csv --rate-gbps 112 --pattern auto \
    --domain electrical --out out.json -p eye.png

# 光域 TDECQ，P802.3dj 草案档（15-tap FFE + 1T DFE，有界候选池默认开启）
python run_tdecq.py --input wave.csv --rate-gbps 106.25 --pattern prbs11 \
    --standard ieee8023dj-draft --out dj.json

# 显式 FFE 几何：被锁定，永不被自动选择覆盖
python run_tdecq.py --input wave.csv --rate-gbps 106.25 --pattern prbs11 \
    --standard ieee8023dj-draft --ffe-taps 9 --ffe-pre-cursor 2 --out dj9.json

# 复现历史发布版行为
python run_tdecq.py --input wave.csv --rate-gbps 112 --pattern auto \
    --tdecq-mode legacy --out legacy.json
```

不带 `--out` 时只把结果打印到终端，不写任何文件。`RuntimeWarning`（如
码型不匹配）会以 `WARNING: ...` 打印到 stderr，并同时记录进 JSON 的
`pattern_match_warning` 字段。

## 3. CLI 参数完整参考

### 3.1 输入与输出

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input PATH` | 必填 | 输入波形 CSV（Virtuoso 导出或 `time,out` 记录，自动识别） |
| `--out PATH` | 无 | 输出 JSON 路径；缺省时只打印结果不写文件 |
| `-p, --output-plot PATH` | 无 | 输出诊断图 PNG（4 面板，见第 9 节） |
| `--no-plot` | 关 | 完全跳过绘图 |
| `--eye-color NAME` | `blue` | 眼图密度配色（matplotlib 色名/cmap 前缀） |
| `--case ID` | 无 | 纯诊断用例标签（只进 JSON `id`，绝不影响算法） |
| `--domain` | `electrical` | `electrical`（TECQ，mV）/ `optical`（TDECQ，mW） |
| `--rate-gbps F` | `112` | PAM4 聚合数据率（Gbps）；UI = 2/rate ns（106.25 → 18.82 ps） |
| `--samples-per-ui N\|auto` | `auto` | auto：速率 ≥120 Gbps 取 32，否则 64；显式整数优先 |
| `--window N\|auto` | `auto` | 测试窗（UI），规则见 6.5 节 |

### 3.2 码型

| 参数 | 说明 |
|---|---|
| `--pattern PATTERN` | 默认 `auto`。可选：`auto`、`ssprq`、`prbs13dual`、`prbs13q`、`prbs31q`、`prbs2`…`prbs31`（含 `prbs11/prbs13/prbs31`），或自定义多项式 `prbs:<poly>`（二进制映射）/ `prbsq:<poly>`（802.3 Gray），如 `'prbs:x^13+x^4+x^3+x+1'`。多项式必须本征（周期 = 2ⁿ−1），否则参数校验直接报错。裸 `prbs`/`prbsn` 已废弃，会被拒绝 |

`prbs2`–`prbs31` 全部走自动 seed/相位/窗长恢复（无需先验 seed 或偏移）；
`prbs11`、`prbs31`、`ssprq` 也可使用发布参考流；`prbs13dual` 自动搜索
802.3 MZM 双平面两组 seed（`seed_a`/`seed_b`）及相位，无硬编码 (1,137)。

### 3.3 测量模式与标准档位

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tdecq-mode` | `standard` | `standard`：参考 FFE（草案档另含 DFE）抽头直接最小化 TDECQ，IEEE 抽头命名，OMA/P_ave 取自未均衡波形。`legacy`：历史发布版候选选择（非最小化；pre/post 命名与 IEEE 相反） |
| `--standard` | `ieee8023-2022` | `ieee8023-2022`（发布标准，5-tap FFE，无 DFE）/ `ieee8023dj-draft`（草案，15-tap FFE + 1T DFE，默认几何 3pre/11post，有界池默认开）。草案特性在 2022 档被拒绝 |

### 3.4 standard 模式专用选项

| 参数 | 默认 | 说明 |
|---|---|---|
| `--sigma-s RMS` | `0` | 仪器（示波器/O-E）噪声 RMS（波形单位），测量时从方差中扣除 |
| `--sigma-combine` | `min` | 左右窗 sigma 组合：`min`（较差窗起限制作用，默认）或 `max` |
| `--ceq-model` | 按档 | FFE 噪声增强模型：`white`（2022 档默认）/ `bt4`（草案档默认，BT4 自相关） |
| `--rx-bw-over-baud RATIO` | `0.5` | standard 模式光域：4 阶 Bessel-Thomson 参考接收机 3 dB 带宽（单位：波特率） |

### 3.5 FFE 几何与有界均衡（草案机制）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ffe-pre-cursor N` | 按档 | 主抽头之前的 pre-cursor 抽头数（0..5；2022 档默认 0，草案档默认 3）。**显式给出即锁定几何** |
| `--ffe-post-cursor M` | 按档 | 主抽头之后的 post-cursor 抽头数（0..30；2022 档默认 4，草案档默认 11） |
| `--ffe-taps T` | 无 | 总抽头数便捷形式（1..36），等价 `--ffe-post-cursor T-1-<pre>`；与显式 post 同时给出时校验一致性 |
| `--ffe-bounds` | 按档 | 启用 IEEE P802.3dj 草案有界多 tap FFE + 1-tap DFE 候选池（`0≤b1<0.3`；bounded QP/active-set/damped/渐进坐标下降；tap 数 N∈{5,7,9} + 显式请求几何）。2022 档下给出此选项直接报错 |
| `--no-ffe-bounds` | — | 草案档下关闭有界池（诊断用） |
| `--ffe-auto-geometry` | 关 | 仅 legacy 模式：允许有界池自动改选 tap 数/主抽头位置（`ffe_geometry_override`）。默认（推荐）：显式请求的几何被锁定。standard 模式下拒绝此选项 |
| `--dfe-b1-unbounded` | 关 | 追加显式受罚的诊断 DFE 候选 `0≤b1<1`（需草案档；规范 `b1<0.3` 候选保持不变） |
| `--no-equalized-validation` | 关 | 关闭闭眼 canonical-fallback 记录上的"已验证均衡"分支（始终报告诚实的 identity 眼） |

### 3.6 其他

| 参数 | 默认 | 说明 |
|---|---|---|
| `--threshold-mode` | `linear` | PAM4 判决阈值模型：`linear`（IEEE 兼容，Pave±OMA/3）/ `class_midpoint`（相邻实测电平中点，适合非线性眼；别名 `midpoint`/`nonlinear`） |
| `--seed-diagnostics` | 关 | 向 stderr 输出 ASCII seed/相位/阈值诊断 |
| `--seed-diagnostics-file PATH` | 无 | 把 seed 诊断追加写入文件（同时开启诊断） |

seed 诊断也可用环境变量 `TDECQ_SEED_DIAGNOSTICS` / `TDECQ_SEED_DIAGNOSTICS_FILE`
开启；诊断流只含事件与标量，绝不输出波形样本。

## 4. 测量模式：standard 与 legacy

| 方面 | `standard`（默认） | `legacy` |
|---|---|---|
| 抽头求解 | 在参考接收机约束下直接最小化 TDECQ（LS 起点 + 无导数模式搜索，交替重新对准眼中心） | 历史发布版候选池选择（identity/MMSE 家族 + 确定性两级选择），非最小化 |
| 抽头命名 | IEEE（pre-cursor 抽头读主抽头之后的采样） | 历史命名（与 IEEE 相反，为逐位复现而保留） |
| OMA / P_ave | 取自**未均衡**波形的稳定电平段（≥6 符号长游程；退化为脉冲响应 LS 拟合），与 FFE 增益无关 | 取自均衡后类均值（另报 `pre_FFE_*` 原始眼参考值） |
| 直方图 sigma | 2048-bin 无标签直方图、相邻阈值各自计数，0.04 UI 宽采样窗 @ 0.45/0.55 UI | `hist`（64 bin）/ `hist_ieee`（512 bin 固定范围）/ `samp` |
| 光域前端 | 4 阶 Bessel-Thomson 参考接收机，3 dB @ `--rx-bw-over-baud`×波特率（默认 0.5），色散不施加（属于信道特性） | 0.8 ps/nm @ 1310 nm 色散（纯延迟）+ 一阶低通（112G 42 GHz，其余 0.75×波特率）+ 1927 UI 稳态窗 |
| 光域结果 | 面向标准一致性 | 非标准一致（GUI 中会显示橙色提示） |

legacy 模式用于与历史发布结果逐位对账；新测量请使用默认 standard 模式。

## 5. 码型（--pattern）与自动识别

支持的码型：

| 名称 | 说明 | 周期（符号） |
|---|---|---|
| `auto` | 从 CSV 表头与波形自动识别（先表头关键字，后记录形状/解码匹配） | — |
| `prbs2`…`prbs31` | 单 LFSR，标准本征多项式表；`prbs11`（x¹¹+x⁹+1）、`prbs13`（x¹³+x¹²+x¹⁰+x⁹+1）、`prbs31`（x³¹+x²⁸+1，seed 0x402） | 2ⁿ−1 |
| `prbs13q` / `prbs31q` | 802.3 Gray 映射（00→0, 01→1, 11→2, 10→3） | 2ⁿ−1 |
| `prbs13dual` | 802.3 MZM 双平面：symbol = 2A + (A⊕B)，两平面同为 x¹³+x¹²+x¹⁰+x⁹+1 | 8191 |
| `ssprq` | 802.3 Clause 120.5.11 SSPRQ，65535 符号（三段 PRBS31 + 反转 Gray），实现与官方序列逐位一致 | 65535 |
| `prbs:<poly>` / `prbsq:<poly>` | 自定义本征多项式（二进制映射 / 802.3 Gray） | 2ⁿ−1 |

auto 识别顺序：表头提示（`ssprq`、`prbs13dual`、`prbs13q`/`prbs31q`、
`prbs13`）→ Virtuoso 记录形状启发式（≥10000 UI 或峰峰值 <700 mV 判 ssprq，
否则 prbs13；双平面必须显式指定）→ 112G 解码匹配（电域先 ssprq FFT，再按
31/11/13/15/16/9/7 阶尝试；只有真正从波形里恢复出 LFSR 状态的阶才会被采信，
防止随机噪声误判）。识别不确定时给出保守结果并附 `pattern_match_warning`。

## 6. 自动行为

1. **CSV 解析**：Virtuoso 导出（`VT(...)` 表头、任意节点名、均匀或自适应
   时间网格）与普通 `time,out` 记录按表头自动区分；时间列 = 单调列，信号列
   = 方差最大列。Virtuoso 电压 ×1000 归一为 mV；psf2csv 电域已是 mV；
   光域峰值 <0.01 按 W 处理 ×1000 归一为 mW。不做幅度启发式，可疑单位只给
   warning。自适应网格按解码符号/LFSR 匹配度扫描选取最优重采样相位。
2. **极性与 DC**：反相波形用补码解码 LFSR 递归探针自动翻转；明显直流偏置
   （mV 级电域、|DC| > 0.2×标称 OMA）自动扣除。
3. **CDR**：过阈值交叉相位圆均值 + spu/2，无参数。
4. **seed 恢复**（单 LFSR prbsN）：多相位/多阈值/多电平映射解码 →
   最长"干净"递归段 + GF(2) 移位回 bit0（失败退回精确 GF(2) 求解；再生流
   必须 ≥95% 复现后续比特，全零态拒绝）→ 折叠/线性判别（检测回绕接缝，
   ≥99.9% 平铺匹配才采纳）→ 置信门限（解码精度、均衡后 SER 交叉校验、
   残差重锚定）。双平面先恢复 A 平面，B = 恢复(LSB⊕A)。恢复失败回退
   canonical seed 并打 warning（`seed_ok=false`），闭眼记录再走"已验证
   均衡"分支（样本外 SER ≤0.05 且跨窗 seed 一致性 ≥0.98 才采纳）。
5. **测试窗**（`--window auto`）：prbs31/ssprq 取完整 65535 UI 周期（记录
   足够长时）；prbs13/prbs13dual 取 min(记录, 6400)；一般 prbsN（n≤16）取
   完整 2ⁿ−1 周期（存在时，至少 16 UI），n>16 取 min(记录, 65535)；其余取
   记录长度。
6. **阈值模型**：`linear` = Pave + (−OMA/3, 0, +OMA/3)（IEEE）；非线性眼可
   用 `class_midpoint`（相邻实测电平中点），两者同样作用于 sigma 拟合与
   输出阈值。

## 7. GUI 使用

```bash
python run_gui.py
```

窗口标题 "TDECQ / TECQ Measurement Tool"（1180×860，可缩放），两个页签：
**Single analysis** 与 **Batch**。`run_gui.py` 启动时自检 Python/numpy/
scipy/matplotlib/tkinter；当前解释器不合格且项目 `.venv` 存在时自动用
`.venv` 解释器重启。分析在独立进程中运行（multiprocessing spawn），主窗口
每 120 ms 轮询日志队列，因此长任务不卡界面、**Stop 可随时硬终止**；若
进程无法启动自动退化为进程内线程模式（此时 Stop 只能丢弃输出）。

### 7.1 Single analysis 页

**Measurement settings** 区（字段与 CLI 一一对应）：

| 控件 | 默认 | 说明 |
|---|---|---|
| Waveform CSV | 空 | Browse... 选择波形 CSV |
| Output JSON (optional) | 空 | 留空 = 不写 JSON |
| Data rate (Gbps) | 112 | > 0 |
| Pattern | `auto` | auto / custom / ssprq / prbs13q / prbs31q / prbs13dual / prbs2…prbs31；选 `custom` 时填写 Custom PRBS poly（可勾选 802.3 Gray mapping），多项式实时校验 |
| Domain | `electrical` | electrical / optical |
| Samples/UI | `auto` | ≥2 整数或 auto |
| Window (UI) | `auto` | ≥1 整数或 auto |
| Standard | `ieee8023-2022` | 选 2022 档时 FFE pre/post 锁定为 0/4（输入框禁用）；草案档可编辑 |
| FFE pre / FFE post | 空（按档） | ≥0 整数；留空 = 档位默认 |
| Threshold mode | `linear` | linear / class_midpoint |
| TDECQ mode | `standard` | standard / legacy |
| Ceq model | `auto` | auto / white / bt4（仅 standard 模式有效） |
| Sigma combine | `min` | min / max（仅 standard） |
| Sigma_S (mW/mV) | `0` | ≥0（仅 standard） |
| RX BW (x baud) | `0.5` | ≥1e-3（仅 standard） |
| 勾选项 | — | auto geometry selection (legacy only)、seed diagnostics (to log)、eye plot |

**Run analysis** 开始分析，**Stop** 终止当前分析。左侧 **Log** 实时显示
后台进程的全部输出；**Result summary** 列出关键指标（TDECQ/TECQ (dB)、
mode、credible（含原因）、no-FFE (dB)、actual SER、C_eq、sigma_G、
OMA/Swing、P_ave、FFE taps 与权重、DFE b1、eye centre、level source、
pattern/seed、RLM、warning、elapsed 等）。右侧 **Equalised eye
(centred, 2 UI)** 为嵌入式均衡后眼图（matplotlib 工具栏可缩放/平移），
**Save eye plot...** 可存为 PNG/SVG/PDF（dpi=200）。

### 7.2 Batch 页

* **Add files... / Add folder...** 把多个 CSV（或整个目录的 `*.csv`）加入
  队列（自动去重）；**Remove selected / Clear** 管理队列。
* 批量运行**沿用 Single 页的全部设置**（速率/码型/域/档位/模式/FFE 几何/
  standard 选项）。
* **JSON output dir (optional)**：给定后每个记录写
  `<目录>/<文件名>_tdecq.json`。
* **Run batch** 逐个运行，进度实时显示（`running 1/5: file.csv` →
  `done: 5/5 records`）；**Stop** 终止后续记录。
* 结果表列：`record`、`dB`、`credible`、`SER`、`C_eq`、`seed ok`、
  `pattern`、`warning`；出错或不可信的行标红，单条失败不中断批处理。
* **Export CSV...** 把结果表导出为 UTF-8 CSV，列：`file, mode, metric,
  credible, ser, ceq, seed_ok, seed, seed_a, seed_b, pattern, taps,
  strategy, warn, warn_text, json, error, path`。

## 8. 输出 JSON 字段

顶层按域改名：电域 `tecq_db`/`Swing_mV`/`sigma_G_mV`…，光域
`tdecq_db`/`OMA_mW`/`sigma_G_mW`…。主要分组：

| 分组 | 字段 |
|---|---|
| 档位 | `standard`、`standard_label`、`standard_is_draft`、`tdecq_mode` |
| 指标 | `<tecq|tdecq>_db`、`*_raw_db`（无 FFE 参考）、`*_pre_db`（无约束 MMSE 参考）、`sigma_G_*`、`R_*`、`C_eq`（无量纲噪声增强）、`Swing_mV`/`OMA_mW`、`P_ave_*` |
| FFE | `tap_weights`、`ffe_tap_sum`、`ffe_pre_cursor`、`ffe_post_cursor`、`ffe_tap_count`、`ffe_main_tap_index`、`ffe_largest_tap_index`、`ffe_optimisation` |
| 眼图 | `RX_levels_*`、`RX_thresholds_*`、`level_source`、`threshold_mode`、`RLM`、`ES1/ES2/ES3`、`level_gaps_*`、`DNL`，以及均衡后镜像 `post_ffe_*` |
| SER/CDR | `cdr_raw/pre/post`、`actual_SER(_raw/_pre)`、`actual_errors`、`total_symbols` |
| 码型/seed | `test_pattern`、`prbs_register`、`prbs_polynomial`、`prbs_seed`、`prbs_seed_a/b`、`seed_source`、`seed_bit_offset*`、`seed_auto_detected`、`pattern_match_warning`、`gray_mapping` |
| 运行信息 | `symbol_rate_GBd`、`data_rate_Gbps`、`spu`、`total_UI`、`test_window_symbols`、`elapsed_sec` |
| 选择诊断（legacy） | `chosen_strategy`、`selection_tier`、`selected_family`、`sigma_method`、`ffe_reason` |
| standard 模式 | `metric_credible(_reason)`、嵌套 `standard_measurement`（lag/window/eye_centre/sigma_L/sigma_R/sigma_s/oma/pave/pth/b1/taps/feasible/level_source/rx_model/candidates/config 等） |
| 草案档专属 | `dfe_b1`、`dfe_enabled`、`dfe_feedback_source`、`ffe_bounds_enabled/used/strategy`、`ffe_geometry_locked/explicit`、（legacy 路径另含）`ffe_bounds_satisfied`、`ffe_bounds_violations*`、`ffe_bounds_warning`、`ffe_geometry_override(_reason)`、`ffe_overlimit_*`、`ffe_eye_lid_*` 等 |
| 光域专属 | `ER_dB`、`post_ffe_ER_dB`、`pre_FFE_OMA_mW`、`reference_receiver`（standard 模式：`bessel_thomson_4` + 带宽；legacy：`dispersion_ps_per_nm=0.8`、`wavelength_nm=1310.0`） |

2022 档绝不输出任何 `dfe_*` / `ffe_bounds_*` 草案字段。

## 9. 诊断图（-p eye.png）

一幅 12×10 英寸 PNG（4 面板）：

1. 均衡后 PAM4 密度眼图（功率归一 2-D 直方图，PowerNorm γ=0.2，深色底），
   标注三条判决阈值与 0.45/0.55 UI 采样窗（横轴 ps）；
2. 0.45/0.55 UI 两处水平幅度直方图（512 bins，Pave±2·OMA）+ sigma_G 高斯核；
3. 前 48 UI 的原始 vs FFE 后波形，标出采样时刻；
4. 理想参考码型（48 符号，优先用实测 RX 电平），标题带 seed 恢复/warning 信息。

大标题：`TDECQ=… dB taps=[…] cdr=… SER=… BER(est)=SER/2`（Gray 假设估计，
仅供参考）。

## 10. 终端诊断行

standard 模式打印 `STD_DIAG main=… taps=… b1=… lag=… eye_centre=…
sigma_L=… sigma_R=… Ceq=… level_source=… credible=…`；开启 `--ffe-bounds`
的 legacy 路径打印 `DFE_DIAG …` 与 `FFE_EYE_DIAG …`。NOTE/WARNING 行包括：

* `WARNING: ...`（RuntimeWarning，如码型不匹配 → `pattern_match_warning`）
* `NOTE: pattern_match_warning = ...` — 码型/seed 置信度不足
* `NOTE: ffe_bounds_warning = ...` — 有界池结果仍违反限值表
* `NOTE: ffe_geometry_override = ...` — 有界池改选了几何（仅
  `--ffe-auto-geometry`）
* `NOTE: ffe_geometry_locked = requested N-tap geometry (pre=…/post=…) was
  honored` — 显式几何被遵守
* `NOTE: result not credible: ...` — `metric_credible=false`（SER>0.1 或
  sigma_G/OMA ≤1e-3）

## 11. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `No module named '_tkinter'` | Python 未带 tkinter，按第 1 节安装；CLI 不受影响 |
| `error: externally-managed-environment` | 直接用了系统 pip；先 `python3 -m venv .venv` 并激活 |
| matplotlib 后端报错（无显示服务器） | CLI 绘图用 Agg 不受影响；GUI 需图形会话，或先用 `-p eye.png` 出图 |
| 结果不可信（`metric_credible=false`） | 检查码型/速率/域是否正确；看 `pattern_match_warning` 与 `standard_measurement.candidates`；确认输入不是噪声段 |
| 想与旧版发布结果对账 | `--tdecq-mode legacy`（注意其 pre/post 命名与 IEEE 相反） |
| 数值末位差异 | 不同 numpy/BLAS 的正常浮点尾差 |
