# TDECQ / TECQ 测量工具（Release V1.3）

对 PAM4 接收波形测量 IEEE 802.3-2022 风格的 **TDECQ（光域）/ TECQ（电域）**，
并支持 IEEE P802.3dj 草案参考均衡器（15-tap FFE + 1-tap DFE、抽头限值表）。

纯 Python 实现（numpy / scipy / matplotlib），无编译依赖；提供命令行与
图形界面两种使用方式。本目录是**自包含发布包**：整个目录拷贝到任何机器，
装好三个第三方库即可运行。

## 核心特性

1. **两种测量模式**（`--tdecq-mode`）
   * `standard`（默认）：在参考接收机约束下**直接最小化 TDECQ** 的参考
     FFE 抽头（草案档另含 1-tap DFE）；IEEE 抽头命名（pre-cursor 抽头读取
     主抽头之后的采样）；OMA / P_ave 取自未均衡波形（与 FFE 增益无关）；
     光域使用 4 阶 Bessel-Thomson 参考接收机（3 dB @ 0.5×波特率，可调）。
     支持仪器噪声 RMS 扣除（`--sigma-s`）、左右窗 sigma 组合方式
     （`--sigma-combine min|max`）、Ceq 噪声增强模型
     （`--ceq-model white|bt4`）。
   * `legacy`：历史发布版的候选选择行为（复现 33 例 release manifest；
     光域为 0.8 ps/nm @ 1310 nm 色散 + 一阶低通前端，非标准一致）。
2. **标准/草案严格分离**：`--standard ieee8023-2022 | ieee8023dj-draft`。
   草案机制（有界 FFE/DFE、抽头限值表）在 2022 档被直接拒绝，两档输出
   JSON 字段互相隔离并带 `standard` 标记（见 [docs/STANDARDS.md](docs/STANDARDS.md)）。
3. **码型全自动**：`auto` 从波形/表头自动识别；`prbs2`–`prbs31` 任意阶
   单 LFSR（自动恢复 seed、符号偏移与测试窗，无需先验）；802.3 双平面
   `prbs13dual`（自动搜索 seed_a/seed_b 与相位）；Gray 映射
   `prbs13q` / `prbs31q`；SSPRQ（Clause 120.5.11，65535 符号，与官方序列
   逐位一致）；以及自定义本征多项式 `prbs:<poly>` / `prbsq:<poly>`。
4. **CSV 全自动解析**：Virtuoso 导出（`VT(...)`、任意节点名、均匀或自适应
   时间网格）与普通 `time,out` 记录自动识别；电压/功率单位自动归一化
   （V→mV、W→mW），不需要按文件做任何配置。
5. **有界 FFE + 1-tap DFE（草案实验特性）**：`--ffe-bounds` 启用抽头限值表
   约束的候选池（bounded QP / active-set / damped MMSE / 渐进坐标下降，
   `0 ≤ b1 < 0.3`）；显式请求的 tap 几何会被锁定，绝不被自动选择覆盖。
6. **GUI**：tkinter + matplotlib（TkAgg），单次分析 + 批处理 + 均衡后眼图
   嵌入显示与导出（PNG/SVG/PDF）、批处理结果表 + CSV 导出；分析在后台
   独立进程中运行，可随时 Stop。

## 环境要求与安装

| 项 | 要求 |
|---|---|
| Python | **3.10+**（开发验证用 3.14） |
| numpy / scipy / matplotlib | `numpy>=1.24`、`scipy>=1.10`、`matplotlib>=3.5` |
| tkinter | 仅 GUI 需要；Windows/macOS 官方安装包自带，Linux 需单独装 |

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy>=1.24 / scipy>=1.10 / matplotlib>=3.5
```

Linux（Debian/Ubuntu）缺 tkinter：`sudo apt install python3-tk`；
Fedora：`sudo dnf install python3-tkinter`。

在 `run_gui.py` 旁边建好 `.venv` 后，即使用了不带依赖的解释器启动 GUI，
`run_gui.py` 也会自动切换到 `.venv` 里的解释器重新启动。

## 快速开始

```bash
# CLI：默认 IEEE 802.3-2022 档 + standard 测量模式（5-tap 参考 FFE）
python run_tdecq.py --input <waveform.csv> --rate-gbps 112 --pattern auto \
    --domain electrical --out out.json -p eye.png

# P802.3dj 草案档（15-tap FFE + 1T DFE 参考均衡器，默认几何 3pre/11post，
# 有界候选池默认开启；tap 限值表 + 0<=b1<0.3 生效）
python run_tdecq.py --input w.csv --rate-gbps 106.25 --pattern prbs11 \
    --standard ieee8023dj-draft --out dj.json

# 显式 tap 几何（任何档位）：几何被锁定，永不被自动选择覆盖
python run_tdecq.py --input w.csv --rate-gbps 106.25 --pattern prbs11 \
    --standard ieee8023dj-draft --ffe-taps 9 --ffe-pre-cursor 2 --out dj9.json

# GUI
python run_gui.py
```

不带 `--out` 时只在终端打印测量结果，不写 JSON。完整参数见
[docs/USAGE.md](docs/USAGE.md)。

## 目录结构

```
Release/
├── run_tdecq.py            # CLI 入口（标准/草案档、几何锁定、绘图）
├── run_gui.py              # GUI 入口（环境自检，必要时自动切换 .venv）
├── gui/
│   ├── tdecq_gui.py        # tkinter GUI（Single analysis + Batch 两页）
│   └── analysis_worker.py  # 后台分析进程（multiprocessing spawn）
├── src/tdecq_tool/
│   ├── pipeline.py         # 端到端分析流水线（analyze 入口）
│   ├── tdecq_std.py        # standard 模式测量核心（直接最小化 TDECQ）
│   ├── eq.py               # FFE/MMSE/DFE 均衡、sigma 拟合、指标公式
│   ├── ffe_bounds.py       # 草案抽头限值表 + 有界求解器族
│   ├── patterns.py         # PRBS/SSPRQ 码型生成、对齐、seed 恢复
│   ├── csv_io.py           # CSV 自动解析、单位归一化、重采样
│   ├── optical.py          # 光域前端（BT4 参考接收机 / legacy 前端）
│   ├── plot.py             # 诊断图 PNG（眼图/直方图/波形/参考码型）
│   ├── seed_diag.py        # opt-in seed 诊断流（ASCII）
│   └── standards.py        # 标准档位注册表（单一权威来源）
└── docs/
    ├── USAGE.md            # CLI/GUI 完整使用说明
    └── STANDARDS.md        # 2022 标准档 vs P802.3dj 草案档对照
```

## 文档

* [docs/USAGE.md](docs/USAGE.md) — 环境安装、CLI 全参数参考、测量模式与
  码型说明、GUI 操作手册、输出 JSON 字段、常见问题。
* [docs/STANDARDS.md](docs/STANDARDS.md) — `ieee8023-2022` 与
  `ieee8023dj-draft` 两档的定义出处、参考接收机差异、抽头限值表与 JSON
  字段映射。
