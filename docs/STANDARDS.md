# 标准档位：IEEE 802.3-2022 与 IEEE P802.3dj 草案

本工具用 `--standard` 显式区分**已发布标准**与**草案**两套参考接收机定义，
实现在 `src/tdecq_tool/standards.py`（单一权威注册表）。两档在一次测量中
不可能混用：2022 档直接拒绝一切草案机制（`--ffe-bounds`、`--dfe-b1-unbounded`
会报错），所有输出 JSON 顶层带 `standard` 标记。

注意区分两个正交的概念：

* **标准档位**（`--standard`）：参考均衡器结构、抽头限值表、DFE 的定义来源；
* **测量模式**（`--tdecq-mode`，默认 `standard`）：抽头如何求解（直接最小化
  TDECQ vs 历史候选选择）以及光域前端（BT4 参考接收机 vs 历史前端）。
  光域前端由测量模式决定，与档位无关——standard 模式在两个档位下都使用
  Bessel-Thomson 参考接收机。

## IEEE 802.3-2022（已发布，默认档 `ieee8023-2022`）

| 项目 | 定义 | 本工具实现 |
|---|---|---|
| 指标 | TDECQ（光域，Clause 121.8.5.3）/ TECQ（电域对应） | `tdecq_db` / `tecq_db` |
| 测试码型 | SSPRQ（Clause 120.5.11，65535 符号）；128G MZM 双平面 PRBS13 | `patterns.ssprq_symbols()`（与官方序列逐位一致）、`prbs13dual` |
| 参考接收机 | 5-tap、T/2 间隔 FFE，单位增益（抽头平方和开根 = Ceq） | standard 模式：5-tap 直接最小化（`tdecq_std`，默认几何 0 pre / 4 post，主抽头固定）；legacy 模式：历史 MMSE 候选族 |
| DFE / 抽头限值表 | **不存在** | `--ffe-bounds`、`--dfe-b1-unbounded` 在本档报错；`dfe_*`、`ffe_bounds_*` 字段不输出 |
| 光域前端（standard 模式） | 参考接收机 = 4 阶 Bessel-Thomson，3 dB @ 0.5×波特率 | `optical.reference_receiver_bt4`（`--rx-bw-over-baud` 可调） |
| Ceq 模型 | 白噪声增强 `sqrt(Σc²)` | `--ceq-model white`（默认） |
| 历史出处 | TDECQ 方法源自 IEEE 802.3bs（king_3bs_01_0317，"5 tap, T/2 spaced FFE"），经 802.3cd/db 并入 802.3-2022 | `standards.py` citation |

## IEEE P802.3dj 草案（`ieee8023dj-draft`，非发布标准，可能变化）

草案出处（D2.1 任务组采纳）：`chayeb_3dj_02_2509` "TDECQ Reference
Equalizer Definition"、`ghiasi_3dj_01a_2509` "Adjusting TDECQ Limit with
Addition of 1T DFE"；抽头限值表来自 `welch_3dj_01_2405` 及 D2.1 裁决
#313/#183。

| 项目 | 草案定义 | 本工具实现 |
|---|---|---|
| 参考均衡器 | 15-tap FFE + 1-tap DFE | 默认几何 3 pre / 11 post；standard 模式下主抽头位置在 0..3 中搜索 |
| FFE 抽头限值表（相对主抽头 n，主抽头归一 ≈1） | 主 0.8–2.5；\|n\|=1 −0.4–0.05；\|n\|=2 −0.1–0.2；\|n\|≥3 −0.1–0.1 | `ffe_bounds._BOUND_TABLE` |
| 归一化 / 差分限 | 单位增益（Σw=1）；\|w(+1)/w(0) − b(1) − w(−1)/w(0)\| ≤ 0.25 | `check_feasible`、`solve_bounded_qp` |
| 结构上限 | 3 pre / 13 post | `FFE_BOUNDS.max_pre/max_post` |
| DFE | 1-tap，反馈 `(OMA_outer/2)·x(n−1)`（前一对齐理想符号），`0 ≤ b(1) < 0.3`（开上界） | `ffe_bounds.pam4_feedback`、`apply_dfe`；`b1` 网格 + 联合求解 |
| 有界候选池 | — | `--ffe-bounds`（草案档默认开启）：tap 数 N∈{5,7,9} + 显式请求几何；bounded QP / eye-lid QP / active-set / damped MMSE / 联合 QP / 渐进坐标下降；显式几何被锁定 |
| Ceq 模型 | BT4 自相关噪声增强 | `--ceq-model bt4`（默认） |
| 光域前端（standard 模式） | 参考接收机 = 4 阶 Bessel-Thomson，3 dB @ 0.5×波特率，色散不施加 | 同 2022 档 standard 模式（`--rx-bw-over-baud` 可调） |
| legacy 诊断 | `--dfe-b1-unbounded` 追加显式受罚的 `0≤b1<1` 候选（仅 legacy 模式） | 规范候选不受影响 |

## 两档行为速查

| 方面 | `ieee8023-2022` | `ieee8023dj-draft` |
|---|---|---|
| 默认 FFE 几何 | 0 pre / 4 post（5 tap） | 3 pre / 11 post（15 tap） |
| standard 模式主抽头 | 固定 index 0 | 在 0..3 中搜索 |
| 有界池（--ffe-bounds） | 拒绝（报错） | 默认开启 |
| DFE | 无 | 1-tap，0≤b1<0.3 |
| Ceq 默认模型 | white | bt4 |
| JSON | 无草案字段 | 输出 `dfe_*`、`ffe_bounds_*`、`ffe_geometry_*`，`standard_is_draft=true` |

## 草案档专属 JSON 字段映射

| 字段 | 含义 |
|---|---|
| `dfe_b1` / `dfe_enabled` | 选中的 DFE 反馈系数 / 是否启用 |
| `dfe_feedback_source` | 恒为 `aligned_ideal_previous_symbol` |
| `dfe_oma_reference` | 恒为 `FFE_input_OMA_over_2` |
| `ffe_bounds_enabled` / `ffe_bounds_used` | 有界池是否请求 / 实际采用 |
| `ffe_bounds_strategy` | 胜出候选求解器（`mmse_un`、`bounded_qp`、`eye_lid_qp`、`active_set`、`damped_0.25/0.5/0.75`、`bounded_qp_dfe`、`joint_qp_dfe`、渐进坐标下降等） |
| `ffe_bounds_satisfied` / `ffe_bounds_violations(_details)` | 胜出抽头是否全部落在限值表内及具体违例 |
| `ffe_geometry_locked` / `ffe_geometry_explicit` | 显式几何被遵守 / 调用方显式给过几何 |
| `ffe_geometry_override(_reason)` | 仅 `--ffe-auto-geometry`：有界池改选几何及原因 |
| `ffe_bounds_warning` | 胜出抽头仍违限时的告警文本 |
| `metric_credible(_reason)` | 结果可信度（SER>0.1 或 sigma_G/OMA ≤1e-3 时为 false） |

## 使用建议

* 出标准一致性结果：`--standard ieee8023-2022`（默认）+ 默认 standard
  测量模式；有仪器本底噪声时给 `--sigma-s`。
* 出草案预研结果：`--standard ieee8023dj-draft`（有界池与 DFE 自动生效）；
  显式指定几何时加 `--ffe-taps/--ffe-pre-cursor`，几何即被锁定。
* 与旧版发布结果对账：加 `--tdecq-mode legacy`（注意其 pre/post 抽头命名
  与 IEEE 相反，光域前端亦为历史前端，非标准一致）。
