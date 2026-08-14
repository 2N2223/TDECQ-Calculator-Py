# TDECQ / TECQ 测量工具 — V1.3.1

对 PAM4 接收波形计算 IEEE 802.3-2022 风格 **TDECQ（光域）/ TECQ（电域）** 指标。
全部参数（CSV 解析、码型识别、CDR 相位、符号偏移、测试窗长度、PRBS seed、FFE
tap 几何）均由波形自动估计，**无逐用例标定、无 case_id 参与算法**；`--case` 仅作
输出 JSON 的 `id` 诊断标签。

## 特性

- **CSV 波形自动识别**：Virtuoso 导出（`VT(...) X,Y` 表头、任意节点名、均匀或
  自适应网格）与普通 `time,out` 记录自动判定时间列/信号列并归一化单位；无需
  `csv_format`、无需指定列号。
- **码型**：`prbs11 / prbs13 / prbs31 / ssprq`，以及通用 `prbsN`（N = 2..31）与
  `--pattern prbs --prbs-order N`；`--pattern auto` 由波形/文件头决定，绝不读取
  `case_id`。
- **任意 seed / 偏移自动恢复**：PRBSn 单 LFSR seed、PRBS13 双位平面
  （802.3 mzm，`symbol = 2*A + (A xor B)`）的 `seed_a/seed_b` 与相位、SSPRQ
  起始偏移（含头部截断/周期回绕）、PRBS31 长记录线性序列任意起点，全部自动搜索。
- **可配置 FFE tap 几何**：`--ffe-pre-cursor 0..5`、`--ffe-post-cursor 0..30`、
  `--ffe-taps T`；默认 `(0,4)` 与 5-tap 版本逐位一致。主抽头锚点在索引 `pre`，
  实际能量峰（受通道群延迟影响）输出为 `ffe_largest_tap_index`。
- **IEEE 802.3dj 抽头边界约束 + 多 tap 候选（V1.3，`--ffe-bounds` 默认关闭）**：
  相对主抽头边界表（主 0.8–2.5、近旁瓣 ±0.05、远旁瓣 ±0.2、|n|≥3 ±0.1；
  Σtaps=1；差分限 |w(+1)-w(-1)|/w(0)≤0.25），tap 数 N ∈ {5,7,9} 自动选择、
  主抽头由脉冲响应峰值对齐；无约束 MMSE 超界时切换有界 QP / active-set /
  damped MMSE / 渐进式坐标下降。选择分数
  `min(hist,samp) + 0.2·D_FFE + 0.1·C_eq + 0.05·max(0,N-5)`。默认关闭时 33 例
  与 V1.2.1 **逐位一致**（仅新增 `ffe_bounds_*` 诊断字段）。绘图 suptitle 主抽头
  **加粗**、触界 tap **下划线/上划线**标注。
- **已验证均衡分支（V1.3）**：closed-eye canonical 回退记录先做 out-of-sample
  SER + cross-window seed 校验，通过才报告均衡后 TECQ
  （`closed-eye-equalized-validated`），否则诚实 identity + 告警；
  `--no-equalized-validation` 可关闭。
- **绘图**：`-p/--output-plot <xxx.png>` 输出三面板诊断图（均衡后密度眼图 +
  probability mass + 波形/采样点概览，IEEE 802.3 风格）；`--no-plot` 显式关闭；
  未指定 `-p` 不绘图、不报错。
- **V1.3.0 多 Tap 缺陷修复（V1_Dev）**：
  - P1 对齐幻影：`align_elec_112` bit 域对齐对奇数 `bo`（半符号相位）按
    `bit_offset_to_symbol_offset` 映射回符号域偏移（`(bo+period)//2`），并加
    低 identity-SER 一致性守卫；`prbs11_e_rc50` 15-tap 在 numpy 2.2.6/1.19.2
    下均 off=5 / SER=0 / TECQ=2.1605；
  - P1b 指标 NaN：`eq.metric_detail` 拒绝 OMA≤0 / 非有限电平 / σ；`analyze`
    对退化均衡点回退诚实 identity（`identity-invalid-metric-fallback`）+ 告警；
  - P2 σ 退化：宽 tap 闭眼分支（`closed-eye-weak-basin-wide`）改按 sample-σ
    排序、IEEE fixed-OMA hist 重算报告值；4 个 rc50 退化 seed 回落 18.28-18.61 dB。
- **闭眼通道性能（V1.2.1）**：闭眼候选池改为“先算符号错误数（SER）、再按需
  计算 σ 拟合”——只有 `ne <= SER 上限` 的行才计算 histogram/sample 两套 σ，
  0 行通过时直接短路走 min-SER 回退、不计算任何 σ。批量 33 例 243s→190s；
  真实闭眼单例 p13_rc50_std 29s→4s、ssprq_full_rc50 374s→11s；输出与 V1.2
  逐位一致（仅 elapsed_sec 不同）。
- **告警**：码型不匹配 / seed 恢复失败 / 高 BER 时输出 `pattern_match_warning`，
  并诚实回退（禁止 `seed_auto_detected=True` 且 SER>0.5 的假阳性）。

## 目录结构

```
V1.3.0/
├── run_tdecq.py            # 单用例 CLI（含 -p 绘图、--ffe-* 与 --ffe-bounds）
├── src/tdecq_tool/         # 实现：csv_io / eq / patterns / pipeline / optical / plot / ffe_bounds
├── docs/USAGE.md           # 使用说明
├── SPEC.md                 # 权威规格（输入 CSV / 输出 JSON schema / 容差）
├── IMPLEMENTATION.md       # 实现说明（自动估计算法 / FFE 配置 / 边界约束 / 版本变更）
```

## 快速开始

```bash
# 环境：Python 3.10+，numpy / scipy / matplotlib
pip install numpy scipy matplotlib

# 单用例
python run_tdecq.py --input w.csv \
    --rate-gbps 128 --pattern auto --domain electrical \
    --out results/prbs13_128g_32glp.json -p plots/prbs13_128g_32glp.png

# 任意 seed 自动恢复
python run_tdecq.py --input w.csv \
    --rate-gbps 112 --pattern prbs --prbs-order 31 --domain electrical \
    --out out.json

# 自定义 FFE tap 几何（pre=2 / post=6，总 9 tap）
python run_tdecq.py --input w.csv --rate-gbps 128 --pattern prbs13 \
    --ffe-pre-cursor 2 --ffe-post-cursor 6 --out out.json -p eye.png

# 启用 IEEE 802.3dj 抽头边界约束 + 多 tap 候选（默认关闭；默认路径逐位一致）
python run_tdecq.py --input w.csv --rate-gbps 128 --pattern prbs13 \
    --ffe-bounds --out out.json -p eye.png

详细使用见 `docs/USAGE.md`；输出 JSON 字段见 `SPEC.md` 
