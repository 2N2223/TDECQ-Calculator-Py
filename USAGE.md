# TDECQ / TECQ 工具使用说明（USAGE）— V1.3.0

本文档说明 V1.3.0 工具（无逐用例标定、无 case_id 参与算法、自动参数估计、可配置
FFE tap 几何、闭眼候选池性能优化、IEEE 802.3dj 抽头边界约束 + 多 tap 候选）的安装、运行与常见问题。输出 JSON 字段与 `SPEC.md` §5 一致。

## 1. 环境与安装

- Python 3.10+（本包在 Python 3.10.4 上验证）。
- 依赖：`numpy`、`scipy`、`matplotlib`（仅绘图需要；未指定 `-p` 时不渲染、不写文件）。
- 无需安装：`src/tdecq_tool/` 直接通过根目录入口脚本 `sys.path` 导入。

```bash
pip install numpy scipy matplotlib     # 若尚未安装
```

## 2. 单用例 CLI

```bash
python run_tdecq.py \
    --input data/virtuoso_export/PRBS13_128G_PAM4_32GLP.csv \
    --rate-gbps 128 --pattern auto --domain electrical \
    --samples-per-ui auto --window auto \
    --out results/prbs13_128g_32glp.json \
    -p plots/prbs13_128g_32glp.png
```

- `--input`：波形 CSV。Virtuoso 导出（表头 `VT("/c") X,VT("/c") Y`，节点名任意）与
  普通 `time,out` 格式都自动识别，**不需要** `csv_format` 参数、不需要指定列号。
- `--rate-gbps`：聚合数据速率（128 或 112），UI = 2/rate。
- `--pattern`：`auto`（默认）、`prbs11 / prbs13 / prbs31 / ssprq`、通用
  `prbsN`（N = 2..31）或 `prbs`（配 `--prbs-order N`）。
- `--domain`：`electrical`（TECQ，mV）或 `optical`（TDECQ，mW）。
- `--samples-per-ui`：`auto`（默认，按速率推断：≥120 Gbps → 32，否则 64）或显式整数。
- `--window`：`auto`（默认）或显式 UI 数。
- `-p/--output-plot`：可选的诊断 PNG；`--no-plot` 显式关闭。未指定 `-p` 时不绘图、不报错。
- `--case`：兼容选项，**仅作为输出 JSON 的 `id` 诊断标签**，不参与任何算法分支，
  不是运行必要条件。

### 2.1 FFE tap 几何（V1.2）

```bash
# pre-cursor 2 / post-cursor 6 -> 总 9 tap（mainTap 锚点在索引 2）
python run_tdecq.py --input w.csv --rate-gbps 128 --pattern prbs13 \
    --ffe-pre-cursor 2 --ffe-post-cursor 6 --out out.json -p eye.png

# 总量便捷形式（等价 --ffe-post-cursor 8，pre=0）
python run_tdecq.py --input w.csv --rate-gbps 112 --pattern prbs31 \
    --ffe-taps 9 --out out.json
```

- `--ffe-pre-cursor N`：mainTap 前的 pre-cursor 数，0..5。
- `--ffe-post-cursor M`：mainTap 后的 post-cursor 数，0..30。
- `--ffe-taps T`：总量 = N+1+M；与显式 `--ffe-post-cursor` 冲突时报错。
- 默认 `(0,4)` 即 5-tap，与 V1.1.1 及此前版本**逐位一致**。
- 输出 JSON 含 `ffe_pre_cursor` / `ffe_post_cursor` / `ffe_tap_count` /
  `ffe_main_tap_index`（= pre 锚点）/ `ffe_largest_tap_index`（实际能量峰；
  受通道群延迟影响可能偏在锚点右侧，参考行为一致），`n_taps` 反映实际 tap 数。

### 2.2 IEEE 802.3dj 抽头边界约束 + 多 tap 候选（V1.3，`--ffe-bounds` 默认关闭）

```bash
python run_tdecq.py --input w.csv --rate-gbps 128 --pattern prbs13 \
    --ffe-bounds --out out.json -p eye.png
```

- 开启后 tap 数 N ∈ {5, 7, 9} 成为候选维度，主抽头位置由脉冲响应峰值自动对齐
  并在其附近 ±1 扫描；无约束 MMSE 超界时自动切换有界求解，不直接 clamp。
- 边界表（相对主抽头偏移 n = i − m，任意 N 通用）：n=0 [0.8, 2.5]；n=±1
  [−0.4, 0.05]；n=±2 [−0.1, 0.2]；|n|≥3 [−0.1, 0.1]；Σtaps=1；主抽头位置上限
  pre≤3 / post≤13。差分限 `|w(+1)-w(-1)|/w(0) ≤ 0.25`。

- 超界求解方法：有界 QP（SLSQP，失败退化为 lsq_linear）、active-set
  （clip + 自由系数重解）、damped MMSE（α=0.25/0.5/0.75）、渐进式约束坐标
  下降（含 tap 扩展与收益回退）。
- 选择分数 `min(hist,samp) + 0.2·D_FFE + 0.1·C_eq + 0.05·max(0,N-5)`；
  仅当分数严格更优且符号错误数不劣于主路径选中行时才替换；
  seed 恢复失败时跳过边界池（诚实回退 + 告警）。
- **默认关闭**：不加 `--ffe-bounds` 时输出与 V1.2.1 逐位一致（仅新增
  `ffe_bounds_*` 诊断字段）。开启后 33 例验收不适用（bound 模式 override
  IEEE 参考接收机，指标会明显低于 manifest 参考，属预期行为）。
- 绘图：启用边界时 suptitle 中主抽头**加粗**，触碰边界的 tap 用**下划线**
  （触下限）/ **上划线**（触上限）标注。

### 2.3 已验证均衡分支（V1.3，`--no-equalized-validation` 可关闭）

closed-eye canonical 回退记录默认先做 out-of-sample SER + cross-window seed
校验：通过才报告均衡后的 TECQ（`equalized_validated=True`、SER=0），否则维持
诚实 identity + 告警。`--no-equalized-validation` 恢复旧行为（一律 identity）。

## 3. 批量运行与验收

```bash
# 跑 manifest 全部 33 个主用例，输出 results/<case_id>.json
python run_all.py --results results

# （可选）同时为每个用例输出绘图
python run_all.py --results results --plots plots

# 验收：33/33 PASS
python tests/validate_metrics.py --results results

# 单元测试（131 个）
python -m unittest discover -s tests -q
```

## 4. 各 --pattern 的自动行为

### 4.1 CSV 波形自动识别（无需配置）
`src/tdecq_tool/csv_io.py` 自动完成：

- 读取文件头与首数据行，启发式判定表头/列含义；
- 时间列 = 表头词元 `time`/`t`/`x` 优先（双列与多列统一），否则取最单调递增列；
  其余列中方差最大者为信号列；
- 单位归一化仅由表头驱动：Virtuoso 电域 ×1000 → mV；`time,out` 电域已是 mV 时
  不做幅度猜测（真实小摆幅 mV 记录不会误放大）；光域保持 mW；
- Virtuoso 判定以 `VT(` 头为准；仅“恰好两列且词元为 `x`/`y`”才按 Virtuoso 风格
  处理，`data_x,data_y` 等普通两列头不会误判；均匀网格 Virtuoso 记录（含 1 ps
  非目标分辨率）自动重采样到目标 spu。

### 4.2 码型自动识别（--pattern auto，完全波形驱动）
- `resolve_pattern` **不读取 case_id**：显式 `--pattern` 直接返回；`auto` 时仅由
  CSV 头部词元（`ssprq`/`prbs13`）或记录本身判定。
- **Virtuoso 128G**：按记录长度 + 摆幅自动判定——PRBS13 记录约 6400 UI、±500 mV；
  SSPRQ 记录约 67200 UI、±225 mV。
- **112G 电/光域**：无文件名线索时，依次对 PRBS11/PRBS31/SSPRQ 做确定性对齐，
  取解码匹配度最高者。

### 4.3 自动 CDR
`src/tdecq_tool/eq.py::cdr_from_crossing`：PAM4 过零点圆均值 + 半 UI，输出采样相位。

### 4.4 自动对齐、符号偏移与 seed 恢复
- PRBSn 单 LFSR：逐相位解码 → 位平面 LFSR 递推一致性 → 恢复记录起点状态
  （预卷/记录起点/瞬态裁剪偏移全部吸收进恢复 seed）。
- PRBS13 双位平面（802.3 mzm）：由 MSB 位平面恢复 `seed_a`，由 LSB 与 A 异或
  得到 B 位平面恢复 `seed_b`；闭眼（LC 谐振）记录用整段 MMSE SER 交叉验证 +
  identity 眼判别拒绝相位平移假阳性；不可恢复时回退 canonical (1,137) + 告警。
- SSPRQ：FFT 符号相关；SSPRQ_Cut（头部截断）按 `%65535` 周期回绕补全后对齐。
- PRBS31 长记录（65535 符号）：按**线性 LFSR 序列**处理，不折叠到 2047 周期；
  对齐支持**从序列任意位置开始**的记录（65535 符号预卷 + 两阶段 MMSE 残差扫描）。

### 4.5 自动测试窗
- PRBS31 / SSPRQ 记录含完整周期 → 65535 UI；
- PRBS13 → 记录长度（128G 记录 6400 UI）；
- 其它 PRBSn → 码型周期或记录长度。语义与 manifest 的 `window=auto` 一致。

### 4.6 FFE 候选池与 σ 拟合
- 候选池（按所选 tap 几何）：identity、单位增益 MMSE、阻尼 MMSE
  （α=0.2/0.4/0.6/0.8）、受限子集 LS、弱 progressive CD；均在自动 CDR ± 窗口、
  自动偏移 ± 窗口内评估。
- σ 拟合法：`hist`（64 分箱垂直直方图卷积高斯，IEEE 风格）与 `samp`（样本级高斯二分），
  按波形特征选择并记录 `sigma_method`。
- 所有选择常量均为全局常量，**不存在按用例 id 查表**。

## 5. 输出 JSON 字段（与 SPEC.md §5 一致）

必填/建议字段全部保留；另加以下诊断字段（不影响验收）：

| 字段 | 说明 |
|---|---|
| `ffe_pre_cursor` / `ffe_post_cursor` | 使用的 FFE pre/post-cursor 数 |
| `ffe_tap_count` / `n_taps` | 实际 tap 总数（默认 5） |
| `ffe_main_tap_index` / `ffe_largest_tap_index` | 主抽头锚点 / 实际能量峰索引 |
| `chosen_strategy` | 选中策略：`identity` / `ls_mmse` / `damped_mmse` / `subset_ls` |
| `ffe_reason` | 选中原因的短说明（如 open-eye / strong-ISI / closed-eye weak-basin） |
| `selection_tier` | 内部选择层级（如 `identity-rc`、`tap-constrained`、`closed-eye-weak-basin`） |
| `selected_family` | 选中 FFE 族（如 `id`、`d0.4`、`s01`、`full`） |
| `sigma_method` | 使用的 σ 拟合法：`hist` 或 `samp` |
| `cdr_estimated` / `offset_estimated` | 自动估计的 CDR 相位与符号偏移（诊断） |
| `prbs_seed_a` / `prbs_seed_b` / `seed_source` | PRBS13 双位平面恢复状态与来源 |
| `pattern_match_warning` | 码型不匹配 / seed 失败 / 高 BER 告警文本（无则 null） |
| `ffe_bounds_enabled` / `ffe_bounds_used` | 边界模式是否启用 / 是否替换了主路径选中行（默认 False） |
| `ffe_bounds_strategy` / `ffe_bounds_tap_count` / `ffe_bounds_main_index` | 边界池选中策略（如 `bounded_qp_N9_m1`）/ tap 数 / 主抽头索引 |
| `ffe_bounds_at_bound` | 逐 tap 触界标注（`lb`/`ub`/null，供绘图下划线/上划线） |
| `equalized_validated` | 已验证均衡分支是否通过校验（默认 False） |

主指标：`tecq_db`（电域）/ `tdecq_db`（光域）；其余字段见 `SPEC.md` §5。
