# V4 曲线化 V/W 结构与前高前低 — 设计说明

## 问题（现状）

`v4_aggressive/strategy_vw.py` 中：

- V 底：`wloc=15` 内必须为最低价；左侧瓶口仅 `low_idx-5:low_idx` 的 high。
- W 底：收盘低点过滤 + 两底间距 8~80 根。
- 前高/前低（`dynamic_weight_engine`）：`lookback` / `pivot_window` 固定根数。

**缺陷**：左侧跌 50 根、每日小阴小阳的大 V 无法识别；3 根急跌的小 V 可能被窗口规则漏掉或误并；子 V 与大 V 关系无法表达。

## 目标

在 **V4 分析当日**，基于**截至当日的全部历史日 K**：

1. 构造四条序列曲线：**最高、最低、收盘、均价**（均价默认 `(O+H+L+C)/4`，可配置为 `(H+L+C)/3`）。
2. 在曲线上用**自适应波动**找摆动点（非固定「前 N 根」定义形态）。
3. 识别所有 **V / W** 候选，维护生命周期：`形成中 → 右侧确认 → 触瓶口 → 已完成/失效`。
4. **剔除已完成**的历史 V/W，但保留**嵌套**：小 V 已完成可仍属于大 V 的左侧或右侧一段。
5. 在此之上判断：**V 底、V反右侧、是否到瓶口、激进底**；旗形/N 字/前高过滤改为读结构注册表。
6. **性能**：单次分析 O(n)～O(n·k)，k 为摆动点数量（通常远小于 n）；结果按 `stock_code` 一次构建、当次 `analyze` 复用。

## 非目标（首期）

- 不改 V2/V3 的 `strategy_vw_bottle_backtest` / `engine_v2`。
- 不重写全部 `dynamic_weight_engine` 规则，仅 V4 必买/前高过滤走新接口。

---

## 核心数据结构

```python
@dataclass
class PriceCurves:
    high, low, close, avg: np.ndarray  # 长度 n

@dataclass
class SwingPoint:
    idx: int
    kind: str   # 'high' | 'low'
    price: float
    curve: str  # 'high'|'low'|'close'|'avg'

@dataclass
class VPattern:
    id: str
    left_peak_idx: int      # 左侧瓶口对应摆动高点（在 high 曲线上）
    bottom_idx: int         # V 底（在 low 曲线上）
    neck_price: float       # 左侧瓶口价 = left_peak 至 bottom 区间 high 最大
    bottom_price: float
    right_entry_idx: int | None
    state: str              # forming | right | neck_touched | completed | broken
    parent_id: str | None   # 嵌套大 V
    children: list[str]

@dataclass
class WPattern:
  # 双底 + 颈线，类似
```

**`V4StructureRegistry`**（每次 `analyze_signals_v4_aggressive` 构建一次）：

- `build(df) -> registry`
- `active_v_at(bar) -> VPattern | None`  # 当日可用的「主 V」
- `v_right_entry_at(bar) -> event dict`   # 兼容现有 `detect_v_right_bottom_events` 输出
- `prior_high_before(bar) -> (idx, price)`  # 基于 high 曲线摆动，非固定 lookback
- `is_neck_touched(v_id, bar) -> bool`

---

## 摆动点检测（推荐：ATR 显著性，非固定窗口）

对每条曲线（尤其 `low` 找底、`high` 找顶）：

1. `atr = ATR(14)` 或滚动 `std(close, 20)`。
2. `scipy.signal.find_peaks`：
   - 谷底：对 `-low` 找峰，`prominence = α * atr[i]`，`distance = 3`（最小间隔防噪声，非形态宽度）。
   - 峰顶：对 `high` 同理。
3. **α** 默认 0.5~1.0，使「慢跌大 V」与「急跌小 V」共用同一套显著性。

左侧瓶口 **不再** 用 `low_idx-5:low_idx`，改为：

```text
neck_price = max(high[left_search_start : bottom_idx])
```

其中 `left_search_start` = 上一个显著摆动高点索引（在 high 曲线上），若无则从头扫描至 bottom。

左侧跌幅：

```text
drop = (neck_price - bottom_price) / neck_price
```

仅要求 `drop >= drop_min`（如 4%），**不限制**左侧天数。

右侧确认（与现逻辑类似，但不限 10 根）：

- 从 `bottom_idx+1` 起扫描至 `bottom_idx + max_right_span`（如 60，或到下一摆动低点为止）。
- 首根：`close` 反弹 ≥ `bounce_min` 且 `close[r] > close[r-1]`。

---

## 已完成 / 嵌套策略

| 状态 | 条件（示例） |
|------|----------------|
| `completed` | 触及瓶口后跌破颈线 fail，或持仓模拟止盈/止损结束（可选） |
| `broken` | 收盘跌破 `bottom_price` 或新建更低摆动低点替代 |
| `right` | 已有 `right_entry_idx`，未到瓶口 |
| `forming` | 仅有 bottom，尚无右侧 |

**嵌套**：

- 若 V_small 的时间区间 ⊆ V_large 的 `[left_peak, bottom]` 或右侧恢复段，则 `V_small.parent_id = V_large.id`。
- **信号归属**：默认只对「**当前 bar 上最外层未完成 V**」发 V反右侧；内层已完成的小 V 不再单独发，除非用户配置 `emit_nested=True`。

---

## 与现有 V4 规则的衔接

| 模块 | 改动 |
|------|------|
| `strategy_vw.detect_v_right_bottom_events` | 薄封装：调用 `registry.v_events()` |
| `iter_v_left_bottoms` | 改为 `registry.iter_v_bottoms()` |
| `v4_v_rev_rules` 激进底 | 用 `registry` 的 V 底与 `v_min` |
| `prior_high_*`（V4 路径） | 新增 `v4_structure_pivots.py`，V4 引擎优先调用 |
| 旗形 | 二期：旗形=收敛通道，用 high/low 曲线摆动切线，非固定根数 |

---

## 性能估算（n=800）

| 步骤 | 复杂度 |
|------|--------|
| 四曲线构造 | O(n) |
| ATR + find_peaks ×2 | O(n) |
| 摆动点两两组 V/W | O(k²)，k≈20~40 |
| 嵌套标记 | O(k²) 或 O(v²) |
| 按 bar 查询活跃 V | O(1) 预计算索引 |

内存：4×n float + 模式列表，可忽略。

---

## 分期实施建议

**Phase 1**（优先）：`v4_structure_curves.py` + `V4StructureRegistry`；替换 V4 的 `detect_v_right_bottom_events` / `iter_v_left_bottoms`；保留旧函数作 fallback 开关。

**Phase 2**：`v4_v_rev_rules`、瓶口/N 字、前高过滤读 registry。

**Phase 3**：旗形、W 形态统一；前端展示四条曲线与 V/W 标注（可选）。

---

## 待确认（1 项）

**均价曲线**：已确认用 **A) `(O+H+L+C)/4`**；**瓶口/前高**一律用 **high 曲线**。

---

## 推荐方案

**方案 B（推荐）**：ATR 显著性摆动 + 全区间扫瓶口 + 生命周期注册表；Phase 1 先落地 V/V反右侧，激进底与前高跟 Phase 2。

**方案 A**：仅把「左侧瓶口窗口」从 5 根改为「上一摆动高点至今」——改动小，解决不了 W/旗形/前高。

**方案 C**：机器学习/手工标注——过重，不采纳。
