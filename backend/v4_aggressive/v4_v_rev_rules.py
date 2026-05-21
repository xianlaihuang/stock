"""
V4 激进专用 V 反 / V 底买入结构检测（与 V2/V3 隔离）。

必买标签：
- V反右侧、V4V反吞没（保留）
- V反激进底部（三条独立子规则）：
  1. V4激进底金针  2. V4激进底止跌  3. V4激进底大低开阳
"""
import numpy as np

from v4_aggressive.strategy_vw import (
    iter_v_left_bottoms,
    detect_v_right_bottom_events,
    merge_w_v_events,
    _dedupe_by_entry,
)

# 经典 V 反右侧 + 吞没
V4_V_REV_ENGULF_RULE_NAME = 'V4V反吞没'

# V反激进底部 — 三条独立子规则（图上须能区分）
V4_AGGR_BOTTOM_NEEDLE = 'V反激进底-金针'
V4_AGGR_BOTTOM_STABILIZE = 'V反激进底-止跌'
V4_AGGR_BOTTOM_GAP_YANG = 'V反激进底-大低开阳'

V4_AGGRESSIVE_BOTTOM_RULE_NAMES = frozenset({
    V4_AGGR_BOTTOM_NEEDLE,
    V4_AGGR_BOTTOM_STABILIZE,
    V4_AGGR_BOTTOM_GAP_YANG,
})

# 三条激进底买入后的动态止损（收盘破 V 底最低则卖）
V4_AGGR_BOTTOM_STOP_SELL_REASON = 'V反激进底破V底止损'

# 仅「纯激进底」买入：连续 2 个交易日收盘跌破 MA5 则卖
V4_AGGR_BOTTOM_MA5_2DAY_SELL_REASON = 'V反激进底连2日破5日线'


def is_exclusive_aggressive_bottom_entry(mandatory_buy_triggered, buy_triggered=None):
    """
    买入当日有且仅有 V反激进底 三条之一（可多条均为激进底子规则），
    且无其它加权/必买规则叠加。
    """
    mb = set(mandatory_buy_triggered or [])
    bt = set(buy_triggered or [])
    if not mb & V4_AGGRESSIVE_BOTTOM_RULE_NAMES:
        return False
    if not mb.issubset(V4_AGGRESSIVE_BOTTOM_RULE_NAMES):
        return False
    if bt:
        return False
    return True

# 兼容旧常量名（引擎/前端若有引用）
V4_NEEDLE_V_REV_RULE_NAME = V4_AGGR_BOTTOM_NEEDLE
V4_V_REV_MOMENTUM_RULE_NAME = V4_AGGR_BOTTOM_STABILIZE

V4_V_REV_BUY_RULE_NAMES = frozenset({
    'V反右侧',
    V4_V_REV_ENGULF_RULE_NAME,
}) | V4_AGGRESSIVE_BOTTOM_RULE_NAMES

V4_V_BOTTOM_ZONE_MAX_AFTER_LOW = 8
V4_V_LEFT_SCAN_BEFORE = 15
V4_V_ENGULF_BODY_RANGE_MIN = 0.52
V4_V_ENGULF_BODY_VS_PREV_MIN = 1.05

# 通用金针/锤子线（较旧版 1.2/45% 放宽；与双针探底针口径一致）
V4_NEEDLE_LOWER_RANGE_MIN = 0.33   # 下影占全日振幅 >= 1/3
V4_NEEDLE_SHADOW_BODY_MIN = 0.75   # 下影 >= 0.75×实体（通用金针，略宽于双针0.8）
V4_NEEDLE_UPPER_RANGE_MAX = 0.45   # 上影占振幅 <= 45%，排除上影过长
V4_PRIOR_LOW_SUPPORT_TOL = 0.005
V4_PRIOR_LOW_LOOKBACK = 5
V4_MA20_PREMIUM_NEEDLE = 0.03
V4_MA20_PREMIUM_GAP = 0.02
V4_MA20_PREMIUM_STABILIZE = 0.03
V4_GAP_DOWN_OPEN_VS_PREV = 0.01
V4_BIG_YANG_BODY_RANGE_MIN = 0.45
V4_STABILIZE_AP_WINDOW = 3
V4_MOMENTUM_DOWN_SHRINK_RATIO = 0.88


def _ma20_premium_ok(high, ma20, idx, min_pct):
    """
    确认买入日相对 MA20 的「收益/空间」：
    (MA20 - 当日最高价) / 当日最高价 >= min_pct
    """
    idx = int(idx)
    if idx < 0 or idx >= len(ma20):
        return False
    m = float(ma20[idx])
    h = float(high[idx])
    if np.isnan(m) or m <= 0 or h <= 0:
        return False
    return (m - h) / h >= float(min_pct)


def _v_zone_min_low(low, low_idx, n, before=None, after=None):
    if before is None:
        before = V4_V_LEFT_SCAN_BEFORE
    if after is None:
        after = 1
    lo_i = max(0, int(low_idx) - int(before))
    hi_i = min(int(n) - 1, int(low_idx) + int(after))
    if hi_i < lo_i:
        return None
    return float(np.min(low[lo_i: hi_i + 1]))


def _low_is_v_bottom(low_val, v_bottom_low, tol=0.005):
    v_bottom_low = float(v_bottom_low)
    if v_bottom_low <= 0:
        return False
    return abs(float(low_val) - v_bottom_low) / v_bottom_low <= tol


def _has_prior_low_support(low, at_idx, lookback=None):
    """金针/大阳线最低点不破前低（前低支撑）。"""
    at_idx = int(at_idx)
    if at_idx < 1:
        return False
    lb = lookback if lookback is not None else V4_PRIOR_LOW_LOOKBACK
    i0 = max(0, at_idx - lb)
    if i0 >= at_idx:
        return False
    prior_min = float(np.min(low[i0:at_idx]))
    if prior_min <= 0:
        return False
    return float(low[at_idx]) >= prior_min * (1.0 - V4_PRIOR_LOW_SUPPORT_TOL)


def _resolve_entry_bar(close, open_, low, signal_idx, n):
    """前低支撑则信号日买；否则次日阳线且收盘高于信号日收盘则次日买。"""
    signal_idx = int(signal_idx)
    if _has_prior_low_support(low, signal_idx):
        return signal_idx
    j = signal_idx + 1
    if j < n and float(close[j]) > float(open_[j]) and float(close[j]) > float(close[signal_idx]):
        return j
    return None


def _bar_is_golden_needle_yang(o, h, l, c):
    """
    金针探底阳线（通用锤子线标准）：
    - 收阳；
    - 下影占振幅 >= 1/3；
    - 下影 >= 0.75×实体（实体极小时仅看下影占比）；
    - 上影不宜过长。
    """
    o, h, l, c = float(o), float(h), float(l), float(c)
    if c <= o:
        return False
    rng = h - l
    if rng <= 1e-12:
        return False
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    if lower <= 0:
        return False
    if lower / rng < V4_NEEDLE_LOWER_RANGE_MIN:
        return False
    if upper / rng > V4_NEEDLE_UPPER_RANGE_MAX:
        return False
    min_body = 1e-10 * max(abs(c), abs(o), 1e-9)
    if body < min_body:
        return True
    return (lower / body) >= V4_NEEDLE_SHADOW_BODY_MIN


def _big_bullish_engulfing_at(open_, high, low, close, idx):
    idx = int(idx)
    if idx < 1:
        return False
    o, c = float(open_[idx]), float(close[idx])
    o1, c1 = float(open_[idx - 1]), float(close[idx - 1])
    if c <= o or c1 >= o1:
        return False
    if o > c1 or c < o1:
        return False
    body = c - o
    prev_body = o1 - c1
    if prev_body <= 1e-12 or body < prev_body * V4_V_ENGULF_BODY_VS_PREV_MIN:
        return False
    rng = float(high[idx]) - float(low[idx])
    if rng <= 1e-12 or body / rng < V4_V_ENGULF_BODY_RANGE_MIN:
        return False
    return True


def _big_gap_down_big_yang(open_, high, low, close, idx):
    """V 左侧大低开的大阳线。"""
    idx = int(idx)
    if idx < 1:
        return False
    o, c = float(open_[idx]), float(close[idx])
    prev_c = float(close[idx - 1])
    if c <= o or prev_c <= 0:
        return False
    if o > prev_c * (1.0 - V4_GAP_DOWN_OPEN_VS_PREV):
        return False
    rng = float(high[idx]) - float(low[idx])
    if rng <= 1e-12:
        return False
    body = c - o
    return body / rng >= V4_BIG_YANG_BODY_RANGE_MIN


def _v_left_momentum_declining(close, low_idx, left_start):
    low_idx, left_start = int(low_idx), int(left_start)
    down_moves = []
    for i in range(max(left_start + 1, 1), low_idx + 1):
        if close[i] < close[i - 1] and close[i - 1] > 1e-12:
            down_moves.append(abs(close[i] - close[i - 1]) / close[i - 1])
    if len(down_moves) < 4:
        return False
    mid = len(down_moves) // 2
    first = float(np.mean(down_moves[:mid])) if mid > 0 else 0.0
    second = float(np.mean(down_moves[mid:]))
    if first <= 1e-12:
        return False
    return second < first * V4_MOMENTUM_DOWN_SHRINK_RATIO


def _avg_price_stabilizing(high, low, close, idx, window=None):
    """均价止跌：近段典型价均值不再创新低。"""
    window = window or V4_STABILIZE_AP_WINDOW
    idx = int(idx)
    if idx < window * 2:
        return False

    def _tp(i):
        return (float(high[i]) + float(low[i]) + float(close[i])) / 3.0

    recent = float(np.mean([_tp(idx - k) for k in range(window)]))
    prior = float(np.mean([_tp(idx - window - k) for k in range(window)]))
    if prior <= 0:
        return False
    return recent >= prior * 0.998


def detect_v4_aggressive_bottom_events(close, open_, high, low, n, ma5, ma20, min_drop=None):
    """
    V反激进底部三条规则，返回买入日 entry 及子规则名 _rule。
    """
    if ma5 is None or ma20 is None:
        return []
    n = int(n)
    events = []
    for ctx in iter_v_left_bottoms(close, high, low, n, min_drop=min_drop, open_=open_):
        low_idx = int(ctx['left_idx'])
        left_ref = float(ctx['ref_high'])
        stop_ref = float(low[low_idx])
        v_min = _v_zone_min_low(low, low_idx, n)
        if v_min is None:
            continue
        left_start = max(0, low_idx - V4_V_LEFT_SCAN_BEFORE)
        scan_end = min(n - 1, low_idx + V4_V_BOTTOM_ZONE_MAX_AFTER_LOW)

        # --- 1. 金针探底：影线最低点为 V 底最低，确认后 (MA20-最高价)/最高价>3% ---
        for j in range(left_start, scan_end + 1):
            if not _bar_is_golden_needle_yang(open_[j], high[j], low[j], close[j]):
                continue
            if not _low_is_v_bottom(float(low[j]), v_min):
                continue
            entry = _resolve_entry_bar(close, open_, low, j, n)
            if entry is None:
                continue
            if not _ma20_premium_ok(high, ma20, entry, V4_MA20_PREMIUM_NEEDLE):
                continue
            events.append({
                'kind': 'V',
                'entry': int(entry),
                'neck': left_ref,
                'v_bottom_low': float(v_min),
                'stop_ref': float(v_min),
                '_rule': V4_AGGR_BOTTOM_NEEDLE,
                '_signal_bar': int(j),
            })
            break

        # --- 3. 大低开大阳线（优先于止跌扫描顺序上在金针之后单独找）---
        for j in range(left_start, scan_end + 1):
            if not _big_gap_down_big_yang(open_, high, low, close, j):
                continue
            entry = _resolve_entry_bar(close, open_, low, j, n)
            if entry is None:
                continue
            if not _ma20_premium_ok(high, ma20, entry, V4_MA20_PREMIUM_GAP):
                continue
            events.append({
                'kind': 'V',
                'entry': int(entry),
                'neck': left_ref,
                'v_bottom_low': float(v_min),
                'stop_ref': float(v_min),
                '_rule': V4_AGGR_BOTTOM_GAP_YANG,
                '_signal_bar': int(j),
            })
            break

        # --- 2. 跌能减弱 + 均价止跌 + 站上 MA5 + (MA20-最高价)/最高价>3% ---
        r_max = min(low_idx + V4_V_BOTTOM_ZONE_MAX_AFTER_LOW, n - 1)
        for idx in range(low_idx + 1, r_max + 1):
            if idx >= len(ma5) or idx >= len(ma20):
                break
            if np.isnan(ma5[idx]) or np.isnan(ma20[idx]):
                continue
            if float(close[idx]) <= float(ma5[idx]):
                continue
            if not _v_left_momentum_declining(close, low_idx, left_start):
                continue
            if not _avg_price_stabilizing(high, low, close, idx):
                continue
            if not _ma20_premium_ok(high, ma20, idx, V4_MA20_PREMIUM_STABILIZE):
                continue
            if float(close[idx]) <= float(open_[idx]):
                continue
            events.append({
                'kind': 'V',
                'entry': int(idx),
                'neck': left_ref,
                'v_bottom_low': float(v_min),
                'stop_ref': float(v_min),
                '_rule': V4_AGGR_BOTTOM_STABILIZE,
            })
            break

    events.sort(key=lambda x: x['entry'])
    return _dedupe_by_entry(events, min_gap=5)


def build_aggressive_bottom_stop_by_entry(
    close, open_, high, low, n, ma5, ma20, min_drop=None,
):
    """买入 bar 索引 -> V 反底部最低点（动态止损价）。"""
    out = {}
    for ev in detect_v4_aggressive_bottom_events(
        close, open_, high, low, n, ma5, ma20, min_drop=min_drop,
    ):
        e = int(ev['entry'])
        floor = float(ev.get('v_bottom_low', ev.get('stop_ref', np.nan)))
        if np.isnan(floor) or floor <= 0:
            continue
        if e in out:
            out[e] = min(out[e], floor)
        else:
            out[e] = floor
    return out


def detect_v4_v_bottom_engulf_signal_indices(close, open_, high, low, n, min_drop=None):
    n = int(n)
    signals = []
    for ctx in iter_v_left_bottoms(close, high, low, n, min_drop=min_drop, open_=open_):
        low_idx = int(ctx['left_idx'])
        left_ref = float(ctx['ref_high'])
        stop_ref = float(low[low_idx])
        hi_end = min(low_idx + V4_V_BOTTOM_ZONE_MAX_AFTER_LOW + 1, n - 1)
        for s in range(low_idx, hi_end):
            if not _big_bullish_engulfing_at(open_, high, low, close, s):
                continue
            if s + 1 >= n:
                continue
            signals.append({
                'signal': int(s),
                'low_idx': low_idx,
                'neck': left_ref,
                'stop_ref': float(min(stop_ref, float(low[s]))),
            })
            break
    return signals


# 兼容旧调用名
def detect_v4_needle_v_bottom_events(close, open_, high, low, n, min_drop=None):
    return []


def detect_v4_v_bottom_momentum_events(close, open_, high, low, vol, n, ma5, ma20, min_drop=None):
    return detect_v4_aggressive_bottom_events(
        close, open_, high, low, n, ma5, ma20, min_drop=min_drop,
    )


def merge_v4_v_rev_events(close, open_, high, low, n, vol=None, ma5=None, ma20=None):
    classic = detect_v_right_bottom_events(close, high, low, n, open_=open_)
    aggressive = (
        detect_v4_aggressive_bottom_events(close, open_, high, low, n, ma5, ma20)
        if ma5 is not None and ma20 is not None else []
    )
    engulf_entries = []
    for sig in detect_v4_v_bottom_engulf_signal_indices(close, open_, high, low, n):
        s = int(sig['signal'])
        if s + 1 < n and float(close[s + 1]) > float(open_[s + 1]):
            engulf_entries.append({
                'kind': 'V',
                'entry': s + 1,
                'neck': float(sig['neck']),
                'stop_ref': float(sig['stop_ref']),
                '_rule': V4_V_REV_ENGULF_RULE_NAME,
                '_signal_bar': s,
            })
    merged = merge_w_v_events(classic, aggressive)
    merged = merge_w_v_events(merged, engulf_entries)
    return merged
