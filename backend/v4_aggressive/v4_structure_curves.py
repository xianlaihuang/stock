"""
V4 曲线化 V/W 结构：基于全历史 high / low / close / avg 四条曲线 + ATR 摆动点。
前高、瓶口等一律用 high 曲线；V 反左侧跌深门槛用均价 avg = (O+H+L+C)/4。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

# 摆动点：相对 ATR 的显著性（非固定「前 N 根」形态宽度）
SWING_ATR_PERIOD = 14
SWING_PROMINENCE_ATR = 0.55
SWING_MIN_DISTANCE = 3

V_DROP_MIN = 0.04
V_BOUNCE_MIN = 0.02
V_MAX_RIGHT_SPAN = 60
V_RIGHT_SCAN_CAP = 120

W_BOTTOM_MAX_GAP = 120
W_BOTTOM_MIN_GAP = 8
W_NECK_MIN_RISE = 0.03
W_BOTTOM_PRICE_TOL = 0.05
W_RIGHT_MAX_SPAN = 60
W_RIGHT_SCAN_CAP = 120
W_EVENT_MIN_GAP = 5

M_TOP_MIN_GAP = 10
M_TOP_MAX_GAP = 60
M_TOP_PRICE_TOL = 0.02
M_TOP_MIN_RETRACE = 0.03
M_TOP_BREAK_RATIO = 0.98
M_TOP_BREAK_SCAN = 30
M_EVENT_MIN_GAP = 5

V_TOP_RISE_MIN = 0.07
V_TOP_DROP_MIN = 0.03
V_TOP_BREAK_RATIO = 0.98
V_TOP_RIGHT_SCAN = 60
V_TOP_EVENT_MIN_GAP = 5

# 头肩顶：high 曲线 ATR 摆动高点，隔 1 个摆动取左肩/头/右肩（与旧 close 隔 2 峰同构）
HS_SHOULDER_HEAD_MIN_GAP = 10
HS_HEAD_RS_MIN_GAP = 10
HS_TOTAL_MAX_GAP = 100
HS_SHOULDER_TOL = 0.05
HS_MIN_TROUGH_RETRACE = 0.03
HS_BREAK_SCAN = 20

_REGISTRY_CACHE: Tuple[Optional[tuple], Optional['V4StructureRegistry']] = (None, None)


@dataclass
class PriceCurves:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    avg: np.ndarray
    n: int


@dataclass
class SwingPoint:
    idx: int
    price: float
    kind: str  # 'high' | 'low'


@dataclass
class VPattern:
    bottom_idx: int
    left_peak_idx: int
    neck_price: float
    bottom_price: float
    drop_pct: float
    right_entry_idx: Optional[int] = None
    state: str = 'forming'  # forming | right | completed | broken
    parent: Optional[int] = None  # index into patterns list


@dataclass
class WPattern:
    left_bottom_idx: int
    right_bottom_idx: int
    neck_price: float
    stop_ref: float
    neck_rise_pct: float
    right_entry_idx: Optional[int] = None


@dataclass
class MPattern:
    left_peak_idx: int
    right_peak_idx: int
    top_price: float
    trough_price: float
    break_idx: Optional[int] = None


@dataclass
class VTopPattern:
    peak_idx: int
    left_trough_idx: int
    peak_price: float
    rise_pct: float
    break_idx: Optional[int] = None


@dataclass
class HSTopPattern:
    left_shoulder_idx: int
    head_idx: int
    right_shoulder_idx: int
    left_shoulder_price: float
    head_price: float
    right_shoulder_price: float
    neckline_price: float
    break_idx: Optional[int] = None


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return out
    import talib
    return talib.ATR(high, low, close, timeperiod=period)


def _build_avg(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    return (open_ + high + low + close) / 4.0


def _find_swings(prices: np.ndarray, atr: np.ndarray, kind: str) -> List[SwingPoint]:
    """在单条价格曲线上找显著摆动高/低点。"""
    n = len(prices)
    if n < 8:
        return []
    med_atr = float(np.nanmedian(atr[atr > 0])) if np.any(atr > 0) else float(np.nanstd(prices) * 0.02)
    prom = max(med_atr * SWING_PROMINENCE_ATR, prices[-1] * 0.008 if n else 0.01)

    if find_peaks is not None:
        series = -prices if kind == 'low' else prices
        idxs, _ = find_peaks(series, distance=SWING_MIN_DISTANCE, prominence=prom)
    else:
        idxs = _find_swings_fallback(prices, kind, SWING_MIN_DISTANCE)

    out: List[SwingPoint] = []
    for i in idxs:
        i = int(i)
        if 0 <= i < n:
            out.append(SwingPoint(idx=i, price=float(prices[i]), kind=kind))
    return out


def _find_swings_fallback(prices: np.ndarray, kind: str, order: int) -> List[int]:
    n = len(prices)
    idxs = []
    for i in range(order, n - order):
        seg = prices[i - order: i + order + 1]
        if kind == 'low' and prices[i] <= seg.min() + 1e-12:
            idxs.append(i)
        elif kind == 'high' and prices[i] >= seg.max() - 1e-12:
            idxs.append(i)
    return idxs


def _neck_high_between(high: np.ndarray, start: int, end: int) -> Tuple[float, int]:
    """瓶口/前高：high 曲线区间最高价。"""
    start, end = max(0, int(start)), int(end)
    if end < start:
        return 0.0, start
    seg = high[start: end + 1]
    if len(seg) == 0:
        return 0.0, start
    j = int(np.argmax(seg))
    return float(seg[j]), start + j


def _v_left_drop_pct_on_avg(avg: np.ndarray, left_start: int, bottom_idx: int) -> float:
    """
    V 反左侧跌深：左侧至 V 底（含）均价曲线最高点 → V 底当日均价。
    (max(avg[left..底]) - avg[底]) / max(avg[left..底])
    """
    left_start, bottom_idx = max(0, int(left_start)), int(bottom_idx)
    if bottom_idx < left_start:
        return 0.0
    seg = avg[left_start: bottom_idx + 1]
    if len(seg) == 0:
        return 0.0
    peak = float(np.max(seg))
    bot = float(avg[bottom_idx])
    if peak <= 1e-12:
        return 0.0
    return max(0.0, (peak - bot) / peak)


class V4StructureRegistry:
    def __init__(self, curves: PriceCurves, atr: np.ndarray):
        self.curves = curves
        self.atr = atr
        self.swing_highs = _find_swings(curves.high, atr, 'high')
        self.swing_lows = _find_swings(curves.low, atr, 'low')
        self.v_patterns: List[VPattern] = []
        self.w_patterns: List[WPattern] = []
        self.m_patterns: List[MPattern] = []
        self.v_top_patterns: List[VTopPattern] = []
        self.hs_top_patterns: List[HSTopPattern] = []
        self.m_break_bars: set = set()
        self.v_top_break_bars: set = set()
        self.hs_break_bars: set = set()
        self._build_v_patterns()
        self._build_w_patterns()
        self._build_m_patterns()
        self._build_v_top_patterns()
        self._build_hs_top_patterns()

    def _prev_swing_high_before(self, idx: int) -> Optional[SwingPoint]:
        cand = [s for s in self.swing_highs if s.idx < idx]
        if not cand:
            return None
        return cand[-1]

    def _build_v_patterns(self):
        c, h, l, avg = self.curves.close, self.curves.high, self.curves.low, self.curves.avg
        n = self.curves.n
        patterns: List[VPattern] = []

        for sw_low in self.swing_lows:
            bi = sw_low.idx
            if bi < 8 or bi >= n - 3:
                continue
            prev_h = self._prev_swing_high_before(bi)
            left_start = prev_h.idx if prev_h else max(0, bi - V_MAX_RIGHT_SPAN)
            neck, _ = _neck_high_between(h, left_start, bi)
            bottom = float(l[bi])
            if neck <= 0 or bottom <= 0:
                continue
            drop = _v_left_drop_pct_on_avg(avg, left_start, bi)
            if drop < V_DROP_MIN:
                continue

            right_entry = None
            r_end = min(n - 1, bi + V_RIGHT_SCAN_CAP)
            for r in range(bi + 1, min(bi + V_MAX_RIGHT_SPAN, r_end) + 1):
                if (float(c[r]) - bottom) / bottom < V_BOUNCE_MIN:
                    continue
                if float(c[r]) <= float(c[r - 1]):
                    continue
                right_entry = r
                break

            state = 'right' if right_entry is not None else 'forming'
            patterns.append(
                VPattern(
                    bottom_idx=bi,
                    left_peak_idx=left_start if prev_h is None else prev_h.idx,
                    neck_price=neck,
                    bottom_price=bottom,
                    drop_pct=drop,
                    right_entry_idx=right_entry,
                    state=state,
                )
            )

        patterns.sort(key=lambda p: p.bottom_idx)
        self._mark_nested_v(patterns)
        self.v_patterns = patterns

    def _mark_nested_v(self, patterns: List[VPattern]):
        """小 V 落在大 V 左侧或底部区间内则标记 parent，发信号时优先外层。"""
        for i, inner in enumerate(patterns):
            for j, outer in enumerate(patterns):
                if i == j:
                    continue
                if outer.bottom_idx <= inner.bottom_idx:
                    continue
                if inner.bottom_idx > outer.bottom_idx:
                    continue
                lo = min(outer.left_peak_idx, outer.bottom_idx)
                hi = max(outer.left_peak_idx, outer.bottom_idx)
                if lo <= inner.bottom_idx <= hi and (
                    outer.right_entry_idx is None
                    or inner.bottom_idx <= outer.right_entry_idx
                ):
                    if inner.parent is None or patterns[inner.parent].bottom_idx < outer.bottom_idx:
                        inner.parent = j

    def _is_active_v(self, p: VPattern, at_bar: int) -> bool:
        if p.state == 'broken':
            return False
        if at_bar < p.bottom_idx:
            return False
        c, l = self.curves.close, self.curves.low
        if at_bar > p.bottom_idx and float(l[at_bar]) < p.bottom_price * 0.995:
            return False
        return True

    def outermost_active_v_at(self, bar: int) -> Optional[VPattern]:
        bar = int(bar)
        cands = [
            p for p in self.v_patterns
            if p.right_entry_idx == bar and p.parent is None and self._is_active_v(p, bar)
        ]
        if cands:
            return cands[-1]
        cands = [
            p for p in self.v_patterns
            if p.right_entry_idx is not None
            and p.right_entry_idx <= bar
            and p.parent is None
            and self._is_active_v(p, bar)
        ]
        if not cands:
            return None
        return max(cands, key=lambda p: p.bottom_idx)

    def iter_v_bottom_contexts(self, min_drop: Optional[float] = None) -> Iterator[dict]:
        md = V_DROP_MIN if min_drop is None else float(min_drop)
        for p in self.v_patterns:
            if p.drop_pct < md:
                continue
            if p.parent is not None:
                continue
            yield {
                'kind': 'V',
                'left_idx': int(p.bottom_idx),
                'ref_high': float(p.neck_price),
                'drop_pct': float(p.drop_pct),
            }

    def to_v_right_events(self, bounce_min: float = V_BOUNCE_MIN, drop_min: float = V_DROP_MIN) -> List[dict]:
        out = []
        last_e = -10 ** 9
        for p in self.v_patterns:
            if p.drop_pct < drop_min or p.right_entry_idx is None:
                continue
            if p.parent is not None:
                continue
            e = int(p.right_entry_idx)
            if e - last_e < 5:
                continue
            out.append({
                'kind': 'V',
                'entry': e,
                'neck': float(p.neck_price),
                'stop_ref': float(p.bottom_price),
                'low_idx': int(p.bottom_idx),
            })
            last_e = e
        return out

    def _prev_swing_low_before(self, idx: int) -> Optional[SwingPoint]:
        cand = [s for s in self.swing_lows if s.idx < idx]
        if not cand:
            return None
        return cand[-1]

    def _build_w_patterns(self):
        """W 底：ATR 摆动低点配对，颈线=两底间 high 最大，右侧逻辑同 V。"""
        c, h, l, avg = self.curves.close, self.curves.high, self.curves.low, self.curves.avg
        n = self.curves.n
        lows = self.swing_lows
        patterns: List[WPattern] = []

        for i in range(len(lows) - 1):
            s1, s2 = lows[i], lows[i + 1]
            idx1, idx2 = s1.idx, s2.idx
            gap = idx2 - idx1
            if gap < W_BOTTOM_MIN_GAP or gap > W_BOTTOM_MAX_GAP:
                continue
            b1, b2 = float(avg[idx1]), float(avg[idx2])
            if max(b1, b2) <= 1e-12:
                continue
            if abs(b1 - b2) / max(b1, b2) > W_BOTTOM_PRICE_TOL:
                continue
            neck, _ = _neck_high_between(h, idx1, idx2)
            bot_ref = max(b1, b2)
            if neck <= 0 or (neck - bot_ref) / bot_ref < W_NECK_MIN_RISE:
                continue
            stop_ref = float(min(l[idx1], l[idx2]))
            right_entry = None
            r_end = min(n - 1, idx2 + W_RIGHT_SCAN_CAP)
            for r in range(idx2 + 1, min(idx2 + W_RIGHT_MAX_SPAN, r_end) + 1):
                if float(c[r]) <= float(c[idx2]):
                    continue
                if float(c[r]) <= float(c[r - 1]):
                    continue
                right_entry = r
                break
            patterns.append(
                WPattern(
                    left_bottom_idx=idx1,
                    right_bottom_idx=idx2,
                    neck_price=neck,
                    stop_ref=stop_ref,
                    neck_rise_pct=(neck - bot_ref) / bot_ref,
                    right_entry_idx=right_entry,
                )
            )

        patterns.sort(key=lambda p: p.right_bottom_idx)
        self.w_patterns = patterns

    def _build_m_patterns(self):
        """M 顶：ATR 摆动高点配对，跌破顶部 98% 为 break 日。"""
        c, h, l = self.curves.close, self.curves.high, self.curves.low
        n = self.curves.n
        highs = self.swing_highs
        patterns: List[MPattern] = []
        break_bars: set = set()

        for i in range(len(highs) - 1):
            s1, s2 = highs[i], highs[i + 1]
            idx1, idx2 = s1.idx, s2.idx
            gap = idx2 - idx1
            if gap < M_TOP_MIN_GAP or gap > M_TOP_MAX_GAP:
                continue
            p1, p2 = float(h[idx1]), float(h[idx2])
            if max(p1, p2) <= 1e-12:
                continue
            if abs(p1 - p2) / max(p1, p2) > M_TOP_PRICE_TOL:
                continue
            top_price = max(p1, p2)
            trough = float(np.min(l[idx1 : idx2 + 1]))
            if (top_price - trough) / top_price < M_TOP_MIN_RETRACE:
                continue
            break_idx = None
            for j in range(idx2 + 1, min(idx2 + M_TOP_BREAK_SCAN, n - 1) + 1):
                if float(c[j]) < top_price * M_TOP_BREAK_RATIO:
                    if j > 0 and float(c[j - 1]) >= top_price * M_TOP_BREAK_RATIO:
                        break_idx = j
                        break
                    if j == idx2 + 1:
                        break_idx = j
                        break
            patterns.append(
                MPattern(
                    left_peak_idx=idx1,
                    right_peak_idx=idx2,
                    top_price=top_price,
                    trough_price=trough,
                    break_idx=break_idx,
                )
            )
            if break_idx is not None:
                break_bars.add(int(break_idx))

        self.m_patterns = patterns
        self.m_break_bars = break_bars

    def _build_v_top_patterns(self):
        """V 反顶部：摆动高点为峰，左侧均价抬升≥7%，右侧回落≥3% 后跌破峰×98%。"""
        c, h, l, avg = self.curves.close, self.curves.high, self.curves.low, self.curves.avg
        n = self.curves.n
        patterns: List[VTopPattern] = []
        break_bars: set = set()

        for sw_hi in self.swing_highs:
            pi = sw_hi.idx
            if pi < 10 or pi >= n - 3:
                continue
            prev_l = self._prev_swing_low_before(pi)
            left_start = prev_l.idx if prev_l else max(0, pi - V_MAX_RIGHT_SPAN)
            seg = avg[left_start : pi + 1]
            if len(seg) == 0:
                continue
            trough_avg = float(np.min(seg))
            peak_avg = float(avg[pi])
            peak_h = float(h[pi])
            if trough_avg <= 1e-12 or peak_avg <= trough_avg:
                continue
            rise_pct = (peak_avg - trough_avg) / trough_avg
            if rise_pct < V_TOP_RISE_MIN:
                continue
            break_idx = None
            for r in range(pi + 1, min(pi + V_TOP_RIGHT_SCAN, n - 1) + 1):
                trail = float(np.min(avg[pi : r + 1]))
                if (peak_avg - trail) / peak_avg < V_TOP_DROP_MIN:
                    continue
                if float(c[r]) >= peak_h * V_TOP_BREAK_RATIO:
                    continue
                if float(c[r]) >= float(c[r - 1]):
                    continue
                break_idx = r
                break
            patterns.append(
                VTopPattern(
                    peak_idx=pi,
                    left_trough_idx=left_start if prev_l is None else prev_l.idx,
                    peak_price=peak_h,
                    rise_pct=rise_pct,
                    break_idx=break_idx,
                )
            )
            if break_idx is not None:
                break_bars.add(int(break_idx))

        self.v_top_patterns = patterns
        self.v_top_break_bars = break_bars

    def _build_hs_top_patterns(self):
        """头肩顶：high 曲线 ATR 摆动高点 ls/head/rs（隔 1 摆动），颈线=两肩间 low 最小，仅首次收盘跌破颈线日触发。"""
        c, h, l = self.curves.close, self.curves.high, self.curves.low
        n = self.curves.n
        swings = self.swing_highs
        patterns: List[HSTopPattern] = []
        break_bars: set = set()

        for i in range(len(swings) - 4):
            s_ls, s_head, s_rs = swings[i], swings[i + 2], swings[i + 4]
            idx_ls, idx_head, idx_rs = s_ls.idx, s_head.idx, s_rs.idx
            if idx_rs - idx_ls > HS_TOTAL_MAX_GAP:
                continue
            if idx_head - idx_ls < HS_SHOULDER_HEAD_MIN_GAP:
                continue
            if idx_rs - idx_head < HS_HEAD_RS_MIN_GAP:
                continue
            p_ls = float(h[idx_ls])
            p_head = float(h[idx_head])
            p_rs = float(h[idx_rs])
            if p_head <= p_ls or p_head <= p_rs:
                continue
            if abs(p_ls - p_rs) / max(p_ls, p_rs) > HS_SHOULDER_TOL:
                continue
            neckline = float(np.min(l[idx_ls : idx_rs + 1]))
            min_sh = min(p_ls, p_rs)
            if min_sh <= 1e-12:
                continue
            if (min_sh - neckline) / min_sh < HS_MIN_TROUGH_RETRACE:
                continue
            break_idx = None
            for j in range(idx_rs + 1, min(idx_rs + HS_BREAK_SCAN, n)):
                if float(c[j]) < neckline:
                    if j > 0 and float(c[j - 1]) >= neckline:
                        break_idx = j
                        break
                    if j == idx_rs + 1:
                        break_idx = j
                        break
            patterns.append(
                HSTopPattern(
                    left_shoulder_idx=idx_ls,
                    head_idx=idx_head,
                    right_shoulder_idx=idx_rs,
                    left_shoulder_price=p_ls,
                    head_price=p_head,
                    right_shoulder_price=p_rs,
                    neckline_price=neckline,
                    break_idx=break_idx,
                )
            )
            if break_idx is not None:
                break_bars.add(int(break_idx))

        self.hs_top_patterns = patterns
        self.hs_break_bars = break_bars

    def to_w_right_events(self) -> List[dict]:
        out = []
        last_e = -10 ** 9
        for p in self.w_patterns:
            if p.right_entry_idx is None:
                continue
            e = int(p.right_entry_idx)
            if e - last_e < W_EVENT_MIN_GAP:
                continue
            out.append({
                'kind': 'W',
                'entry': e,
                'neck': float(p.neck_price),
                'stop_ref': float(p.stop_ref),
                'idx1': int(p.left_bottom_idx),
                'idx2': int(p.right_bottom_idx),
            })
            last_e = e
        return out

    def is_m_top_break_bar(self, idx: int) -> bool:
        return int(idx) in self.m_break_bars

    def is_v_top_break_bar(self, idx: int) -> bool:
        return int(idx) in self.v_top_break_bars

    def is_hs_top_break_bar(self, idx: int) -> bool:
        return int(idx) in self.hs_break_bars

    def prior_high_on_high_curve(
        self, before_idx: int, search_from: int = 0,
    ) -> Tuple[Optional[int], Optional[float]]:
        """前高：before_idx 之前最近一次 high 曲线摆动高点；若无则用区间 high 最大。"""
        before_idx = int(before_idx)
        highs = [s for s in self.swing_highs if search_from <= s.idx < before_idx]
        if highs:
            s = highs[-1]
            return s.idx, s.price
        if before_idx <= search_from:
            return None, None
        seg = self.curves.high[search_from:before_idx]
        if len(seg) == 0:
            return None, None
        j = int(np.argmax(seg))
        return search_from + j, float(seg[j])


def build_structure_registry(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: Optional[int] = None,
) -> V4StructureRegistry:
    n = int(n) if n is not None else len(close)
    o = np.asarray(open_, dtype=float)[:n]
    h = np.asarray(high, dtype=float)[:n]
    l = np.asarray(low, dtype=float)[:n]
    c = np.asarray(close, dtype=float)[:n]
    avg = _build_avg(o, h, l, c)
    curves = PriceCurves(open=o, high=h, low=l, close=c, avg=avg, n=n)
    atr = _atr(h, l, c, SWING_ATR_PERIOD)
    return V4StructureRegistry(curves, atr)


def get_structure_registry(
    open_,
    high,
    low,
    close,
    n: Optional[int] = None,
    *,
    reset_cache: bool = False,
) -> V4StructureRegistry:
    """单次 analyze 内复用 registry（避免 iter_v_left 与 detect_v_right 重复构建）。"""
    global _REGISTRY_CACHE
    n = int(n) if n is not None else len(close)
    key = (
        n,
        round(float(close[0]), 4),
        round(float(close[-1]), 4),
        round(float(high[0]), 4),
        round(float(low[-1]), 4),
    )
    if reset_cache:
        _REGISTRY_CACHE = (None, None)
    if _REGISTRY_CACHE[0] == key and _REGISTRY_CACHE[1] is not None:
        return _REGISTRY_CACHE[1]
    reg = build_structure_registry(open_, high, low, close, n)
    _REGISTRY_CACHE = (key, reg)
    return reg


def reset_structure_registry_cache():
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = (None, None)
