"""
V 左肩几何 — 三层因果判定（不依赖固定 K 根数，也不在单股上堆阈值补丁）

Layer 1 结构不变量（硬约束，不可绕过）
  - 左肩 bar 必须早于 V 底 bar（禁止单 K 线 high→low 冒充左段）
  - 左肩→V底 的 ATR 路径 >= MIN_V_LEFT_ATR_PATH（左段是「过程」不是 event）
  - 跌深 >= drop_min，且 (跌深 >= BLUNT_DROP_MIN 或 ATR路 >= BLUNT_ATR_PATH_MIN)

Layer 2 去噪（因果摆动底）
  - 仅在 ATR 窗口内创阶段新低的 bar 上评估（滤掉跌途中的毛刺）

Layer 3 在线确认锁定（标注不可回溯修改）
  - bar b 为候选 V 底（Layer1+2 通过）；之后若 low 再破 b 则候选作废
  - bar t：自 b 起反弹 ≥ CONFIRM_BOUNCE_ATR → 在 t 确认，锁定 V 左 [peak..b]
  - 已锁定标注不随后续更深底/同 leg 去重而删除或改写

event 假 V = 违反 Layer 1 的浅/event 簇（含长下影单根暴跌）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np

VLeftKind = Literal['v_left', 'v_left_event_fake', 'v_left_weak', 'none']

# event 簇：左肩 high→V底 的 ATR 路径上限（非 K 根数）
EVENT_ATR_PATH_MAX = 1.85
# 左肩→V底 净跌占左肩 high 比例，event 簇通常较浅
EVENT_DROP_MAX = 0.06
BIG_YANG_BODY_PCT = 0.025
GAP_DOWN_OPEN_PCT = 0.012
BOTTOM_REJECTION_MIN = 0.42
BOTTOM_SHADOW_BODY_MIN = 0.65
# 钝底：ATR 路径或跌深（比例，非 bar 数）
BLUNT_ATR_PATH_MIN = 2.2
BLUNT_DROP_MIN = 0.10
# V 左有效：左段至少走过的 ATR 路径（单根/极短 event 一律否）
MIN_V_LEFT_ATR_PATH = 1.5
# 左肩与 V 底至少相隔 bar 数（0=同根 K，结构不成立）
MIN_V_LEFT_BAR_SPAN = 1
# 两跌段分界：自前谷底反弹 ≥ 此 ATR 倍数 → 新 leg，左肩重新计
LEG_BOUNCE_ATR = 1.0
# 因果扫描起始 bar：≥ ATR(14) 预热即可；60 会漏掉前 ~3 个月
V_LEFT_SCAN_START = 25
# V 底后反弹 ≥ 此 ATR 倍数 → 在 bar t 确认并锁定 V 左（不要求 V 右走完）
V_LEFT_CONFIRM_BOUNCE_ATR = 0.75
# 确认前 V 底后至少经过 bar 数（n+8 底，n+10 确认 → ≥1）
V_LEFT_CONFIRM_MIN_BARS = 1


@dataclass
class VLeftGeometry:
    left_peak_idx: int
    bottom_idx: int
    effective_peak_idx: int
    net_drop: float
    drop_pct: float
    path_length: float
    path_efficiency: float
    drop_herfindahl: float
    max_bar_share: float
    atr_path: float
    bar_span: int
    bottom_rejection: float
    kind: VLeftKind
    leg_start_idx: int = 0
    confirm_idx: int = -1
    tags: List[str] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)

    @property
    def pass_geometry(self) -> bool:
        return self.kind == 'v_left'

    @property
    def is_v_left(self) -> bool:
        return self.kind == 'v_left'

    def summary_cn(self) -> str:
        kind_cn = {
            'v_left': 'V左有效',
            'v_left_event_fake': 'event假V',
            'v_left_weak': '左肩弱',
            'none': '—',
        }[self.kind]
        parts = [
            kind_cn,
            f'ATR路={self.atr_path:.1f}',
            f'跌深={self.drop_pct:.0%}',
            f'η={self.path_efficiency:.2f}',
            f'探底={self.bottom_rejection:.0%}',
        ]
        if self.tags:
            parts.append(','.join(self.tags))
        if self.reject_reasons:
            parts.append('✗' + ';'.join(self.reject_reasons))
        return ' · '.join(parts)


def _atr_walkback_start(avg: np.ndarray, atr: np.ndarray, end: int, min_atr: float) -> int:
    """从 end 向左累计 ATR 路径，返回窗口起点（因果，不含未来）。"""
    end = int(end)
    if end < 1:
        return 0
    path = 0.0
    i = end
    while i > 0 and path < min_atr:
        a = float(atr[i]) if i < len(atr) and np.isfinite(atr[i]) and atr[i] > 0 else 1.0
        path += abs(float(avg[i]) - float(avg[i - 1])) / max(a, 1e-12)
        i -= 1
    return max(0, i)


def _passes_v_left_structure(
    bar_span: int,
    atr_path: float,
    drop_pct: float,
) -> Tuple[bool, List[str]]:
    """Layer 1：结构不变量，返回 (通过, 违反原因)。"""
    reasons: List[str] = []
    if bar_span < MIN_V_LEFT_BAR_SPAN:
        reasons.append('单K无左段')
    if atr_path < MIN_V_LEFT_ATR_PATH:
        reasons.append(f'左段ATR<{MIN_V_LEFT_ATR_PATH}')
    has_magnitude = drop_pct >= BLUNT_DROP_MIN or atr_path >= BLUNT_ATR_PATH_MIN
    if not has_magnitude:
        reasons.append('跌深与ATR路径均不足')
    return (len(reasons) == 0, reasons)


def _is_causal_atr_trough(
    low: np.ndarray,
    avg: np.ndarray,
    atr: np.ndarray,
    t: int,
    *,
    min_atr: float = MIN_V_LEFT_ATR_PATH,
    tol: float = 0.001,
) -> bool:
    """Layer 2：low[t] 为 ATR 窗口内阶段新低（去毛刺）。"""
    t = int(t)
    if t < 1:
        return False
    start = _atr_walkback_start(avg, atr, t, min_atr)
    seg = low[start: t + 1]
    if len(seg) == 0:
        return False
    return float(low[t]) <= float(np.min(seg)) * (1.0 + tol)


def _leg_start_before_trough(
    t: int,
    low: np.ndarray,
    high: np.ndarray,
    atr: np.ndarray,
    *,
    bounce_atr: float = LEG_BOUNCE_ATR,
) -> int:
    """
    当前跌段 leg 起点：从 V 底 t 向左，找最近一次「前谷底 + 显著反弹」后的反弹峰位置。
    因果：仅用 [0..t]；若无 leg 分界则返回 0。
    """
    t = int(t)
    if t < 2:
        return 0
    bot_t = float(low[t])
    for i in range(t - 1, 0, -1):
        trough = float(low[i])
        if bot_t >= trough * (1.0 - 0.004):
            continue
        if i + 1 > t:
            continue
        rebound_high = float(np.max(high[i + 1: t + 1]))
        a = float(atr[i]) if i < len(atr) and np.isfinite(atr[i]) and atr[i] > 0 else 1.0
        if rebound_high - trough >= bounce_atr * a:
            sub = high[i + 1: t]
            if len(sub) == 0:
                return i + 1
            return i + 1 + int(np.argmax(sub))
    return 0


def _leg_reset_between(
    earlier_bottom: int,
    later_bottom: int,
    low: np.ndarray,
    high: np.ndarray,
    atr: np.ndarray,
) -> bool:
    """两 V 底之间是否存在 ATR 级反弹 → 分属不同 leg。"""
    a, b = int(earlier_bottom), int(later_bottom)
    if b <= a + 1:
        return False
    trough = float(low[a])
    rebound_high = float(np.max(high[a + 1: b + 1]))
    ai = float(atr[a]) if a < len(atr) and np.isfinite(atr[a]) and atr[a] > 0 else 1.0
    return rebound_high - trough >= LEG_BOUNCE_ATR * ai


def _v_bottom_bounce_confirmed(
    bottom_idx: int,
    t: int,
    low: np.ndarray,
    high: np.ndarray,
    atr: np.ndarray,
    *,
    bounce_atr: float = V_LEFT_CONFIRM_BOUNCE_ATR,
    min_bars: int = V_LEFT_CONFIRM_MIN_BARS,
) -> bool:
    """bar t 能否确认 bottom_idx 为 V 底：仅看 [bottom..t]，不要求 V 右走完。"""
    b, t = int(bottom_idx), int(t)
    if t <= b + min_bars - 1:
        return False
    a = float(atr[b]) if b < len(atr) and np.isfinite(atr[b]) and atr[b] > 0 else 1.0
    rebound = float(np.max(high[b + 1: t + 1])) - float(low[b])
    return rebound >= bounce_atr * a


def _classify_v_left_kind(
    *,
    struct_ok: bool,
    struct_reasons: List[str],
    drop_pct: float,
    atr_path: float,
    bar_span: int,
    path_efficiency: float,
    max_bar_share: float,
    has_rejection: bool,
    rally_start: bool,
) -> Tuple[VLeftKind, List[str], List[str]]:
    """Layer 1 结果 + 形态标签 → kind / tags / reject_reasons。"""
    tags: List[str] = []
    reasons: List[str] = list(struct_reasons)

    if not struct_ok:
        kind: VLeftKind = 'v_left_event_fake'
        if bar_span < MIN_V_LEFT_BAR_SPAN:
            tags.append('event簇')
        elif atr_path <= EVENT_ATR_PATH_MAX and drop_pct <= EVENT_DROP_MAX:
            tags.append('event簇')
        else:
            tags.append('结构不足')
        return kind, tags, reasons

    if has_rejection:
        kind = 'v_left'
        tags.append('不规则V左')
        if path_efficiency >= 0.85:
            tags.append('急跌')
        if atr_path >= BLUNT_ATR_PATH_MIN:
            tags.append('左肩充分')
    elif drop_pct >= BLUNT_DROP_MIN or atr_path >= BLUNT_ATR_PATH_MIN:
        kind = 'v_left'
        tags.append('钝底V左')
        if drop_pct >= 0.12:
            tags.append('大级别')
    elif max_bar_share > 0.55 and rally_start:
        kind = 'v_left_event_fake'
        reasons.append('单根暴跌event')
        tags.append('event簇')
    else:
        kind = 'v_left_weak'
        if not has_rejection:
            reasons.append('无探底拒绝')
    return kind, tags, reasons


def _body_pct(open_: float, close: float) -> float:
    if open_ <= 1e-12:
        return 0.0
    return (close - open_) / open_


def _lower_shadow_ratio(open_: float, high: float, low: float, close: float) -> float:
    body = abs(close - open_)
    lower = min(open_, close) - low
    if body < 1e-12:
        return lower / max(high - low, 1e-12)
    return lower / body


def _bottom_rejection(open_: float, high: float, low: float, close: float) -> float:
    rng = high - low
    if rng <= 1e-12:
        return 0.0
    return (close - low) / rng


def _macro_decline_peak(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bottom_idx: int,
    atr: np.ndarray,
    *,
    drop_min: float = 0.04,
    max_atr_lookback: float = 18.0,
) -> int:
    """
    从 V 底沿 ATR 路径向左扫描，取区间内 high 最大点 = 宏观左肩。
    扫描深度用 ATR 路径（非固定 K 根数）；不在中间反弹处提前停止。
    """
    bi = int(bottom_idx)
    peak_i = bi
    peak_h = float(high[bi])
    atr_path = 0.0
    i = bi - 1
    while i >= 0 and atr_path < max_atr_lookback:
        hi = float(high[i])
        if hi > peak_h:
            peak_h = hi
            peak_i = i
        if i + 1 < len(close):
            a = float(atr[i + 1]) if np.isfinite(atr[i + 1]) and atr[i + 1] > 0 else 1.0
            atr_path += abs(float(close[i + 1]) - float(close[i])) / a
        i -= 1
    if peak_h > 1e-12 and (peak_h - float(low[bi])) / peak_h < drop_min:
        return bi
    return peak_i


def _effective_left_peak(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    left_start: int,
    bottom_idx: int,
    drop_min: float = 0.04,
) -> int:
    macro = _macro_decline_peak(high, low, close, bottom_idx, atr, drop_min=drop_min)
    a, b = int(left_start), int(bottom_idx)
    if b < a:
        a = b
    seg = high[a: b + 1]
    local = a + int(np.argmax(seg)) if len(seg) else a
    return macro if float(high[macro]) >= float(high[local]) else local


def _atr_path_segment(avg: np.ndarray, atr: np.ndarray, start: int, end: int) -> float:
    total = 0.0
    for i in range(int(start) + 1, int(end) + 1):
        a = float(atr[i]) if i < len(atr) and np.isfinite(atr[i]) and atr[i] > 0 else 1.0
        total += abs(float(avg[i]) - float(avg[i - 1])) / max(a, 1e-12)
    return total


def _is_rally_then_crash_at_start(
    open_: np.ndarray,
    close: np.ndarray,
    peak_idx: int,
    bottom_idx: int,
) -> bool:
    """峰后 1~2 根内大阳+缺口/大阴，且整体 ATR 路径极短 → event 簇。"""
    pi, bi = int(peak_idx), int(bottom_idx)
    if bi <= pi:
        return False
    for k in range(pi + 1, min(bi, pi + 3) + 1):
        if k < 1:
            continue
        prev = k - 1
        yang = close[prev] > open_[prev] and _body_pct(float(open_[prev]), float(close[prev])) >= BIG_YANG_BODY_PCT
        gap_dn = float(open_[k]) <= float(close[prev]) * (1.0 - GAP_DOWN_OPEN_PCT)
        big_yin = float(close[k]) < float(open_[k]) and float(close[k]) < float(close[prev]) * 0.985
        if yang and (gap_dn or big_yin):
            return True
    return False


def _has_bottom_rejection_candle(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bottom_idx: int,
) -> Tuple[bool, float]:
    bi = int(bottom_idx)
    o, h, l, c = float(open_[bi]), float(high[bi]), float(low[bi]), float(close[bi])
    rej = _bottom_rejection(o, h, l, c)
    shadow = _lower_shadow_ratio(o, h, l, c)
    ok = rej >= BOTTOM_REJECTION_MIN or (c > o and shadow >= BOTTOM_SHADOW_BODY_MIN)
    return ok, rej


def compute_v_left_geometry(
    avg: np.ndarray,
    atr: np.ndarray,
    left_peak_idx: int,
    bottom_idx: int,
    *,
    open_: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    close: Optional[np.ndarray] = None,
    drop_min: float = 0.04,
    force_peak_idx: Optional[int] = None,
    force_leg_start: Optional[int] = None,
) -> Optional[VLeftGeometry]:
    n = len(avg)
    lp, bi = int(left_peak_idx), int(bottom_idx)
    if bi < 1 or lp < 0 or bi >= n:
        return None

    o = open_ if open_ is not None else avg
    h = high if high is not None else avg
    l = low if low is not None else avg
    c = close if close is not None else avg

    left_start = min(lp, bi)
    if force_peak_idx is not None:
        eff_peak = int(force_peak_idx)
    else:
        eff_peak = _effective_left_peak(h, l, c, atr, left_start, bi, drop_min=drop_min)
        if eff_peak >= bi:
            eff_peak = lp

    peak_high = float(h[eff_peak])
    bot_low = float(l[bi])
    drop_pct = (peak_high - bot_low) / peak_high if peak_high > 1e-12 else 0.0
    if drop_pct < drop_min:
        return None

    seg_avg = avg[min(eff_peak, bi): bi + 1].astype(float)
    peak_avg = float(np.max(avg[min(eff_peak, bi): bi + 1]))
    bot_avg = float(avg[bi])
    net_drop = max(0.0, peak_avg - bot_avg)

    path_length = float(np.sum(np.abs(np.diff(seg_avg)))) if len(seg_avg) > 1 else net_drop
    path_efficiency = net_drop / path_length if path_length > 1e-12 else 1.0

    bar_drops = []
    for i in range(min(eff_peak, bi) + 1, bi + 1):
        bar_drops.append(max(0.0, float(avg[i - 1]) - float(avg[i])))
    total_bar_drop = sum(bar_drops) or 1e-12
    shares = [d / total_bar_drop for d in bar_drops]
    max_bar_share = max(shares) if shares else 0.0
    drop_herf = float(sum(s * s for s in shares))

    atr_path = _atr_path_segment(avg, atr, min(eff_peak, bi), bi)
    bar_span = bi - eff_peak

    has_rejection, bottom_rej = _has_bottom_rejection_candle(o, h, l, c, bi)
    rally_start = _is_rally_then_crash_at_start(o, c, eff_peak, bi)

    struct_ok, struct_reasons = _passes_v_left_structure(bar_span, atr_path, drop_pct)
    kind, tags, reasons = _classify_v_left_kind(
        struct_ok=struct_ok,
        struct_reasons=struct_reasons,
        drop_pct=drop_pct,
        atr_path=atr_path,
        bar_span=bar_span,
        path_efficiency=path_efficiency,
        max_bar_share=max_bar_share,
        has_rejection=has_rejection,
        rally_start=rally_start,
    )

    return VLeftGeometry(
        left_peak_idx=lp,
        bottom_idx=bi,
        effective_peak_idx=eff_peak,
        leg_start_idx=int(force_leg_start if force_leg_start is not None else 0),
        net_drop=net_drop,
        drop_pct=drop_pct,
        path_length=path_length,
        path_efficiency=path_efficiency,
        drop_herfindahl=drop_herf,
        max_bar_share=max_bar_share,
        atr_path=atr_path,
        bar_span=bar_span,
        bottom_rejection=bottom_rej,
        kind=kind,
        tags=tags,
        reject_reasons=reasons,
    )


def find_causal_peak_at(
    t: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    *,
    drop_min: float = 0.04,
    max_atr_lookback: float = 18.0,
    trough_tol: float = 0.002,
) -> Optional[Tuple[int, float, int]]:
    """
    bar t 因果左肩：在当前跌段 leg 内（上次 ATR 反弹之后）取 high 最大点。
    返回 (peak_idx, drop_pct, leg_start_idx)；peak 严格早于 t。
    """
    t = int(t)
    if t < 1:
        return None
    bot = float(low[t])
    leg_start = _leg_start_before_trough(t, low, high, atr)
    search_lo = max(0, int(leg_start))
    if search_lo >= t:
        return None

    seg_high = high[search_lo:t]
    if len(seg_high) == 0:
        return None
    peak_i = search_lo + int(np.argmax(seg_high))
    peak_h = float(high[peak_i])

    seg_min = float(np.min(low[peak_i: t + 1]))
    if seg_min < bot * (1.0 - trough_tol):
        return None
    if peak_h <= 1e-12:
        return None
    drop = (peak_h - bot) / peak_h
    if drop < drop_min:
        return None
    return peak_i, drop, leg_start


def evaluate_v_left_at(
    t: int,
    avg: np.ndarray,
    atr: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    drop_min: float = 0.04,
) -> Optional[VLeftGeometry]:
    """bar t 因果评估：只使用 t 及以前的数据。"""
    found = find_causal_peak_at(t, high, low, close, atr, drop_min=drop_min)
    if found is None:
        return None
    peak_i, _, leg_start = found
    return compute_v_left_geometry(
        avg[: t + 1],
        atr[: t + 1],
        peak_i,
        t,
        open_=open_[: t + 1],
        high=high[: t + 1],
        low=low[: t + 1],
        close=close[: t + 1],
        drop_min=drop_min,
        force_peak_idx=peak_i,
        force_leg_start=leg_start,
    )


def scan_v_left_causal(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    avg: np.ndarray,
    atr: np.ndarray,
    *,
    start_offset: int = V_LEFT_SCAN_START,
    drop_min: float = 0.04,
) -> List[VLeftGeometry]:
    """
    在线因果扫描 V 左：确认后锁定，已标注不随后续 K 线回溯修改。

    流程（bar t 逐日推进，只用 [0..t]）：
      1. 作废 pending：若 low[t] 跌破候选底 b → b 移除
      2. bar t 若为 ATR 阶段新低且 Layer1 通过 → 加入 pending
      3. 对每个 pending 的 b：若 t 处反弹确认 → 锁定 V左(peak..b)，移出 pending
      4. locked 列表只增不改
    """
    n = len(close)
    start = int(start_offset)
    locked: List[VLeftGeometry] = []
    locked_bottoms: set = set()
    pending: List[int] = []
    tol = 0.001

    for t in range(start, n):
        pending = [
            b for b in pending
            if b not in locked_bottoms and float(low[t]) >= float(low[b]) * (1.0 - tol)
        ]

        if _is_causal_atr_trough(low, avg, atr, t):
            found = find_causal_peak_at(t, high, low, close, atr, drop_min=drop_min)
            if found is not None:
                peak_i, _, _ = found
                if peak_i < t and t not in locked_bottoms:
                    geo = evaluate_v_left_at(
                        t, avg, atr, open_, high, low, close, drop_min=drop_min,
                    )
                    if geo is not None and geo.kind == 'v_left' and t not in pending:
                        pending.append(t)

        still_pending: List[int] = []
        for b in sorted(pending):
            if b in locked_bottoms:
                continue
            if not _v_bottom_bounce_confirmed(b, t, low, high, atr):
                still_pending.append(b)
                continue
            geo = evaluate_v_left_at(
                b, avg, atr, open_, high, low, close, drop_min=drop_min,
            )
            if geo is None or geo.kind != 'v_left':
                still_pending.append(b)
                continue
            geo = VLeftGeometry(
                left_peak_idx=geo.left_peak_idx,
                bottom_idx=geo.bottom_idx,
                effective_peak_idx=geo.effective_peak_idx,
                net_drop=geo.net_drop,
                drop_pct=geo.drop_pct,
                path_length=geo.path_length,
                path_efficiency=geo.path_efficiency,
                drop_herfindahl=geo.drop_herfindahl,
                max_bar_share=geo.max_bar_share,
                atr_path=geo.atr_path,
                bar_span=geo.bar_span,
                bottom_rejection=geo.bottom_rejection,
                kind=geo.kind,
                leg_start_idx=geo.leg_start_idx,
                confirm_idx=t,
                tags=geo.tags + ['已锁定'],
                reject_reasons=list(geo.reject_reasons),
            )
            locked.append(geo)
            locked_bottoms.add(b)
        pending = still_pending

    locked.sort(key=lambda g: g.bottom_idx)
    return locked
