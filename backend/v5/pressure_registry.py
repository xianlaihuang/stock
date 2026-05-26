"""
V5 压力位注册表：静态 locked + 持仓期 dynamic（统一对外称「压力位」）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np

from v4_aggressive.v4_structure_curves import _build_avg
from v4_aggressive.v_left_geometry import (
    VLeftGeometry,
    _atr_walkback_start,
    scan_v_left_causal,
)

PressureMode = Literal['locked', 'dynamic']
PressureSource = Literal['swing_high', 'v_left_shoulder', 'gap_up', 'ma20']

PRESSURE_SCAN_START = 25
PRESSURE_CONFIRM_DROP_ATR = 0.75
PRESSURE_BAND_ATR = 0.5
PRESSURE_BAND_MIN_PCT = 0.02
PRESSURE_STAGE_HIGH_ATR = 1.5
PRESSURE_MIN_BAR_SPAN = 1

DYN_MA_PERIOD = 20
DYN_MA_MIN_BARS = 20
DYN_BAND_ATR = 0.3


@dataclass
class PressureLevel:
    mode: PressureMode
    anchor_idx: int
    P: float
    zone_lo: float
    zone_hi: float
    confirm_idx: int
    source: PressureSource
    bar_idx: int = -1
    open_idx: int = -1
    close_idx: int = -1
    broken_idx: int = -1
    tags: List[str] = field(default_factory=list)

    def summary_cn(self) -> str:
        mode_cn = '压力位(锁定)' if self.mode == 'locked' else '压力位(动态)'
        return (
            f'{mode_cn} · P={self.P:.2f} · 带[{self.zone_lo:.2f},{self.zone_hi:.2f}]'
            f' · 确认={self.confirm_idx}'
        )


def _pressure_band(P: float, atr_val: float) -> Tuple[float, float]:
    band = max(PRESSURE_BAND_ATR * atr_val, PRESSURE_BAND_MIN_PCT * P)
    return P - band, P + band


def _is_causal_atr_peak(
    high: np.ndarray,
    avg: np.ndarray,
    atr: np.ndarray,
    t: int,
    *,
    min_atr: float = PRESSURE_STAGE_HIGH_ATR,
    tol: float = 0.001,
) -> bool:
    t = int(t)
    if t < 1:
        return False
    start = _atr_walkback_start(avg, atr, t, min_atr)
    seg = high[start: t + 1]
    if len(seg) == 0:
        return False
    return float(high[t]) >= float(np.max(seg)) * (1.0 - tol)


def _pressure_drop_confirmed(
    peak_idx: int,
    t: int,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
) -> bool:
    h, t = int(peak_idx), int(t)
    if t <= h + PRESSURE_MIN_BAR_SPAN - 1:
        return False
    peak_h = float(high[h])
    if peak_h <= 1e-12:
        return False
    if float(high[t]) > peak_h * (1.001):
        return False
    drop = peak_h - float(np.min(low[h: t + 1]))
    a = float(atr[h]) if h < len(atr) and np.isfinite(atr[h]) and atr[h] > 0 else 1.0
    return drop >= PRESSURE_CONFIRM_DROP_ATR * a


def _is_up_gap_at(low: np.ndarray, high: np.ndarray, n: int) -> bool:
    n = int(n)
    if n < 1:
        return False
    return float(low[n - 1]) > float(high[n]) * (1.0 + 1e-9)


def _gap_unfilled_at(low: np.ndarray, high: np.ndarray, gap_n: int, t: int) -> bool:
    gap_n, t = int(gap_n), int(t)
    if not _is_up_gap_at(low, high, gap_n) or t < gap_n:
        return False
    gap_top = float(low[gap_n - 1])
    seg = high[gap_n: t + 1]
    if len(seg) == 0:
        return False
    return float(np.max(seg)) < gap_top * (1.0 - 1e-9)


def _dedup_locked(levels: List[PressureLevel], tol_pct: float = 0.008) -> List[PressureLevel]:
    kept: List[PressureLevel] = []
    for g in sorted(levels, key=lambda x: (x.confirm_idx, x.anchor_idx)):
        dup = False
        for f in kept:
            if abs(g.P - f.P) / max(f.P, 1e-12) < tol_pct and abs(g.anchor_idx - f.anchor_idx) <= 3:
                dup = True
                break
        if not dup:
            kept.append(g)
    return kept


def scan_locked_swing_pressure(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    avg: np.ndarray,
    atr: np.ndarray,
    *,
    start_offset: int = PRESSURE_SCAN_START,
) -> List[PressureLevel]:
    """摆动高点 + 回落确认 → locked 压力位。"""
    n = len(close)
    locked: List[PressureLevel] = []
    locked_peaks: set = set()
    pending: List[int] = []

    for t in range(int(start_offset), n):
        pending = [
            h for h in pending
            if h not in locked_peaks and float(high[t]) <= float(high[h]) * 1.001
        ]

        if _is_causal_atr_peak(high, avg, atr, t):
            if t not in locked_peaks and t not in pending:
                pending.append(t)

        still: List[int] = []
        for h in sorted(pending):
            if h in locked_peaks:
                continue
            if not _pressure_drop_confirmed(h, t, high, low, atr):
                still.append(h)
                continue
            P = float(high[h])
            a = float(atr[h]) if np.isfinite(atr[h]) and atr[h] > 0 else 1.0
            zlo, zhi = _pressure_band(P, a)
            locked.append(PressureLevel(
                mode='locked',
                anchor_idx=h,
                P=P,
                zone_lo=zlo,
                zone_hi=zhi,
                confirm_idx=t,
                source='swing_high',
                bar_idx=h,
                tags=['已锁定', '摆动高点'],
            ))
            locked_peaks.add(h)
        pending = still

    return locked


def scan_locked_from_v_left(
    v_lefts: List[VLeftGeometry],
    high: np.ndarray,
    atr: np.ndarray,
) -> List[PressureLevel]:
    """V 左肩锁定 → 结构确认压力位（左肩 high）。"""
    out: List[PressureLevel] = []
    for g in v_lefts:
        if g.kind != 'v_left' or g.confirm_idx < 0:
            continue
        h = int(g.effective_peak_idx)
        P = float(high[h])
        a = float(atr[h]) if h < len(atr) and np.isfinite(atr[h]) and atr[h] > 0 else 1.0
        zlo, zhi = _pressure_band(P, a)
        out.append(PressureLevel(
            mode='locked',
            anchor_idx=h,
            P=P,
            zone_lo=zlo,
            zone_hi=zhi,
            confirm_idx=int(g.confirm_idx),
            source='v_left_shoulder',
            bar_idx=h,
            tags=['已锁定', 'V左肩'],
        ))
    return out


def scan_locked_gap_pressure(
    low: np.ndarray,
    high: np.ndarray,
    *,
    start_offset: int = PRESSURE_SCAN_START,
) -> List[PressureLevel]:
    """上方缺口：形成且截至 t 未回补 → 在 t 锁定。"""
    n = len(high)
    locked: List[PressureLevel] = []
    locked_gaps: set = set()

    for t in range(int(start_offset), n):
        if not _is_up_gap_at(low, high, t):
            continue
        key = t
        if key in locked_gaps:
            continue
        if not _gap_unfilled_at(low, high, t, t):
            continue
        gap_top = float(low[t - 1])
        gap_bot = float(high[t])
        P = gap_top
        zlo, zhi = gap_bot, gap_top
        if zlo > zhi:
            zlo, zhi = zhi, zlo
        locked.append(PressureLevel(
            mode='locked',
            anchor_idx=t - 1,
            P=P,
            zone_lo=zlo,
            zone_hi=zhi,
            confirm_idx=t,
            source='gap_up',
            bar_idx=t - 1,
            tags=['已锁定', '上方缺口'],
        ))
        locked_gaps.add(key)

    return locked


def scan_locked_pressure_causal(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    avg: np.ndarray,
    atr: np.ndarray,
    *,
    v_lefts: Optional[List[VLeftGeometry]] = None,
) -> List[PressureLevel]:
    """汇总静态 locked 压力位。"""
    if v_lefts is None:
        from v4_aggressive.v4_structure_curves import V_DROP_MIN
        v_lefts = scan_v_left_causal(open_, high, low, close, avg, atr, drop_min=V_DROP_MIN)

    parts = []
    parts.extend(scan_locked_swing_pressure(high, low, close, avg, atr))
    parts.extend(scan_locked_from_v_left(v_lefts, high, atr))
    parts.extend(scan_locked_gap_pressure(low, high))
    return _dedup_locked(parts)


def scan_dynamic_pressure_ma(
    close: np.ndarray,
    atr: np.ndarray,
    positions: List[Tuple[int, int]],
    *,
    period: int = DYN_MA_PERIOD,
    min_bars: int = DYN_MA_MIN_BARS,
) -> List[PressureLevel]:
    """持仓期内每日 MA 压力位快照（当日落库不溯及）。"""
    import talib

    ma = talib.SMA(close.astype(float), timeperiod=period)
    out: List[PressureLevel] = []
    for pos_id, (open_i, close_i) in enumerate(positions):
        open_i, close_i = int(open_i), int(close_i)
        for t in range(open_i, close_i + 1):
            if t < min_bars - 1:
                continue
            if not np.isfinite(ma[t]):
                continue
            P = float(ma[t])
            a = float(atr[t]) if np.isfinite(atr[t]) and atr[t] > 0 else P * 0.02
            band = DYN_BAND_ATR * a
            out.append(PressureLevel(
                mode='dynamic',
                anchor_idx=t,
                P=P,
                zone_lo=P - band,
                zone_hi=P + band,
                confirm_idx=t,
                source='ma20',
                bar_idx=t,
                open_idx=open_i,
                close_idx=close_i,
                tags=['持仓期', f'MA{period}'],
            ))
    return out


def mark_pressure_broken(
    levels: List[PressureLevel],
    high: np.ndarray,
    close: np.ndarray,
    *,
    use_close: bool = True,
) -> None:
    """因果追加突破日（不改 P/zone）。"""
    for lv in levels:
        if lv.broken_idx >= 0:
            continue
        start = lv.confirm_idx
        for t in range(start, len(close)):
            px = float(close[t]) if use_close else float(high[t])
            if px > lv.zone_hi * (1.0 + 1e-4):
                lv.broken_idx = t
                break
