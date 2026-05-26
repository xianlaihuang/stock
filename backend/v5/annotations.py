"""V5 标注：V 左 + 压力位（静态 + 动态）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from v4_aggressive.v4_structure_curves import V_DROP_MIN, _build_avg
from v5.positions import holding_periods_from_paired
from v5.pressure_registry import (
    mark_pressure_broken,
    scan_dynamic_pressure_ma,
    scan_locked_pressure_causal,
)
from v5.v_left import scan_v_left_causal


def scan_v5_annotations(
    df: pd.DataFrame,
    paired_signals: Optional[list] = None,
    *,
    drop_min: float = V_DROP_MIN,
    mark_broken: bool = True,
) -> Dict[str, Any]:
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    avg = _build_avg(o, h, l, c)
    import talib
    atr = talib.ATR(h, l, c, timeperiod=14)

    v_lefts = scan_v_left_causal(o, h, l, c, avg, atr, drop_min=drop_min)
    locked = scan_locked_pressure_causal(o, h, l, c, avg, atr, v_lefts=v_lefts)

    positions = holding_periods_from_paired(df, paired_signals or [])
    dynamic = scan_dynamic_pressure_ma(c, atr, positions) if positions else []

    if mark_broken:
        mark_pressure_broken(locked, h, c, use_close=True)

    dates = [str(d)[:10] for d in df['date']] if 'date' in df.columns else [str(i) for i in range(len(df))]

    return {
        'dates': dates,
        'v_lefts': v_lefts,
        'pressure_locked': locked,
        'pressure_dynamic': dynamic,
        'holding_periods': positions,
        'counts': {
            'v_left': sum(1 for g in v_lefts if g.kind == 'v_left'),
            'pressure_locked': len(locked),
            'pressure_dynamic': len(dynamic),
            'holding_segments': len(positions),
        },
    }
