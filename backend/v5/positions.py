"""从买卖信号序列推导持仓区间（有仓即算）。"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def _date_to_idx(df: pd.DataFrame) -> dict:
    if 'date' not in df.columns:
        return {str(i): i for i in range(len(df))}
    out = {}
    for i, d in enumerate(df['date']):
        out[str(d)[:10]] = i
    return out


def holding_periods_from_paired(
    df: pd.DataFrame,
    paired_signals: list,
) -> List[Tuple[int, int]]:
    """
    返回 [(open_idx, close_idx), ...]。
    按 paired 顺序：B 开仓，S 平仓；末尾仍持仓则 close_idx = 最后一根 K。
    """
    if not paired_signals:
        return []
    d2i = _date_to_idx(df)
    n = len(df)
    periods: List[Tuple[int, int]] = []
    open_idx = None

    for row in paired_signals:
        t = row.get('type')
        if t == 'B':
            if open_idx is not None:
                periods.append((open_idx, n - 1))
            di = row.get('idx')
            if di is None:
                di = d2i.get(str(row.get('date', ''))[:10])
            if di is not None:
                open_idx = int(di)
        elif t == 'S' and open_idx is not None:
            di = row.get('idx')
            if di is None:
                di = d2i.get(str(row.get('date', ''))[:10])
            if di is not None:
                periods.append((open_idx, int(di)))
            open_idx = None

    if open_idx is not None:
        periods.append((open_idx, n - 1))
    return periods
