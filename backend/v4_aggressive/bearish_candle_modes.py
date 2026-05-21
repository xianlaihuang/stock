"""V4 看跌K线形态：legacy（改动前）与 slim（改动后，去掉黄昏星/十字/射击之星）。"""
import numpy as np
import pandas as pd
import talib

EVENING_STAR_CN = '黄昏星'
EVENING_DOJI_CN = '黄昏十字'
SHOOTING_STAR_CN = '射击之星'

MA5_DEFER_KINDS = frozenset({EVENING_STAR_CN, EVENING_DOJI_CN, SHOOTING_STAR_CN})
MA5_BREAK_SELL_REASON = '看跌K线形态破5日线'

_LEGACY_DEFS = (
    ('CDLENGULFING', '看跌吞没'),
    ('CDLEVENINGSTAR', EVENING_STAR_CN),
    ('CDLHANGINGMAN', '上吊线'),
    ('CDLDARKCLOUDCOVER', '乌云盖顶'),
    ('CDLEVENINGDOJISTAR', EVENING_DOJI_CN),
    ('CDLSHOOTINGSTAR', SHOOTING_STAR_CN),
)

_SLIM_DEFS = (
    ('CDLENGULFING', '看跌吞没'),
    ('CDLHANGINGMAN', '上吊线'),
    ('CDLDARKCLOUDCOVER', '乌云盖顶'),
)


def _detect_kind(df, idx, defs):
    if idx < 3:
        return None
    o = df['open'].iloc[: idx + 1].values.astype(float)
    h = df['high'].iloc[: idx + 1].values.astype(float)
    l = df['low'].iloc[: idx + 1].values.astype(float)
    c = df['close'].iloc[: idx + 1].values.astype(float)
    for pname, label in defs:
        func = getattr(talib, pname, None)
        if func is None:
            continue
        try:
            result = func(o, h, l, c)
            if len(result) > 0 and result[-1] == -100:
                return label
        except Exception:
            pass
    return None


def detect_bearish_kind(df, idx, mode='slim'):
    defs = _LEGACY_DEFS if mode == 'legacy' else _SLIM_DEFS
    return _detect_kind(df, idx, defs)


def _ma5_at(df, idx):
    idx = int(idx)
    if idx < 5:
        return None
    close = df['close'].values.astype(float)
    ma5 = talib.MA(close[: idx + 1], timeperiod=5)
    v = ma5[idx]
    return None if pd.isna(v) else float(v)


def close_still_on_ma5(df, idx, eps=1e-9):
    m5 = _ma5_at(df, idx)
    if m5 is None:
        return False
    return float(df['close'].values.astype(float)[idx]) >= m5 * (1.0 - eps)


HANGING_MAN_CN = '上吊线'


def close_above_ma20(df, idx, eps=1e-9):
    """上吊线须在 20 日均线上方（上涨末端反转语义）。"""
    idx = int(idx)
    if idx < 20:
        return False
    close = df['close'].values.astype(float)
    ma20 = talib.MA(close[: idx + 1], timeperiod=20)
    v = ma20[idx]
    if pd.isna(v):
        return False
    return float(close[idx]) > float(v) * (1.0 + eps)


def close_broke_below_ma5(df, idx, eps=1e-9):
    m5 = _ma5_at(df, idx)
    if m5 is None:
        return False
    return float(df['close'].values.astype(float)[idx]) < m5 * (1.0 - eps)


def evening_star_prev_yang_shrink_vol(df, idx):
    if idx < 1 or 'volume' not in df.columns:
        return False
    try:
        if float(df['close'].iloc[idx - 1]) <= float(df['open'].iloc[idx - 1]):
            return False
        v_cur = float(df['volume'].iloc[idx])
        v_prev = float(df['volume'].iloc[idx - 1])
    except Exception:
        return False
    return np.isfinite(v_cur) and np.isfinite(v_prev) and v_prev > 0 and v_cur < v_prev


def bearish_kind_defers_sell_until_ma5_break(df, idx, kind):
    if kind not in MA5_DEFER_KINDS:
        return False
    return close_still_on_ma5(df, idx)


def sell_bearish_pattern_at(df, idx, mode='slim'):
    kind = detect_bearish_kind(df, idx, mode)
    if kind is None:
        return False
    if kind == HANGING_MAN_CN and not close_above_ma20(df, idx):
        return False
    if mode != 'legacy':
        return True
    if kind == EVENING_STAR_CN and evening_star_prev_yang_shrink_vol(df, idx):
        return False
    if bearish_kind_defers_sell_until_ma5_break(df, idx, kind):
        return False
    return True


def precompute_bearish_pattern_series(df, start_offset, mode='slim'):
    n = len(df)
    arr = np.zeros(n, dtype=bool)
    for idx in range(int(start_offset), n):
        arr[idx] = sell_bearish_pattern_at(df, idx, mode)
    return arr
