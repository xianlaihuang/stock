import pandas as pd
import numpy as np
import talib
from scipy import stats
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

BUY_RULES = {}
SELL_RULES = {}
MANDATORY_SELL_RULES = {}
MANDATORY_BUY_RULES = {}
BUY_RESTRICTION_RULES = {}


def _register_buy(name):
    def decorator(func):
        BUY_RULES[name] = func
        return func
    return decorator


def _register_sell(name):
    def decorator(func):
        SELL_RULES[name] = func
        return func
    return decorator


def _register_mandatory_sell(name):
    def decorator(func):
        MANDATORY_SELL_RULES[name] = func
        return func
    return decorator


def _register_mandatory_buy(name):
    def decorator(func):
        MANDATORY_BUY_RULES[name] = func
        return func
    return decorator


def _register_buy_restriction(name):
    def decorator(func):
        BUY_RESTRICTION_RULES[name] = func
        return func
    return decorator


@_register_buy('MA金叉')
def buy_ma_cross(df, idx):
    if idx < 20:
        return False
    ma5 = talib.MA(df['close'].iloc[:idx+1].values.astype(float), timeperiod=5)
    ma20 = talib.MA(df['close'].iloc[:idx+1].values.astype(float), timeperiod=20)
    if len(ma5) < 2 or len(ma20) < 2:
        return False
    return (ma5[-1] > ma20[-1]) and (ma5[-2] <= ma20[-2])


@_register_sell('MA死叉')
def sell_ma_cross(df, idx):
    if idx < 20:
        return False
    ma5 = talib.MA(df['close'].iloc[:idx+1].values.astype(float), timeperiod=5)
    ma20 = talib.MA(df['close'].iloc[:idx+1].values.astype(float), timeperiod=20)
    if len(ma5) < 2 or len(ma20) < 2:
        return False
    return (ma5[-1] < ma20[-1]) and (ma5[-2] >= ma20[-2])


@_register_buy('MACD金叉')
def buy_macd_cross(df, idx):
    if idx < 26 + 9:
        return False
    macd, sig, _ = talib.MACD(df['close'].iloc[:idx+1].values.astype(float))
    if len(macd) < 2 or len(sig) < 2 or pd.isna(macd[-1]) or pd.isna(sig[-1]):
        return False
    return (macd[-1] > sig[-1]) and (macd[-2] <= sig[-2])


@_register_sell('MACD死叉')
def sell_macd_cross(df, idx):
    if idx < 26 + 9:
        return False
    macd, sig, _ = talib.MACD(df['close'].iloc[:idx+1].values.astype(float))
    if len(macd) < 2 or len(sig) < 2 or pd.isna(macd[-1]) or pd.isna(sig[-1]):
        return False
    return (macd[-1] < sig[-1]) and (macd[-2] >= sig[-2])


@_register_buy('RSI超卖上穿')
def buy_rsi_oversold(df, idx):
    if idx < 14:
        return False
    rsi = talib.RSI(df['close'].iloc[:idx+1].values.astype(float), timeperiod=14)
    if len(rsi) < 2 or pd.isna(rsi[-1]) or pd.isna(rsi[-2]):
        return False
    return (rsi[-1] > 30) and (rsi[-2] <= 30)


@_register_sell('RSI超买下穿')
def sell_rsi_overbought(df, idx):
    if idx < 14:
        return False
    rsi = talib.RSI(df['close'].iloc[:idx+1].values.astype(float), timeperiod=14)
    if len(rsi) < 2 or pd.isna(rsi[-1]) or pd.isna(rsi[-2]):
        return False
    return (rsi[-1] < 70) and (rsi[-2] >= 70)


@_register_buy('KDJ金叉')
def buy_kdj_cross(df, idx):
    if idx < 15:
        return False
    k, d = talib.STOCH(
        df['high'].iloc[:idx+1].values.astype(float),
        df['low'].iloc[:idx+1].values.astype(float),
        df['close'].iloc[:idx+1].values.astype(float),
        fastk_period=9, slowk_period=3, slowd_period=3
    )
    if len(k) < 2 or len(d) < 2 or pd.isna(k[-1]) or pd.isna(d[-1]):
        return False
    return (k[-1] > d[-1]) and (k[-2] <= d[-2]) and (k[-1] < 20)


@_register_sell('KDJ死叉')
def sell_kdj_cross(df, idx):
    if idx < 15:
        return False
    k, d = talib.STOCH(
        df['high'].iloc[:idx+1].values.astype(float),
        df['low'].iloc[:idx+1].values.astype(float),
        df['close'].iloc[:idx+1].values.astype(float),
        fastk_period=9, slowk_period=3, slowd_period=3
    )
    if len(k) < 2 or len(d) < 2 or pd.isna(k[-1]) or pd.isna(d[-1]):
        return False
    return (k[-1] < d[-1]) and (k[-2] >= d[-2]) and (k[-1] > 80)


@_register_buy('MACD底背离')
def buy_macd_divergence(df, idx):
    if idx < 60:
        return False
    close = df['close'].iloc[:idx+1].values.astype(float)
    _, _, hist = talib.MACD(close)
    n = len(close)
    lows = []
    for i in range(5, n - 5):
        if close[i] == close[i-5:i+6].min():
            lows.append(i)
    if len(lows) >= 2:
        recent_lows = [i for i in lows if i >= n - 40]
        older_lows = [i for i in lows if i < n - 40 and i >= n - 80]
        if recent_lows and older_lows:
            rl = recent_lows[-1]
            ol = older_lows[-1]
            if (close[rl] < close[ol] and
                not pd.isna(hist[rl]) and not pd.isna(hist[ol]) and
                hist[rl] > hist[ol]):
                return abs(idx - rl) <= 10
    return False


@_register_sell('MACD顶背离')
def sell_macd_divergence(df, idx):
    if idx < 60:
        return False
    close = df['close'].iloc[:idx+1].values.astype(float)
    _, _, hist = talib.MACD(close)
    n = len(close)
    highs = []
    for i in range(5, n - 5):
        if close[i] == close[i-5:i+6].max():
            highs.append(i)
    if len(highs) >= 2:
        recent_highs = [i for i in highs if i >= n - 40]
        older_highs = [i for i in highs if i < n - 40 and i >= n - 80]
        if recent_highs and older_highs:
            rh = recent_highs[-1]
            oh = older_highs[-1]
            if (close[rh] > close[oh] and
                not pd.isna(hist[rh]) and not pd.isna(hist[oh]) and
                hist[rh] < hist[oh]):
                return abs(idx - rh) <= 10
    return False


# TA-Lib 看涨/看跌 K 线子形态（与 buy_bullish_pattern / sell_bearish_pattern 顺序一致，先命中先返回）
_BULLISH_CANDLE_DEFS = (
    ('CDLENGULFING', '看涨吞没'),
    ('CDLMORNINGSTAR', '晨星'),
    ('CDLHAMMER', '锤头'),
    ('CDLPIERCING', '刺透'),
    ('CDLMORNINGDOJISTAR', '晨十字'),
    ('CDLINVERTEDHAMMER', '倒锤头'),
)
from v4_aggressive.bearish_candle_modes import (
    MA5_BREAK_SELL_REASON as V4_BEARISH_MA5_BREAK_SELL_REASON,
    MA5_DEFER_KINDS as BEARISH_CANDLE_MA5_DEFER_KINDS,
    detect_bearish_kind,
    close_broke_below_ma5,
    close_still_on_ma5,
    bearish_kind_defers_sell_until_ma5_break,
    sell_bearish_pattern_at,
)

# 默认 slim：不含黄昏星/黄昏十字/射击之星
_BEARISH_CANDLE_DEFS = (
    ('CDLENGULFING', '看跌吞没'),
    ('CDLHANGINGMAN', '上吊线'),
    ('CDLDARKCLOUDCOVER', '乌云盖顶'),
)


def detect_bullish_candle_pattern_kind(df, idx):
    """返回当日 TA-Lib 识别的看涨子形态中文名；未识别则 None。"""
    if idx < 3:
        return None
    o = df['open'].iloc[:idx + 1].values.astype(float)
    h = df['high'].iloc[:idx + 1].values.astype(float)
    l = df['low'].iloc[:idx + 1].values.astype(float)
    c = df['close'].iloc[:idx + 1].values.astype(float)
    for pname, label in _BULLISH_CANDLE_DEFS:
        func = getattr(talib, pname, None)
        if func is None:
            continue
        try:
            result = func(o, h, l, c)
            if len(result) > 0 and result[-1] == 100:
                return label
        except Exception:
            pass
    return None


def detect_bullish_candle_pattern_labels_at_bar(df, idx):
    """同一根 K 上 TA-Lib 判定为看涨(100)的全部子形态中文名集合（与 _BULLISH_CANDLE_DEFS 一致）。"""
    if idx < 3:
        return frozenset()
    o = df['open'].iloc[:idx + 1].values.astype(float)
    h = df['high'].iloc[:idx + 1].values.astype(float)
    l = df['low'].iloc[:idx + 1].values.astype(float)
    c = df['close'].iloc[:idx + 1].values.astype(float)
    out = []
    for pname, label in _BULLISH_CANDLE_DEFS:
        func = getattr(talib, pname, None)
        if func is None:
            continue
        try:
            result = func(o, h, l, c)
            if len(result) > 0 and result[-1] == 100:
                out.append(label)
        except Exception:
            pass
    return frozenset(out)


def detect_bearish_candle_pattern_kind(df, idx, mode='slim'):
    """返回当日 TA-Lib 识别的看跌子形态中文名；默认 slim（无黄昏星/十字/射击之星）。"""
    return detect_bearish_kind(df, idx, mode)


@_register_buy('看涨K线形态')
def buy_bullish_pattern(df, idx):
    return detect_bullish_candle_pattern_kind(df, idx) is not None


@_register_sell('看跌K线形态')
def sell_bearish_pattern(df, idx):
    return sell_bearish_pattern_at(df, idx, mode='slim')


def _near_high_pressure_zone_at_idx(df, idx, pivot_min_bars=25, pivot_max_bars=220):
    """
    历史压力位：局部高点 P 形成压力带（P≥30 为 ±2 元，否则 ±2%），
    自该日起至昨日区间内最高价未突破带上沿；当日 high 切入压力带。
    """
    if idx < 80:
        return False
    high = df['high'].values.astype(float) if 'high' in df.columns else df['close'].values.astype(float)
    pivot_i = None
    pivot_p = -1.0
    for i in range(idx - pivot_min_bars, max(10, idx - pivot_max_bars), -1):
        if i < 5 or i + 5 > len(high):
            continue
        hh = float(high[i])
        if hh < float(np.max(high[i - 5 : i + 6])) - 1e-9:
            continue
        zhi = (hh + 2.0) if hh >= 30.0 else hh * 1.02
        if i + 1 < idx:
            mx_after = float(np.max(high[i + 1 : idx]))
            if mx_after >= zhi * (1.0 - 1e-9):
                continue
        pivot_i = i
        pivot_p = hh
        break
    if pivot_i is None or pivot_p <= 0:
        return False
    if pivot_p >= 30.0:
        zone_lo = pivot_p - 2.0
    else:
        zone_lo = pivot_p * 0.98
    return float(high[idx]) >= zone_lo * 0.995


def _is_yizi_limit_up_bar(df, idx, flat_range_max_rel=0.003):
    """
    一字涨停：涨停价封死且振幅极窄（近似 open=high=low=close）。
    主板约 +10%、创业板/科创板约 +20%。
    """
    if idx < 1:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    c, h, lo, o = float(close[idx]), float(high[idx]), float(low[idx]), float(open_[idx])
    prev_c = float(close[idx - 1])
    if not all(np.isfinite(x) for x in (c, h, lo, o, prev_c)) or prev_c <= 0 or c <= 0:
        return False
    gain = c / prev_c - 1.0
    if gain < 0.099 and gain < 0.199:
        return False
    if (h - lo) / c > flat_range_max_rel:
        return False
    if abs(c - o) / c > flat_range_max_rel:
        return False
    if c < h * (1.0 - 1e-4) or o < h * (1.0 - 1e-4):
        return False
    if lo < min(o, c) * (1.0 - flat_range_max_rel):
        return False
    return True


def _heavy_volume_yin_bar(df, idx, prev_vol_mult=1.5):
    """当日收阴且成交量 > 前一日成交量 × prev_vol_mult。"""
    if idx < 1:
        return False
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    if vol is None:
        return False
    if close[idx] >= open_[idx]:
        return False
    v_cur = float(vol[idx])
    v_prev = float(vol[idx - 1])
    if not (np.isfinite(v_cur) and np.isfinite(v_prev)) or v_prev <= 0:
        return False
    return v_cur > prev_vol_mult * v_prev


@_register_mandatory_sell('高位放量阴线看跌')
def mandatory_sell_high_heavy_bearish_pattern(df, idx):
    """
    接近历史前高压力带 + 放巨量阴线 → 必卖，不须等跌破 MA5/沿均线主升破位。
    """
    if not _near_high_pressure_zone_at_idx(df, idx):
        return False
    return _heavy_volume_yin_bar(df, idx)


@_register_mandatory_sell('一字涨停次日放量阴')
def mandatory_sell_after_yizi_limit_up_heavy_yin(df, idx):
    """前一日一字涨停，次日放巨量阴线 → 必卖。"""
    if idx < 1:
        return False
    if not _is_yizi_limit_up_bar(df, idx - 1):
        return False
    return _heavy_volume_yin_bar(df, idx)


@_register_mandatory_buy('W底突破')
def buy_double_bottom(df, idx):
    if idx < 30:
        return False
    close = df['close'].iloc[:idx+1].values.astype(float)
    vol = df['volume'].iloc[:idx+1].values.astype(float) if 'volume' in df.columns else None
    n = len(close)
    lows = []
    for i in range(8, n):
        if close[i] == close[max(0, i-8):min(n, i+4)].min():
            lows.append(i)
    if len(lows) < 2:
        return False
    filtered_lows = [lows[0]]
    for i in range(1, len(lows)):
        if lows[i] - filtered_lows[-1] >= 5:
            filtered_lows.append(lows[i])
    for i in range(len(filtered_lows) - 1):
        idx1, idx2 = filtered_lows[i], filtered_lows[i + 1]
        if idx2 - idx1 < 8 or idx2 - idx1 > 80:
            continue
        p1, p2 = close[idx1], close[idx2]
        if abs(p1 - p2) / max(p1, p2) > 0.05:
            continue
        between_high = close[idx1:idx2+1].max()
        retrace = (between_high - max(p1, p2)) / max(p1, p2)
        if retrace < 0.03:
            continue
        neckline = between_high
        breakout_end = min(idx2 + 30, n)
        breakout_start = idx2 + 1
        for j in range(breakout_start, breakout_end):
            if close[j] > neckline and j == idx:
                if j > 0 and close[j-1] <= neckline:
                    if vol is not None and idx >= 1:
                        vol_ma5 = np.mean(vol[max(0, idx-5):idx])
                        if vol_ma5 > 0 and vol[idx] > vol_ma5 * 1.2:
                            return True
                        elif close[idx] > close[idx-1]:
                            return True
                    else:
                        return True
                elif j == breakout_start:
                    if vol is not None and idx >= 1:
                        vol_ma5 = np.mean(vol[max(0, idx-5):idx])
                        if vol_ma5 > 0 and vol[idx] > vol_ma5 * 1.2:
                            return True
                        elif close[idx] > close[idx-1]:
                            return True
                    else:
                        return True
    return False


@_register_mandatory_sell('M顶跌破')
def sell_double_top(df, idx):
    """M 顶跌破：ATR 摆动高点双顶 + 跌破顶部 98%（与 V 结构同 registry）。"""
    if idx < 30:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    from v4_aggressive.v4_structure_curves import get_structure_registry
    reg = get_structure_registry(open_, high, low, close, len(close))
    return reg.is_m_top_break_bar(idx)


@_register_mandatory_buy('头肩底突破')
def buy_head_shoulders_bottom(df, idx):
    if idx < 50:
        return False
    close = df['close'].iloc[:idx+1].values.astype(float)
    n = len(close)
    lows = []
    for i in range(5, n - 5):
        if close[i] == close[i-5:i+6].min():
            lows.append(i)
    for i in range(len(lows) - 4):
        ls = lows[i]
        head = lows[i + 2]
        rs = lows[i + 4]
        if rs - ls > 100 or head - ls < 10 or rs - head < 10:
            continue
        if close[head] < close[ls] and close[head] < close[rs]:
            diff = abs(close[ls] - close[rs]) / max(close[ls], close[rs])
            if diff < 0.05:
                neckline = close[ls:rs+1].max()
                for j in range(rs + 1, min(n, rs + 20)):
                    if close[j] > neckline and j == idx:
                        return True
    return False


@_register_sell('头肩顶跌破')
def sell_head_shoulders_top(df, idx):
    """头肩顶：high 曲线 ATR 摆动肩/头/肩，颈线=两肩间 low 最小，仅首次收盘跌破颈线日（同 M 顶 registry）。"""
    if idx < 50:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    from v4_aggressive.v4_structure_curves import get_structure_registry
    reg = get_structure_registry(open_, high, low, close, len(close))
    return reg.is_hs_top_break_bar(idx)


@_register_buy('价升量增')
def buy_price_up_volume_up(df, idx):
    if idx < 21:
        return False
    close = df['close'].values
    vol = df['volume'].values.astype(float)
    vol_ma = pd.Series(vol).rolling(window=5).mean().values
    if idx >= len(vol_ma) or pd.isna(vol_ma[idx]):
        return False
    return (close[idx] > close[idx-1]) and (vol[idx] > vol_ma[idx])


@_register_sell('价跌量增')
def sell_price_down_volume_up(df, idx):
    if idx < 21:
        return False
    close = df['close'].values.astype(float)
    vol = df['volume'].values.astype(float)
    vol_ma = pd.Series(vol).rolling(window=5).mean().values
    if idx >= len(vol_ma) or pd.isna(vol_ma[idx]):
        return False
    drop_pct = (close[idx-1] - close[idx]) / close[idx-1]
    if drop_pct < 0.01:
        return False
    if vol[idx] <= vol_ma[idx] * 1.5:
        return False
    return True


@_register_buy('放量突破高点')
def buy_volume_breakout_high(df, idx):
    """前 20 根（不含当日）high 最高价为前高；当日 high 突破且放量."""
    if idx < 21:
        return False
    high = df['high'].values.astype(float) if 'high' in df.columns else df['close'].values.astype(float)
    vol = df['volume'].values.astype(float)
    period_high = high[idx - 20 : idx]
    period_vol = vol[idx - 20 : idx]
    if len(period_high) < 20:
        return False
    highest = float(np.max(period_high))
    if highest <= 0:
        return False
    avg_vol = float(np.mean(period_vol))
    if avg_vol <= 0:
        return False
    return (float(high[idx]) > highest) and (float(vol[idx]) > avg_vol * 1.5)


@_register_sell('放量跌破低点')
def sell_volume_breakout_low(df, idx):
    if idx < 21:
        return False
    close = df['close'].values
    vol = df['volume'].values.astype(float)
    if idx < 20:
        return False
    period_close = close[idx-20:idx]
    period_vol = vol[idx-20:idx]
    lowest = period_close.min()
    avg_vol = period_vol.mean()
    if avg_vol == 0:
        return False
    return (close[idx] < lowest) and (vol[idx] > avg_vol * 1.5)


@_register_buy('地量地价')
def buy_low_volume_low_price(df, idx):
    if idx < 61:
        return False
    close = df['close'].values
    vol = df['volume'].values.astype(float)
    period_close = close[idx-60:idx]
    period_vol = vol[idx-60:idx]
    lowest_price = period_close.min()
    lowest_vol = period_vol.min()
    return (close[idx] <= lowest_price * 1.02) and (vol[idx] <= lowest_vol * 1.1)


@_register_sell('天量天价')
def sell_high_volume_high_price(df, idx):
    if idx < 61:
        return False
    close = df['close'].values
    vol = df['volume'].values.astype(float)
    period_close = close[idx-60:idx]
    period_vol = vol[idx-60:idx]
    highest_price = period_close.max()
    highest_vol = period_vol.max()
    return (close[idx] >= highest_price * 0.98) and (vol[idx] >= highest_vol * 0.9)


@_register_buy('趋势线支撑')
def buy_trendline_support(df, idx):
    if idx < 31:
        return False
    close = df['close'].values
    x = np.arange(30).reshape(-1, 1)
    y = close[idx-30:idx]
    try:
        slope, intercept, r_val, _, _ = stats.linregress(x.flatten(), y)
        trend_at_current = slope * 30 + intercept
        if trend_at_current <= 0:
            return False
        deviation = (close[idx] - trend_at_current) / trend_at_current
        return slope > 0 and abs(deviation) < 0.02 and close[idx] > trend_at_current
    except Exception:
        return False


@_register_sell('趋势线阻力')
def sell_trendline_resistance(df, idx):
    if idx < 31:
        return False
    close = df['close'].values
    x = np.arange(30).reshape(-1, 1)
    y = close[idx-30:idx]
    try:
        slope, intercept, r_val, _, _ = stats.linregress(x.flatten(), y)
        trend_at_current = slope * 30 + intercept
        if trend_at_current <= 0:
            return False
        deviation = (close[idx] - trend_at_current) / trend_at_current
        return slope < 0 and abs(deviation) < 0.02 and close[idx] < trend_at_current
    except Exception:
        return False


@_register_sell('均线趋势打破')
def sell_ma_trend_break(df, idx):
    if idx < 30:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    for period in [5, 10, 20]:
        ma = talib.MA(close[:idx+1], timeperiod=period)
        if len(ma) < period + 3 or pd.isna(ma[idx]) or pd.isna(ma[idx-1]):
            continue
        if ma[idx] <= ma[idx-1]:
            continue
        if close[idx] < ma[idx] * 0.95:
            continue
        recent_high = high[max(0, idx-period):idx+1].max()
        if close[idx] < recent_high * 0.95:
            continue
        support_count = 0
        check_start = max(period, idx - period * 2)
        for j in range(check_start, idx):
            if pd.isna(ma[j]):
                continue
            if close[j] >= ma[j] * 0.98:
                support_count += 1
        total_days = idx - check_start
        if total_days <= 0:
            continue
        support_ratio = support_count / total_days
        if support_ratio >= 0.7 and close[idx] < ma[idx] * 0.99:
            return True
    return False


@_register_mandatory_sell('均线趋势打破必卖')
def mandatory_sell_ma_trend_break(df, idx):
    if idx < 30:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    for period in [5, 10, 20]:
        ma = talib.MA(close[:idx+1], timeperiod=period)
        if len(ma) < period + 3 or pd.isna(ma[idx]) or pd.isna(ma[idx-1]):
            continue
        if ma[idx] <= ma[idx-1]:
            continue
        if close[idx] < ma[idx] * 0.95:
            continue
        recent_high = high[max(0, idx-period):idx+1].max()
        if close[idx] < recent_high * 0.95:
            continue
        support_count = 0
        check_start = max(period, idx - period * 2)
        for j in range(check_start, idx):
            if pd.isna(ma[j]):
                continue
            if close[j] >= ma[j] * 0.98:
                support_count += 1
        total_days = idx - check_start
        if total_days <= 0:
            continue
        support_ratio = support_count / total_days
        if support_ratio >= 0.7 and close[idx] < ma[idx] * 0.99:
            return True
    return False


@_register_sell('沿均线主升破位')
def sell_rail_break(df, idx):
    """
    主升段「贴」哪条均线，则由跌破该均线贡献卖出信号（加权计数卖路径，非必卖，减少过早离场对收益的侵蚀）：
    - 贴 MA5 主升 → 首次有效跌破 MA5；
    - 贴 MA10 / MA20 更明显 → 分别在跌破 MA10 / MA20 时触发。
    分类：回看窗口内各日，在「收盘不低于该均线」的候选中取 |收盘/均线−1| 最小为主轨，统计 5/10/20 票数；平票时优先更短周期。
    """
    if idx < 28:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    ma5 = talib.MA(close[: idx + 1], timeperiod=5)
    ma10 = talib.MA(close[: idx + 1], timeperiod=10)
    ma20 = talib.MA(close[: idx + 1], timeperiod=20)

    win_beg = max(20, idx - 17)
    counts = {5: 0, 10: 0, 20: 0}
    for j in range(win_beg, idx):
        if any(pd.isna(ma5[j]) or pd.isna(ma10[j]) or pd.isna(ma20[j])):
            continue
        if ma5[j] <= 0 or ma10[j] <= 0 or ma20[j] <= 0:
            continue
        candidates = []
        if close[j] >= ma5[j] * 0.996:
            candidates.append((5, abs(close[j] / ma5[j] - 1.0)))
        if close[j] >= ma10[j] * 0.994:
            candidates.append((10, abs(close[j] / ma10[j] - 1.0)))
        if close[j] >= ma20[j] * 0.992:
            candidates.append((20, abs(close[j] / ma20[j] - 1.0)))
        if not candidates:
            continue
        rail_day = min(candidates, key=lambda x: x[1])[0]
        counts[rail_day] += 1

    best = max(counts.keys(), key=lambda p: (counts[p], -p))
    if counts[best] < 6 or sum(counts.values()) < 8:
        return False

    ma_ch = {5: ma5, 10: ma10, 20: ma20}[best]
    if pd.isna(ma_ch[idx]) or pd.isna(ma_ch[idx - 1]) or pd.isna(ma_ch[idx - 8]):
        return False
    if ma_ch[idx - 1] < ma_ch[idx - 8] * 1.006:
        return False

    if close[idx] >= ma_ch[idx] * 0.997:
        return False
    if close[idx - 1] < ma_ch[idx - 1] * 0.995:
        return False

    run_low = float(close[max(0, idx - 20) : idx].min())
    run_high = float(high[max(0, idx - 20) : idx + 1].max())
    if run_low <= 0 or run_high / run_low < 1.08:
        return False
    return True


@_register_sell('横盘整理向下突破')
def sell_sideways_box_breakdown(df, idx):
    """近端窄幅横盘后放量向下跌破；由引擎在 V3 下计入卖出，V2 忽略。"""
    if idx < 35:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    if vol is None:
        return False
    box_n = 20
    seg_high = float(high[idx - box_n : idx].max())
    seg_low = float(low[idx - box_n : idx].min())
    mid = (seg_high + seg_low) / 2.0
    if mid <= 0:
        return False
    if (seg_high - seg_low) / mid > 0.07:
        return False
    if close[idx] >= seg_low * 0.999:
        return False
    if close[idx] >= close[idx - 1]:
        return False
    vol_ma = np.mean(vol[max(0, idx - 5) : idx])
    if vol_ma <= 0 or vol[idx] < vol_ma * 1.2:
        return False
    return True


def is_down_gap_pair(low, high, n, eps=1e-9):
    """上方缺口（向下跳空）形成：左棒(n-1)最低价 > 右棒(n)最高价。"""
    n = int(n)
    if n < 1:
        return False
    return float(low[n - 1]) > float(high[n]) * (1.0 + eps)


def is_upper_gap_unfilled(low, high, n, idx, eps=1e-9):
    """
    未回补：自 n 日（含）至当前 idx 日（含），最高价均未达到左棒最低价（缺口上沿）。
    """
    n, idx = int(n), int(idx)
    if n < 1 or idx < n:
        return False
    if not is_down_gap_pair(low, high, n, eps):
        return False
    gap_top = float(low[n - 1])
    seg = high[n : idx + 1]
    if len(seg) == 0:
        return False
    return float(np.max(seg)) < gap_top * (1.0 - eps)


def find_recent_unfilled_upper_gap(close, high, low, idx, lookback=200, min_bars_after=0):
    """自 idx 向左找最近一处未回补的上方缺口；返回 gap 元信息或 None。"""
    idx = int(idx)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    start = max(1, idx - int(lookback) + 1)
    for n in range(idx, start - 1, -1):
        if idx - n < min_bars_after:
            continue
        if not is_upper_gap_unfilled(low, high, n, idx):
            continue
        return {
            'n': n,
            'n_minus_1': n - 1,
            'gap_top': float(low[n - 1]),
            'gap_bottom': float(high[n]),
        }
    return None


@_register_sell('历史缺口阻力')
def sell_gap_resistance(df, idx):
    """
    上方缺口：左棒最低 > 右棒最高；未回补至左棒最低（缺口上沿）。
    当前切入缺口上沿附近且走弱时卖出。
    """
    if idx < 45 or 'open' not in df.columns:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    open_ = df['open'].values.astype(float)
    gap = find_recent_unfilled_upper_gap(close, high, low, idx, lookback=200, min_bars_after=5)
    if gap is None:
        return False
    prev_c = gap['gap_top']
    if high[idx] < prev_c * 0.985:
        return False
    if close[idx] >= close[idx - 1]:
        return False
    if close[idx] >= open_[idx]:
        return False
    rng = max(high[idx] - low[idx], 1e-9)
    upper_shadow = (high[idx] - max(close[idx], open_[idx])) / rng
    if upper_shadow >= 0.35:
        return True
    if close[idx] < prev_c * 0.995:
        return True
    return False


@_register_sell('接近前高巨量阴增')
def sell_near_high_heavy_volume(df, idx):
    """
    历史压力位：历史上形成局部高点 P（压力带：P≥30 为 ±2 元，否则 ±2%），
    自该日起至昨日区间内最高价从未突破过压力带上沿；
    当前进入压力带、近端阴线偏多、当日收阴且成交量>前一日1.5倍。仅 V3 计入。
    """
    if not _near_high_pressure_zone_at_idx(df, idx):
        return False
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    n_yin = sum(
        1 for j in range(max(0, idx - 4), idx + 1)
        if close[j] < open_[j]
    )
    if n_yin < 3:
        return False
    return _heavy_volume_yin_bar(df, idx)


@_register_buy_restriction('未突破5日线不买')
def buy_restriction_above_ma5(df, idx):
    if idx < 5:
        return False
    close = df['close'].values.astype(float)
    ma5 = talib.MA(close[:idx+1], timeperiod=5)
    if pd.isna(ma5[idx]):
        return False
    return close[idx] <= ma5[idx]


@_register_buy_restriction('阴线不买')
def buy_restriction_no_bearish(df, idx):
    if idx < 1:
        return False
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float)
    return close[idx] <= open_[idx]


# V反底部收紧：弱反弹（未收复左侧跌前压力参考）、站上 MA20 却无回踩、价升量缩 不触发必买
V_REV_BOTTOM_LEFT_RECLAIM_MIN = 0.985
V_REV_BOTTOM_LEFT_LOOKBACK = 5
V_REV_BOTTOM_MA20_TOUCH_MAX = 1.028
V_REV_BOTTOM_MA20_CLOSE_MIN = 0.991
V_REV_BOTTOM_VOL_SHRINK_RATIO = 0.93

# 黄金坑：收盘价与 MA20 间距至少「3 个点」（MA20≥阈值时用绝对价差，元）；低价股改用相对 MA20 的 3%。
# MA20 在坑口阶段常缓跌：允许较前一日缓跌、与 5 日前偏差在 DRIFT 内视为基本持平。
GOLDEN_PIT_GAP_ABS_POINTS = 3.0
GOLDEN_PIT_GAP_USE_ABS_M20_MIN = 15.0
GOLDEN_PIT_GAP_LOWP_FR = 0.03
GOLDEN_PIT_MA20_MIN_VS_PREV = 0.988
GOLDEN_PIT_MA20_MAX_5D_DRIFT = 0.060


def _v_reversal_bottom_ma20_pullback_ok(close, low, ma20, recent_low_idx, idx):
    """自近期低点以来已收盘站上 MA20 时：须曾出现一次对 MA20 的靠近且收盘仍在其上（突破后回踩确认）。"""
    rli = int(recent_low_idx)
    if rli >= idx or idx < 1:
        return True
    mcur = ma20[idx]
    if pd.isna(mcur) or mcur <= 0 or close[idx] < mcur * 0.998:
        return True
    first_above = None
    for j in range(rli, idx + 1):
        mj = ma20[j]
        if pd.isna(mj) or mj <= 0:
            continue
        if close[j] >= mj:
            first_above = j
            break
    if first_above is None:
        return False
    for k in range(first_above, idx + 1):
        mk = ma20[k]
        if pd.isna(mk) or mk <= 0:
            continue
        if low[k] <= mk * V_REV_BOTTOM_MA20_TOUCH_MAX and close[k] >= mk * V_REV_BOTTOM_MA20_CLOSE_MIN:
            return True
    return False


@_register_mandatory_buy('V反底部')
def buy_v_reversal_bottom(df, idx):
    if idx < 20:
        return False
    close = df['close'].values.astype(float)
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    lookback = min(20, idx)
    recent_low_idx = idx - lookback + np.argmin(low[idx-lookback:idx+1])
    if recent_low_idx >= idx:
        return False
    if recent_low_idx < idx - 10:
        return False
    low_price = low[recent_low_idx]
    left_start = max(0, recent_low_idx - V_REV_BOTTOM_LEFT_LOOKBACK)
    if 'high' in df.columns:
        h = df['high'].values.astype(float)
        left_ref = float(np.max(h[left_start:recent_low_idx]))
    else:
        left_ref = float(np.max(close[left_start:recent_low_idx]))
    if left_ref <= 0:
        return False
    drop_pct = (left_ref - low_price) / left_ref
    if drop_pct < 0.05:
        return False
    bounce_pct = (close[idx] - low_price) / low_price
    if bounce_pct < 0.02:
        return False
    if close[idx] <= close[idx-1]:
        return False
    # 须收复左侧跌前压力区（用跌前窗口最高价）一定比例，避免弱弹必买
    if close[idx] < left_ref * V_REV_BOTTOM_LEFT_RECLAIM_MIN:
        return False
    ma20 = talib.MA(close[: idx + 1], timeperiod=20)
    if not _v_reversal_bottom_ma20_pullback_ok(close, low, ma20, recent_low_idx, idx):
        return False
    if vol is not None:
        vol_ma5 = np.mean(vol[max(0, idx - 5) : idx])
        if vol_ma5 > 0 and vol[idx] < vol_ma5 * 0.8:
            return False
        if idx >= 2 and close[idx] > close[idx - 1] and close[idx - 1] > close[idx - 2]:
            if (
                vol[idx] < vol[idx - 1] * V_REV_BOTTOM_VOL_SHRINK_RATIO
                and vol[idx - 1] < vol[idx - 2] * V_REV_BOTTOM_VOL_SHRINK_RATIO
            ):
                return False
    return True


@_register_mandatory_sell('V反顶部')
def sell_v_reversal_top(df, idx):
    """V 反顶部：摆动高点 + 左侧均价抬升 + 右侧回落跌破（与 V 底同 registry）。"""
    if idx < 20:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    from v4_aggressive.v4_structure_curves import get_structure_registry
    reg = get_structure_registry(open_, high, low, close, len(close))
    return reg.is_v_top_break_bar(idx)


@_register_mandatory_buy('旗形突破')
def buy_flag_breakout(df, idx):
    if idx < 30:
        return False
    close = df['close'].values.astype(float)
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    pole_start = max(0, idx - 30)
    pole_high_idx = pole_start + np.argmax(close[pole_start:idx-5])
    pole_high = close[pole_high_idx]
    if (pole_high - close[pole_start]) / close[pole_start] < 0.05:
        return False
    flag_start = pole_high_idx + 1
    if flag_start >= idx - 3:
        return False
    flag_close = close[flag_start:idx]
    flag_x = np.arange(len(flag_close))
    if len(flag_x) < 3:
        return False
    try:
        slope, _, _, _, _ = stats.linregress(flag_x, flag_close)
    except Exception:
        return False
    if slope >= 0:
        return False
    flag_range = np.max(flag_close) - np.min(flag_close)
    flag_mid = np.mean(flag_close)
    if flag_mid > 0 and flag_range / flag_mid > 0.08:
        return False
    if close[idx] > pole_high and close[idx] > close[idx-1]:
        if vol is not None:
            vol_ma5 = np.mean(vol[max(0, idx-5):idx])
            if vol_ma5 > 0 and vol[idx] > vol_ma5 * 1.3:
                return True
        else:
            return True
    return False


# 双针探底：第一根针在信号日前；第二根针=信号日；两针低点均为 V/W 底区最低价且相差<0.5%
_DOUBLE_NEEDLE_LOOKBACK = 8
_DOUBLE_NEEDLE_MAX_GAP_BARS = 5
_DOUBLE_NEEDLE_PAIR_LOW_MAX_DIFF = 0.005
_DOUBLE_NEEDLE_SIGNAL_MIN_SHADOW_BODY_RATIO = 0.8
_DOUBLE_NEEDLE_MIN_SHADOW_RANGE_RATIO = 0.45
_DOUBLE_NEEDLE_MIN_SHADOW_BODY_RATIO = 0.85


def _bar_is_lower_needle(o, h, l, c):
    """长下影探底针：下影占振幅足够大，且不低于实体一定比例。"""
    if not all(np.isfinite(x) for x in (o, h, l, c)):
        return False
    rng = h - l
    if rng <= 0:
        return False
    body_bot = min(o, c)
    lower = body_bot - l
    if lower <= 0:
        return False
    if lower / rng < _DOUBLE_NEEDLE_MIN_SHADOW_RANGE_RATIO:
        return False
    body = abs(c - o)
    min_body = 1e-10 * max(abs(c), 1e-9)
    if body >= min_body and lower < body * _DOUBLE_NEEDLE_MIN_SHADOW_BODY_RATIO:
        return False
    return True


def _lower_shadow_over_body_ratio(o, h, l, c, min_ratio=0.8):
    """下影线长度 / 实体 >= min_ratio（阴/阳线均适用）。"""
    o, h, l, c = float(o), float(h), float(l), float(c)
    body = abs(c - o)
    lower = min(o, c) - l
    if lower <= 0:
        return False
    min_body = 1e-10 * max(abs(c), 1e-9)
    if body < min_body:
        return lower >= min_body * min_ratio
    return (lower / body) >= min_ratio


def _yang_lower_shadow_over_body(o, h, l, c, min_ratio=1.0):
    """阳线：下影线长度 / 实体 >= min_ratio。"""
    o, h, l, c = float(o), float(h), float(l), float(c)
    if c <= o:
        return False
    return _lower_shadow_over_body_ratio(o, h, l, c, min_ratio=min_ratio)


def _v_bottom_zone_min_low(lows, left_idx, n):
    """V/W 左底允许窗口内的最低价（双针低点须贴此底）。"""
    from strategy_vw_bottle_backtest import (
        DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_AFTER,
        DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_BEFORE,
    )
    lo_i = max(0, int(left_idx) - int(DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_BEFORE))
    hi_i = min(n - 1, int(left_idx) + int(DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_AFTER))
    if hi_i < lo_i:
        return None
    seg = lows[lo_i : hi_i + 1]
    if len(seg) == 0:
        return None
    return float(np.min(seg))


def _low_at_v_bottom(low_val, v_bottom_low, tol=_DOUBLE_NEEDLE_PAIR_LOW_MAX_DIFF):
    v_bottom_low = float(v_bottom_low)
    if v_bottom_low <= 0:
        return False
    return abs(float(low_val) - v_bottom_low) / v_bottom_low <= tol


def _double_needle_pair_ok(lows, j1, j2, v_bottom_low):
    """两针低点均在 V/W 底区最低附近，且彼此相差 <0.5%；j2 为信号日（第二根针）。"""
    if j2 - j1 > _DOUBLE_NEEDLE_MAX_GAP_BARS:
        return False
    low1, low2 = float(lows[j1]), float(lows[j2])
    if not _low_at_v_bottom(low1, v_bottom_low):
        return False
    if not _low_at_v_bottom(low2, v_bottom_low):
        return False
    mid = (low1 + low2) / 2.0
    if mid <= 0:
        return False
    return abs(low1 - low2) / mid <= _DOUBLE_NEEDLE_PAIR_LOW_MAX_DIFF


def _find_first_needle_before_signal(o, h, l, c, start, signal_idx, v_bottom_low):
    """第一根针在信号日前；多候选时取最近信号日的一根。"""
    best_j1 = None
    for j in range(start, signal_idx):
        if not _bar_is_lower_needle(o[j], h[j], l[j], c[j]):
            continue
        if not _lower_shadow_over_body_ratio(
            o[j], h[j], l[j], c[j],
            min_ratio=_DOUBLE_NEEDLE_SIGNAL_MIN_SHADOW_BODY_RATIO,
        ):
            continue
        if not _double_needle_pair_ok(l, j, signal_idx, v_bottom_low):
            continue
        if best_j1 is None or j > best_j1:
            best_j1 = j
    return best_j1


def detect_double_needle_bottom(df, idx):
    """
    双针探底：须在 V/W 左侧底部附近；跌段总跌幅>=4%；
    第一根探底针在信号日前（下影/实体>=0.8），第二根针=信号日（阳线且下影/实体>=0.8）；
    两针低点均为 V/W 底区最低价且彼此相差<0.5%。
    """
    if idx < 20:
        return False
    from strategy_vw_bottle_backtest import signal_near_vw_left_bottom
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float) if 'high' in df.columns else df['close'].values.astype(float)
    l = df['low'].values.astype(float) if 'low' in df.columns else df['close'].values.astype(float)
    c = df['close'].values.astype(float)
    n = len(c)
    ctx = signal_near_vw_left_bottom(c, h, l, n, idx)
    if not ctx:
        return False
    v_bottom_low = _v_bottom_zone_min_low(l, ctx['left_idx'], n)
    if v_bottom_low is None:
        return False
    if not _bar_is_lower_needle(o[idx], h[idx], l[idx], c[idx]):
        return False
    if c[idx] <= o[idx]:
        return False
    if not _lower_shadow_over_body_ratio(
        o[idx], h[idx], l[idx], c[idx],
        min_ratio=_DOUBLE_NEEDLE_SIGNAL_MIN_SHADOW_BODY_RATIO,
    ):
        return False
    start = max(0, idx - _DOUBLE_NEEDLE_LOOKBACK + 1)
    return _find_first_needle_before_signal(o, h, l, c, start, idx, v_bottom_low) is not None


def confirm_double_needle_t1(df, confirm_idx):
    """双针探底 T+1：阳线(收盘>开盘)且收盘在 MA5 上方。"""
    if confirm_idx < 5 or confirm_idx >= len(df):
        return False
    o = df['open'].values.astype(float)
    c = df['close'].values.astype(float)
    if c[confirm_idx] <= o[confirm_idx]:
        return False
    ma5 = talib.MA(c[: confirm_idx + 1], timeperiod=5)
    if pd.isna(ma5[confirm_idx]):
        return False
    return float(c[confirm_idx]) > float(ma5[confirm_idx])


@_register_buy('双针探底')
def buy_double_needle_bottom(df, idx):
    return detect_double_needle_bottom(df, idx)


@_register_buy('横盘整理向上突破')
def buy_sideways_box_breakout(df, idx):
    """近端窄幅横盘后放量向上突破；V3 仅记 pending，下一交易日阳线确认后出 B。"""
    if idx < 35:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    if vol is None:
        return False
    box_n = 20
    seg_high = float(high[idx - box_n : idx].max())
    seg_low = float(low[idx - box_n : idx].min())
    mid = (seg_high + seg_low) / 2.0
    if mid <= 0:
        return False
    if (seg_high - seg_low) / mid > 0.07:
        return False
    if close[idx] <= seg_high * 1.001:
        return False
    if close[idx] <= close[idx - 1]:
        return False
    vol_ma = np.mean(vol[max(0, idx - 5) : idx])
    if vol_ma <= 0 or vol[idx] < vol_ma * 1.2:
        return False
    return True


@_register_buy('黄金坑')
def buy_golden_pit(df, idx):
    """
    黄金坑：
    - 前几个交易日内有明显大跌（近 10 根振幅 ≥10%）；
    - 收盘站上 MA5，且仍在 MA20 下方；
    - 收盘价与 MA20 的间距：MA20≥GOLDEN_PIT_GAP_USE_ABS_M20_MIN 时须 > GOLDEN_PIT_GAP_ABS_POINTS（元），
      否则须 > MA20×GOLDEN_PIT_GAP_LOWP_FR（低价相对口径）；
    - MA20 未急杀：较前一日不低于 GOLDEN_PIT_MA20_MIN_VS_PREV，且与 5 日前偏差在 GOLDEN_PIT_MA20_MAX_5D_DRIFT 内（缓跌/基本持平）；
    - 近端曾有收在 MA20 下方的「坑」，当日反弹收阳、带量。
    """
    if idx < 45:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    if vol is None:
        return False
    ma5 = talib.MA(close[: idx + 1], timeperiod=5)
    ma20 = talib.MA(close[: idx + 1], timeperiod=20)
    if pd.isna(ma5[idx]) or pd.isna(ma20[idx]) or pd.isna(ma20[idx - 1]) or pd.isna(ma20[idx - 5]):
        return False
    m20 = float(ma20[idx])
    if m20 <= 0:
        return False
    m20_1 = float(ma20[idx - 1])
    m20_5 = float(ma20[idx - 5])
    if m20 < m20_1 * GOLDEN_PIT_MA20_MIN_VS_PREV:
        return False
    if abs(m20 - m20_5) / m20 > GOLDEN_PIT_MA20_MAX_5D_DRIFT:
        return False
    lb = 10
    hi_r = float(high[idx - lb : idx].max())
    lo_r = float(low[idx - lb : idx].min())
    if hi_r <= 0 or (hi_r - lo_r) / hi_r < 0.10:
        return False
    if close[idx] <= ma5[idx]:
        return False
    if close[idx] >= m20:
        return False
    gap_abs = m20 - close[idx]
    if m20 >= GOLDEN_PIT_GAP_USE_ABS_M20_MIN:
        if gap_abs <= GOLDEN_PIT_GAP_ABS_POINTS:
            return False
    else:
        if gap_abs / m20 <= GOLDEN_PIT_GAP_LOWP_FR:
            return False
    pit_ok = False
    pit_close = None
    for j in range(idx - 12, idx):
        if j < 20:
            continue
        mj = ma20[j]
        if pd.isna(mj) or mj <= 0:
            continue
        if close[j] < mj * 0.995:
            pit_ok = True
            pit_close = float(close[j])
            break
    if not pit_ok or pit_close is None:
        return False
    if close[idx] <= pit_close:
        return False
    if close[idx] <= close[idx - 1]:
        return False
    vol_ma = np.mean(vol[max(0, idx - 5) : idx])
    if vol_ma <= 0 or vol[idx] < vol_ma * 1.05:
        return False
    return True


@_register_buy('N字形突破')
def buy_n_shape_breakout(df, idx):
    """
    W/V 右侧瓶口背景下的 N 字回踩突破（不可脱离 W底右侧 / V反右侧 单独成立）。
    前高取右侧 entry 至信号日的最高价 high；实现见 strategy_vw_bottle_backtest。
    """
    if idx < 40:
        return False
    from strategy_vw_bottle_backtest import n_shape_vw_bottle_buy_signal
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    return n_shape_vw_bottle_buy_signal(close, high, low, vol, idx, open_=open_)


@_register_mandatory_sell('旗形跌破')
def sell_flag_breakdown(df, idx):
    if idx < 30:
        return False
    close = df['close'].values.astype(float)
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    pole_start = max(0, idx - 30)
    pole_low_idx = pole_start + np.argmin(close[pole_start:idx-5])
    pole_low = close[pole_low_idx]
    if (close[pole_start] - pole_low) / close[pole_start] < 0.05:
        return False
    flag_start = pole_low_idx + 1
    if flag_start >= idx - 3:
        return False
    flag_close = close[flag_start:idx]
    flag_x = np.arange(len(flag_close))
    if len(flag_x) < 3:
        return False
    try:
        slope, _, _, _, _ = stats.linregress(flag_x, flag_close)
    except Exception:
        return False
    if slope <= 0:
        return False
    flag_range = np.max(flag_close) - np.min(flag_close)
    flag_mid = np.mean(flag_close)
    if flag_mid > 0 and flag_range / flag_mid > 0.08:
        return False
    if close[idx] < pole_low and close[idx] < close[idx-1]:
        if vol is not None:
            vol_ma5 = np.mean(vol[max(0, idx-5):idx])
            if vol_ma5 > 0 and vol[idx] > vol_ma5 * 1.3:
                return True
        else:
            return True
    return False


def get_all_rules():
    return dict(BUY_RULES), dict(SELL_RULES)


# 加权买入（非必买）：下列形态可单日单条触发；其余为技术指标/量价/趋势线类，
# 须同日至少 2 条命中方可进入买入候选。形态学类必买仍走 MANDATORY_BUY_RULES。
BUY_STANDALONE_WEIGHTED_RULE_NAMES = frozenset({
    '看涨K线形态',
    '横盘整理向上突破',
    '黄金坑',
    '双针探底',
})

# MACD / 价升量增：不得单独触发，须同日有看涨K线、形态学或突破类支持
BUY_MACD_VP_REQUIRES_PATTERN_BREAKOUT = frozenset({
    'MACD金叉',
    'MACD底背离',
    '价升量增',
})

BUY_PATTERN_BREAKOUT_SUPPORT_NAMES = frozenset({
    '看涨K线形态',
    '横盘整理向上突破',
    '黄金坑',
    'N字形突破',
    '双针探底',
    '放量突破高点',
    'W底突破',
    '头肩底突破',
    'V反底部',
    '旗形突破',
})


def buy_weighted_combo_gate_ok(triggered_buy):
    """同日触发的加权买入规则是否满足「单 K 线形态 或 多规则组合」."""
    if not triggered_buy:
        return False
    # N 字形：规则内已强制 W/V 右侧瓶口回踩突破，允许单日仅命中本条即可过 gate
    if len(triggered_buy) == 1 and triggered_buy[0] == 'N字形突破':
        return True
    if any(n in BUY_STANDALONE_WEIGHTED_RULE_NAMES for n in triggered_buy):
        return True
    combo = [n for n in triggered_buy if n not in BUY_STANDALONE_WEIGHTED_RULE_NAMES]
    return len(combo) >= 2


def get_all_rules_extended():
    return dict(BUY_RULES), dict(SELL_RULES), dict(MANDATORY_SELL_RULES), dict(BUY_RESTRICTION_RULES), dict(MANDATORY_BUY_RULES)


def init_weights(buy_rules, sell_rules):
    n_buy = len(buy_rules)
    n_sell = len(sell_rules)
    buy_w = {name: 1.0 / n_buy for name in buy_rules}
    sell_w = {name: 1.0 / n_sell for name in sell_rules}
    return buy_w, sell_w


def _find_extremes_by_depth(df, depth):
    close = df['close'].values
    n = len(close)
    high_indices = []
    low_indices = []
    half = depth
    for i in range(half, n - half):
        window_high = close[i - half:i + half + 1].max()
        window_low = close[i - half:i + half + 1].min()
        if close[i] == window_high:
            high_indices.append(i)
        if close[i] == window_low:
            low_indices.append(i)
    return high_indices, low_indices


def _tag_rules_at_extremes(df, extremes, rules_dict, window=10):
    tags = {name: 0 for name in rules_dict}
    for ext_idx in extremes:
        start = max(60, ext_idx - window)
        end = min(len(df), ext_idx + window + 1)
        for t in range(start, end):
            for name, rule_func in rules_dict.items():
                try:
                    if rule_func(df, t):
                        tags[name] += 1
                except Exception:
                    pass
    return tags


def _tag_from_precomputed(rule_signals, extremes, n, window=10):
    tags = {name: 0 for name in rule_signals}
    for ext_idx in extremes:
        start = max(0, ext_idx - window)
        end = min(n, ext_idx + window + 1)
        for name, sig in rule_signals.items():
            tags[name] += int(np.sum(sig[start:end]))
    return tags


def _tag_single_rule_at_extremes(df, extremes, rule_func, window=10):
    count = 0
    for ext_idx in extremes:
        start = max(60, ext_idx - window)
        end = min(len(df), ext_idx + window + 1)
        for t in range(start, end):
            try:
                if rule_func(df, t):
                    count += 1
            except Exception:
                pass
    return count


def _tag_rules_at_extremes_parallel(df, extremes, rules_dict, window=10, max_workers=None):
    if max_workers is None:
        max_workers = min(len(rules_dict), 6)

    if len(extremes) == 0 or len(rules_dict) <= 2:
        return _tag_rules_at_extremes(df, extremes, rules_dict, window)

    tags = {name: 0 for name in rules_dict}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, func in rules_dict.items():
            futures[executor.submit(_tag_single_rule_at_extremes, df, extremes, func, window)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                tags[name] = future.result()
            except Exception:
                tags[name] = 0

    return tags


def _backtest_single_rule(df, rule_func, direction='buy', hold_days=5):
    returns = []
    min_start = max(60, hold_days + 5)
    for idx in range(min_start, len(df) - hold_days):
        try:
            if rule_func(df, idx):
                entry_price = df['close'].iloc[idx]
                exit_price = df['close'].iloc[idx + hold_days]
                ret = (exit_price - entry_price) / entry_price
                if direction == 'sell':
                    ret = -ret
                returns.append(ret)
        except Exception:
            continue
    if not returns:
        return 0.0, 0.0, 0
    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    avg_return = np.mean(returns)
    return win_rate, avg_return, len(returns)


def _backtest_rules_parallel(df, rules_dict, direction='buy', hold_days=5, max_workers=None):
    if max_workers is None:
        max_workers = min(len(rules_dict), 6)

    results = {}
    if len(rules_dict) <= 2:
        for name, func in rules_dict.items():
            wr, ar, cnt = _backtest_single_rule(df, func, direction, hold_days)
            results[name] = {'win_rate': wr, 'avg_return': ar, 'count': cnt}
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, func in rules_dict.items():
            futures[executor.submit(_backtest_single_rule, df, func, direction, hold_days)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                wr, ar, cnt = future.result()
                results[name] = {'win_rate': wr, 'avg_return': ar, 'count': cnt}
            except Exception:
                results[name] = {'win_rate': 0.0, 'avg_return': 0.0, 'count': 0}

    return results


def classify_market_regime(df, idx, lookback=60):
    if idx < lookback:
        return 'unknown'
    close = df['close'].values.astype(float)
    ma20 = talib.MA(close[:idx+1], timeperiod=20)
    ma60 = talib.MA(close[:idx+1], timeperiod=60)
    if pd.isna(ma60[idx]) or ma60[idx] == 0:
        return 'unknown'
    trend = (ma20[idx] - ma60[idx]) / ma60[idx]
    atr = talib.ATR(df['high'].values.astype(float)[:idx+1],
                    df['low'].values.astype(float)[:idx+1],
                    df['close'].values.astype(float)[:idx+1], timeperiod=14)
    if len(atr) == 0 or pd.isna(atr[-1]) or close[idx] == 0:
        return 'unknown'
    vol_ratio = atr[-1] / close[idx]
    if trend > 0.02 and vol_ratio < 0.03:
        return 'bull_trend'
    elif trend > 0.02 and vol_ratio >= 0.03:
        return 'bull_volatile'
    elif trend < -0.02 and vol_ratio < 0.03:
        return 'bear_trend'
    elif trend < -0.02 and vol_ratio >= 0.03:
        return 'bear_volatile'
    else:
        return 'sideways'


def _precompute_quality_scores(df, buy_rule_signals, sell_rule_signals, start_offset=60):
    n = len(df)
    close = df['close'].values.astype(float)
    buy_quality = {name: np.zeros(n, dtype=np.float32) for name in buy_rule_signals}
    sell_quality = {name: np.zeros(n, dtype=np.float32) for name in sell_rule_signals}
    n_buy = max(len(buy_rule_signals), 1)
    n_sell = max(len(sell_rule_signals), 1)
    for idx in range(start_offset, n):
        buy_triggered_count = sum(1 for name in buy_rule_signals if buy_rule_signals[name][idx])
        sell_triggered_count = sum(1 for name in sell_rule_signals if sell_rule_signals[name][idx])
        buy_confluence = buy_triggered_count / n_buy
        sell_confluence = sell_triggered_count / n_sell
        if idx >= 5:
            momentum = (close[idx] - close[idx-5]) / close[idx-5] if close[idx-5] > 0 else 0
        else:
            momentum = 0
        buy_momentum = 1.0 / (1.0 + np.exp(momentum * 20))
        sell_momentum = 1.0 / (1.0 + np.exp(-momentum * 20))
        for name in buy_rule_signals:
            if buy_rule_signals[name][idx]:
                buy_quality[name][idx] = 0.45 * buy_confluence + 0.35 * buy_momentum + 0.20
        for name in sell_rule_signals:
            if sell_rule_signals[name][idx]:
                sell_quality[name][idx] = 0.45 * sell_confluence + 0.35 * sell_momentum + 0.20
    return buy_quality, sell_quality


def _backtest_adaptive(df, rule_func, direction='buy',
                       min_hold=3, max_hold=20,
                       stop_loss_pct=0.05, take_profit_pct=0.10):
    close = df['close'].values.astype(float)
    n = len(close)
    returns = []
    for idx in range(60, n - max_hold):
        try:
            if not rule_func(df, idx):
                continue
        except Exception:
            continue
        entry_price = close[idx]
        if entry_price <= 0:
            continue
        best_return = 0
        worst_return = 0
        exit_price = entry_price
        exit_day = min_hold
        for hold in range(min_hold, max_hold + 1):
            if idx + hold >= n:
                break
            current_price = close[idx + hold]
            ret = (current_price - entry_price) / entry_price
            if direction == 'sell':
                ret = -ret
            best_return = max(best_return, ret)
            worst_return = min(worst_return, ret)
            if ret < -stop_loss_pct:
                exit_price = current_price
                exit_day = hold
                break
            if ret > take_profit_pct:
                exit_price = current_price
                exit_day = hold
                break
            exit_price = current_price
            exit_day = hold
        final_ret = (exit_price - entry_price) / entry_price
        if direction == 'sell':
            final_ret = -final_ret
        returns.append({
            'return': final_ret,
            'max_drawdown': worst_return,
            'max_profit': best_return,
            'hold_days': exit_day,
        })
    if not returns:
        return {'win_rate': 0, 'avg_return': 0, 'count': 0,
                'avg_hold': 0, 'profit_factor': 0, 'avg_max_drawdown': 0}
    wins = [r for r in returns if r['return'] > 0]
    losses = [r for r in returns if r['return'] <= 0]
    total_profit = sum(r['return'] for r in wins) if wins else 0
    total_loss = abs(sum(r['return'] for r in losses)) if losses else 0.001
    return {
        'win_rate': len(wins) / len(returns),
        'avg_return': np.mean([r['return'] for r in returns]),
        'count': len(returns),
        'avg_hold': np.mean([r['hold_days'] for r in returns]),
        'profit_factor': total_profit / total_loss,
        'avg_max_drawdown': np.mean([r['max_drawdown'] for r in returns]),
    }


def _backtest_rules_adaptive_parallel(df, rules_dict, direction='buy',
                                       min_hold=3, max_hold=20,
                                       stop_loss_pct=0.05, take_profit_pct=0.10,
                                       max_workers=None):
    if max_workers is None:
        max_workers = min(len(rules_dict), 6)
    results = {}
    if len(rules_dict) <= 2:
        for name, func in rules_dict.items():
            results[name] = _backtest_adaptive(df, func, direction, min_hold, max_hold,
                                                stop_loss_pct, take_profit_pct)
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_backtest_adaptive, df, func, direction,
                                   min_hold, max_hold, stop_loss_pct, take_profit_pct): name
                   for name, func in rules_dict.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = {'win_rate': 0, 'avg_return': 0, 'count': 0,
                                 'avg_hold': 0, 'profit_factor': 0, 'avg_max_drawdown': 0}
    return results


def _graduated_penalty(win_rate, avg_return, profit_factor, count, min_count=3):
    if count < min_count:
        return 0.3, f'样本{count}<{min_count}，权重×0.3'
    wr_mult = 1.0
    wr_reason = ''
    if win_rate >= 0.65:
        wr_mult = 1.0
    elif win_rate >= 0.55:
        wr_mult = 0.92
        wr_reason = f'胜率{win_rate*100:.0f}%∈[55%,65%)，×0.92'
    elif win_rate >= 0.45:
        wr_mult = 0.80
        wr_reason = f'胜率{win_rate*100:.0f}%∈[45%,55%)，×0.80'
    elif win_rate >= 0.35:
        wr_mult = 0.60
        wr_reason = f'胜率{win_rate*100:.0f}%∈[35%,45%)，×0.60'
    else:
        wr_mult = 0.35
        wr_reason = f'胜率{win_rate*100:.0f}%<35%，×0.35'
    ret_mult = 1.0
    ret_reason = ''
    if avg_return >= 0.03:
        ret_mult = 1.0
    elif avg_return >= 0.01:
        ret_mult = 0.95
        ret_reason = f'收益{avg_return*100:.1f}%∈[1%,3%)，×0.95'
    elif avg_return >= 0:
        ret_mult = 0.85
        ret_reason = f'收益{avg_return*100:.1f}%∈[0%,1%)，×0.85'
    else:
        ret_mult = 0.50
        ret_reason = f'收益{avg_return*100:.1f}%<0%，×0.50'
    pf_mult = 1.0
    pf_reason = ''
    if profit_factor >= 1.5:
        pf_mult = 1.0
    elif profit_factor >= 1.0:
        pf_mult = 0.95
        pf_reason = f'盈亏比{profit_factor:.2f}∈[1.0,1.5)，×0.95'
    elif profit_factor >= 0.7:
        pf_mult = 0.80
        pf_reason = f'盈亏比{profit_factor:.2f}∈[0.7,1.0)，×0.80'
    else:
        pf_mult = 0.45
        pf_reason = f'盈亏比{profit_factor:.2f}<0.7，×0.45'
    total_mult = wr_mult * ret_mult * pf_mult
    total_mult = max(0.05, min(1.0, total_mult))
    reasons = [r for r in [wr_reason, ret_reason, pf_reason] if r]
    reason_str = '；'.join(reasons) if reasons else '表现优秀，无惩罚'
    return total_mult, reason_str


def optimize_weights_v2(df, buy_rules=None, sell_rules=None,
                        max_depth=200, step=10,
                        window=5, decay_factor=3.0,
                        min_hold=3, max_hold=20,
                        stop_loss_pct=0.05, take_profit_pct=0.10):
    if buy_rules is None:
        buy_rules = BUY_RULES
    if sell_rules is None:
        sell_rules = SELL_RULES

    print(f"[V2] 预计算规则信号表...")
    buy_rule_signals = _precompute_rule_signals(df, buy_rules, start_offset=60)
    sell_rule_signals = _precompute_rule_signals(df, sell_rules, start_offset=60)
    print(f"[V2] 规则信号表完成 (买入{len(buy_rules)}, 卖出{len(sell_rules)})")

    print(f"[V2] 预计算质量评分...")
    buy_quality, sell_quality = _precompute_quality_scores(df, buy_rule_signals, sell_rule_signals)
    print(f"[V2] 质量评分完成")

    print(f"[V2] 自适应回测...")
    buy_backtest = _backtest_rules_adaptive_parallel(df, buy_rules, 'buy',
                                                      min_hold, max_hold,
                                                      stop_loss_pct, take_profit_pct)
    sell_backtest = _backtest_rules_adaptive_parallel(df, sell_rules, 'sell',
                                                       min_hold, max_hold,
                                                       stop_loss_pct, take_profit_pct)
    print(f"[V2] 回测完成")

    n = len(df)
    best_buy_weights = {name: 0.0 for name in buy_rules}
    best_sell_weights = {name: 0.0 for name in sell_rules}
    best_depth = step
    best_buy_stats = {}
    best_sell_stats = {}
    best_score = -1

    for depth in range(step, max_depth + 1, step):
        high_indices, low_indices = _find_extremes_by_depth(df, depth)
        if not high_indices and not low_indices:
            continue

        buy_quality_sums = {}
        for name in buy_rules:
            total_q = 0.0
            for ext_idx in low_indices:
                for offset in range(-window, window + 1):
                    day = ext_idx + offset
                    if 0 <= day < n and buy_rule_signals[name][day]:
                        timeliness = np.exp(-abs(offset) / decay_factor)
                        q = buy_quality[name][day]
                        total_q += q * timeliness
            buy_quality_sums[name] = total_q

        sell_quality_sums = {}
        for name in sell_rules:
            total_q = 0.0
            for ext_idx in high_indices:
                for offset in range(-window, window + 1):
                    day = ext_idx + offset
                    if 0 <= day < n and sell_rule_signals[name][day]:
                        timeliness = np.exp(-abs(offset) / decay_factor)
                        q = sell_quality[name][day]
                        total_q += q * timeliness
            sell_quality_sums[name] = total_q

        total_buy_q = sum(buy_quality_sums.values())
        total_sell_q = sum(sell_quality_sums.values())

        raw_buy_w = {}
        raw_sell_w = {}
        if total_buy_q > 0:
            for name in buy_rules:
                raw_buy_w[name] = buy_quality_sums[name] / total_buy_q
        else:
            for name in buy_rules:
                raw_buy_w[name] = 0.0
        if total_sell_q > 0:
            for name in sell_rules:
                raw_sell_w[name] = sell_quality_sums[name] / total_sell_q
        else:
            for name in sell_rules:
                raw_sell_w[name] = 0.0

        buy_stats = {}
        for name in buy_rules:
            bt = buy_backtest.get(name, {})
            buy_stats[name] = {
                'win_rate': bt.get('win_rate', 0),
                'avg_return': bt.get('avg_return', 0),
                'count': bt.get('count', 0),
                'profit_factor': bt.get('profit_factor', 0),
                'avg_hold': bt.get('avg_hold', 0),
                'quality_sum': buy_quality_sums.get(name, 0),
            }
        sell_stats = {}
        for name in sell_rules:
            bt = sell_backtest.get(name, {})
            sell_stats[name] = {
                'win_rate': bt.get('win_rate', 0),
                'avg_return': bt.get('avg_return', 0),
                'count': bt.get('count', 0),
                'profit_factor': bt.get('profit_factor', 0),
                'avg_hold': bt.get('avg_hold', 0),
                'quality_sum': sell_quality_sums.get(name, 0),
            }

        active_buy = {name for name in buy_rules if raw_buy_w.get(name, 0) > 0}
        active_sell = {name for name in sell_rules if raw_sell_w.get(name, 0) > 0}
        if not active_buy or not active_sell:
            continue

        buy_wr = np.mean([buy_stats[n]['win_rate'] for n in active_buy])
        sell_wr = np.mean([sell_stats[n]['win_rate'] for n in active_sell])
        buy_ret = np.mean([buy_stats[n]['avg_return'] for n in active_buy])
        sell_ret = np.mean([sell_stats[n]['avg_return'] for n in active_sell])
        score = (buy_wr + sell_wr) / 2 * 0.5 + (buy_ret + sell_ret) / 2 * 100 * 0.5

        if score > best_score:
            best_score = score
            best_buy_weights = dict(raw_buy_w)
            best_sell_weights = dict(raw_sell_w)
            best_depth = depth
            best_buy_stats = dict(buy_stats)
            best_sell_stats = dict(sell_stats)

    return {
        'buy_weights': best_buy_weights,
        'sell_weights': best_sell_weights,
        'depth_used': best_depth,
        'all_buy_stats': best_buy_stats,
        'all_sell_stats': best_sell_stats,
        'best_score': best_score,
    }


def optimize_weights(df, buy_rules=None, sell_rules=None,
                     max_depth=200, step=10,
                     win_rate_threshold=0.9, return_threshold=0.2,
                     window=10, hold_days=5):
    if buy_rules is None:
        buy_rules = BUY_RULES
    if sell_rules is None:
        sell_rules = SELL_RULES

    print(f"[优化] 预计算规则信号表 (买入{len(buy_rules)}规则, 卖出{len(sell_rules)}规则)...")
    buy_rule_signals = _precompute_rule_signals(df, buy_rules, start_offset=60)
    sell_rule_signals = _precompute_rule_signals(df, sell_rules, start_offset=60)
    print(f"[优化] 规则信号表完成")

    print(f"[优化] 预计算回测统计...")
    buy_backtest = _backtest_rules_parallel(df, buy_rules, 'buy', hold_days)
    sell_backtest = _backtest_rules_parallel(df, sell_rules, 'sell', hold_days)
    print(f"[优化] 回测统计完成")

    n = len(df)

    best_buy_rules = {}
    best_sell_rules = {}
    best_buy_weights = {name: 0.0 for name in buy_rules}
    best_sell_weights = {name: 0.0 for name in sell_rules}
    best_depth = step
    best_buy_stats = {}
    best_sell_stats = {}
    found_strict = False

    for depth in range(step, max_depth + 1, step):
        high_indices, low_indices = _find_extremes_by_depth(df, depth)

        if not high_indices and not low_indices:
            continue

        buy_tags = _tag_from_precomputed(buy_rule_signals, low_indices, n, window)
        sell_tags = _tag_from_precomputed(sell_rule_signals, high_indices, n, window)

        total_buy_tags = sum(buy_tags.values())
        total_sell_tags = sum(sell_tags.values())

        raw_buy_w = {}
        raw_sell_w = {}

        if total_buy_tags > 0:
            for name in buy_rules:
                raw_buy_w[name] = buy_tags[name] / total_buy_tags
        else:
            for name in buy_rules:
                raw_buy_w[name] = 0.0

        if total_sell_tags > 0:
            for name in sell_rules:
                raw_sell_w[name] = sell_tags[name] / total_sell_tags
        else:
            for name in sell_rules:
                raw_sell_w[name] = 0.0

        buy_stats = {}
        for name in buy_rules:
            bt = buy_backtest.get(name, {})
            buy_stats[name] = {
                'win_rate': bt.get('win_rate', 0),
                'avg_return': bt.get('avg_return', 0),
                'count': bt.get('count', 0),
                'tags': buy_tags.get(name, 0),
            }

        sell_stats = {}
        for name in sell_rules:
            bt = sell_backtest.get(name, {})
            sell_stats[name] = {
                'win_rate': bt.get('win_rate', 0),
                'avg_return': bt.get('avg_return', 0),
                'count': bt.get('count', 0),
                'tags': sell_tags.get(name, 0),
            }

        strict_buy = {name: buy_rules[name] for name in buy_rules
                      if (buy_stats[name]['win_rate'] >= win_rate_threshold and
                          buy_stats[name]['avg_return'] >= return_threshold and
                          buy_stats[name]['count'] >= 3)}
        strict_sell = {name: sell_rules[name] for name in sell_rules
                       if (sell_stats[name]['win_rate'] >= win_rate_threshold and
                           sell_stats[name]['avg_return'] >= return_threshold and
                           sell_stats[name]['count'] >= 3)}

        if strict_buy and strict_sell:
            active_bw = {n: raw_buy_w.get(n, 0) for n in strict_buy if raw_buy_w.get(n, 0) > 0}
            active_sw = {n: raw_sell_w.get(n, 0) for n in strict_sell if raw_sell_w.get(n, 0) > 0}

            total_abw = sum(active_bw.values())
            total_asw = sum(active_sw.values())

            final_buy_w = {n: (v / total_abw) if total_abw > 0 else (1.0/len(active_bw))
                          for n, v in active_bw.items()}
            final_sell_w = {n: (v / total_asw) if total_asw > 0 else (1.0/len(active_sw))
                           for n, v in active_sw.items()}

            best_buy_rules = strict_buy
            best_sell_rules = strict_sell
            best_buy_weights = final_buy_w
            best_sell_weights = final_sell_w
            best_depth = depth
            best_buy_stats = buy_stats
            best_sell_stats = sell_stats
            found_strict = True
            break
        else:
            triggered_buy = {name: buy_rules[name] for name in buy_rules
                             if buy_stats[name]['count'] > 0 or buy_tags.get(name, 0) > 0}
            triggered_sell = {name: sell_rules[name] for name in sell_rules
                              if sell_stats[name]['count'] > 0 or sell_tags.get(name, 0) > 0}

            if triggered_buy and triggered_sell:
                active_bw = {n: raw_buy_w.get(n, 0) for n in triggered_buy}
                active_sw = {n: raw_sell_w.get(n, 0) for n in triggered_sell}

                total_abw = sum(active_bw.values())
                total_asw = sum(active_sw.values())

                relaxed_buy_w = {n: (v / total_abw) if total_abw > 0 else (1.0/len(active_bw))
                                 for n, v in active_bw.items()}
                relaxed_sell_w = {n: (v / total_asw) if total_asw > 0 else (1.0/len(active_sw))
                                  for n, v in active_sw.items()}

                best_buy_rules = triggered_buy
                best_sell_rules = triggered_sell
                best_buy_weights = relaxed_buy_w
                best_sell_weights = relaxed_sell_w
                best_depth = depth
                best_buy_stats = buy_stats
                best_sell_stats = sell_stats

    zero_buy = {name: 0.0 for name in buy_rules}
    zero_sell = {name: 0.0 for name in sell_rules}

    result = {
        'buy_rules': best_buy_rules if best_buy_rules else dict(buy_rules),
        'sell_rules': best_sell_rules if best_sell_rules else dict(sell_rules),
        'buy_weights': best_buy_weights if any(v > 0 for v in best_buy_weights.values()) else zero_buy,
        'sell_weights': best_sell_weights if any(v > 0 for v in best_sell_weights.values()) else zero_sell,
        'depth_used': best_depth,
        'all_buy_stats': best_buy_stats,
        'all_sell_stats': best_sell_stats,
        'found_strict': found_strict,
    }
    return result


def apply_filters(df, idx, candidate_signal, trade_history,
                   min_interval=3, ma_period=20):
    if candidate_signal is None:
        return None

    close = df['close'].values
    n = len(close)

    if idx < ma_period:
        return None

    ma = talib.MA(close[:idx+1], timeperiod=ma_period)
    if pd.isna(ma[-1]) or pd.isna(ma[-2]):
        return None

    trend_up = (ma[-1] > ma[-2]) and (close[idx] > ma[-1])
    trend_down = (ma[-1] < ma[-2]) and (close[idx] < ma[-1])

    if candidate_signal == 'B' and trend_down:
        return None
    if candidate_signal == 'S' and trend_up:
        return None

    if trade_history:
        last_idx, last_signal = trade_history[-1]
        if idx - last_idx < min_interval:
            return None
        if last_signal == candidate_signal:
            return None

    return candidate_signal


def _precompute_rule_signals(df, rules_dict, start_offset=60):
    rule_signals = {}
    n = len(df)

    def _eval_rule(name_func):
        name, func = name_func
        signals = np.zeros(n, dtype=np.bool_)
        for idx in range(start_offset, n):
            try:
                if func(df, idx):
                    signals[idx] = True
            except Exception:
                pass
        return name, signals

    if len(rules_dict) <= 2:
        for name, func in rules_dict.items():
            _, sig = _eval_rule((name, func))
            rule_signals[name] = sig
    else:
        with ThreadPoolExecutor(max_workers=min(len(rules_dict), 6)) as executor:
            futures = [executor.submit(_eval_rule, (name, func)) for name, func in rules_dict.items()]
            for future in as_completed(futures):
                try:
                    name, sig = future.result()
                    rule_signals[name] = sig
                except Exception:
                    pass

    return rule_signals


# 「刺透」中文名须与 _BULLISH_CANDLE_DEFS 中 CDLPIERCING 标签一致
BULLISH_CANDLE_PIERCING_CN = '刺透'

# 突破前高类买入：信号日 high 至「下一前高」空间不足则否决
BREAK_PRIOR_HIGH_BUY_RULE_NAMES = frozenset({
    '放量突破高点',
    'W底突破',
    '头肩底突破',
    '横盘整理向上突破',
    'N字形突破',
    '旗形突破',
})
MIN_UPSIDE_TO_NEXT_PRIOR_HIGH_PCT = 0.03
NEXT_PRIOR_HIGH_LOOKFORWARD_BARS = 120
PRIOR_HIGH_LOOKBACK_BARS = 120
NEXT_PRIOR_HIGH_PIVOT_WINDOW = 5
T_HIGH_PRECEDING_BARS = 5  # T高前须连续5个交易日最高价均低于T高当日最高价


def bar_index_from_date_str(df, date_str):
    """将 YYYY-MM-DD 对齐到 df 行索引；未找到返回 None。"""
    if date_str is None or 'date' not in df.columns:
        return None
    target = str(date_str).split(' ')[0].split('T')[0][:10]
    for i, d in enumerate(df['date'].values):
        try:
            if pd.Timestamp(d).strftime('%Y-%m-%d') == target:
                return int(i)
        except Exception:
            s = str(d).split(' ')[0].split('T')[0][:10]
            if s == target:
                return int(i)
    return None


def bar_date_str_from_df(df, bar_idx):
    if bar_idx is None or bar_idx < 0 or bar_idx >= len(df):
        return None
    if 'date' not in df.columns:
        return str(int(bar_idx))
    d = df['date'].iloc[int(bar_idx)]
    try:
        return pd.Timestamp(d).strftime('%Y-%m-%d')
    except Exception:
        return str(d).split(' ')[0].split('T')[0][:10]


def _is_local_high_bar(high, i, window=None):
    w = window if window is not None else NEXT_PRIOR_HIGH_PIVOT_WINDOW
    if i < w or i + w >= len(high):
        return False
    return float(high[i]) >= float(np.max(high[i - w : i + w + 1])) - 1e-9


def _is_local_low_bar(low, i, window=None):
    w = window if window is not None else NEXT_PRIOR_HIGH_PIVOT_WINDOW
    if i < w or i + w >= len(low):
        return False
    return float(low[i]) <= float(np.min(low[i - w : i + w + 1])) + 1e-9


def _pct_upside_ref_minus_current(current_price, ref_price):
    """(参考价 - 当前价) / 当前价 × 100，前高类空间（%）。"""
    if ref_price is None or current_price is None:
        return None
    ref_price = float(ref_price)
    current_price = float(current_price)
    if current_price <= 0 or not np.isfinite(ref_price) or not np.isfinite(current_price):
        return None
    return round((ref_price - current_price) / current_price * 100.0, 2)


def _pct_return_vs_ref(anchor_close, ref_price):
    """自参考价至锚定棒收盘价的涨跌幅（%），用于前低等。"""
    if ref_price is None or anchor_close is None:
        return None
    ref_price = float(ref_price)
    anchor_close = float(anchor_close)
    if ref_price <= 0 or not np.isfinite(ref_price) or not np.isfinite(anchor_close):
        return None
    return round((anchor_close - ref_price) / ref_price * 100.0, 2)


def _is_breakthrough_prior_high_buy(buy_triggered):
    """B 是否由突破前高类规则触发（此类标注仍用旧前高扫描）。"""
    if not buy_triggered:
        return False
    for name in buy_triggered:
        if name in BREAK_PRIOR_HIGH_BUY_RULE_NAMES:
            return True
    return False


def _t_high_preceding_bars_ok(high, t_idx, preceding_bars=None):
    """T高前 preceding_bars 个交易日的最高价均低于 T高当日最高价。"""
    preceding_bars = preceding_bars if preceding_bars is not None else T_HIGH_PRECEDING_BARS
    t_idx = int(t_idx)
    if t_idx < preceding_bars:
        return False
    t_h = float(high[t_idx])
    prev_max = float(np.max(high[t_idx - preceding_bars : t_idx]))
    return prev_max < t_h - 1e-9


def find_prior_high_before_signal_bar(high, signal_idx, lookback=None, pivot_window=None):
    """信号日之前最近一处前高（局部 high 峰）；若无局部峰则取回看窗内最高 high。"""
    lookback = lookback if lookback is not None else PRIOR_HIGH_LOOKBACK_BARS
    pivot_window = pivot_window if pivot_window is not None else NEXT_PRIOR_HIGH_PIVOT_WINDOW
    signal_idx = int(signal_idx)
    end = int(signal_idx)
    start = max(0, end - lookback)
    if end <= start:
        return None, None
    for i in range(end - 1, start + pivot_window - 1, -1):
        if _is_local_high_bar(high, i, pivot_window):
            return i, float(high[i])
    bi = int(np.argmax(high[start:end]))
    return start + bi, float(high[start + bi])


def find_qualified_t_high_before(
    high, before_idx, min_high_exclusive, lookback=None, pivot_window=None, preceding_bars=None,
):
    """
    非突破前高类 B 的历史前高（T高）：
    1) 当日最高价 > min_high_exclusive（B 信号日最高价）；
    2) T高前 preceding_bars 个交易日，各日最高价的最大值 < T高当日最高价。
    自 before_idx 之前由近及远扫描，先局部峰再任意棒。
    """
    before_idx = int(before_idx)
    min_high_exclusive = float(min_high_exclusive)
    lookback = lookback if lookback is not None else PRIOR_HIGH_LOOKBACK_BARS
    pivot_window = pivot_window if pivot_window is not None else NEXT_PRIOR_HIGH_PIVOT_WINDOW
    start = max(0, before_idx - lookback)
    end = before_idx
    if end <= start:
        return None, None

    def _try_scan(require_local):
        for i in range(end - 1, start + (pivot_window - 1 if require_local else 0), -1):
            if float(high[i]) <= min_high_exclusive + 1e-9:
                continue
            if require_local and not _is_local_high_bar(high, i, pivot_window):
                continue
            if _t_high_preceding_bars_ok(high, i, preceding_bars):
                return i, float(high[i])
        return None, None

    found = _try_scan(require_local=True)
    if found[0] is not None:
        return found
    return _try_scan(require_local=False)


def find_prior_low_before_signal_bar(low, signal_idx, lookback=None, pivot_window=None):
    """信号日之前最近一处前低（局部 low 谷）；若无局部谷则取回看窗内最低 low。"""
    lookback = lookback if lookback is not None else PRIOR_HIGH_LOOKBACK_BARS
    pivot_window = pivot_window if pivot_window is not None else NEXT_PRIOR_HIGH_PIVOT_WINDOW
    signal_idx = int(signal_idx)
    start = max(0, signal_idx - lookback)
    end = signal_idx
    if end <= start:
        return None, None
    for i in range(end - 1, start + pivot_window - 1, -1):
        if _is_local_low_bar(low, i, pivot_window):
            return i, float(low[i])
    bi = int(np.argmin(low[start:end]))
    return start + bi, float(low[start + bi])


def find_next_prior_high_bar(high, signal_idx, lookforward=None, pivot_window=None):
    """
    信号日之后遇到的下一处前高（局部 high 峰，含上影线）。
    若无局部峰，则用信号日后窗口内最高 high（须高于信号日 high）。
    返回 (bar_index, high_price)。
    """
    lookforward = lookforward if lookforward is not None else NEXT_PRIOR_HIGH_LOOKFORWARD_BARS
    pivot_window = pivot_window if pivot_window is not None else NEXT_PRIOR_HIGH_PIVOT_WINDOW
    signal_idx = int(signal_idx)
    n = len(high)
    start = signal_idx + 1
    end = min(n, signal_idx + lookforward + 1)
    if start >= n:
        return None, None
    signal_h = float(high[signal_idx])
    for i in range(start, end):
        if not _is_local_high_bar(high, i, pivot_window):
            continue
        hp = float(high[i])
        if hp > signal_h + 1e-9:
            return i, hp
    if end > start:
        rel = int(np.argmax(high[start:end]))
        fut_max = float(high[start + rel])
        if fut_max > signal_h + 1e-9:
            return start + rel, fut_max
    return None, None


def find_next_prior_high_price(high, signal_idx, lookforward=None, pivot_window=None):
    _, price = find_next_prior_high_bar(high, signal_idx, lookforward, pivot_window)
    return price


def signal_extreme_annotation(df, signal_bar_idx, anchor_bar_idx=None, buy_triggered=None):
    """
    B/S 标注：历史前高、前高的前高、历史前低。
    突破前高类买入：前高仍用局部峰扫描。
    其他 B：T高规则（>信号日最高价，且 T高前5日最高价均低于 T高当日最高价）。
    anchor_bar_idx：当前价棒（确认棒收盘）；前高收益 = (前高价-现价)/现价。
    """
    signal_bar_idx = int(signal_bar_idx)
    if signal_bar_idx < 0 or signal_bar_idx >= len(df):
        return {}
    anchor_bar_idx = int(anchor_bar_idx if anchor_bar_idx is not None else signal_bar_idx)
    if anchor_bar_idx < 0 or anchor_bar_idx >= len(df):
        anchor_bar_idx = signal_bar_idx
    high = df['high'].values.astype(float) if 'high' in df.columns else df['close'].values.astype(float)
    low = df['low'].values.astype(float) if 'low' in df.columns else df['close'].values.astype(float)
    close = df['close'].values.astype(float)
    anchor_close = float(close[anchor_bar_idx])
    signal_h = float(high[signal_bar_idx])
    out = {'signal_bar_high': round(signal_h, 2)}

    use_t_high = not _is_breakthrough_prior_high_buy(buy_triggered)
    out['prior_high_mode'] = 't_high' if use_t_high else 'breakthrough'

    if use_t_high:
        prior_i, prior_p = find_qualified_t_high_before(high, signal_bar_idx, signal_h)
    else:
        prior_i, prior_p = find_prior_high_before_signal_bar(high, signal_bar_idx)
    if prior_i is not None:
        out['prior_high_date'] = bar_date_str_from_df(df, prior_i)
        out['prior_high_price'] = round(float(prior_p), 2)
        ret = _pct_upside_ref_minus_current(anchor_close, prior_p)
        if ret is not None:
            out['return_vs_prior_high_pct'] = ret

    if prior_i is not None:
        if use_t_high:
            pp_i, pp_p = find_qualified_t_high_before(high, prior_i, signal_h)
        else:
            pp_i, pp_p = find_prior_high_before_signal_bar(high, prior_i)
        if pp_i is not None:
            out['prior_prior_high_date'] = bar_date_str_from_df(df, pp_i)
            out['prior_prior_high_price'] = round(float(pp_p), 2)
            ret_pp = _pct_upside_ref_minus_current(anchor_close, pp_p)
            if ret_pp is not None:
                out['return_vs_prior_prior_high_pct'] = ret_pp

    low_i, low_p = find_prior_low_before_signal_bar(low, signal_bar_idx)
    if low_i is not None:
        out['prior_low_date'] = bar_date_str_from_df(df, low_i)
        out['prior_low_price'] = round(float(low_p), 2)
        ret_l = _pct_return_vs_ref(anchor_close, low_p)
        if ret_l is not None:
            out['return_vs_prior_low_pct'] = ret_l

    return out


def prior_next_high_annotation(df, signal_bar_idx, anchor_bar_idx=None, buy_triggered=None):
    """兼容旧名；买入过滤仍用 find_next_prior_high_*，此处仅输出历史高低点标注。"""
    return signal_extreme_annotation(df, signal_bar_idx, anchor_bar_idx, buy_triggered)


def breakthrough_has_room_to_next_prior_high(df, signal_idx, min_upside_pct=None):
    """信号日最高价 → 下一前高的涨幅须 >= min_upside_pct（默认 3 个点=3%）。"""
    if min_upside_pct is None:
        min_upside_pct = MIN_UPSIDE_TO_NEXT_PRIOR_HIGH_PCT
    signal_idx = int(signal_idx)
    if signal_idx < 0 or signal_idx >= len(df):
        return True
    high = df['high'].values.astype(float) if 'high' in df.columns else df['close'].values.astype(float)
    signal_h = float(high[signal_idx])
    if signal_h <= 0 or not np.isfinite(signal_h):
        return True
    next_h = find_next_prior_high_price(high, signal_idx)
    if next_h is None:
        return True
    upside = (next_h - signal_h) / signal_h
    return upside >= float(min_upside_pct) - 1e-12


def breakthrough_prior_high_vetoes_buy(df, signal_idx):
    """空间不足时返回 True（应放弃 B）。"""
    return not breakthrough_has_room_to_next_prior_high(df, signal_idx)


def apply_all_buy_next_high_room_filter(
    df, buy_signals_pre, buy_rules_map, start_offset, mandatory_buy_pre=None,
):
    """全部买入（含必买）：信号日 high 至下一前高空间 < 3% 则置 False。"""
    n = len(df)
    pools = []
    if buy_signals_pre is not None:
        pools.append(buy_signals_pre)
    if mandatory_buy_pre is not None:
        pools.append(mandatory_buy_pre)
    for pool in pools:
        for name, arr in pool.items():
            if arr is None:
                continue
            for idx in range(start_offset, n):
                if arr[idx] and breakthrough_prior_high_vetoes_buy(df, idx):
                    arr[idx] = False


# 兼容旧名
apply_breakthrough_next_high_room_filter = apply_all_buy_next_high_room_filter


def _idx_has_pattern_breakout_buy_support(buy_signals_pre, mandatory_buy_pre, idx, exclude_names=None):
    """当日是否存在看涨K线、形态学或突破类买入支持（不含 exclude_names 自身）。"""
    exclude = set(exclude_names or ())
    for name, arr in (buy_signals_pre or {}).items():
        if name in exclude or name not in BUY_PATTERN_BREAKOUT_SUPPORT_NAMES:
            continue
        if arr is not None and idx < len(arr) and bool(arr[idx]):
            return True
    for name, arr in (mandatory_buy_pre or {}).items():
        if name in BUY_PATTERN_BREAKOUT_SUPPORT_NAMES:
            if arr is not None and idx < len(arr) and bool(arr[idx]):
                return True
    return False


def apply_macd_vp_requires_pattern_breakout_buy(
    df, buy_signals_pre, buy_rules_map, start_offset, mandatory_buy_pre=None,
):
    """MACD金叉/底背离、价升量增须同日有看涨K线形态、形态学或突破，否则置 False。"""
    n = len(df)
    for name in BUY_MACD_VP_REQUIRES_PATTERN_BREAKOUT:
        arr = buy_signals_pre.get(name) if buy_signals_pre else None
        if arr is None:
            continue
        for idx in range(start_offset, n):
            if not arr[idx]:
                continue
            if _idx_has_pattern_breakout_buy_support(
                buy_signals_pre, mandatory_buy_pre, idx, exclude_names={name},
            ):
                continue
            arr[idx] = False


def apply_piercing_requires_confluence_buy(df, buy_signals_pre, buy_rules_map, start_offset,
                                           mandatory_buy_pre=None):
    """
    刺透不得单独作为「看涨K线形态」：须同日至少还有一条其它加权买入规则命中，
    或任一「必买」规则命中（如 W 底突破），或同一根 K 上另有其它看涨子形态与刺透并存
    （TA-Lib 多重 100）；不含「看涨K线形态」加权键自身。
    """
    bull = '看涨K线形态'
    if bull not in buy_signals_pre or bull not in buy_rules_map:
        return
    arr = buy_signals_pre[bull]
    n = len(arr)
    other_keys = [k for k in buy_rules_map.keys() if k != bull]
    for idx in range(start_offset, n):
        if not arr[idx]:
            continue
        kind = detect_bullish_candle_pattern_kind(df, idx)
        if kind != BULLISH_CANDLE_PIERCING_CN:
            continue
        kinds_here = detect_bullish_candle_pattern_labels_at_bar(df, idx)
        # 先命中为刺透时，若同日另有晨星/锤头等其它看涨子形态同为 100，视为共振
        has_other = len(kinds_here) > 1
        if not has_other:
            for ok in other_keys:
                sig = buy_signals_pre.get(ok)
                if sig is not None and idx < len(sig) and bool(sig[idx]):
                    has_other = True
                    break
        if not has_other and mandatory_buy_pre:
            for sig in mandatory_buy_pre.values():
                if sig is not None and idx < len(sig) and bool(sig[idx]):
                    has_other = True
                    break
        if not has_other:
            arr[idx] = False


def _dwe_signal_day_volume_effective(df, idx, mult=1.05):
    if 'volume' not in df.columns or idx < 1:
        return True
    v = df['volume'].values.astype(float)
    base = float(np.mean(v[max(0, idx - 5):idx]))
    if base <= 0 or not np.isfinite(base):
        return True
    return float(v[idx]) >= base * mult


# 上影线否决 B：① 相对实体 ② 相对全日振幅（覆盖小实体+长上影，如 600519 类长上影日）
_UPPER_SHADOW_VS_BODY = 0.75
_UPPER_SHADOW_VS_RANGE = 0.40
_UPPER_SHADOW_MIN_RANGE_REL = 0.008  # 振幅低于收盘价×该比例时不启用「占比」条款，减少极窄日误判


def upper_shadow_vetoes_buy_signal(df, idx):
    """当日 K 线：上影过长则否决 B（用于信号日及 pending 确认日）.
    - 上影线 = high - max(open, close)；实体 = |close-open|；振幅 = high-low
    - 满足任一即否决：上影 > 实体×3/4；或（振幅足够大时）上影 ≥ 振幅×40%
    """
    if idx < 0 or idx >= len(df):
        return False
    try:
        row = df.iloc[idx]
        o = float(row['open'])
        h = float(row['high'])
        lo = float(row['low'])
        c = float(row['close'])
    except Exception:
        return False
    if not all(np.isfinite(x) for x in (o, h, lo, c)):
        return False
    ref = max(abs(c), 1e-9)
    range_ = h - lo
    if range_ <= 0:
        return False
    upper = h - max(o, c)
    if upper <= 0:
        return False
    body = abs(c - o)
    min_body = 1e-10 * ref
    if body >= min_body and upper > _UPPER_SHADOW_VS_BODY * body:
        return True
    if range_ >= _UPPER_SHADOW_MIN_RANGE_REL * ref and upper >= _UPPER_SHADOW_VS_RANGE * range_:
        return True
    return False


def confirm_buy_vetoes_shrink_volume_without_close_break(df, confirm_idx, signal_idx):
    """pending 确认日：相对信号日缩量时，须确认收盘 > 信号日收盘，否则否决 B（返回 True 表示否决）."""
    if confirm_idx < 0 or signal_idx < 0 or confirm_idx >= len(df) or signal_idx >= len(df):
        return False
    if 'volume' not in df.columns:
        return False
    try:
        v_c = float(df['volume'].iloc[confirm_idx])
        v_s = float(df['volume'].iloc[signal_idx])
        c_c = float(df['close'].iloc[confirm_idx])
        c_s = float(df['close'].iloc[signal_idx])
    except Exception:
        return False
    if not all(np.isfinite(x) for x in (v_c, v_s, c_c, c_s)):
        return False
    if v_s <= 0:
        return False
    if v_c >= v_s:
        return False
    return c_c <= c_s


# pending 二次确认：前低回看根数（与 V反左侧压力窗口一致为 5）
PENDING_PRELOW_LOOKBACK_BARS = 5
# 与 engine V2 ATR 止损倍数对齐，供非 engine 路径（如 generate_signals_with_weights）使用
PENDING_ATR_STOP_MULT_DEFAULT = 2.5


def pending_buy_support_floor(close, low, signal_idx, atr_arr=None, atr_mult=None):
    """pending 多头：前低带（信号日前若干日至信号日最低低）与 信号收盘−ATR×倍数 的较强位，日 low 不得跌破。"""
    if atr_mult is None:
        atr_mult = PENDING_ATR_STOP_MULT_DEFAULT
    s = int(signal_idx)
    if s < 0 or s >= len(low):
        return float('-inf')
    i0 = max(0, s - PENDING_PRELOW_LOOKBACK_BARS)
    pre_low = float(np.min(low[i0 : s + 1]))
    floor = pre_low
    if atr_arr is not None and s < len(atr_arr) and s < len(close):
        ae = atr_arr[s]
        if not pd.isna(ae) and ae > 0:
            stop_px = float(close[s]) - float(atr_mult) * float(ae)
            floor = max(floor, float(stop_px))
    return floor


def pending_buy_lows_hold_since_signal(low, signal_idx, through_idx, floor):
    """自 signal 的次一交易日至 through_idx（含），各日 low 均 >= floor。"""
    if not np.isfinite(floor):
        return True
    for j in range(signal_idx + 1, through_idx + 1):
        if j < 0 or j >= len(low):
            return False
        if float(low[j]) + 1e-12 < float(floor):
            return False
    return True


def generate_signals_with_weights(df, buy_rules, buy_weights,
                                  sell_rules, sell_weights,
                                  min_interval=3, ma_period=20,
                                  start_offset=60,
                                  confidence_medium=0.55, confidence_strong=0.70,
                                  min_score_threshold=0.05,
                                  buy_details=None, sell_details=None,
                                  min_triggered_win_rate=0.50,
                                  mandatory_sell_rules=None,
                                  buy_restriction_rules=None,
                                  sell_trigger_count=2):
    buy_rule_signals = _precompute_rule_signals(df, buy_rules, start_offset)
    sell_rule_signals = _precompute_rule_signals(df, sell_rules, start_offset)
    mandatory_buy_signals = _precompute_rule_signals(df, MANDATORY_BUY_RULES, start_offset)

    mandatory_sell_signals = {}
    if mandatory_sell_rules:
        mandatory_sell_signals = _precompute_rule_signals(df, mandatory_sell_rules, start_offset)

    restriction_signals = {}
    if buy_restriction_rules:
        restriction_signals = _precompute_rule_signals(df, buy_restriction_rules, start_offset)

    apply_piercing_requires_confluence_buy(df, buy_rule_signals, buy_rules, start_offset,
                                           mandatory_buy_pre=mandatory_buy_signals)
    apply_macd_vp_requires_pattern_breakout_buy(
        df, buy_rule_signals, buy_rules, start_offset, mandatory_buy_pre=mandatory_buy_signals)
    apply_all_buy_next_high_room_filter(
        df, buy_rule_signals, buy_rules, start_offset, mandatory_buy_pre=mandatory_buy_signals)

    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    dates_arr = df['date'].values if 'date' in df.columns else None
    n = len(df)
    ma = talib.MA(close, timeperiod=ma_period)
    hi_a = df['high'].values.astype(float) if 'high' in df.columns else close
    lo_a = df['low'].values.astype(float) if 'low' in df.columns else close
    atr_pending = talib.ATR(hi_a, lo_a, close, timeperiod=14)

    def _dwe_bar_date_str(j):
        if dates_arr is None:
            return str(j)
        try:
            return pd.Timestamp(dates_arr[j]).strftime('%Y-%m-%d')
        except Exception:
            s = str(dates_arr[j])
            return s.split(' ')[0].split('T')[0][:10]

    signals = []
    trade_history = []
    reasons_list = []
    pending_buy = None

    for idx in range(start_offset, n):
        if pending_buy is not None:
            signal_idx, prev_reasons, prev_confidence, prev_level = pending_buy[:4]
            pending_meta = dict(pending_buy[4]) if len(pending_buy) > 4 else {}
            pending_buy = None
            rel = idx - signal_idx
            if 1 <= rel <= 2:
                floor = pending_buy_support_floor(
                    close, lo_a, signal_idx, atr_pending, PENDING_ATR_STOP_MULT_DEFAULT)
                hold = pending_buy_lows_hold_since_signal(lo_a, signal_idx, idx, floor)
                await_second = pending_meta.get('await_second_confirm', False)
                if rel == 2 and not await_second:
                    pass
                elif hold:
                    if pending_meta.get('bull_pattern_defer'):
                        gap_ok = (rel == 1 and not await_second) or (rel == 2 and await_second)
                        is_bullish = gap_ok and (close[idx] > open_[idx])
                    else:
                        is_bullish = (
                            close[idx] > open_[idx] and close[idx] > close[signal_idx])
                    ok = (
                        is_bullish
                        and not upper_shadow_vetoes_buy_signal(df, idx)
                        and not confirm_buy_vetoes_shrink_volume_without_close_break(
                            df, idx, signal_idx))
                    if ok:
                        trade_history.append((idx, 'B'))
                        signals.append((idx, 'B'))
                        row = {**prev_reasons, 'final_signal': 'B',
                               'confidence': round(prev_confidence, 3), 'level': prev_level,
                               'confirmed': True}
                        row['signal_date'] = _dwe_bar_date_str(signal_idx)
                        if rel == 2:
                            row['confirm_note'] = (
                                '≥2加权买入:第二交易日确认B' if not pending_meta.get('bull_pattern_defer')
                                else (
                                    '看涨K线形态:T+2阳线确认B;量能不足'
                                    if pending_meta.get('weak_vol_bull_pattern')
                                    else '看涨K线形态:T+2阳线确认B(收盘>开盘)'))
                        elif pending_meta.get('bull_pattern_defer'):
                            row['confirm_note'] = (
                                '看涨K线形态:下一交易日阳线确认B;量能不足'
                                if pending_meta.get('weak_vol_bull_pattern')
                                else '看涨K线形态:下一交易日阳线确认B')
                        reasons_list.append(row)
                    elif rel == 1 and not ok and len(prev_reasons.get('buy_triggered', [])) >= 2 and not await_second:
                        pending_meta['await_second_confirm'] = True
                        pending_buy = (signal_idx, prev_reasons, prev_confidence, prev_level, pending_meta)

        buy_score = 0.0
        sell_score = 0.0
        triggered_buy = []
        triggered_sell = []
        triggered_mandatory_sell = []
        triggered_restriction = []

        for name in buy_rules:
            if buy_rule_signals.get(name, np.zeros(n, dtype=np.bool_))[idx]:
                w = buy_weights.get(name, 0)
                buy_score += w
                triggered_buy.append(name)

        for name in sell_rules:
            if sell_rule_signals.get(name, np.zeros(n, dtype=np.bool_))[idx]:
                w = sell_weights.get(name, 0)
                sell_score += w
                triggered_sell.append(name)

        if mandatory_sell_signals:
            for name in mandatory_sell_rules:
                if mandatory_sell_signals.get(name, np.zeros(n, dtype=np.bool_))[idx]:
                    triggered_mandatory_sell.append(name)

        if restriction_signals:
            for name in buy_restriction_rules:
                if restriction_signals.get(name, np.zeros(n, dtype=np.bool_))[idx]:
                    triggered_restriction.append(name)

        is_mandatory_sell = len(triggered_mandatory_sell) > 0
        has_weighted_sell = any(sell_weights.get(name, 0) > 0 for name in triggered_sell)
        weighted_sell_count = sum(1 for name in triggered_sell if sell_weights.get(name, 0) > 0)
        is_buy_restricted = len(triggered_restriction) > 0
        in_position = trade_history and trade_history[-1][1] == 'B'

        is_sell_by_count = (len(triggered_sell) >= sell_trigger_count
                           and in_position
                           and weighted_sell_count >= 1
                           and sell_score > buy_score)

        candidate = None
        confidence = 0
        sell_reason_type = ''

        if is_mandatory_sell and in_position:
            candidate = 'S'
            confidence = 1.0
            sell_reason_type = '必卖'
        elif is_sell_by_count and in_position:
            candidate = 'S'
            total = buy_score + sell_score
            confidence = sell_score / total if total > 0 else 0.8
            sell_reason_type = f'{len(triggered_sell)}条规则'
        elif buy_score > 0 and buy_score > sell_score and buy_weighted_combo_gate_ok(triggered_buy):
            candidate = 'B'
            total = buy_score + sell_score
            confidence = buy_score / total if total > 0 else 0

        if candidate == 'B' and is_buy_restricted:
            candidate = None

        if candidate is not None:
            if idx < ma_period or pd.isna(ma[idx]) or pd.isna(ma[idx - 1]):
                candidate = None
            else:
                trend_down = (ma[idx] < ma[idx - 1]) and (close[idx] < ma[idx])
                if candidate == 'B' and trend_down:
                    candidate = None

        if candidate is not None and not is_mandatory_sell and not is_sell_by_count:
            dominant_score = buy_score if candidate == 'B' else sell_score
            if dominant_score < min_score_threshold:
                candidate = None

        if candidate is not None and not is_mandatory_sell and not is_sell_by_count:
            if buy_details is not None and sell_details is not None:
                if candidate == 'B' and triggered_buy:
                    weighted_wr = 0.0
                    weighted_pf = 0.0
                    total_w = 0.0
                    for name in triggered_buy:
                        w = buy_weights.get(name, 0)
                        detail = buy_details.get(name, {})
                        weighted_wr += w * detail.get('win_rate', 0)
                        weighted_pf += w * detail.get('profit_factor', 0)
                        total_w += w
                    avg_wr = weighted_wr / total_w if total_w > 0 else 0
                    avg_pf = weighted_pf / total_w if total_w > 0 else 0
                    if avg_wr < min_triggered_win_rate and avg_pf < 1.0:
                        candidate = None
                elif candidate == 'S' and triggered_sell:
                    weighted_wr = 0.0
                    weighted_pf = 0.0
                    total_w = 0.0
                    for name in triggered_sell:
                        w = sell_weights.get(name, 0)
                        detail = sell_details.get(name, {})
                        weighted_wr += w * detail.get('win_rate', 0)
                        weighted_pf += w * detail.get('profit_factor', 0)
                        total_w += w
                    avg_wr = weighted_wr / total_w if total_w > 0 else 0
                    avg_pf = weighted_pf / total_w if total_w > 0 else 0
                    if avg_wr < min_triggered_win_rate and avg_pf < 1.0:
                        candidate = None

        if candidate is not None and trade_history:
            last_idx, last_signal = trade_history[-1]
            if not is_mandatory_sell and not is_sell_by_count:
                if idx - last_idx < min_interval:
                    candidate = None
                elif last_signal == candidate:
                    candidate = None
            else:
                if last_signal == 'B' and candidate == 'S':
                    pass
                elif idx - last_idx < min_interval:
                    candidate = None

        if candidate == 'B' and upper_shadow_vetoes_buy_signal(df, idx):
            candidate = None

        if candidate == 'B' and breakthrough_prior_high_vetoes_buy(df, idx):
            candidate = None

        reasons = {
            'buy_triggered': triggered_buy,
            'sell_triggered': triggered_sell,
            'buy_score': round(buy_score, 4),
            'sell_score': round(sell_score, 4),
            'mandatory_sell_triggered': triggered_mandatory_sell,
            'buy_restriction_triggered': triggered_restriction,
        }

        if candidate and confidence >= confidence_medium:
            level = 'strong' if confidence >= confidence_strong else 'medium'
            if candidate == 'B':
                if pending_buy is None:
                    pending_meta = {}
                    if '看涨K线形态' in reasons.get('buy_triggered', []):
                        pending_meta['bull_pattern_defer'] = True
                        if not _dwe_signal_day_volume_effective(df, idx):
                            pending_meta['weak_vol_bull_pattern'] = True
                    pending_buy = (idx, reasons, confidence, level, pending_meta)
            else:
                trade_history.append((idx, candidate))
                signals.append((idx, candidate))
                reasons_list.append({**reasons, 'final_signal': candidate,
                                     'confidence': round(confidence, 3), 'level': level,
                                     'sell_reason_type': sell_reason_type})

    return signals, reasons_list


def calculate_weights(stock_data, max_depth=200, step=10,
                      window=5, decay_factor=3.0,
                      min_hold=3, max_hold=20,
                      stop_loss_pct=0.05, take_profit_pct=0.10):
    df = stock_data.copy()
    if isinstance(df, pd.DataFrame):
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

    buy_rules, sell_rules = get_all_rules()

    opt_result = optimize_weights_v2(
        df, buy_rules, sell_rules,
        max_depth=max_depth, step=step,
        window=window, decay_factor=decay_factor,
        min_hold=min_hold, max_hold=max_hold,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
    )

    raw_buy_weights = dict(opt_result['buy_weights'])
    raw_sell_weights = dict(opt_result['sell_weights'])
    buy_stats = opt_result['all_buy_stats']
    sell_stats = opt_result['all_sell_stats']

    penalized_buy_weights = {}
    penalized_sell_weights = {}
    buy_details = {}
    sell_details = {}

    adjusted_buy_quality = {}
    for name, raw_w in raw_buy_weights.items():
        stats = buy_stats.get(name, {})
        win_rate = stats.get('win_rate', 0)
        avg_return = stats.get('avg_return', 0)
        profit_factor = stats.get('profit_factor', 0)
        count = stats.get('count', 0)
        quality_sum = stats.get('quality_sum', 0)
        mult, _ = _graduated_penalty(win_rate, avg_return, profit_factor, count)
        adjusted_buy_quality[name] = quality_sum * mult

    total_adjusted_buy_q = sum(adjusted_buy_quality.values())
    for name, raw_w in raw_buy_weights.items():
        stats = buy_stats.get(name, {})
        win_rate = stats.get('win_rate', 0)
        avg_return = stats.get('avg_return', 0)
        count = stats.get('count', 0)
        profit_factor = stats.get('profit_factor', 0)
        avg_hold = stats.get('avg_hold', 0)
        quality_sum = stats.get('quality_sum', 0)
        adjusted_q = adjusted_buy_quality.get(name, 0)
        mult, penalty_reason = _graduated_penalty(win_rate, avg_return, profit_factor, count)
        if total_adjusted_buy_q > 0:
            final_w = adjusted_q / total_adjusted_buy_q
        else:
            final_w = 0.0
        penalized = mult < 0.95
        level = _classify_level(final_w, raw_buy_weights)
        buy_details[name] = {
            'raw_weight': round(raw_w, 4),
            'final_weight': round(final_w, 4),
            'win_rate': round(win_rate, 4),
            'avg_return': round(avg_return, 4),
            'count': count,
            'profit_factor': round(profit_factor, 4),
            'avg_hold': round(avg_hold, 1),
            'quality_sum': round(quality_sum, 4),
            'adjusted_quality': round(adjusted_q, 4),
            'penalty_multiplier': round(mult, 3),
            'penalized': penalized,
            'penalty_reason': penalty_reason,
            'level': level,
        }
        penalized_buy_weights[name] = round(final_w, 4)

    adjusted_sell_quality = {}
    for name, raw_w in raw_sell_weights.items():
        stats = sell_stats.get(name, {})
        win_rate = stats.get('win_rate', 0)
        avg_return = stats.get('avg_return', 0)
        profit_factor = stats.get('profit_factor', 0)
        count = stats.get('count', 0)
        quality_sum = stats.get('quality_sum', 0)
        mult, _ = _graduated_penalty(win_rate, avg_return, profit_factor, count)
        adjusted_sell_quality[name] = quality_sum * mult

    total_adjusted_sell_q = sum(adjusted_sell_quality.values())
    for name, raw_w in raw_sell_weights.items():
        stats = sell_stats.get(name, {})
        win_rate = stats.get('win_rate', 0)
        avg_return = stats.get('avg_return', 0)
        count = stats.get('count', 0)
        profit_factor = stats.get('profit_factor', 0)
        avg_hold = stats.get('avg_hold', 0)
        quality_sum = stats.get('quality_sum', 0)
        adjusted_q = adjusted_sell_quality.get(name, 0)
        mult, penalty_reason = _graduated_penalty(win_rate, avg_return, profit_factor, count)
        if total_adjusted_sell_q > 0:
            final_w = adjusted_q / total_adjusted_sell_q
        else:
            final_w = 0.0
        penalized = mult < 0.95
        level = _classify_level(final_w, raw_sell_weights)
        sell_details[name] = {
            'raw_weight': round(raw_w, 4),
            'final_weight': round(final_w, 4),
            'win_rate': round(win_rate, 4),
            'avg_return': round(avg_return, 4),
            'count': count,
            'profit_factor': round(profit_factor, 4),
            'avg_hold': round(avg_hold, 1),
            'quality_sum': round(quality_sum, 4),
            'adjusted_quality': round(adjusted_q, 4),
            'penalty_multiplier': round(mult, 3),
            'penalized': penalized,
            'penalty_reason': penalty_reason,
            'level': level,
        }
        penalized_sell_weights[name] = round(final_w, 4)

    total_bw = sum(penalized_buy_weights.values())
    total_sw = sum(penalized_sell_weights.values())
    if total_bw > 0:
        penalized_buy_weights = {n: round(w / total_bw, 4) for n, w in penalized_buy_weights.items()}
    if total_sw > 0:
        penalized_sell_weights = {n: round(w / total_sw, 4) for n, w in penalized_sell_weights.items()}

    for name in buy_details:
        buy_details[name]['normalized_weight'] = penalized_buy_weights.get(name, 0)
    for name in sell_details:
        sell_details[name]['normalized_weight'] = penalized_sell_weights.get(name, 0)

    return {
        'buy_weights': penalized_buy_weights,
        'sell_weights': penalized_sell_weights,
        'buy_details': buy_details,
        'sell_details': sell_details,
        'buy_rules': buy_rules,
        'sell_rules': sell_rules,
        'depth_used': opt_result['depth_used'],
        'found_strict': True,
    }


def _classify_level(weight, all_weights):
    positive_weights = [w for w in all_weights.values() if w > 0]
    if not positive_weights or weight <= 0:
        return '无效'
    max_w = max(positive_weights)
    if max_w == 0:
        return '无效'
    ratio = weight / max_w
    if ratio >= 0.8:
        return '核心'
    elif ratio >= 0.5:
        return '重要'
    elif ratio >= 0.2:
        return '辅助'
    else:
        return '弱'


def run_system(stock_data, max_depth=200, step=10,
               window=5, decay_factor=3.0,
               min_hold=3, max_hold=20,
               stop_loss_pct=0.05, take_profit_pct=0.10):
    df = stock_data.copy()
    if isinstance(df, pd.DataFrame):
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

    buy_rules, sell_rules = get_all_rules()
    _, _, mandatory_sell_rules, buy_restriction_rules, _ = get_all_rules_extended()

    print(f"规则库加载完成: 买入规则{len(buy_rules)}个, 卖出规则{len(sell_rules)}个, "
          f"必卖规则{len(mandatory_sell_rules)}个, 买入限制{len(buy_restriction_rules)}个")
    print(f"开始V2权重优化 (max_depth={max_depth}, step={step})...")

    weight_result = calculate_weights(
        df, max_depth=max_depth, step=step,
        window=window, decay_factor=decay_factor,
        min_hold=min_hold, max_hold=max_hold,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
    )

    final_buy_weights = weight_result['buy_weights']
    final_sell_weights = weight_result['sell_weights']
    depth_used = weight_result['depth_used']

    print(f"\n权重优化完成! 使用递归深度: {depth_used}")
    print(f"\n--- 最终买入规则及权重 ---")
    for name, w in sorted(final_buy_weights.items(), key=lambda x: -x[1]):
        detail = weight_result['buy_details'].get(name, {})
        print(f"  {name}: 权重={w:.4f}, 胜率={detail.get('win_rate',0):.2%}, "
              f"均收益={detail.get('avg_return',0):.2%}, 盈亏比={detail.get('profit_factor',0):.2f}")

    print(f"\n--- 最终卖出规则及权重 ---")
    for name, w in sorted(final_sell_weights.items(), key=lambda x: -x[1]):
        detail = weight_result['sell_details'].get(name, {})
        print(f"  {name}: 权重={w:.4f}, 胜率={detail.get('win_rate',0):.2%}, "
              f"均收益={detail.get('avg_return',0):.2%}, 盈亏比={detail.get('profit_factor',0):.2f}")

    print(f"\n开始生成信号...")
    signals, reasons = generate_signals_with_weights(
        df, buy_rules, final_buy_weights,
        sell_rules, final_sell_weights,
        buy_details=weight_result.get('buy_details'),
        sell_details=weight_result.get('sell_details'),
        mandatory_sell_rules=mandatory_sell_rules,
        buy_restriction_rules=buy_restriction_rules,
        sell_trigger_count=2,
    )

    df['_signal'] = ''
    df['_signal_score'] = 0.0
    df['_buy_score'] = 0.0
    df['_sell_score'] = 0.0

    for i, (sig_idx, sig_type) in enumerate(signals):
        df.iloc[sig_idx, df.columns.get_loc('_signal')] = sig_type
        if i < len(reasons):
            r = reasons[i]
            df.iloc[sig_idx, df.columns.get_loc('_signal_score')] = r.get(
                'buy_score' if sig_type == 'B' else 'sell_score', 0)
            df.iloc[sig_idx, df.columns.get_loc('_buy_score')] = r.get('buy_score', 0)
            df.iloc[sig_idx, df.columns.get_loc('_sell_score')] = r.get('sell_score', 0)

    buy_signals = [(s[0], s[1], reasons[i] if i < len(reasons) else {})
                   for i, s in enumerate(signals) if s[1] == 'B']
    sell_signals = [(s[0], s[1], reasons[i] if i < len(reasons) else {})
                    for i, s in enumerate(signals) if s[1] == 'S']

    paired_signals = []
    all_b = [s for s in buy_signals]
    all_s = [s for s in sell_signals]

    bi, si = 0, 0
    while bi < len(all_b) and si < len(all_s):
        b_idx, _, b_r = all_b[bi]
        s_idx, _, s_r = all_s[si]
        if b_idx < s_idx:
            paired_signals.append({
                'buy_date': str(df.iloc[b_idx]['date']) if 'date' in df.columns else str(b_idx),
                'sell_date': str(df.iloc[s_idx]['date']) if 'date' in df.columns else str(s_idx),
                'buy_price': float(df.iloc[b_idx]['close']),
                'sell_price': float(df.iloc[s_idx]['close']),
                'return_pct': round((float(df.iloc[s_idx]['close']) - float(df.iloc[b_idx]['close'])) /
                                    float(df.iloc[b_idx]['close']) * 100, 2),
                'buy_reasons': b_r.get('buy_triggered', []),
                'sell_reasons': s_r.get('sell_triggered', []),
            })
            bi += 1
            si += 1
        elif s_idx < b_idx:
            si += 1
        else:
            si += 1

    opt_result = {
        'buy_weights': final_buy_weights,
        'sell_weights': final_sell_weights,
        'depth_used': depth_used,
        'all_buy_stats': {n: {'win_rate': d.get('win_rate', 0), 'avg_return': d.get('avg_return', 0),
                               'count': d.get('count', 0), 'profit_factor': d.get('profit_factor', 0),
                               'avg_hold': d.get('avg_hold', 0)}
                          for n, d in weight_result.get('buy_details', {}).items()},
        'all_sell_stats': {n: {'win_rate': d.get('win_rate', 0), 'avg_return': d.get('avg_return', 0),
                                'count': d.get('count', 0), 'profit_factor': d.get('profit_factor', 0),
                                'avg_hold': d.get('avg_hold', 0)}
                           for n, d in weight_result.get('sell_details', {}).items()},
    }

    result = {
        'df': df,
        'signals': signals,
        'reasons': reasons,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'paired_signals': paired_signals,
        'optimization': opt_result,
        'summary': {
            'total_signals': len(signals),
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'paired_count': len(paired_signals),
            'depth_used': depth_used,
            'active_buy_rules': list(buy_rules.keys()),
            'active_sell_rules': list(sell_rules.keys()),
            'buy_weights': final_buy_weights,
            'sell_weights': final_sell_weights,
        }
    }

    print(f"\n信号生成完成!")
    print(f"  总信号数: {len(signals)}")
    print(f"  买入信号: {len(buy_signals)}")
    print(f"  卖出信号: {len(sell_signals)}")
    print(f"  B-S配对: {len(paired_signals)}对")

    if paired_signals:
        profits = [p['return_pct'] for p in paired_signals]
        wins = [p for p in profits if p > 0]
        print(f"  配对胜率: {len(wins)/len(profits):.2%}")
        print(f"  平均收益率: {np.mean(profits):.2%}")
        print(f"  总收益率: {sum(profits):.2f}%")

    return result


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("动态权重 B/S 信号系统 - 测试运行")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_path = os.path.join(data_dir, '000001_daily.csv')

    if os.path.exists(csv_path):
        print(f"\n从本地CSV加载数据: {csv_path}")
        df = pd.read_csv(csv_path)
        if 'date' not in df.columns and df.shape[1] >= 6:
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume'] + list(df.columns[6:])
    else:
        print("\n使用模拟数据进行测试...")
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        base_price = 15.0
        trend = np.linspace(0, 0.8, n)
        noise = np.random.normal(0, 0.02, n)
        returns = 0.0005 * np.ones(n) + trend * 0.002 + noise
        prices = base_price * np.cumprod(1 + returns)
        closes = prices
        opens = closes * (1 + np.random.uniform(-0.015, 0.015, n))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.012, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.012, n)))
        volumes = (1000000 + np.random.randint(-300000, 3000000, n)).astype(float)

        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': np.round(opens, 2),
            'high': np.round(highs, 2),
            'low': np.round(lows, 2),
            'close': np.round(closes, 2),
            'volume': volumes,
        })

    print(f"数据行数: {len(df)}")
    print(f"日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    system_result = run_system(
        df,
        max_depth=200,
        step=10,
        win_rate_threshold=0.90,
        return_threshold=0.20
    )

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
