import pandas as pd
import numpy as np
import talib
from scipy import stats


def ma_cross_signal(df, short=5, long=20):
    short_ma = talib.MA(df['close'], timeperiod=short)
    long_ma = talib.MA(df['close'], timeperiod=long)
    buy = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    sell = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    return buy.fillna(False), sell.fillna(False)


def macd_cross_signal(df, fast=12, slow=26, signal_period=9):
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal_period)
    buy = (macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))
    sell = (macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))
    return buy.fillna(False), sell.fillna(False)


def rsi_ob_os_signal(df, period=14, oversold=30, overbought=70):
    rsi = talib.RSI(df['close'], timeperiod=period)
    buy = (rsi > oversold) & (rsi.shift(1) <= oversold)
    sell = (rsi < overbought) & (rsi.shift(1) >= overbought)
    return buy.fillna(False), sell.fillna(False)


def kdj_cross_signal(df):
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                fastk_period=9, slowk_period=3, slowd_period=3)
    buy = (slowk > slowd) & (slowk.shift(1) <= slowd.shift(1)) & (slowk < 20)
    sell = (slowk < slowd) & (slowk.shift(1) >= slowd.shift(1)) & (slowk > 80)
    return buy.fillna(False), sell.fillna(False)


def _find_local_extremes(series, window=5):
    lows = []
    highs = []
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i - window:i + window + 1].min():
            lows.append(i)
        if series.iloc[i] == series.iloc[i - window:i + window + 1].max():
            highs.append(i)
    return lows, highs


def macd_divergence_signal(df, lookback=60):
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    close = df['close'].values
    hist = macdhist.values
    n = len(close)
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)

    low_idx, high_idx = _find_local_extremes(pd.Series(close), window=5)

    if len(low_idx) >= 2:
        recent_lows = [i for i in low_idx if i >= n - lookback]
        older_lows = [i for i in low_idx if i < n - lookback and i >= n - lookback * 2]
        if len(recent_lows) >= 1 and len(older_lows) >= 1:
            recent_low_price = close[recent_lows[-1]]
            older_low_price = close[older_lows[-1]]
            recent_low_hist = hist[recent_lows[-1]]
            older_low_hist = hist[older_lows[-1]]
            if recent_low_price < older_low_price and recent_low_hist > older_low_hist and not np.isnan(recent_low_hist) and not np.isnan(older_low_hist):
                buy.iloc[recent_lows[-1]] = True

    if len(high_idx) >= 2:
        recent_highs = [i for i in high_idx if i >= n - lookback]
        older_highs = [i for i in high_idx if i < n - lookback and i >= n - lookback * 2]
        if len(recent_highs) >= 1 and len(older_highs) >= 1:
            recent_high_price = close[recent_highs[-1]]
            older_high_price = close[older_highs[-1]]
            recent_high_hist = hist[recent_highs[-1]]
            older_high_hist = hist[older_highs[-1]]
            if recent_high_price > older_high_price and recent_high_hist < older_high_hist and not np.isnan(recent_high_hist) and not np.isnan(older_high_hist):
                sell.iloc[recent_highs[-1]] = True

    return buy, sell


def candle_pattern_signal(df):
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)

    bullish_patterns = {
        'CDLENGULFING': '看涨吞没',
        'CDLMORNINGSTAR': '早晨之星',
        'CDLHAMMER': '锤头线',
        'CDLPIERCING': '刺透形态',
        'CDLMORNINGDOJISTAR': '十字早晨之星',
        'CDLINVERTEDHAMMER': '倒锤子线',
        'CDLBELTHOLD': '大阳烛',
        'CDL3WHITESOLDIERS': '三白兵',
    }

    bearish_patterns = {
        'CDLENGULFING': '看跌吞没',
        'CDLEVENINGSTAR': '黄昏之星',
        'CDLHANGINGMAN': '上吊线',
        'CDLDARKCLOUDCOVER': '乌云盖顶',
        'CDLEVENINGDOJISTAR': '十字黄昏之星',
        'CDLSHOOTINGSTAR': '射击之星',
        'CDL3BLACKCROWS': '三只乌鸦',
        'CDLABANDONEDBABY': '弃婴(顶)',
    }

    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)

    for pname in bullish_patterns:
        func = getattr(talib, pname, None)
        if func is None:
            continue
        try:
            result = func(o, h, l, c)
            buy = buy | (pd.Series(result, index=df.index) == 100)
        except Exception:
            pass

    for pname in bearish_patterns:
        func = getattr(talib, pname, None)
        if func is None:
            continue
        try:
            result = func(o, h, l, c)
            sell = sell | (pd.Series(result, index=df.index) == -100)
        except Exception:
            pass

    return buy, sell


def double_bottom_signal(df, lookback=60, tolerance=0.03):
    close = df['close'].values
    n = len(close)
    buy = pd.Series(False, index=df.index)

    low_idx, _ = _find_local_extremes(pd.Series(close), window=5)

    for i in range(len(low_idx) - 1):
        idx1 = low_idx[i]
        idx2 = low_idx[i + 1]
        if idx2 - idx1 > lookback or idx2 - idx1 < 5:
            continue
        price1 = close[idx1]
        price2 = close[idx2]
        if abs(price1 - price2) / price1 > tolerance:
            continue

        neck_start = min(idx1, idx2)
        neck_end = max(idx1, idx2)
        neckline = close[neck_start:neck_end + 1].max()

        for j in range(idx2 + 1, min(n, idx2 + lookback // 2)):
            if close[j] > neckline:
                buy.iloc[j] = True
                break

    return buy, pd.Series(False, index=df.index)


def double_top_signal(df, lookback=60, tolerance=0.03):
    close = df['close'].values
    n = len(close)
    sell = pd.Series(False, index=df.index)

    _, high_idx = _find_local_extremes(pd.Series(close), window=5)

    for i in range(len(high_idx) - 1):
        idx1 = high_idx[i]
        idx2 = high_idx[i + 1]
        if idx2 - idx1 > lookback or idx2 - idx1 < 5:
            continue
        price1 = close[idx1]
        price2 = close[idx2]
        if abs(price1 - price2) / price1 > tolerance:
            continue

        neck_start = min(idx1, idx2)
        neck_end = max(idx1, idx2)
        neckline = close[neck_start:neck_end + 1].min()

        for j in range(idx2 + 1, min(n, idx2 + lookback // 2)):
            if close[j] < neckline:
                sell.iloc[j] = True
                break

    return pd.Series(False, index=df.index), sell


def head_shoulders_signal(df, lookback=100):
    close = df['close'].values
    n = len(close)
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)

    low_idx, high_idx = _find_local_extremes(pd.Series(close), window=5)

    def find_head_shoulder_bottom(extremes, direction='bottom'):
        signals = []
        for i in range(len(extremes) - 4):
            left_shoulder = extremes[i]
            head = extremes[i + 2]
            right_shoulder = extremes[i + 4]

            if right_shoulder - left_shoulder > lookback:
                continue
            if head - left_shoulder < 10 or right_shoulder - head < 10:
                continue

            ls_price = close[left_shoulder]
            h_price = close[head]
            rs_price = close[right_shoulder]

            if direction == 'bottom':
                if h_price < ls_price and h_price < rs_price:
                    diff = abs(ls_price - rs_price) / max(ls_price, rs_price)
                    if diff < 0.05:
                        neck_region = close[left_shoulder:right_shoulder + 1]
                        neckline = neck_region.max()
                        for j in range(right_shoulder + 1, min(n, right_shoulder + 20)):
                            if close[j] > neckline:
                                signals.append(j)
                                break
            else:
                if h_price > ls_price and h_price > rs_price:
                    diff = abs(ls_price - rs_price) / max(ls_price, rs_price)
                    if diff < 0.05:
                        neck_region = close[left_shoulder:right_shoulder + 1]
                        neckline = neck_region.min()
                        for j in range(right_shoulder + 1, min(n, right_shoulder + 20)):
                            if close[j] < neckline:
                                signals.append(j)
                                break
        return signals

    bottom_signals = find_head_shoulder_bottom(low_idx, 'bottom')
    top_signals = find_head_shoulder_bottom(high_idx, 'top')

    for s in bottom_signals:
        buy.iloc[s] = True
    for s in top_signals:
        sell.iloc[s] = True

    return buy, sell


def volume_price_trend_signal(df, volume_factor=1.2):
    vol = df['volume'].values.astype(float)
    close = df['close'].values.astype(float)
    n = len(close)
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)

    vol_ma = pd.Series(vol).rolling(window=20).mean().values

    for i in range(1, n):
        if np.isnan(vol_ma[i]):
            continue
        price_up = close[i] > close[i - 1]
        price_down = close[i] < close[i - 1]
        vol_surge = vol[i] > vol_ma[i] * volume_factor

        if price_up and vol_surge:
            buy.iloc[i] = True
        elif price_down and vol_surge:
            sell.iloc[i] = True

    return buy, sell


def volume_breakout_signal(df, lookback=20, volume_mult=1.5):
    close = df['close'].values.astype(float)
    vol = df['volume'].values.astype(float)
    n = len(close)
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)

    for i in range(lookback, n):
        period_close = close[i - lookback:i]
        period_vol = vol[i - lookback:i]
        highest = period_close.max()
        lowest = period_close.min()
        avg_vol = period_vol.mean()

        if avg_vol == 0:
            continue

        if close[i] > highest and vol[i] > avg_vol * volume_mult:
            buy.iloc[i] = True
        elif close[i] < lowest and vol[i] > avg_vol * volume_mult:
            sell.iloc[i] = True

    return buy, sell


def extreme_volume_price_signal(df, lookback=60):
    close = df['close'].values.astype(float)
    vol = df['volume'].values.astype(float)
    n = len(close)
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)

    for i in range(lookback, n):
        period_close = close[i - lookback:i]
        period_vol = vol[i - lookback:i]

        lowest_price = period_close.min()
        highest_price = period_close.max()
        lowest_vol = period_vol.min()
        highest_vol = period_vol.max()

        price_near_low = close[i] <= lowest_price * 1.02
        price_near_high = close[i] >= highest_price * 0.98
        vol_is_min = vol[i] <= lowest_vol * 1.1
        vol_is_max = vol[i] >= highest_vol * 0.9

        if price_near_low and vol_is_min:
            buy.iloc[i] = True
        elif price_near_high and vol_is_max:
            sell.iloc[i] = True

    return buy, sell


def trendline_support_resistance_signal(df, lookback=30, tolerance=0.02):
    close = df['close'].values.astype(float)
    n = len(close)
    buy = pd.Series(False, index=df.index)
    sell = pd.Series(False, index=df.index)

    for i in range(lookback, n):
        x = np.arange(lookback).reshape(-1, 1)
        y = close[i - lookback:i]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)

        trend_at_current = slope * lookback + intercept
        if trend_at_current == 0:
            continue

        deviation = (close[i] - trend_at_current) / trend_at_current

        if slope > 0 and abs(deviation) < tolerance and close[i] > trend_at_current:
            buy.iloc[i] = True
        elif slope < 0 and abs(deviation) < tolerance and close[i] < trend_at_current:
            sell.iloc[i] = True

    return buy, sell


def generate_all_signals(df):
    df = df.copy()
    df['buy_signal'] = False
    df['sell_signal'] = False

    rule_results = []

    ma_buy, ma_sell = ma_cross_signal(df)
    rule_results.append(('MA金叉/死叉', ma_buy, ma_sell))

    macd_buy, macd_sell = macd_cross_signal(df)
    rule_results.append(('MACD金叉/死叉', macd_buy, macd_sell))

    rsi_buy, rsi_sell = rsi_ob_os_signal(df)
    rule_results.append(('RSI超买超卖', rsi_buy, rsi_sell))

    kdj_buy, kdj_sell = kdj_cross_signal(df)
    rule_results.append(('KDJ金叉死叉', kdj_buy, kdj_sell))

    div_buy, div_sell = macd_divergence_signal(df)
    rule_results.append(('MACD背离', div_buy, div_sell))

    cp_buy, cp_sell = candle_pattern_signal(df)
    rule_results.append(('K线形态', cp_buy, cp_sell))

    db_buy, db_sell = double_bottom_signal(df)
    rule_results.append(('W底/M顶', db_buy, db_sell))

    hs_buy, hs_sell = head_shoulders_signal(df)
    rule_results.append(('头肩形态', hs_buy, hs_sell))

    vpt_buy, vpt_sell = volume_price_trend_signal(df)
    rule_results.append(('量价趋势', vpt_buy, vpt_sell))

    vb_buy, vb_sell = volume_breakout_signal(df)
    rule_results.append(('放量突破', vb_buy, vb_sell))

    evp_buy, evp_sell = extreme_volume_price_signal(df)
    rule_results.append(('地量地价/天量天价', evp_buy, evp_sell))

    tl_buy, tl_sell = trendline_support_resistance_signal(df)
    rule_results.append(('趋势线突破', tl_buy, tl_sell))

    total_buy = pd.Series(False, index=df.index)
    total_sell = pd.Series(False, index=df.index)

    for name, b, s in rule_results:
        total_buy = total_buy | b
        total_sell = total_sell | s

    conflict = total_buy & total_sell
    total_buy = total_buy & ~conflict
    total_sell = total_sell & ~conflict

    df['buy_signal'] = total_buy
    df['sell_signal'] = total_sell

    return df, rule_results


if __name__ == "__main__":
    import os
    print("=" * 60)
    print("股票交易信号生成模块 - 测试运行")
    print("=" * 60)

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
        returns = np.random.normal(0.001, 0.02, n)
        prices = base_price * np.cumprod(1 + returns)
        closes = prices
        opens = closes * (1 + np.random.uniform(-0.015, 0.015, n))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, n)))
        volumes = np.random.randint(1000000, 10000000, n).astype(float)

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

    result_df, all_rules = generate_all_signals(df)

    buy_count = result_df['buy_signal'].sum()
    sell_count = result_df['sell_signal'].sum()
    print(f"\n信号统计:")
    print(f"  买入信号总数: {buy_count} ({buy_count / len(df) * 100:.2f}%)")
    print(f"  卖出信号总数: {sell_count} ({sell_count / len(df) * 100:.2f}%)")

    print(f"\n各规则触发次数:")
    for name, b, s in all_rules:
        b_cnt = b.sum()
        s_cnt = s.sum()
        print(f"  {name}: 买入={b_cnt}, 卖出={s_cnt}")

    print(f"\n最近10个交易日信号摘要:")
    recent = result_df.tail(10)[['date', 'close', 'buy_signal', 'sell_signal']].copy()
    for _, row in recent.iterrows():
        sig_str = ""
        if row['buy_signal']:
            sig_str = "【买入】"
        elif row['sell_signal']:
            sig_str = "【卖出】"
        else:
            sig_str = "----"
        print(f"  {row['date']}  收盘:{row['close']:.2f}  {sig_str}")

    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'signal_output.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
