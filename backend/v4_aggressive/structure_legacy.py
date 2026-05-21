"""V4 改动前：W/M/V反顶 旧版固定窗口识别（用于 A/B 对比）。"""
import numpy as np


def detect_w_right_bottom_events_legacy(close, high, low, n):
    lows = []
    for i in range(8, n):
        if close[i] == close[max(0, i - 8) : min(n, i + 4)].min():
            lows.append(i)
    if len(lows) < 2:
        return []
    filtered = [lows[0]]
    for i in range(1, len(lows)):
        if lows[i] - filtered[-1] >= 5:
            filtered.append(lows[i])
    events = []
    for k in range(len(filtered) - 1):
        idx1, idx2 = filtered[k], filtered[k + 1]
        if idx2 - idx1 < 8 or idx2 - idx1 > 80:
            continue
        p1, p2 = close[idx1], close[idx2]
        if abs(p1 - p2) / max(p1, p2) > 0.05:
            continue
        between_high = float(np.max(high[idx1 : idx2 + 1]))
        if (between_high - max(p1, p2)) / max(p1, p2) < 0.03:
            continue
        neck = between_high
        stop_ref = float(min(low[idx1], low[idx2]))
        r_max = min(idx2 + 10, n - 1)
        for r in range(idx2 + 1, r_max + 1):
            if close[r] > close[idx2]:
                events.append({
                    'kind': 'W',
                    'entry': r,
                    'neck': neck,
                    'stop_ref': stop_ref,
                    'idx1': idx1,
                    'idx2': idx2,
                })
                break
    events.sort(key=lambda x: x['entry'])
    return _dedupe(events)


def is_m_top_break_legacy(df, idx) -> bool:
    if idx < 30:
        return False
    close = df['close'].iloc[: idx + 1].values.astype(float)
    high = df['high'].iloc[: idx + 1].values.astype(float) if 'high' in df.columns else close
    n = len(close)
    highs = []
    for i in range(8, n):
        if high[i] == high[max(0, i - 8) : min(n, i + 4)].max():
            highs.append(i)
    filtered = [highs[0]] if highs else []
    for i in range(1, len(highs)):
        if highs[i] - filtered[-1] >= 5:
            filtered.append(highs[i])
    for i in range(len(filtered) - 1):
        idx1, idx2 = filtered[i], filtered[i + 1]
        if idx2 - idx1 < 10 or idx2 - idx1 > 60:
            continue
        p1, p2 = high[idx1], high[idx2]
        if abs(p1 - p2) / max(p1, p2) > 0.02:
            continue
        between_low = close[idx1 : idx2 + 1].min()
        retrace = (max(p1, p2) - between_low) / max(p1, p2)
        if retrace < 0.03:
            continue
        top_price = max(p1, p2)
        for j in range(idx2 + 1, min(idx2 + 30, n)):
            if close[j] < top_price * 0.98 and j == idx:
                if j > 0 and close[j - 1] >= top_price * 0.98:
                    return True
                if j == idx2 + 1:
                    return True
    return False


def is_v_top_break_legacy(df, idx) -> bool:
    if idx < 20:
        return False
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    vol = df['volume'].values.astype(float) if 'volume' in df.columns else None
    n = len(close)
    lookback = min(20, idx)
    if lookback < 8:
        return False
    window_high = high[idx - lookback : idx]
    if window_high.size < 6:
        return False
    rel_peak = int(np.argmax(window_high))
    recent_high_idx = idx - lookback + rel_peak
    if recent_high_idx >= idx - 3 or recent_high_idx < idx - 10:
        return False
    high_price = float(high[recent_high_idx])
    lo = max(0, recent_high_idx - 2)
    hi = min(n, recent_high_idx + 3)
    if high_price + 1e-12 < float(np.max(high[lo:hi])):
        return False
    left_start = max(0, recent_high_idx - 10)
    left_low = float(close[left_start:recent_high_idx].min())
    if left_low <= 0:
        return False
    if (high_price - left_low) / left_low < 0.07:
        return False
    trail_low = float(low[recent_high_idx : idx + 1].min())
    if (high_price - trail_low) / high_price < 0.03:
        return False
    if close[idx] > high_price * 0.98:
        return False
    if close[idx] >= close[idx - 1]:
        return False
    if vol is not None:
        vol_ma5 = np.mean(vol[max(0, idx - 5) : idx])
        if vol_ma5 > 0 and vol[idx] < vol_ma5 * 0.8:
            return False
    return True


def detect_v_right_bottom_events_legacy(close, high, low, n, bounce_min=0.02, drop_min=0.05):
    """改动前：15 根内最低 + 左侧 5 根 high 瓶口 + 右侧 10 根。"""
    hi = high if high is not None else close
    used_low = set()
    events = []
    wloc = 15
    for low_idx in range(12, n - 12):
        if low_idx in used_low:
            continue
        lo = max(0, low_idx - wloc)
        seg = low[lo : low_idx + 2]
        if low_idx - lo != int(np.argmin(seg)):
            continue
        low_price = float(low[low_idx])
        i0 = max(0, low_idx - 5)
        left_ref = float(np.max(hi[i0:low_idx]))
        if left_ref <= 0 or (left_ref - low_price) / left_ref < drop_min:
            continue
        for r in range(low_idx + 1, min(low_idx + 11, n)):
            if (close[r] - low_price) / low_price < bounce_min:
                continue
            if close[r] <= close[r - 1]:
                continue
            events.append({
                'kind': 'V',
                'entry': r,
                'neck': left_ref,
                'stop_ref': low_price,
                'low_idx': low_idx,
            })
            used_low.add(low_idx)
            break
    events.sort(key=lambda x: x['entry'])
    return _dedupe(events)


def _dedupe(events, min_gap=5):
    out = []
    last_e = -10 ** 9
    for ev in events:
        if ev['entry'] - last_e < min_gap:
            continue
        out.append(ev)
        last_e = ev['entry']
    return out
