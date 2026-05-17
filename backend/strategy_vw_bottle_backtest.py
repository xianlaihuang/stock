"""
日线：W / V 结构「右侧底全仓 → 触及瓶口减半 → 突破失败清仓 / 突破则继续持有」回测。
用于与 engine_v2.analyze_signals_v2 的 B→S 配对收益做对比，不修改主引擎逻辑。
"""
import numpy as np
import pandas as pd


def _filtered_close_lows(close, n):
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
    return filtered


def detect_w_right_bottom_events(close, high, low, n):
    """
    W：两底 idx1、idx2 与颈线 neck（两底间收盘最高），右侧底 = idx2 之后首根 close > close[idx2] 的日 r。
    瓶口 = neck。止损参考 = min(low[idx1], low[idx2])。
    """
    events = []
    fl = _filtered_close_lows(close, n)
    for k in range(len(fl) - 1):
        idx1, idx2 = fl[k], fl[k + 1]
        if idx2 - idx1 < 8 or idx2 - idx1 > 80:
            continue
        p1, p2 = close[idx1], close[idx2]
        if abs(p1 - p2) / max(p1, p2) > 0.05:
            continue
        between_high = close[idx1 : idx2 + 1].max()
        if (between_high - max(p1, p2)) / max(p1, p2) < 0.03:
            continue
        neck = float(between_high)
        stop_ref = float(min(low[idx1], low[idx2]))
        r_max = min(idx2 + 10, n - 1)
        for r in range(idx2 + 1, r_max + 1):
            if close[r] > close[idx2]:
                events.append(
                    {
                        "kind": "W",
                        "entry": r,
                        "neck": neck,
                        "stop_ref": stop_ref,
                        "idx2": idx2,
                    }
                )
                break
    events.sort(key=lambda x: x["entry"])
    return _dedupe_by_entry(events)


def detect_v_right_bottom_events(close, high, low, n, bounce_min=0.02, drop_min=0.05):
    """
    V：low_idx 为过去约 15 根内最低价日；左侧瓶口 left_ref = high[low_idx-5:low_idx) 最高；
    右侧底 r = low_idx 之后首根满足反弹 bounce_min 且收阳。
    每个 low_idx 至多产生一笔。
    """
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
            events.append(
                {
                    "kind": "V",
                    "entry": r,
                    "neck": left_ref,
                    "stop_ref": low_price,
                    "low_idx": low_idx,
                }
            )
            used_low.add(low_idx)
            break
    events.sort(key=lambda x: x["entry"])
    return _dedupe_by_entry(events, min_gap=5)


def _dedupe_by_entry(events, min_gap=5):
    out = []
    last_e = -10**9
    for ev in events:
        if ev["entry"] - last_e < min_gap:
            continue
        out.append(ev)
        last_e = ev["entry"]
    return out


def simulate_trade(
    close,
    high,
    low,
    entry,
    neck,
    stop_ref,
    n,
    neck_touch_tol=0.998,
    break_eps=1.008,
    fail_below=0.992,
    max_post_half_bars=25,
    max_hold_after_break=60,
):
    """
    初始资金 1，entry 日收盘全仓买入；触及瓶口日收盘卖出一半仓位；
    未突破：若收盘跌回颈线 fail_below 以下或超时，剩余收盘清仓；
    已突破：继续持有直至收盘跌回 fail_below 以下或持仓超时。
    若日内 low 跌破硬止损 stop_ref，按 stop_ref 全部平仓。
    返回 (组合收益率, 入场索引, 出场索引, meta)。
    meta: half_day 减半日索引或 None, breakout 是否曾有效突破颈线, stopped 是否止损离场。
    """
    px0 = float(close[entry])
    if px0 <= 0 or entry >= n - 1:
        meta = {"half_day": None, "breakout": False, "stopped": False}
        return 0.0, entry, entry, meta

    shares = 1.0 / px0
    cash = 0.0
    half_done = False
    breakout = False
    half_day = None
    exit_d = entry
    stopped = False

    for d in range(entry + 1, n):
        exit_d = d
        c = float(close[d])
        h = float(high[d])
        lo = float(low[d])

        if lo <= float(stop_ref) * 0.9995:
            px = float(stop_ref)
            cash += shares * px
            shares = 0.0
            stopped = True
            break

        if not half_done:
            if h >= neck * neck_touch_tol:
                sell_q = shares * 0.5
                cash += sell_q * c
                shares -= sell_q
                half_done = True
                half_day = d
            continue

        if half_done and not breakout:
            if c > neck * break_eps:
                breakout = True
                continue
            if d - half_day > max_post_half_bars:
                cash += shares * c
                shares = 0.0
                break
            if c < neck * fail_below:
                cash += shares * c
                shares = 0.0
                break
            continue

        if breakout:
            if c < neck * fail_below:
                cash += shares * c
                shares = 0.0
                break
            if d - half_day >= max_hold_after_break:
                cash += shares * c
                shares = 0.0
                break

    if shares > 1e-12:
        cash += shares * float(close[n - 1])
        shares = 0.0
        exit_d = n - 1

    meta = {
        "half_day": half_day,
        "breakout": bool(breakout),
        "stopped": stopped,
    }
    return cash - 1.0, entry, exit_d, meta


def merge_w_v_events(w_events, v_events):
    all_e = sorted(w_events + v_events, key=lambda x: x["entry"])
    return _dedupe_by_entry(all_e, min_gap=5)


# 双针探底：须在 V/W 左侧底部附近，且左侧跌段总跌幅 >= 4%
DOUBLE_NEEDLE_MIN_TOTAL_DROP = 0.04
DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_AFTER = 4
DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_BEFORE = 1


def _drop_pct_from_ref_to_low(ref_high, bottom_low):
    if ref_high is None or bottom_low is None:
        return 0.0
    ref_high = float(ref_high)
    bottom_low = float(bottom_low)
    if ref_high <= 0:
        return 0.0
    return (ref_high - bottom_low) / ref_high


def iter_w_left_bottoms(close, high, low, n, min_drop=None):
    """W 结构左侧底 idx1：跌段为 idx1 前高至 idx1 低点。"""
    if min_drop is None:
        min_drop = DOUBLE_NEEDLE_MIN_TOTAL_DROP
    fl = _filtered_close_lows(close, n)
    hi = high if high is not None else close
    lo = low if low is not None else close
    for k in range(len(fl) - 1):
        idx1, idx2 = fl[k], fl[k + 1]
        if idx2 - idx1 < 8 or idx2 - idx1 > 80:
            continue
        p1, p2 = close[idx1], close[idx2]
        if abs(p1 - p2) / max(p1, p2) > 0.05:
            continue
        between_high = close[idx1 : idx2 + 1].max()
        if (between_high - max(p1, p2)) / max(p1, p2) < 0.03:
            continue
        ref_high = float(np.max(hi[max(0, idx1 - 20) : idx1 + 1]))
        bottom_low = float(lo[idx1])
        drop = _drop_pct_from_ref_to_low(ref_high, bottom_low)
        if drop < min_drop:
            continue
        yield {
            "kind": "W",
            "left_idx": int(idx1),
            "right_idx": int(idx2),
            "ref_high": ref_high,
            "drop_pct": drop,
        }


def iter_v_left_bottoms(close, high, low, n, min_drop=None, wloc=15):
    """V 结构左侧底 low_idx：跌段为低点前左侧高点至 V 底。"""
    if min_drop is None:
        min_drop = DOUBLE_NEEDLE_MIN_TOTAL_DROP
    hi = high if high is not None else close
    lo = low if low is not None else close
    used = set()
    for low_idx in range(12, n - 12):
        if low_idx in used:
            continue
        seg_lo = max(0, low_idx - wloc)
        seg = lo[seg_lo : low_idx + 2]
        if low_idx - seg_lo != int(np.argmin(seg)):
            continue
        i0 = max(0, low_idx - 5)
        ref_high = float(np.max(hi[i0:low_idx]))
        bottom_low = float(lo[low_idx])
        drop = _drop_pct_from_ref_to_low(ref_high, bottom_low)
        if drop < min_drop:
            continue
        used.add(low_idx)
        yield {
            "kind": "V",
            "left_idx": int(low_idx),
            "ref_high": ref_high,
            "drop_pct": drop,
        }


def signal_near_vw_left_bottom(close, high, low, n, signal_idx, min_drop=None):
    """信号日是否落在某次 V/W 左侧底部允许窗口内。"""
    signal_idx = int(signal_idx)
    for ctx in iter_w_left_bottoms(close, high, low, n, min_drop=min_drop):
        left = ctx["left_idx"]
        if left - DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_BEFORE <= signal_idx <= left + DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_AFTER:
            return ctx
    for ctx in iter_v_left_bottoms(close, high, low, n, min_drop=min_drop):
        left = ctx["left_idx"]
        if left - DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_BEFORE <= signal_idx <= left + DOUBLE_NEEDLE_LEFT_BOTTOM_MAX_AFTER:
            return ctx
    return None


# N 字形买入：须在 W/V 右侧背景下，前高取右侧起点至当日的最高价 high（非收盘）
N_SHAPE_MAX_SPAN_FROM_ENTRY = 60
N_SHAPE_NECK_TOUCH_TOL = 0.998
N_SHAPE_BREAK_PRIOR_HIGH_EPS = 1.001
N_SHAPE_PULLBACK_MAX_VS_PEAK = 0.98
N_SHAPE_PULLBACK_MIN_VS_PEAK = 0.88
N_SHAPE_PULLBACK_NEAR_NECK_UP = 1.025


def prior_high_high_right_to_day(high, entry, idx):
    """前高：V/W 右侧 entry 日至 idx 日（含）的最高价 high。"""
    entry, idx = int(entry), int(idx)
    if entry > idx or idx < 0:
        return None
    seg = high[entry : idx + 1]
    if len(seg) == 0:
        return None
    return float(np.max(seg))


def prior_high_high_before_signal(high, entry, idx):
    """突破参照前高：右侧起点至信号日前一日（含）的最高价 high。"""
    if idx <= entry:
        return None
    return prior_high_high_right_to_day(high, entry, idx - 1)


def find_vw_right_context_before(close, high, low, n, idx, max_span=None):
    """
    取 idx 之前最近一笔 W/V 右侧事件（entry < idx），且间隔不超过 max_span。
  返回 dict: kind, entry, neck, stop_ref, prior_high_to_day
    """
    if max_span is None:
        max_span = N_SHAPE_MAX_SPAN_FROM_ENTRY
    idx = int(idx)
    if idx < 15:
        return None
    w_e = detect_w_right_bottom_events(close, high, low, n)
    v_e = detect_v_right_bottom_events(close, high, low, n)
    best = None
    for ev in merge_w_v_events(w_e, v_e):
        e = int(ev["entry"])
        if e >= idx or idx - e > max_span:
            continue
        if best is None or e > best["entry"]:
            best = {
                "kind": ev["kind"],
                "entry": e,
                "neck": float(ev["neck"]),
                "stop_ref": float(ev["stop_ref"]),
            }
    if best is None:
        return None
    ph = prior_high_high_right_to_day(high, best["entry"], idx)
    if ph is None or ph <= 0:
        return None
    best["prior_high_to_day"] = ph
    best["prior_high_before"] = prior_high_high_before_signal(high, best["entry"], idx)
    return best


def neck_touched_between_bars(high, low, entry, end_idx, neck, tol=None):
    """entry+1 .. end_idx-1 内曾触及/贴近瓶口（瓶颈）。"""
    if tol is None:
        tol = N_SHAPE_NECK_TOUCH_TOL
    entry, end_idx = int(entry), int(end_idx)
    neck = float(neck)
    if neck <= 0 or end_idx <= entry + 1:
        return False
    for j in range(entry + 1, end_idx):
        if float(high[j]) >= neck * tol:
            return True
        lj = float(low[j])
        if lj <= neck * N_SHAPE_PULLBACK_NEAR_NECK_UP and lj >= neck * 0.95:
            return True
    return False


def n_shape_vw_bottle_buy_signal(close, high, low, vol, idx, open_=None, vol_mult=1.12):
    """
    N 字形买入（不可脱离 W/V 右侧瓶口单独成立）：
    - 须在 W 右侧或 V 反右侧背景下（最近一笔右侧 entry 至信号日不超过 max_span）；
    - 前高 = 右侧 entry 至信号日（含）的最高价 high 最大值（非收盘）；
    - entry 至信号日前须曾触及瓶口（颈线/左侧瓶口）；
    - N 字：右侧区间内前高—回踩瓶口区—再走强；
    - W：当日 high 突破「右侧至昨日」的前高；V：瓶口回踩后收阳走强。
    """
    idx = int(idx)
    n = len(close)
    if idx < 40 or vol is None:
        return False
    if open_ is None:
        open_ = close
    ctx = find_vw_right_context_before(close, high, low, n, idx)
    if ctx is None:
        return False
    entry = ctx["entry"]
    neck = ctx["neck"]
    kind = ctx["kind"]
    prior_before = ctx.get("prior_high_before")
    if prior_before is None or prior_before <= 0:
        return False
    if not neck_touched_between_bars(high, low, entry, idx, neck):
        return False
    if idx <= entry + 2:
        return False
    # 前高腿：右侧起点至信号日前一日的 high 峰值（与 prior 区间一致，不用固定 28～16 窗）
    peak_seg = high[entry:idx]
    if len(peak_seg) < 2:
        return False
    rel_peak = int(np.argmax(peak_seg))
    leg_peak_idx = entry + rel_peak
    leg_peak = float(peak_seg[rel_peak])
    if leg_peak <= 0:
        return False
    if leg_peak_idx >= idx - 1:
        return False
    seg_low = float(np.min(low[leg_peak_idx + 1 : idx]))
    if seg_low > leg_peak * N_SHAPE_PULLBACK_MAX_VS_PEAK:
        return False
    if seg_low < leg_peak * N_SHAPE_PULLBACK_MIN_VS_PEAK:
        return False
    if seg_low > neck * N_SHAPE_PULLBACK_NEAR_NECK_UP:
        return False
    o = float(open_[idx])
    c = float(close[idx])
    if c <= o:
        return False
    if c <= float(close[idx - 1]):
        return False
    if kind == "W":
        if float(high[idx]) <= prior_before * N_SHAPE_BREAK_PRIOR_HIGH_EPS:
            return False
    vol_ma = float(np.mean(vol[max(0, idx - 5) : idx]))
    if vol_ma <= 0 or float(vol[idx]) < vol_ma * vol_mult:
        return False
    return True


def _df_bar_date_str(df, idx):
    if idx is None or idx < 0 or idx >= len(df):
        return ""
    if "date" not in df.columns:
        return str(int(idx))
    d = df["date"].iloc[idx]
    try:
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    except Exception:
        s = str(d)
        return s.split(" ")[0].split("T")[0][:10]


def build_vw_bottle_tracks_for_df(df):
    """
    供 API 双轨展示：每笔 W/V 瓶口轨的入场、减半、出场、是否突破、模拟收益率%。
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float) if "high" in df.columns else close.copy()
    low = df["low"].values.astype(float) if "low" in df.columns else close.copy()
    n = len(close)
    w_e = detect_w_right_bottom_events(close, high, low, n)
    v_e = detect_v_right_bottom_events(close, high, low, n)
    events = merge_w_v_events(w_e, v_e)

    tracks = []
    last_exit = -1
    for ev in events:
        if ev["entry"] <= last_exit:
            continue
        r, neck, stop = ev["entry"], ev["neck"], ev["stop_ref"]
        if r >= n - 2:
            continue
        ret, e0, e1, meta = simulate_trade(close, high, low, r, neck, stop, n)
        hd = meta.get("half_day")
        tracks.append(
            {
                "pattern": ev["kind"],
                "entry_date": _df_bar_date_str(df, e0),
                "half_date": _df_bar_date_str(df, hd) if hd is not None else None,
                "exit_date": _df_bar_date_str(df, e1),
                "neck": round(float(neck), 4),
                "breakout": bool(meta.get("breakout")),
                "stopped": bool(meta.get("stopped")),
                "return_pct": round(float(ret) * 100.0, 2),
            }
        )
        last_exit = e1

    wins = [t for t in tracks if t["return_pct"] > 0]
    summary = {
        "track_count": len(tracks),
        "win_rate": round(len(wins) / len(tracks), 4) if tracks else 0.0,
        "avg_return_pct": round(float(np.mean([t["return_pct"] for t in tracks])), 4) if tracks else 0.0,
    }
    return {"tracks": tracks, "summary": summary}


def run_vw_bottle_backtest(df):
    """非重叠逐笔串联，返回每笔收益率列表。"""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float) if "high" in df.columns else close.copy()
    low = df["low"].values.astype(float) if "low" in df.columns else close.copy()
    n = len(close)

    w_e = detect_w_right_bottom_events(close, high, low, n)
    v_e = detect_v_right_bottom_events(close, high, low, n)
    events = merge_w_v_events(w_e, v_e)

    rets = []
    exits = []
    last_exit = -1
    for ev in events:
        if ev["entry"] <= last_exit:
            continue
        r, neck, stop = ev["entry"], ev["neck"], ev["stop_ref"]
        if r >= n - 2:
            continue
        ret, e0, e1, _meta = simulate_trade(close, high, low, r, neck, stop, n)
        rets.append(ret)
        exits.append((e0, e1, ev["kind"]))
        last_exit = e1

    return rets, exits


def summarize_returns(rets):
    if not rets:
        return {"count": 0, "win_rate": 0.0, "avg": 0.0, "sum": 0.0, "pf": 0.0}
    arr = np.array(rets, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    tp = float(wins.sum()) if len(wins) else 0.0
    tl = float(abs(losses.sum())) if len(losses) else 1e-12
    return {
        "count": len(arr),
        "win_rate": float((arr > 0).mean()),
        "avg": float(arr.mean()),
        "sum": float(arr.sum()),
        "pf": tp / tl if tl > 0 else 0.0,
    }


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from models import KlineData
    import engine_v2 as ev2
    import dynamic_weight_engine_v2 as dwe2

    def klines_to_df(klines):
        d = pd.DataFrame(klines)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        return d

    stocks = [("600519", "茅台"), ("000001", "平安"), ("002347", "泰尔")]
    vw_all = []
    bs_all = []
    for code, name in stocks:
        kl = KlineData.get(code, period="day")
        if not kl or len(kl) < 200:
            continue
        df = klines_to_df(kl)
        rets, _ = run_vw_bottle_backtest(df)
        s_vw = summarize_returns(rets)
        opt = dwe2.calculate_weights_v2(df)
        pw = {"buy_weights": opt["buy_weights"], "sell_weights": opt["sell_weights"]}
        out = ev2.analyze_signals_v2(df, precomputed_weights=pw)
        br = [
            float(s["return_pct"]) / 100.0
            for s in out["paired_signals"]
            if s.get("type") == "S" and s.get("return_pct") is not None
        ]
        s_bs = summarize_returns(br)
        vw_all.extend(rets)
        bs_all.extend(br)
        print(f"=== {name}({code}) ===")
        print(
            f"  VW瓶口策略: 笔数={s_vw['count']} 胜率={s_vw['win_rate']*100:.1f}% "
            f"均值={s_vw['avg']*100:.2f}% 合计(简单加总)={s_vw['sum']*100:.2f}% PF={s_vw['pf']:.2f}"
        )
        print(
            f"  现engine B→S: 笔数={s_bs['count']} 胜率={s_bs['win_rate']*100:.1f}% "
            f"均值={s_bs['avg']*100:.2f}% 合计={s_bs['sum']*100:.2f}% PF={s_bs['pf']:.2f}"
        )

    print("\n--- 三股合并 ---")
    print(
        f"  VW瓶口: {summarize_returns(vw_all)}"
    )
    print(
        f"  engine: {summarize_returns(bs_all)}"
    )
    sv = summarize_returns(vw_all)
    sb = summarize_returns(bs_all)
    if sv["count"] and sb["count"]:
        better_avg = sv["avg"] > sb["avg"]
        print(
            f"\n结论(样本内三股、单笔简单收益率): "
            f"VW瓶口均值={sv['avg']*100:.3f}% , engine B→S均值={sb['avg']*100:.3f}% —— "
            f"{'VW略高' if better_avg else 'engine略高'}；"
            f"胜率 VW={sv['win_rate']*100:.1f}% vs engine={sb['win_rate']*100:.1f}%；"
            f"盈亏比 PF VW={sv['pf']:.2f} vs engine={sb['pf']:.2f}。"
            f"（笔数不同，「合计%」简单相加不可直接对比仓位规模）"
        )
