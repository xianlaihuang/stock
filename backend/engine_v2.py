import pandas as pd
import numpy as np
import talib
from scipy import stats
import warnings
import importlib
import dynamic_weight_engine as dwe
from db import db
from datetime import datetime

warnings.filterwarnings('ignore')

weights_v2_collection = db['stock_weights_v2']
signals_v2_collection = db['stock_signals_v2']

weights_v2_collection.create_index('stock_code', unique=True)
signals_v2_collection.create_index('stock_code', unique=True)

# ========== PRD《B-S打标与公式权重优化方案》落地开关与参数 ==========
V2_EXECUTION_MODEL = 'next_open'
V2_SLIPPAGE_EACH_SIDE = 0.0005
V2_COMMISSION_ROUNDTRIP = 0.0008
# §4.1 效用 U = G − λ1·MDD − λ2·σd − λ3·N（迭代选优，关闭则退回 max sum r）
V2_PRD_UTILITY_OBJECTIVE = True
V2_UTILITY_LAM_MDD = 0.55
V2_UTILITY_LAM_DOWN = 0.12
V2_UTILITY_LAM_TURN = 0.00025
# §5.2 硬止损 / 时间止损
V2_ATR_STOP_ON = True
V2_ATR_PERIOD = 14
V2_ATR_STOP_MULT = 2.5
V2_MAX_HOLD_DAYS = 55
# 浮亏达阈值或自持仓以来高点回撤达阈值时，计数卖由「≥2条」降为「≥1条」，避免见顶后仍等第二条规则（如 002347 类走势 S 偏晚）
V2_FLOAT_LOSS_FOR_FAST_SELL = -0.055
V2_PEAK_DRAWDOWN_FOR_FAST_SELL = -0.045
# §5.3 高波动：抬高买入置信门槛、略延长最小持有
V2_HIGH_VOL_CONFIDENCE = 0.58
V2_HIGH_VOL_MIN_HOLD = 3
# §5.4 同类规则分组 cap（当日该组加权得分上限）
V2_GROUP_SCORE_CAP = 0.38
# §4.2 L1 风格：对归一化权重向量施加轻微惩罚（计入 U）
V2_UTILITY_LAM_L1 = 0.02

PRD_BUY_RULE_GROUP = {
    'MA金叉': 'ma', 'MACD金叉': 'macd', 'MACD底背离': 'macd',
    'RSI超卖上穿': 'osc', 'KDJ金叉': 'osc',
    '看涨K线形态': 'candle',
    '价升量增': 'vp', '放量突破高点': 'vp', '地量地价': 'vp',
    '趋势线支撑': 'trend',
    '横盘整理向上突破': 'pat', '黄金坑': 'pat', 'N字形突破': 'pat', '双针探底': 'pat',
}

# V3：下一交易日阳线(收盘>开盘)确认类 pending 的 meta 键与 confirm_note
_V3_NEXTDAY_BULL_PENDING = (
    ('v3_engulf_pending', 'V3看涨吞没:下一交易日阳线确认B(收盘>开盘)'),
    ('v3_upper_shadow_pending', 'V3长上影:下一交易日阳线确认B(收盘>开盘)'),
    ('v3_box_break_pending', 'V3横盘整理向上突破:下一交易日阳线确认B(收盘>开盘)'),
    ('v3_n_shape_pending', 'V3N字形突破:下一交易日阳线确认B(收盘>开盘)'),
    ('v3_flag_pending', 'V3旗形突破:下一交易日阳线确认B(收盘>开盘)'),
)
PRD_SELL_RULE_GROUP = {
    'MA死叉': 'ma', 'MACD死叉': 'macd', 'MACD顶背离': 'macd',
    'RSI超买下穿': 'osc', 'KDJ死叉': 'osc',
    '看跌K线形态': 'candle',
    '价跌量增': 'vp', '放量跌破低点': 'vp', '天量天价': 'vp',
    '趋势线阻力': 'trend', '头肩顶跌破': 'pat', '均线趋势打破': 'pat',
    '沿均线主升破位': 'ma',
    '横盘整理向下突破': 'pat', '历史缺口阻力': 'pat', '接近前高巨量阴增': 'pat',
}

# 下列卖出仅参与 V2 打标时从 triggered_sell 剔除（预计算仍存在，供 V3 使用）
_V3_ONLY_WEIGHTED_SELL_RULE_NAMES = frozenset({
    '横盘整理向下突破',
    '历史缺口阻力',
    '接近前高巨量阴增',
})


def _exec_buy_fill(b_idx, open_, close, n):
    if V2_EXECUTION_MODEL != 'next_open':
        return b_idx, float(close[b_idx])
    j = b_idx + 1
    if j < n:
        return j, float(open_[j])
    return b_idx, float(close[b_idx])


def _exec_sell_fill(s_idx, open_, close, n):
    if V2_EXECUTION_MODEL != 'next_open':
        return s_idx, float(close[s_idx])
    j = s_idx + 1
    if j < n:
        return j, float(open_[j])
    return s_idx, float(close[s_idx])


def _net_ret_entry_exit(entry_raw, exit_raw):
    if entry_raw <= 0 or exit_raw <= 0:
        return 0.0
    ein = entry_raw * (1.0 + V2_SLIPPAGE_EACH_SIDE)
    eout = exit_raw * (1.0 - V2_SLIPPAGE_EACH_SIDE)
    if ein <= 0:
        return 0.0
    return (eout / ein) - 1.0 - V2_COMMISSION_ROUNDTRIP


def _capped_weight_sum(triggered, weights, group_map, cap):
    if not triggered:
        return 0.0
    from collections import defaultdict
    gsum = defaultdict(float)
    for name in triggered:
        gsum[group_map.get(name, 'misc')] += weights.get(name, 0.0)
    total = 0.0
    for name in triggered:
        g = group_map.get(name, 'misc')
        s = gsum[g]
        f = min(1.0, cap / s) if s > cap else 1.0
        total += weights.get(name, 0.0) * f
    return total


def _trade_net_returns(signals, close, open_):
    n = len(close)
    open_ = close if open_ is None else open_
    rets = []
    bi = None
    for idx, sig in signals:
        if sig == 'B':
            bi = idx
        elif sig == 'S' and bi is not None:
            _, ea = _exec_buy_fill(bi, open_, close, n)
            _, ex = _exec_sell_fill(idx, open_, close, n)
            rets.append(_net_ret_entry_exit(ea, ex))
            bi = None
    return np.asarray(rets, dtype=float) if rets else np.array([], dtype=float)


def _max_dd_on_rets(rets):
    if rets.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    return float(-np.min(dd))


def _downside_sigma(rets):
    if rets.size == 0:
        return 0.0
    neg = np.minimum(0.0, rets)
    return float(np.sqrt(np.mean(neg ** 2)))


def _l1_norm_weights(buy_w, sell_w):
    s = sum(abs(v) for v in buy_w.values()) + sum(abs(v) for v in sell_w.values())
    return float(s)


def _prd_utility(signals, close, open_, buy_w, sell_w):
    rets = _trade_net_returns(signals, close, open_)
    if rets.size == 0:
        return -1e9, 0.0, 0.0, 0.0, 0
    G = float(np.sum(rets))
    mdd = _max_dd_on_rets(rets)
    sd = _downside_sigma(rets)
    n = int(rets.size)
    l1 = _l1_norm_weights(buy_w, sell_w)
    U = G - V2_UTILITY_LAM_MDD * mdd - V2_UTILITY_LAM_DOWN * sd - V2_UTILITY_LAM_TURN * n - V2_UTILITY_LAM_L1 * l1
    return U, G, mdd, sd, n


def _precompute_high_vol_mask(close, high, low, start_offset):
    n = len(close)
    atr = talib.ATR(high, low, close, timeperiod=V2_ATR_PERIOD)
    pct = atr / np.maximum(close, 1e-12)
    sub = pct[start_offset:n]
    if sub.size == 0 or np.all(np.isnan(sub)):
        return None
    med = np.nanmedian(sub)
    if np.isnan(med):
        return None
    mask = np.zeros(n, dtype=np.bool_)
    v = ~np.isnan(pct)
    mask[v] = pct[v] > med
    return mask


def _klines_to_df(klines):
    df = pd.DataFrame(klines)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _precompute_rule_signals(df, rules, start_offset=60):
    n = len(df)
    signals = {}
    for name, func in rules.items():
        arr = np.zeros(n, dtype=np.bool_)
        for idx in range(start_offset, n):
            try:
                if func(df, idx):
                    arr[idx] = True
            except Exception:
                pass
        signals[name] = arr
    return signals


def _preprocess_data(df):
    n = len(df)
    if n <= 15:
        return df.iloc[0:0]
    start = 10
    end = n - 5
    close = df['close'].values.astype(float)
    valid_mask = close[start:end] > 0
    result = df.iloc[start:end].copy()
    result = result[valid_mask]
    return result


def _v2_decorate_triggered_names(names, reasons, *, buy_side):
    """将「看涨/看跌K线形态」展开为带 TA-Lib 子形态中文名的展示串（用于 API rules/reasons）。"""
    out = []
    for name in names:
        if buy_side and name == '看涨K线形态' and reasons.get('bullish_pattern_kind'):
            out.append(f"看涨K线形态·{reasons['bullish_pattern_kind']}")
        elif (not buy_side) and name == '看跌K线形态' and reasons.get('bearish_pattern_kind'):
            out.append(f"看跌K线形态·{reasons['bearish_pattern_kind']}")
        else:
            out.append(name)
    return out


# ============================================================
# 第一大层：计算各买卖指标有效性
# ============================================================

def _extract_s_array(close, max_points=50):
    n = len(close)
    if n < 2:
        return []
    remaining = close.copy()
    offset_map = list(range(n))
    s_points = []

    while len(s_points) < max_points and len(remaining) > 0:
        idx_in_remaining = np.argmax(remaining)
        actual_idx = offset_map[idx_in_remaining]
        s_points.append(actual_idx)

        remaining_list = remaining.tolist()
        offset_map_list = offset_map[:]
        del remaining_list[idx_in_remaining]
        del offset_map_list[idx_in_remaining]
        remaining = np.array(remaining_list)
        offset_map = offset_map_list

    if len(s_points) <= 1:
        return s_points

    s_points.sort()
    filtered = [s_points[0]]

    for i in range(1, len(s_points)):
        candidate = s_points[i]
        last = filtered[-1]
        if abs(candidate - last) < 5:
            continue
        seg = close[last:candidate + 1]
        if len(seg) < 3:
            filtered.append(candidate)
            continue
        x = np.arange(len(seg))
        try:
            coeffs = np.polyfit(x, seg, 2)
            a = coeffs[0]
            if a <= 0:
                continue
        except Exception:
            continue
        filtered.append(candidate)

    return filtered


def _extract_b_array(close, max_points=50):
    n = len(close)
    if n < 2:
        return []
    remaining = close.copy()
    offset_map = list(range(n))
    b_points = []

    while len(b_points) < max_points and len(remaining) > 0:
        idx_in_remaining = np.argmin(remaining)
        actual_idx = offset_map[idx_in_remaining]
        b_points.append(actual_idx)

        remaining_list = remaining.tolist()
        offset_map_list = offset_map[:]
        del remaining_list[idx_in_remaining]
        del offset_map_list[idx_in_remaining]
        remaining = np.array(remaining_list)
        offset_map = offset_map_list

    if len(b_points) <= 1:
        return b_points

    b_points.sort()
    filtered = [b_points[0]]

    for i in range(1, len(b_points)):
        candidate = b_points[i]
        last = filtered[-1]
        if abs(candidate - last) < 5:
            continue
        seg = close[last:candidate + 1]
        if len(seg) < 3:
            filtered.append(candidate)
            continue
        x_quad = np.arange(len(seg))
        try:
            coeffs = np.polyfit(x_quad, seg, 2)
            a = coeffs[0]
            if a < 0:
                continue
        except Exception:
            continue
        filtered.append(candidate)

    return filtered


def layer1_calculate_effectiveness(df, buy_rules, sell_rules, start_offset=60):
    processed = _preprocess_data(df)
    if len(processed) < 30:
        return {}, {}

    orig_start = 10
    close = processed['close'].values.astype(float)
    n = len(close)

    s_array = _extract_s_array(close, max_points=50)
    b_array = _extract_b_array(close, max_points=50)

    buy_rewards = {name: 0 for name in buy_rules}
    sell_rewards = {name: 0 for name in sell_rules}

    for s_idx in s_array:
        orig_idx = s_idx + orig_start
        window_start = max(start_offset, orig_idx - 90)
        window_end = min(len(df), orig_idx + 91)
        for name, func in sell_rules.items():
            for check_idx in range(window_start, window_end):
                try:
                    if func(df, check_idx):
                        sell_rewards[name] += 1
                        break
                except Exception:
                    pass

    for b_idx in b_array:
        orig_idx = b_idx + orig_start
        window_start = max(start_offset, orig_idx - 90)
        window_end = min(len(df), orig_idx + 91)
        for name, func in buy_rules.items():
            for check_idx in range(window_start, window_end):
                try:
                    if func(df, check_idx):
                        buy_rewards[name] += 1
                        break
                except Exception:
                    pass

    return buy_rewards, sell_rewards


# 信号日成交量相对前 5 日均量达到该倍数视为「有效放大」，否则「看涨K线形态」走弱量确认路径
V2_BUY_VOL_EFFECTIVE_MULT = 1.05


def _signal_day_volume_effective(df, idx, mult=None):
    if mult is None:
        mult = V2_BUY_VOL_EFFECTIVE_MULT
    if 'volume' not in df.columns or idx < 1:
        return True
    vol = df['volume'].values.astype(float)
    base = float(np.mean(vol[max(0, idx - 5):idx]))
    if base <= 0 or not np.isfinite(base):
        return True
    return float(vol[idx]) >= base * mult


def _bar_date_str(dates, idx):
    """与 K 线 `date.split(' ')[0]` 对齐，避免 Mongo/JSON 带时分导致前端错位误标 B。"""
    if dates is None:
        return str(idx)
    d = dates[idx]
    try:
        return pd.Timestamp(d).strftime('%Y-%m-%d')
    except Exception:
        s = str(d)
        return s.split(' ')[0].split('T')[0][:10]


def _portfolio_sim_from_paired(paired_signals, initial_capital=1_000_000.0):
    """按每笔 S 的 return_pct 复利模拟全仓滚动（与列表配对顺序一致）。"""
    equity = float(initial_capital)
    closed = 0
    wins = 0
    for row in paired_signals or []:
        if row.get('type') != 'S':
            continue
        r = row.get('return_pct')
        if r is None:
            continue
        equity *= (1.0 + float(r) / 100.0)
        closed += 1
        if float(r) > 0:
            wins += 1
    total_ret_pct = (equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0
    win_rate = (wins / closed * 100.0) if closed else 0.0
    return {
        'initial_capital': float(initial_capital),
        'final_equity': round(equity, 2),
        'total_return_pct': round(total_ret_pct, 2),
        'closed_trades': closed,
        'win_rate_pct': round(win_rate, 1),
    }


def _mandatory_buy_rules_v3(df, start_offset):
    """
    V3：必买里用「W底右侧 / V反右侧」替代「W底突破 / V反底部」（右侧确认日即触发，不要求突破颈线）。
    检测逻辑与 strategy_vw_bottle_backtest 中事件定义一致。
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    n = len(close)
    from strategy_vw_bottle_backtest import detect_w_right_bottom_events, detect_v_right_bottom_events
    w_flags = np.zeros(n, dtype=bool)
    v_flags = np.zeros(n, dtype=bool)
    for ev in detect_w_right_bottom_events(close, high, low, n):
        e = int(ev['entry'])
        if start_offset <= e < n:
            w_flags[e] = True
    for ev in detect_v_right_bottom_events(close, high, low, n):
        e = int(ev['entry'])
        if start_offset <= e < n:
            v_flags[e] = True
    _, _, _, _, base_mb = dwe.get_all_rules_extended()
    out = {k: v for k, v in base_mb.items() if k not in ('W底突破', 'V反底部')}

    def _mk_flag(arr):
        def _f(df_, idx):
            i = int(idx)
            return 0 <= i < len(arr) and bool(arr[i])
        return _f

    out['W底右侧'] = _mk_flag(w_flags)
    out['V反右侧'] = _mk_flag(v_flags)
    return out


# ============================================================
# 第二大层：迭代优化
# ============================================================

def _mark_bs_points(df, buy_rules, sell_rules, mandatory_buy_rules,
                    mandatory_sell_rules, buy_restriction_rules,
                    buy_weights, sell_weights,
                    start_offset=60, min_hold=2,
                    sell_trigger_count=2, confidence_medium=0.55,
                    precomputed_signals=None, atr_arr=None, high_vol_mask=None,
                    engine_mode='v2'):
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    n = len(df)
    bar_dates = df['date'].values if 'date' in df.columns else None
    ma20 = talib.MA(close, timeperiod=20)
    ma5 = talib.MA(close, timeperiod=5)

    if precomputed_signals:
        buy_signals_pre = precomputed_signals['buy']
        sell_signals_pre = precomputed_signals['sell']
        mandatory_buy_pre = precomputed_signals['mandatory_buy']
        mandatory_sell_pre = precomputed_signals['mandatory_sell']
        restriction_pre = precomputed_signals['restriction']
    else:
        buy_signals_pre = _precompute_rule_signals(df, buy_rules, start_offset)
        sell_signals_pre = _precompute_rule_signals(df, sell_rules, start_offset)
        mandatory_buy_pre = _precompute_rule_signals(df, mandatory_buy_rules, start_offset)
        mandatory_sell_pre = _precompute_rule_signals(df, mandatory_sell_rules, start_offset)
        restriction_pre = _precompute_rule_signals(df, buy_restriction_rules, start_offset)

    dwe.apply_piercing_requires_confluence_buy(
        df, buy_signals_pre, buy_rules, start_offset, mandatory_buy_pre=mandatory_buy_pre)
    dwe.apply_all_buy_next_high_room_filter(
        df, buy_signals_pre, buy_rules, start_offset, mandatory_buy_pre=mandatory_buy_pre)

    signals = []
    trade_history = []
    reasons_list = []
    pending_buy = None
    v3_golden_pit_entry = False
    v3_gp_wait_break_m20 = False

    for idx in range(start_offset, n):
        if pending_buy is not None:
            signal_idx, prev_reasons, prev_confidence, prev_level = pending_buy[:4]
            pending_meta = dict(pending_buy[4]) if len(pending_buy) > 4 else {}
            pending_buy = None
            rel = idx - signal_idx

            # V3：双针探底 T+1 须阳线且收盘>MA5
            if engine_mode == 'v3' and pending_meta.get('v3_double_needle_pending'):
                if rel == 1 and dwe.confirm_double_needle_t1(df, idx):
                    if not dwe.upper_shadow_vetoes_buy_signal(df, idx):
                        trade_history.append((idx, 'B'))
                        signals.append((idx, 'B'))
                        v3_golden_pit_entry = (
                            '黄金坑' in prev_reasons.get('buy_triggered', []))
                        v3_gp_wait_break_m20 = False
                        row = {
                            **prev_reasons, 'final_signal': 'B',
                            'confidence': round(prev_confidence, 3),
                            'level': prev_level, 'confirmed': True,
                            'confirm_note': 'V3双针探底:T+1阳线且收盘>MA5确认B',
                        }
                        row['signal_date'] = _bar_date_str(bar_dates, signal_idx)
                        reasons_list.append(row)
                continue

            # V3：各类「仅下一交易日阳线确认」pending（收盘>开盘）
            if engine_mode == 'v3':
                v3_nd_handled = False
                for meta_key, cnote in _V3_NEXTDAY_BULL_PENDING:
                    if not pending_meta.get(meta_key):
                        continue
                    v3_nd_handled = True
                    if rel == 1 and close[idx] > open_[idx]:
                        trade_history.append((idx, 'B'))
                        signals.append((idx, 'B'))
                        if engine_mode == 'v3':
                            v3_golden_pit_entry = (
                                '黄金坑' in prev_reasons.get('buy_triggered', []))
                            v3_gp_wait_break_m20 = False
                        row = {
                            **prev_reasons, 'final_signal': 'B',
                            'confidence': round(prev_confidence, 3),
                            'level': prev_level, 'confirmed': True,
                            'confirm_note': cnote,
                        }
                        row['signal_date'] = _bar_date_str(bar_dates, signal_idx)
                        reasons_list.append(row)
                    break
                if v3_nd_handled:
                    continue

            if engine_mode != 'v2':
                continue

            if 1 <= rel <= 2:
                floor = dwe.pending_buy_support_floor(
                    close, low, signal_idx, atr_arr, V2_ATR_STOP_MULT)
                hold = dwe.pending_buy_lows_hold_since_signal(low, signal_idx, idx, floor)
                await_second = pending_meta.get('await_second_confirm', False)
                if rel == 2 and not await_second:
                    pass
                elif hold:
                    if pending_meta.get('double_needle_defer'):
                        gap_ok = (rel == 1 and not await_second)
                        is_bullish = gap_ok and dwe.confirm_double_needle_t1(df, idx)
                    elif pending_meta.get('bull_pattern_defer'):
                        gap_ok = (rel == 1 and not await_second) or (rel == 2 and await_second)
                        is_bullish = gap_ok and (close[idx] > open_[idx])
                    else:
                        is_bullish = (
                            close[idx] > open_[idx] and close[idx] > close[signal_idx])
                    ok = (
                        is_bullish
                        and not dwe.upper_shadow_vetoes_buy_signal(df, idx)
                        and not dwe.confirm_buy_vetoes_shrink_volume_without_close_break(
                            df, idx, signal_idx))
                    if ok:
                        trade_history.append((idx, 'B'))
                        signals.append((idx, 'B'))
                        row = {**prev_reasons, 'final_signal': 'B',
                               'confidence': round(prev_confidence, 3),
                               'level': prev_level, 'confirmed': True}
                        row['signal_date'] = _bar_date_str(bar_dates, signal_idx)
                        if rel == 2:
                            row['confirm_note'] = (
                                '≥2加权买入:第二交易日确认B' if not pending_meta.get('bull_pattern_defer')
                                else (
                                    '看涨K线形态:T+2阳线确认B(收盘>开盘);量能不足'
                                    if pending_meta.get('weak_vol_bull_pattern')
                                    else '看涨K线形态:T+2阳线确认B(收盘>开盘)'))
                        elif pending_meta.get('double_needle_defer'):
                            row['confirm_note'] = '双针探底:T+1阳线且收盘>MA5确认B'
                        elif pending_meta.get('bull_pattern_defer'):
                            row['confirm_note'] = (
                                '看涨K线形态:下一交易日阳线确认B(收盘>开盘);量能不足'
                                if pending_meta.get('weak_vol_bull_pattern')
                                else '看涨K线形态:下一交易日阳线确认B(收盘>开盘)'
                            )
                        reasons_list.append(row)
                    elif rel == 1 and not ok and len(prev_reasons.get('buy_triggered', [])) >= 2 and not await_second:
                        pending_meta['await_second_confirm'] = True
                        pending_buy = (signal_idx, prev_reasons, prev_confidence, prev_level, pending_meta)

        in_position = trade_history and trade_history[-1][1] == 'B'

        if atr_arr is not None and V2_ATR_STOP_ON and in_position:
            entry_idx, _ = trade_history[-1]
            _, epx = _exec_buy_fill(entry_idx, open_, close, n)
            ae = atr_arr[entry_idx] if entry_idx < n else np.nan
            if idx > entry_idx and not pd.isna(ae) and ae > 0 and low[idx] <= epx - V2_ATR_STOP_MULT * ae:
                v3_golden_pit_entry = False
                v3_gp_wait_break_m20 = False
                trade_history.append((idx, 'S'))
                signals.append((idx, 'S'))
                reasons_list.append({
                    'buy_triggered': [], 'sell_triggered': [], 'mandatory_buy_triggered': [],
                    'mandatory_sell_triggered': [], 'buy_restriction_triggered': [],
                    'buy_score': 0.0, 'sell_score': 0.0, 'final_signal': 'S',
                    'confidence': 1.0, 'level': 'strong', 'sell_reason_type': 'ATR止损',
                })
                continue

        if in_position:
            entry_idx, _ = trade_history[-1]
            if idx > entry_idx and (idx - entry_idx) >= V2_MAX_HOLD_DAYS:
                v3_golden_pit_entry = False
                v3_gp_wait_break_m20 = False
                trade_history.append((idx, 'S'))
                signals.append((idx, 'S'))
                reasons_list.append({
                    'buy_triggered': [], 'sell_triggered': [], 'mandatory_buy_triggered': [],
                    'mandatory_sell_triggered': [], 'buy_restriction_triggered': [],
                    'buy_score': 0.0, 'sell_score': 0.0, 'final_signal': 'S',
                    'confidence': 1.0, 'level': 'strong', 'sell_reason_type': '时间止损',
                })
                continue

        if engine_mode == 'v3' and in_position and v3_golden_pit_entry:
            entry_idx, _ = trade_history[-1]
            if idx > entry_idx:
                m20v = ma20[idx]
                if not pd.isna(m20v):
                    m20f = float(m20v)
                    if v3_gp_wait_break_m20:
                        if close[idx] < m20f * 0.998:
                            v3_golden_pit_entry = False
                            v3_gp_wait_break_m20 = False
                            trade_history.append((idx, 'S'))
                            signals.append((idx, 'S'))
                            reasons_list.append({
                                'buy_triggered': [], 'sell_triggered': [],
                                'mandatory_buy_triggered': [],
                                'mandatory_sell_triggered': [],
                                'buy_restriction_triggered': [],
                                'buy_score': 0.0, 'sell_score': 0.0, 'final_signal': 'S',
                                'confidence': 1.0, 'level': 'strong',
                                'sell_reason_type': 'V3黄金坑破20日线卖出',
                            })
                            continue
                    else:
                        reached = close[idx] >= m20f * 0.998
                        breakout = close[idx] > m20f
                        if reached and breakout:
                            v3_gp_wait_break_m20 = True
                        elif reached:
                            v3_golden_pit_entry = False
                            trade_history.append((idx, 'S'))
                            signals.append((idx, 'S'))
                            reasons_list.append({
                                'buy_triggered': [], 'sell_triggered': [],
                                'mandatory_buy_triggered': [],
                                'mandatory_sell_triggered': [],
                                'buy_restriction_triggered': [],
                                'buy_score': 0.0, 'sell_score': 0.0, 'final_signal': 'S',
                                'confidence': 1.0, 'level': 'strong',
                                'sell_reason_type': 'V3黄金坑达到20日线卖出',
                            })
                            continue

        triggered_buy = [name for name in buy_rules if buy_signals_pre.get(name, np.zeros(n, dtype=np.bool_))[idx]]
        triggered_sell = [name for name in sell_rules if sell_signals_pre.get(name, np.zeros(n, dtype=np.bool_))[idx]]
        if engine_mode != 'v3':
            triggered_sell = [
                x for x in triggered_sell if x not in _V3_ONLY_WEIGHTED_SELL_RULE_NAMES
            ]
        triggered_mandatory_buy = [name for name in mandatory_buy_rules if mandatory_buy_pre.get(name, np.zeros(n, dtype=np.bool_))[idx]]
        triggered_mandatory_sell = [name for name in mandatory_sell_rules if mandatory_sell_pre.get(name, np.zeros(n, dtype=np.bool_))[idx]]
        triggered_restriction = [name for name in buy_restriction_rules if restriction_pre.get(name, np.zeros(n, dtype=np.bool_))[idx]]

        buy_score = _capped_weight_sum(triggered_buy, buy_weights, PRD_BUY_RULE_GROUP, V2_GROUP_SCORE_CAP)
        sell_score = _capped_weight_sum(triggered_sell, sell_weights, PRD_SELL_RULE_GROUP, V2_GROUP_SCORE_CAP)

        is_mandatory_sell = len(triggered_mandatory_sell) > 0
        is_mandatory_buy = len(triggered_mandatory_buy) > 0
        is_buy_restricted = len(triggered_restriction) > 0

        eff_sell_n = sell_trigger_count
        if in_position:
            entry_idx, _ = trade_history[-1]
            if idx > entry_idx:
                upl = close[idx] / max(close[entry_idx], 1e-12) - 1.0
                if upl <= V2_FLOAT_LOSS_FOR_FAST_SELL:
                    eff_sell_n = 1
                peak_since = float(high[entry_idx:idx + 1].max())
                if peak_since > 1e-12:
                    dd_from_peak = close[idx] / peak_since - 1.0
                    if dd_from_peak <= V2_PEAK_DRAWDOWN_FOR_FAST_SELL:
                        eff_sell_n = 1

        weighted_sell_count = sum(1 for name in triggered_sell if sell_weights.get(name, 0) > 0)
        is_sell_by_count = (len(triggered_sell) >= eff_sell_n
                           and in_position
                           and weighted_sell_count >= 1
                           and sell_score > buy_score)

        candidate = None
        confidence = 0
        signal_reason_type = ''

        if is_mandatory_sell and in_position:
            candidate = 'S'
            confidence = 1.0
            signal_reason_type = '必卖'
        elif is_sell_by_count and in_position:
            candidate = 'S'
            total = buy_score + sell_score
            confidence = sell_score / total if total > 0 else 0.8
            signal_reason_type = f'{len(triggered_sell)}条规则'
        elif is_mandatory_buy and not in_position:
            candidate = 'B'
            confidence = 1.0
            signal_reason_type = '必买'
        elif (buy_score > 0 and buy_score > sell_score and not in_position
              and dwe.buy_weighted_combo_gate_ok(triggered_buy)):
            candidate = 'B'
            total = buy_score + sell_score
            confidence = buy_score / total if total > 0 else 0

        if candidate == 'B' and is_buy_restricted:
            candidate = None

        if candidate is not None:
            if idx < 20 or pd.isna(ma20[idx]) or pd.isna(ma20[idx - 1]):
                candidate = None
            else:
                trend_down = (ma20[idx] < ma20[idx - 1]) and (close[idx] < ma20[idx])
                if candidate == 'B' and trend_down:
                    if engine_mode == 'v3' and '黄金坑' in triggered_buy:
                        pass
                    else:
                        candidate = None

        if candidate is not None and not is_mandatory_sell and not is_sell_by_count and not is_mandatory_buy:
            dominant_score = buy_score if candidate == 'B' else sell_score
            if dominant_score < 0.05:
                candidate = None

        mh = V2_HIGH_VOL_MIN_HOLD if (
            high_vol_mask is not None and idx < len(high_vol_mask) and high_vol_mask[idx]
        ) else min_hold

        if candidate is not None and trade_history:
            last_idx, last_signal = trade_history[-1]
            if not is_mandatory_sell and not is_sell_by_count and not is_mandatory_buy:
                if idx - last_idx < mh:
                    candidate = None
                elif last_signal == candidate:
                    candidate = None
            else:
                if last_signal == 'B' and candidate == 'S':
                    pass
                elif idx - last_idx < mh:
                    candidate = None

        if candidate == 'B' and dwe.upper_shadow_vetoes_buy_signal(df, idx):
            if engine_mode != 'v3':
                candidate = None

        if candidate == 'B' and dwe.breakthrough_prior_high_vetoes_buy(df, idx):
            candidate = None

        reasons = {
            'buy_triggered': triggered_buy,
            'sell_triggered': triggered_sell,
            'mandatory_buy_triggered': triggered_mandatory_buy,
            'mandatory_sell_triggered': triggered_mandatory_sell,
            'buy_restriction_triggered': triggered_restriction,
            'buy_score': round(buy_score, 4),
            'sell_score': round(sell_score, 4),
        }
        if '看涨K线形态' in triggered_buy:
            try:
                _bk = dwe.detect_bullish_candle_pattern_kind(df, idx)
                if _bk:
                    reasons['bullish_pattern_kind'] = _bk
            except Exception:
                pass
        if '看跌K线形态' in triggered_sell:
            try:
                _sk = dwe.detect_bearish_candle_pattern_kind(df, idx)
                if _sk:
                    reasons['bearish_pattern_kind'] = _sk
            except Exception:
                pass

        cm = confidence_medium
        if high_vol_mask is not None and idx < len(high_vol_mask) and high_vol_mask[idx]:
            cm = max(confidence_medium, V2_HIGH_VOL_CONFIDENCE)

        if candidate and confidence >= cm:
            level = 'strong' if confidence >= 0.75 else 'medium'
            if candidate == 'B':
                if engine_mode == 'v3':
                    mbuy = reasons.get('mandatory_buy_triggered', [])
                    bt = reasons.get('buy_triggered', [])
                    if reasons.get('bullish_pattern_kind') == '看涨吞没':
                        pending_buy = (
                            idx, reasons, confidence, level,
                            {'v3_engulf_pending': True},
                        )
                    elif dwe.upper_shadow_vetoes_buy_signal(df, idx):
                        pending_buy = (
                            idx, reasons, confidence, level,
                            {'v3_upper_shadow_pending': True},
                        )
                    elif '横盘整理向上突破' in bt:
                        pending_buy = (
                            idx, reasons, confidence, level,
                            {'v3_box_break_pending': True},
                        )
                    elif 'N字形突破' in bt:
                        pending_buy = (
                            idx, reasons, confidence, level,
                            {'v3_n_shape_pending': True},
                        )
                    elif '旗形突破' in mbuy:
                        pending_buy = (
                            idx, reasons, confidence, level,
                            {'v3_flag_pending': True},
                        )
                    elif '双针探底' in bt:
                        pending_buy = (
                            idx, reasons, confidence, level,
                            {'v3_double_needle_pending': True},
                        )
                    else:
                        trade_history.append((idx, 'B'))
                        signals.append((idx, 'B'))
                        row = {**reasons, 'final_signal': 'B',
                               'confidence': round(confidence, 3),
                               'level': level, 'confirmed': True,
                               'confirm_note': 'V3即时买入(无T+1/T+2确认)'}
                        reasons_list.append(row)
                        if engine_mode == 'v3':
                            v3_golden_pit_entry = (
                                '黄金坑' in reasons.get('buy_triggered', []))
                            v3_gp_wait_break_m20 = False
                elif pending_buy is None:
                    pending_meta = {}
                    bt_trig = reasons.get('buy_triggered', [])
                    if '双针探底' in bt_trig:
                        pending_meta['double_needle_defer'] = True
                    elif '看涨K线形态' in bt_trig:
                        pending_meta['bull_pattern_defer'] = True
                        if not _signal_day_volume_effective(df, idx):
                            pending_meta['weak_vol_bull_pattern'] = True
                    pending_buy = (idx, reasons, confidence, level, pending_meta)
            else:
                v3_golden_pit_entry = False
                v3_gp_wait_break_m20 = False
                trade_history.append((idx, candidate))
                signals.append((idx, candidate))
                reasons_list.append({**reasons, 'final_signal': candidate,
                                     'confidence': round(confidence, 3),
                                     'level': level,
                                     'sell_reason_type': signal_reason_type})

    return signals, reasons_list


def _verify_buy_effectiveness(signals, reasons_list, close, high, open_=None):
    buy_rewards = {}
    buy_penalties = {}
    n = len(close)
    open_ = close if open_ is None else open_

    buy_indices = []
    sell_indices = []
    for i, (idx, sig) in enumerate(signals):
        if sig == 'B':
            buy_indices.append((i, idx))
        else:
            sell_indices.append((i, idx))

    for bi, (i, buy_idx) in enumerate(buy_indices):
        next_sell_idx = None
        for si, (j, sell_idx) in enumerate(sell_indices):
            if sell_idx > buy_idx:
                next_sell_idx = sell_idx
                break

        if next_sell_idx is None:
            continue

        buy_fill_bar, entry_raw = _exec_buy_fill(buy_idx, open_, close, n)
        if buy_fill_bar <= next_sell_idx:
            max_price = float(high[buy_fill_bar:next_sell_idx + 1].max())
        else:
            max_price = float(high[buy_idx])
        profit_pct = (max_price - entry_raw) / max(entry_raw, 1e-12) * 100

        reasons = reasons_list[i] if i < len(reasons_list) else {}
        triggered = reasons.get('buy_triggered', []) + reasons.get('mandatory_buy_triggered', [])

        for name in triggered:
            if name not in buy_rewards:
                buy_rewards[name] = 0
                buy_penalties[name] = 0
            if profit_pct < 0:
                buy_penalties[name] += 1
            else:
                reward = int(profit_pct / 10) + 1
                buy_rewards[name] += reward

    return buy_rewards, buy_penalties


def _verify_sell_effectiveness(signals, reasons_list, close, low, open_=None):
    sell_rewards = {}
    sell_penalties = {}
    n = len(close)
    open_ = close if open_ is None else open_

    buy_indices = []
    sell_indices = []
    for i, (idx, sig) in enumerate(signals):
        if sig == 'B':
            buy_indices.append((i, idx))
        else:
            sell_indices.append((i, idx))

    for si, (i, sell_idx) in enumerate(sell_indices):
        next_buy_idx = None
        for bi, (j, buy_idx) in enumerate(buy_indices):
            if buy_idx > sell_idx:
                next_buy_idx = buy_idx
                break

        if next_buy_idx is None:
            continue

        sell_fill_bar, exit_raw = _exec_sell_fill(sell_idx, open_, close, n)
        if sell_fill_bar < next_buy_idx:
            min_price = float(low[sell_fill_bar:next_buy_idx + 1].min())
        else:
            min_price = float(low[sell_idx])
        avoided_pct = (exit_raw - min_price) / max(exit_raw, 1e-12) * 100

        reasons = reasons_list[i] if i < len(reasons_list) else {}
        triggered = reasons.get('sell_triggered', []) + reasons.get('mandatory_sell_triggered', [])

        for name in triggered:
            if name not in sell_rewards:
                sell_rewards[name] = 0
                sell_penalties[name] = 0
            if avoided_pct < 0:
                sell_penalties[name] += 1
            else:
                reward = int(avoided_pct / 10) + 1
                sell_rewards[name] += reward

    return sell_rewards, sell_penalties


def _calculate_final_weights(buy_rewards, buy_penalties, sell_rewards, sell_penalties):
    buy_weights = {}
    buy_details = {}
    sell_weights = {}
    sell_details = {}

    all_buy_names = set(list(buy_rewards.keys()) + list(buy_penalties.keys()))
    buy_total_effective = 0
    for name in all_buy_names:
        r = buy_rewards.get(name, 0)
        p = buy_penalties.get(name, 0)
        raw = r - p
        effective = max(raw, 0)
        buy_total_effective += effective
        buy_weights[name] = effective

    if buy_total_effective > 0:
        for name in all_buy_names:
            buy_weights[name] = buy_weights[name] / buy_total_effective

    for name in all_buy_names:
        r = buy_rewards.get(name, 0)
        p = buy_penalties.get(name, 0)
        raw = r - p
        effective = max(raw, 0)
        total = r + p
        win_rate = r / total if total > 0 else 0
        normalized = buy_weights.get(name, 0)
        score = normalized * win_rate
        level = 'strong' if score >= 0.15 else ('medium' if score >= 0.05 else 'weak')
        buy_details[name] = {
            'reward': r, 'penalty': p, 'raw_weight': raw,
            'effective_weight': effective, 'normalized_weight': round(normalized, 4),
            'win_rate': round(win_rate, 4), 'score': round(score, 4),
            'level': level
        }

    all_sell_names = set(list(sell_rewards.keys()) + list(sell_penalties.keys()))
    sell_total_effective = 0
    for name in all_sell_names:
        r = sell_rewards.get(name, 0)
        p = sell_penalties.get(name, 0)
        raw = r - p
        effective = max(raw, 0)
        sell_total_effective += effective
        sell_weights[name] = effective

    if sell_total_effective > 0:
        for name in all_sell_names:
            sell_weights[name] = sell_weights[name] / sell_total_effective

    for name in all_sell_names:
        r = sell_rewards.get(name, 0)
        p = sell_penalties.get(name, 0)
        raw = r - p
        effective = max(raw, 0)
        total = r + p
        win_rate = r / total if total > 0 else 0
        normalized = sell_weights.get(name, 0)
        score = normalized * win_rate
        level = 'strong' if score >= 0.15 else ('medium' if score >= 0.05 else 'weak')
        sell_details[name] = {
            'reward': r, 'penalty': p, 'raw_weight': raw,
            'effective_weight': effective, 'normalized_weight': round(normalized, 4),
            'win_rate': round(win_rate, 4), 'score': round(score, 4),
            'level': level
        }

    return buy_weights, buy_details, sell_weights, sell_details


def _calculate_total_return(signals, close, open_=None):
    if not signals:
        return 0.0, 0.0
    n = len(close)
    open_ = close if open_ is None else open_
    total_return = 0.0
    win_count = 0
    total_count = 0
    buy_idx = None
    for idx, sig in signals:
        if sig == 'B':
            buy_idx = idx
        elif sig == 'S' and buy_idx is not None:
            _, ea = _exec_buy_fill(buy_idx, open_, close, n)
            _, ex = _exec_sell_fill(idx, open_, close, n)
            ret = _net_ret_entry_exit(ea, ex)
            total_return += ret
            if ret > 0:
                win_count += 1
            total_count += 1
            buy_idx = None
    win_rate = win_count / total_count if total_count > 0 else 0
    return total_return, win_rate


# ============================================================
# 主入口：公式生成
# ============================================================

def calculate_weights_v2(df, max_iterations=100):
    buy_rules, sell_rules = dwe.get_all_rules()
    _, _, mandatory_sell_rules, buy_restriction_rules, mandatory_buy_rules = dwe.get_all_rules_extended()

    start_offset = 60
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close

    print("[V2] 第一大层：计算各买卖指标有效性...")
    buy_rewards_l1, sell_rewards_l1 = layer1_calculate_effectiveness(
        df, buy_rules, sell_rules, start_offset)

    buy_penalties = {name: 0 for name in buy_rules}
    sell_penalties = {name: 0 for name in sell_rules}
    buy_rewards = dict(buy_rewards_l1)
    sell_rewards = dict(sell_rewards_l1)

    buy_weights, buy_details, sell_weights, sell_details = _calculate_final_weights(
        buy_rewards, buy_penalties, sell_rewards, sell_penalties)

    open_vals = df['open'].values.astype(float) if 'open' in df.columns else close
    atr_arr = talib.ATR(high, low, close, timeperiod=V2_ATR_PERIOD) if V2_ATR_STOP_ON else None
    hvm = _precompute_high_vol_mask(close, high, low, start_offset)

    precomputed_all = {
        'buy': _precompute_rule_signals(df, buy_rules, start_offset),
        'sell': _precompute_rule_signals(df, sell_rules, start_offset),
        'mandatory_buy': _precompute_rule_signals(df, mandatory_buy_rules, start_offset),
        'mandatory_sell': _precompute_rule_signals(df, mandatory_sell_rules, start_offset),
        'restriction': _precompute_rule_signals(df, buy_restriction_rules, start_offset),
    }

    best_return = -np.inf
    best_win_rate = 0
    best_buy_weights = dict(buy_weights)
    best_sell_weights = dict(sell_weights)
    best_buy_details = dict(buy_details)
    best_sell_details = dict(sell_details)
    best_iteration = 0
    best_utility = -np.inf
    best_utility_components = None
    prev_return = -np.inf
    converge_count = 0

    print("[V2] 第二大层：迭代优化（PRD：执行价+效用U+分组cap+ATR/时间止损+波动门控）...")
    for iteration in range(1, max_iterations + 1):
        U = -np.inf
        signals, reasons_list = _mark_bs_points(
            df, buy_rules, sell_rules,
            mandatory_buy_rules, mandatory_sell_rules, buy_restriction_rules,
            buy_weights, sell_weights, start_offset,
            precomputed_signals=precomputed_all,
            atr_arr=atr_arr,
            high_vol_mask=hvm)

        if not signals:
            print(f"  迭代{iteration}: 无信号生成，跳过")
            continue

        b_rewards, b_penalties = _verify_buy_effectiveness(
            signals, reasons_list, close, high, open_vals)
        s_rewards, s_penalties = _verify_sell_effectiveness(
            signals, reasons_list, close, low, open_vals)

        for name in buy_rules:
            buy_rewards[name] = buy_rewards_l1.get(name, 0) + b_rewards.get(name, 0)
            buy_penalties[name] = b_penalties.get(name, 0)
        for name in sell_rules:
            sell_rewards[name] = sell_rewards_l1.get(name, 0) + s_rewards.get(name, 0)
            sell_penalties[name] = s_penalties.get(name, 0)

        buy_weights, buy_details, sell_weights, sell_details = _calculate_final_weights(
            buy_rewards, buy_penalties, sell_rewards, sell_penalties)

        total_return, win_rate = _calculate_total_return(signals, close, open_vals)
        U, G, mdd, sd, nt = _prd_utility(signals, close, open_vals, buy_weights, sell_weights)

        if V2_PRD_UTILITY_OBJECTIVE:
            if (U > best_utility) or (
                abs(U - best_utility) <= 1e-12 and total_return > best_return
            ):
                best_utility = U
                best_return = total_return
                best_win_rate = win_rate
                best_buy_weights = dict(buy_weights)
                best_sell_weights = dict(sell_weights)
                best_buy_details = dict(buy_details)
                best_sell_details = dict(sell_details)
                best_iteration = iteration
                best_utility_components = {'U': U, 'G': G, 'mdd': mdd, 'downside': sd, 'n_trades': nt}
        else:
            if total_return > best_return:
                best_return = total_return
                best_win_rate = win_rate
                best_buy_weights = dict(buy_weights)
                best_sell_weights = dict(sell_weights)
                best_buy_details = dict(buy_details)
                best_sell_details = dict(sell_details)
                best_iteration = iteration

        if iteration % 10 == 0:
            ex = f", U={U:.4f}" if V2_PRD_UTILITY_OBJECTIVE else ''
            print(f"  迭代{iteration}: sum收益={total_return*100:.2f}%, 胜率={win_rate*100:.1f}%{ex}")

        if abs(total_return - prev_return) < 0.001:
            converge_count += 1
        else:
            converge_count = 0
        prev_return = total_return

        if converge_count >= 3:
            print(f"  迭代{iteration}: 收敛，停止迭代")
            break

    suf = ''
    if V2_PRD_UTILITY_OBJECTIVE and best_utility_components:
        suf = f", U={best_utility_components['U']:.4f}, MDD={best_utility_components['mdd']:.3f}"
    print(f"[V2] 优化完成: 最佳迭代={best_iteration}, sum收益={best_return*100:.2f}%, "
          f"胜率={best_win_rate*100:.1f}%{suf}")

    out = {
        'buy_weights': best_buy_weights,
        'sell_weights': best_sell_weights,
        'buy_details': best_buy_details,
        'sell_details': best_sell_details,
        'iteration_count': best_iteration,
        'total_return': best_return,
        'win_rate': best_win_rate,
    }
    if V2_PRD_UTILITY_OBJECTIVE and best_utility_components:
        out['prd_utility'] = round(best_utility_components['U'], 6)
        out['prd_max_drawdown'] = round(best_utility_components['mdd'], 6)
    return out


# ============================================================
# 分析功能
# ============================================================

def analyze_signals_v2(df, precomputed_weights=None, start_date=None, end_date=None,
                       engine_mode='v2', mandatory_buy_rules_override=None):
    buy_rules, sell_rules = dwe.get_all_rules()
    _, _, mandatory_sell_rules, buy_restriction_rules, mandatory_buy_rules = dwe.get_all_rules_extended()
    if mandatory_buy_rules_override is not None:
        mandatory_buy_rules = mandatory_buy_rules_override

    if precomputed_weights:
        buy_weights = precomputed_weights.get('buy_weights', {})
        sell_weights = precomputed_weights.get('sell_weights', {})
    else:
        buy_weights = {name: 1.0 / len(buy_rules) for name in buy_rules}
        sell_weights = {name: 1.0 / len(sell_rules) for name in sell_rules}

    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    open_vals = df['open'].values.astype(float) if 'open' in df.columns else close
    dates = df['date'].values if 'date' in df.columns else None
    n = len(close)

    start_offset = 60
    atr_arr = talib.ATR(high, low, close, timeperiod=V2_ATR_PERIOD) if V2_ATR_STOP_ON else None
    hvm = _precompute_high_vol_mask(close, high, low, start_offset)
    precomputed = {
        'buy': _precompute_rule_signals(df, buy_rules, start_offset),
        'sell': _precompute_rule_signals(df, sell_rules, start_offset),
        'mandatory_buy': _precompute_rule_signals(df, mandatory_buy_rules, start_offset),
        'mandatory_sell': _precompute_rule_signals(df, mandatory_sell_rules, start_offset),
        'restriction': _precompute_rule_signals(df, buy_restriction_rules, start_offset),
    }

    signals, reasons_list = _mark_bs_points(
        df, buy_rules, sell_rules,
        mandatory_buy_rules, mandatory_sell_rules, buy_restriction_rules,
        buy_weights, sell_weights, start_offset,
        precomputed_signals=precomputed,
        atr_arr=atr_arr,
        high_vol_mask=hvm,
        engine_mode=engine_mode)

    paired_signals = []
    buy_idx = None
    for i, (idx, sig) in enumerate(signals):
        reasons = reasons_list[i] if i < len(reasons_list) else {}
        date_str = _bar_date_str(dates, idx)
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue

        if sig == 'B':
            buy_idx = idx
            _, fill_b = _exec_buy_fill(buy_idx, open_vals, close, n)
            buy_disp = _v2_decorate_triggered_names(reasons.get('buy_triggered', []), reasons, buy_side=True)
            mand_buy = reasons.get('mandatory_buy_triggered', [])
            b_row = {
                'type': 'B',
                # date / buy_bar_date: 确认棒（与 signals 里 B 的 idx 一致）；signal_date 为 pending 信号日
                'date': date_str,
                'buy_bar_date': date_str,
                'close': round(close[idx], 2),
                'price': round(close[idx], 2),
                'fill_price': round(fill_b, 3),
                'score': reasons.get('confidence', 0),
                'confidence': reasons.get('confidence', 0),
                'reasons': buy_disp + mand_buy,
                'rules': buy_disp + mand_buy,
                'level': reasons.get('level', ''),
                'return_rate_pct': None,
                'return_pct': None,
            }
            if reasons.get('bullish_pattern_kind'):
                b_row['bullish_pattern_kind'] = reasons['bullish_pattern_kind']
            if reasons.get('confirm_note'):
                b_row['confirm_note'] = reasons['confirm_note']
            if reasons.get('signal_date'):
                b_row['signal_date'] = reasons['signal_date']
            sig_bar_idx = idx
            if reasons.get('signal_date'):
                bi = dwe.bar_index_from_date_str(df, reasons['signal_date'])
                if bi is not None:
                    sig_bar_idx = bi
            b_row.update(dwe.signal_extreme_annotation(
                df, sig_bar_idx, anchor_bar_idx=idx, buy_triggered=buy_disp + mand_buy,
            ))
            paired_signals.append(b_row)
        elif sig == 'S' and buy_idx is not None:
            _, ea = _exec_buy_fill(buy_idx, open_vals, close, n)
            _, ex_px = _exec_sell_fill(idx, open_vals, close, n)
            ret_pct = _net_ret_entry_exit(ea, ex_px) * 100
            sell_disp = _v2_decorate_triggered_names(reasons.get('sell_triggered', []), reasons, buy_side=False)
            mand_sell = reasons.get('mandatory_sell_triggered', [])
            s_row = {
                'type': 'S', 'date': date_str,
                'close': round(close[idx], 2),
                'price': round(close[idx], 2),
                'fill_price': round(ex_px, 3),
                'score': reasons.get('confidence', 0),
                'confidence': reasons.get('confidence', 0),
                'reasons': sell_disp + mand_sell,
                'rules': sell_disp + mand_sell,
                'sell_reason_type': reasons.get('sell_reason_type', ''),
                'return_pct': round(ret_pct, 2),
                'return_rate_pct': round(ret_pct, 2),
                'level': reasons.get('level', ''),
            }
            if reasons.get('bearish_pattern_kind'):
                s_row['bearish_pattern_kind'] = reasons['bearish_pattern_kind']
            sig_bar_idx_s = idx
            if reasons.get('signal_date'):
                bi_s = dwe.bar_index_from_date_str(df, reasons['signal_date'])
                if bi_s is not None:
                    sig_bar_idx_s = bi_s
            s_row.update(dwe.signal_extreme_annotation(df, sig_bar_idx_s, anchor_bar_idx=idx))
            paired_signals.append(s_row)
            buy_idx = None

    today_buy = False
    today_sell = False
    today_reasons = []
    today_score = 0
    today_buy_score = 0
    today_sell_score = 0

    if signals:
        last_idx, last_sig = signals[-1]
        last_reasons = reasons_list[-1] if reasons_list else {}
        last_date_str = _bar_date_str(dates, last_idx) if dates is not None else ''
        if last_sig == 'B':
            today_buy = True
            today_reasons = (
                _v2_decorate_triggered_names(last_reasons.get('buy_triggered', []), last_reasons, buy_side=True)
                + last_reasons.get('mandatory_buy_triggered', [])
            )
            today_buy_score = last_reasons.get('buy_score', 0)
        elif last_sig == 'S':
            today_sell = True
            today_reasons = (
                _v2_decorate_triggered_names(last_reasons.get('sell_triggered', []), last_reasons, buy_side=False)
                + last_reasons.get('mandatory_sell_triggered', [])
            )
            today_sell_score = last_reasons.get('sell_score', 0)
        today_score = max(today_buy_score, today_sell_score)

    conditions = get_conditions()

    rets = _trade_net_returns(signals, close, open_vals)
    prd_metrics = {}
    if rets.size > 0:
        prd_metrics = {
            'prd_utility': round(float(np.sum(rets) - V2_UTILITY_LAM_MDD * _max_dd_on_rets(rets)
                - V2_UTILITY_LAM_DOWN * _downside_sigma(rets) - V2_UTILITY_LAM_TURN * rets.size), 6),
            'prd_max_drawdown': round(_max_dd_on_rets(rets), 6),
        }

    return {
        'paired_signals': paired_signals,
        'all_signals': paired_signals,
        'prd_metrics': prd_metrics,
        'today_buy': today_buy,
        'today_sell': today_sell,
        'today_reasons': today_reasons,
        'today_score': today_score,
        'today_buy_score': today_buy_score,
        'today_sell_score': today_sell_score,
        'conditions': conditions,
        'today_rules': today_reasons,
        'rule_stats': {},
        'depth_used': len(df),
        'summary': {'total_signals': len(paired_signals), 'buy_count': sum(1 for s in paired_signals if s['type'] == 'B'), 'sell_count': sum(1 for s in paired_signals if s['type'] == 'S')},
        'engine_mode': engine_mode,
    }


def analyze_signals_dual(df, precomputed_weights=None, start_date=None, end_date=None):
    start_offset = 60
    mb_v3 = _mandatory_buy_rules_v3(df, start_offset)
    r2 = analyze_signals_v2(df, precomputed_weights, start_date, end_date, engine_mode='v2')
    r3 = analyze_signals_v2(df, precomputed_weights, start_date, end_date,
                            engine_mode='v3', mandatory_buy_rules_override=mb_v3)
    r2['portfolio_sim'] = _portfolio_sim_from_paired(r2['paired_signals'])
    r3['portfolio_sim'] = _portfolio_sim_from_paired(r3['paired_signals'])
    return {'v2': r2, 'v3': r3}


def get_conditions():
    return {
        'buy_necessary': [
            {'name': '非阴线', 'description': '当日收盘价 > 开盘价'},
            {'name': '上影线不过长', 'description': '信号日或确认买入日：上影线>实体×3/4，或上影线≥当日振幅×40%（振幅过窄时不启用占比条款），则不确认/不产生B'},
            {'name': '缩量确认须价涨', 'description': '确认买入日：若成交量相对信号日缩小，则确认日收盘价须高于信号日收盘价，否则不确认B'},
            {'name': '突破5日线', 'description': '当日收盘价 > MA5'},
            {'name': '至下一前高空间', 'description': '全部买入：信号日最高价至其后下一局部前高（high峰）涨幅须≥3%，否则放弃B'},
            {'name': '阳线确认', 'description': '买入信号后等1天，下一根交易日为阳线才确认B点（成交价为下一根K线开盘价可配）；若当日命中「看涨K线形态」，则须在「日线序列中紧邻的下一根交易日」以收盘>开盘确认（含周五信号→周一确认）；其它规则仍要求收盘高于信号日。若加权买入规则同日命中≥2条、第一确认日未通过且未跌破前低带与信号收盘−ATR止损较强位，则顺延至再下一交易日最多多确认1天'},
            {'name': '非下跌趋势', 'description': 'MA20上升或价格 > MA20'},
            {'name': '高波动', 'description': 'ATR/收盘高于历史中位数时，买入置信阈值提高、最小持有略延长'},
        ],
        'buy_sufficient': [
            {'name': '关键形态必买', 'description': 'W底突破、头肩底突破、V反底部、旗形突破；其中「V反底部」须收复左侧跌前参考高点一定比例（左侧压力取低点前最多5根K之最高）、若已站上MA20则须自低点以来出现过对MA20的回踩且收盘仍在其上，且排除连阳缩量链'},
            {'name': '加权买入', 'description': '分组 cap 后 buy_score > sell_score 且 buy_score > 0；其中「看涨K线形态」在结果中附带 TA-Lib 子形态名（看涨吞没/晨星/锤头/刺透/晨十字/倒锤头，先命中为准）；子形态「刺透」为先命中时须同日共振：其它加权买入、必买、或同一根K上另有其它看涨子形态同时为真'},
            {'name': '双针探底', 'description': '须在V/W左侧底部附近、跌段总跌幅≥10%；两根探底针+信号日阳线且下影/实体>1；T+1阳线且收盘>MA5确认B'},
            {'name': 'N字形突破', 'description': '不可单独脱离结构：须在最近 W底右侧 或 V反右侧 背景下，前高=右侧起点至信号日(含)的 high 最高值；曾触瓶口后 N 字回踩再突破（W 须 high 突破至昨前高，V 为瓶口回踩收阳），带量'},
        ],
        'sell_necessary': [
            {'name': '当前持仓', 'description': '最近一个信号为B'},
        ],
        'sell_sufficient': [
            {'name': '关键形态必卖', 'description': 'M顶跌破、V反顶部、旗形跌破、均线趋势打破必卖'},
            {'name': '沿均线主升破位', 'description': '识别主升贴 MA5/10/20 后首次跌破对应均线，计入卖出规则加权分（需满足计数卖等条件）'},
            {'name': '计数卖出', 'description': '≥2条卖出规则触发且sell_score > buy_score；浮亏超阈值时降为1条'},
            {'name': '黄昏星缩量过滤', 'description': '「看跌K线形态」子形态为黄昏星时：若前一日为阳线(收盘>开盘)且当日成交量小于前一日，则不因该形态触发卖出（其它卖出规则仍可S）'},
            {'name': 'ATR止损', 'description': '跌破入场价−k×ATR(14)强制平仓'},
            {'name': '时间止损', 'description': f'持仓超过约{V2_MAX_HOLD_DAYS}个交易日强制平仓'},
        ],
        'optimization': [
            {'name': 'PRD效用U', 'description': '迭代选优以 U=Σr−λ1·MDD−λ2·下行波动−λ3·笔数−λ4·L1权重 为主（可关 V2_PRD_UTILITY_OBJECTIVE）'},
        ],
    }


# ============================================================
# 持久化
# ============================================================

def save_weights(stock_code, result):
    doc = {
        'stock_code': stock_code,
        'buy_weights': result['buy_weights'],
        'sell_weights': result['sell_weights'],
        'buy_details': result['buy_details'],
        'sell_details': result['sell_details'],
        'iteration_count': result['iteration_count'],
        'total_return': result['total_return'],
        'win_rate': result['win_rate'],
        'updated_at': datetime.now(),
    }
    for k in ('prd_utility', 'prd_max_drawdown'):
        if k in result:
            doc[k] = result[k]
    weights_v2_collection.update_one(
        {'stock_code': stock_code},
        {'$set': doc, '$setOnInsert': {'created_at': datetime.now()}},
        upsert=True
    )


def load_weights(stock_code):
    return weights_v2_collection.find_one({'stock_code': stock_code}, {'_id': 0})


def delete_weights(stock_code):
    weights_v2_collection.delete_one({'stock_code': stock_code})


def save_signals(stock_code, signals_data):
    if isinstance(signals_data, dict) and 'v2' in signals_data and 'v3' in signals_data:
        doc = {
            'stock_code': stock_code,
            'signals': signals_data['v2']['paired_signals'],
            'signals_v3': signals_data['v3']['paired_signals'],
            'portfolio_v2': signals_data['v2'].get('portfolio_sim'),
            'portfolio_v3': signals_data['v3'].get('portfolio_sim'),
            'updated_at': datetime.now(),
        }
    else:
        doc = {
            'stock_code': stock_code,
            'signals': signals_data,
            'updated_at': datetime.now(),
        }
    signals_v2_collection.update_one(
        {'stock_code': stock_code},
        {'$set': doc, '$setOnInsert': {'created_at': datetime.now()}},
        upsert=True
    )


def load_signals(stock_code):
    return signals_v2_collection.find_one({'stock_code': stock_code}, {'_id': 0})


def delete_signals(stock_code):
    signals_v2_collection.delete_one({'stock_code': stock_code})
