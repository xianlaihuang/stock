import pandas as pd
import numpy as np
import talib
from scipy import stats
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import dynamic_weight_engine as dwe_v1

warnings.filterwarnings('ignore')


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


def _precompute_regimes(df, start_offset=60):
    n = len(df)
    regimes = ['unknown'] * n
    for idx in range(start_offset, n):
        regimes[idx] = classify_market_regime(df, idx)
    return regimes


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


def _find_extremes_by_depth(df, depth):
    close = df['close'].values.astype(float)
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


def _graduated_penalty(win_rate, avg_return, profit_factor, count,
                       min_count=3):
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
        buy_rules = dwe_v1.BUY_RULES
    if sell_rules is None:
        sell_rules = dwe_v1.SELL_RULES

    print(f"[V2] 预计算规则信号表...")
    buy_rule_signals = _precompute_rule_signals(df, buy_rules, start_offset=60)
    sell_rule_signals = _precompute_rule_signals(df, sell_rules, start_offset=60)
    mandatory_buy_signals = _precompute_rule_signals(df, dwe_v1.MANDATORY_BUY_RULES, start_offset=60)
    dwe_v1.apply_piercing_requires_confluence_buy(
        df, buy_rule_signals, buy_rules, 60, mandatory_buy_pre=mandatory_buy_signals)
    dwe_v1.apply_macd_vp_requires_pattern_breakout_buy(
        df, buy_rule_signals, buy_rules, 60, mandatory_buy_pre=mandatory_buy_signals)
    dwe_v1.apply_all_buy_next_high_room_filter(
        df, buy_rule_signals, buy_rules, 60, mandatory_buy_pre=mandatory_buy_signals)
    print(f"[V2] 规则信号表完成 (买入{len(buy_rules)}, 卖出{len(sell_rules)})")

    print(f"[V2] 预计算质量评分...")
    buy_quality, sell_quality = _precompute_quality_scores(df, buy_rule_signals, sell_rule_signals)
    print(f"[V2] 质量评分完成")

    print(f"[V2] 预计算市场状态...")
    regimes = _precompute_regimes(df)
    print(f"[V2] 市场状态完成")

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


def calculate_weights_v2(stock_data, max_depth=200, step=10,
                         window=5, decay_factor=3.0,
                         min_hold=3, max_hold=20,
                         stop_loss_pct=0.05, take_profit_pct=0.10):
    df = stock_data.copy()
    if isinstance(df, pd.DataFrame):
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

    buy_rules, sell_rules = dwe_v1.get_all_rules()

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
        'buy_rules': opt_result.get('buy_rules', dict(buy_rules)),
        'sell_rules': opt_result.get('sell_rules', dict(sell_rules)),
        'depth_used': opt_result['depth_used'],
        'found_strict': True,
    }


def _classify_level(weight, all_weights):
    if weight <= 0:
        return '无效'
    vals = sorted([v for v in all_weights.values() if v > 0], reverse=True)
    if not vals:
        return '无效'
    rank = 0
    for v in vals:
        if weight >= v:
            break
        rank += 1
    ratio = rank / max(len(vals), 1)
    if ratio < 0.3:
        return '核心'
    elif ratio < 0.6:
        return '重要'
    else:
        return '辅助'


def generate_signals_v2(df, buy_rules, buy_weights, sell_rules, sell_weights,
                        min_interval=3, ma_period=20, start_offset=60,
                        confidence_medium=0.55, confidence_strong=0.70,
                        min_score_threshold=0.05,
                        buy_details=None, sell_details=None,
                        min_triggered_win_rate=0.45):
    buy_rule_signals = _precompute_rule_signals(df, buy_rules, start_offset)
    sell_rule_signals = _precompute_rule_signals(df, sell_rules, start_offset)
    mandatory_buy_signals = _precompute_rule_signals(df, dwe_v1.MANDATORY_BUY_RULES, start_offset)
    dwe_v1.apply_piercing_requires_confluence_buy(
        df, buy_rule_signals, buy_rules, start_offset, mandatory_buy_pre=mandatory_buy_signals)
    dwe_v1.apply_macd_vp_requires_pattern_breakout_buy(
        df, buy_rule_signals, buy_rules, start_offset, mandatory_buy_pre=mandatory_buy_signals)
    dwe_v1.apply_all_buy_next_high_room_filter(
        df, buy_rule_signals, buy_rules, start_offset, mandatory_buy_pre=mandatory_buy_signals)

    close = df['close'].values.astype(float)
    n = len(df)
    ma = talib.MA(close, timeperiod=ma_period)

    signals = []
    trade_history = []

    for idx in range(start_offset, n):
        buy_score = 0.0
        sell_score = 0.0
        triggered_buy = []
        triggered_sell = []

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

        candidate = None
        confidence = 0
        if buy_score > 0 and buy_score > sell_score and dwe_v1.buy_weighted_combo_gate_ok(triggered_buy):
            candidate = 'B'
            total = buy_score + sell_score
            confidence = buy_score / total if total > 0 else 0
        elif sell_score > 0 and sell_score > buy_score:
            candidate = 'S'
            total = buy_score + sell_score
            confidence = sell_score / total if total > 0 else 0

        if candidate is not None:
            if idx < ma_period or pd.isna(ma[idx]) or pd.isna(ma[idx - 1]):
                candidate = None
            else:
                trend_up = (ma[idx] > ma[idx - 1]) and (close[idx] > ma[idx])
                trend_down = (ma[idx] < ma[idx - 1]) and (close[idx] < ma[idx])
                if candidate == 'B' and trend_down:
                    candidate = None
                if candidate == 'S' and trend_up:
                    candidate = None

        if candidate is not None:
            dominant_score = buy_score if candidate == 'B' else sell_score
            if dominant_score < min_score_threshold:
                candidate = None

        if candidate is not None and buy_details is not None and sell_details is not None:
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
            if idx - last_idx < min_interval:
                candidate = None
            elif last_signal == candidate:
                candidate = None

        if candidate == 'B' and dwe_v1.upper_shadow_vetoes_buy_signal(df, idx):
            candidate = None

        if candidate and confidence >= confidence_medium:
            level = 'strong' if confidence >= confidence_strong else 'medium'
            trade_history.append((idx, candidate))
            signals.append({
                'idx': idx,
                'date': df['date'].iloc[idx] if 'date' in df.columns else str(idx),
                'type': candidate,
                'confidence': round(confidence, 3),
                'level': level,
                'buy_score': round(buy_score, 4),
                'sell_score': round(sell_score, 4),
                'buy_triggered': triggered_buy,
                'sell_triggered': triggered_sell,
            })

    return signals


def evaluate_signals(signals, df, direction='buy',
                     min_hold=3, max_hold=20,
                     stop_loss_pct=0.05, take_profit_pct=0.10):
    close = df['close'].values.astype(float)
    n = len(close)
    results = []

    for sig in signals:
        idx = sig['idx']
        if idx + min_hold >= n:
            continue
        entry_price = close[idx]
        if entry_price <= 0:
            continue

        exit_price = entry_price
        exit_day = min_hold
        for hold in range(min_hold, max_hold + 1):
            if idx + hold >= n:
                break
            current_price = close[idx + hold]
            ret = (current_price - entry_price) / entry_price
            if sig['type'] == 'S':
                ret = -ret
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
        if sig['type'] == 'S':
            final_ret = -final_ret
        results.append({
            **sig,
            'return': final_ret,
            'hold_days': exit_day,
        })

    if not results:
        return {'win_rate': 0, 'avg_return': 0, 'count': 0,
                'profit_factor': 0, 'strong_wr': 0, 'medium_wr': 0}

    wins = [r for r in results if r['return'] > 0]
    losses = [r for r in results if r['return'] <= 0]
    total_profit = sum(r['return'] for r in wins) if wins else 0
    total_loss = abs(sum(r['return'] for r in losses)) if losses else 0.001

    strong_signals = [r for r in results if r.get('level') == 'strong']
    medium_signals = [r for r in results if r.get('level') == 'medium']
    strong_wr = sum(1 for r in strong_signals if r['return'] > 0) / len(strong_signals) if strong_signals else 0
    medium_wr = sum(1 for r in medium_signals if r['return'] > 0) / len(medium_signals) if medium_signals else 0

    return {
        'win_rate': len(wins) / len(results),
        'avg_return': np.mean([r['return'] for r in results]),
        'count': len(results),
        'profit_factor': total_profit / total_loss,
        'strong_wr': strong_wr,
        'medium_wr': medium_wr,
        'strong_count': len(strong_signals),
        'medium_count': len(medium_signals),
    }
