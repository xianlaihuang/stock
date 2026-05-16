import numpy as np
import pandas as pd
import talib
from datetime import datetime
from db import db
from unified_rule_generator import (
    get_or_generate_unified_rules,
    _compute_all_pattern_results,
    load_unified_config,
)
from signal_dtw import DTWDetector
from sell_dtw import SellDTWDetector
import signal_rules

buy_signals_col = db['buy_signals']
sell_signals_col = db['sell_signals']


def clear_all_signals(stock_code):
    buy_signals_col.delete_many({'stock_code': stock_code})
    sell_signals_col.delete_many({'stock_code': stock_code})


def _klines_to_df(klines):
    df = pd.DataFrame({
        'date': [k['date'][:10] if len(k['date']) > 10 else k['date'] for k in klines],
        'open': [float(k['open']) for k in klines],
        'high': [float(k['high']) for k in klines],
        'low': [float(k['low']) for k in klines],
        'close': [float(k['close']) for k in klines],
        'volume': [float(k['volume']) for k in klines],
    })
    return df


def _get_trend_direction(closes, window=20):
    if len(closes) < window:
        return 0
    recent = closes[-window:]
    x = np.arange(window)
    slope, intercept = np.polyfit(x, recent, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((recent - y_pred) ** 2)
    ss_tot = np.sum((recent - np.mean(recent)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    if slope > 0 and r_squared > 0.3:
        return 1
    elif slope < 0 and r_squared > 0.3:
        return -1
    return 0


def _compute_tech_signal_details(df):
    details = {}

    ma_b, ma_s = signal_rules.ma_cross_signal(df)
    details['MA金叉/死叉'] = {'buy': ma_b, 'sell': ma_s}

    macd_b, macd_s = signal_rules.macd_cross_signal(df)
    details['MACD金叉/死叉'] = {'buy': macd_b, 'sell': macd_s}

    rsi_b, rsi_s = signal_rules.rsi_ob_os_signal(df)
    details['RSI超买超卖'] = {'buy': rsi_b, 'sell': rsi_s}

    kdj_b, kdj_s = signal_rules.kdj_cross_signal(df)
    details['KDJ金叉死叉'] = {'buy': kdj_b, 'sell': kdj_s}

    div_b, div_s = signal_rules.macd_divergence_signal(df)
    details['MACD背离'] = {'buy': div_b, 'sell': div_s}

    cp_b, cp_s = signal_rules.candle_pattern_signal(df)
    details['K线形态(技术)'] = {'buy': cp_b, 'sell': cp_s}

    db_b, db_s = signal_rules.double_bottom_signal(df)
    details['W底/M顶'] = {'buy': db_b, 'sell': db_s}

    hs_b, hs_s = signal_rules.head_shoulders_signal(df)
    details['头肩形态'] = {'buy': hs_b, 'sell': hs_s}

    vpt_b, vpt_s = signal_rules.volume_price_trend_signal(df)
    details['量价趋势'] = {'buy': vpt_b, 'sell': vpt_s}

    vb_b, vb_s = signal_rules.volume_breakout_signal(df)
    details['放量突破'] = {'buy': vb_b, 'sell': vb_s}

    evp_b, evp_s = signal_rules.extreme_volume_price_signal(df)
    details['地量地价/天量天价'] = {'buy': evp_b, 'sell': evp_s}

    tl_b, tl_s = signal_rules.trendline_support_resistance_signal(df)
    details['趋势线突破'] = {'buy': tl_b, 'sell': tl_s}

    return details


def scan_enhanced_signals(stock_code, klines, start_date=None, end_date=None):
    config = load_unified_config()
    cfg = config['engine']
    dtw_bottom_cfg = config['dtw_bottom_templates']
    dtw_top_cfg = config['dtw_top_templates']
    names_cn = config.get('CDL_NAMES_CN', {})

    if len(klines) < 30:
        return {
            'stock_code': stock_code,
            'today_buy': False,
            'today_sell': False,
            'today_reasons': [],
            'paired_signals': [],
            'all_signals': [],
            'rule_stats': {},
        }

    rule = get_or_generate_unified_rules(stock_code, klines)
    if not rule:
        rule = {
            'top10_bullish': [], 'top10_bearish': [],
            'dtw_bottom_patterns': [], 'dtw_bottom_threshold': cfg['dtw_default_threshold'],
            'dtw_top_patterns': [], 'dtw_top_threshold': cfg['dtw_default_threshold'],
            'config_snapshot': {},
        }

    top10_bullish = rule.get('top10_bullish', [])
    top10_bearish = rule.get('top10_bearish', [])
    dtw_bottom_patterns = rule.get('dtw_bottom_patterns', [])
    dtw_bottom_threshold = rule.get('dtw_bottom_threshold', cfg['dtw_default_threshold'])
    dtw_top_patterns = rule.get('dtw_top_patterns', [])
    dtw_top_threshold = rule.get('dtw_top_threshold', cfg['dtw_default_threshold'])
    rule_config = rule.get('config_snapshot', {})

    volume_ratio = rule_config.get('volume_ratio', cfg['volume_ratio'])
    volume_ma_period = rule_config.get('volume_ma_period', cfg['volume_ma_period'])
    dtw_window = rule_config.get('dtw_window', cfg['dtw_window'])

    opens = np.array([k['open'] for k in klines], dtype=float)
    highs = np.array([k['high'] for k in klines], dtype=float)
    lows = np.array([k['low'] for k in klines], dtype=float)
    closes = np.array([k['close'] for k in klines], dtype=float)
    volumes = np.array([k['volume'] for k in klines], dtype=float)

    df = _klines_to_df(klines)

    tech_details = _compute_tech_signal_details(df)

    bullish_pattern_names = [p['name'] for p in top10_bullish]
    bearish_pattern_names = [p['name'] for p in top10_bearish]
    all_pattern_names = list(set(bullish_pattern_names + bearish_pattern_names))

    pattern_results = {}
    if all_pattern_names:
        pattern_results = _compute_all_pattern_results(all_pattern_names, opens, highs, lows, closes)

    bullish_info_map = {p['name']: p for p in top10_bullish}
    bearish_info_map = {p['name']: p for p in top10_bearish}

    dtw_bottom_templates_dict = {}
    for name in dtw_bottom_patterns:
        if name in dtw_bottom_cfg:
            dtw_bottom_templates_dict[name] = dtw_bottom_cfg[name]
    buy_detector = DTWDetector(
        dtw_bottom_templates_dict,
        window=dtw_window,
        default_threshold=dtw_bottom_threshold,
    )

    dtw_top_templates_dict = {}
    for name in dtw_top_patterns:
        if name in dtw_top_cfg:
            dtw_top_templates_dict[name] = dtw_top_cfg[name]
    sell_detector = SellDTWDetector(
        dtw_top_templates_dict,
        window=dtw_window,
        default_threshold=dtw_top_threshold,
    )

    total_days = len(klines)
    raw_b_candidates = []
    raw_s_candidates = []

    RULE_WEIGHTS = {
        'MACD金叉/死叉': 3,
        'MA金叉/死叉': 3,
        'RSI超买超卖': 2,
        'KDJ金叉死叉': 2,
        'MACD背离': 4,
        'K线形态(技术)': 2,
        'W底/M顶': 4,
        '头肩形态': 4,
        '量价趋势': 1,
        '放量突破': 3,
        '地量地价/天量天价': 3,
        '趋势线突破': 2,
        '形态(DTW)': 3,
        '形态(CDL)': 2,
    }

    for i in range(total_days):
        date_str = klines[i]['date']
        if len(date_str) > 10:
            date_str = date_str[:10]

        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue

        current_close = float(closes[i])
        is_yangxian = closes[i] > opens[i]
        is_yinxian = closes[i] < opens[i]

        buy_score = 0
        sell_score = 0
        triggered_buy_rules = []
        triggered_sell_rules = []

        for rule_name, sig_data in tech_details.items():
            b_sig = sig_data['buy'].iloc[i] if i < len(sig_data['buy']) else False
            s_sig = sig_data['sell'].iloc[i] if i < len(sig_data['sell']) else False

            if b_sig and not pd.isna(b_sig) and b_sig:
                w = RULE_WEIGHTS.get(rule_name, 1)
                buy_score += w
                triggered_buy_rules.append(rule_name)
            if s_sig and not pd.isna(s_sig) and s_sig:
                w = RULE_WEIGHTS.get(rule_name, 1)
                sell_score += w
                triggered_sell_rules.append(rule_name)

        cdl_triggered_bullish = []
        for pname in bullish_pattern_names:
            if pname in pattern_results:
                result_arr = pattern_results[pname]
                if i < len(result_arr) and result_arr[i] == 100:
                    cdl_triggered_bullish.append(pname)
                    w = RULE_WEIGHTS.get('形态(CDL)', 2)
                    buy_score += w
                    triggered_buy_rules.append(f'CDL-{names_cn.get(pname, pname)}')

        dtw_bottom_matched = []
        if i >= dtw_window - 1:
            window_closes = closes[max(0, i - dtw_window + 1):i + 1]
            matches = buy_detector.match(window_closes.tolist(), threshold=dtw_bottom_threshold)
            for m in matches:
                dtw_bottom_matched.append(m['pattern'])
                w = RULE_WEIGHTS.get('形态(DTW)', 3)
                buy_score += w
                cn = 'V型底' if m['pattern'] == 'V' else 'W底'
                triggered_buy_rules.append(f'DTW-{cn}')

        cdl_triggered_bearish = []
        for pname in bearish_pattern_names:
            if pname in pattern_results:
                result_arr = pattern_results[pname]
                if i < len(result_arr) and result_arr[i] == -100:
                    cdl_triggered_bearish.append(pname)
                    w = RULE_WEIGHTS.get('形态(CDL)', 2)
                    sell_score += w
                    triggered_sell_rules.append(f'CDL-{names_cn.get(pname, pname)}')

        dtw_top_matched = []
        if i >= dtw_window - 1:
            window_closes = closes[max(0, i - dtw_window + 1):i + 1]
            matches = sell_detector.match(window_closes.tolist(), threshold=dtw_top_threshold)
            for m in matches:
                dtw_top_matched.append(m['pattern'])
                w = RULE_WEIGHTS.get('形态(DTW)', 3)
                sell_score += w
                cn = '倒V顶' if m['pattern'] == 'INVERSE_V' else 'M头'
                triggered_sell_rules.append(f'DTW-{cn}')

        vol_enough = False
        vol_ratio_actual = 0
        if i >= volume_ma_period:
            avg_vol = np.mean(volumes[max(0, i - volume_ma_period):i])
            if avg_vol > 0:
                vol_ratio_actual = volumes[i] / avg_vol
                if vol_ratio_actual >= volume_ratio:
                    vol_enough = True
        elif i > 0:
            avg_vol = np.mean(volumes[:i])
            if avg_vol > 0:
                vol_ratio_actual = volumes[i] / avg_vol
                if vol_ratio_actual >= volume_ratio:
                    vol_enough = True

        trend_dir = _get_trend_direction(closes[:i + 1])

        BUY_THRESHOLD = 5
        SELL_THRESHOLD = 5

        final_buy = False
        final_sell = False

        if buy_score >= BUY_THRESHOLD:
            has_high_weight = any(
                r in ['MACD背离', 'W底/M顶', '头肩形态', 'MACD金叉/死叉', 'MA金叉/死叉', '放量突破', '地量地价/天量天价', '趋势线突破', '形态(DTW)']
                for r in triggered_buy_rules
            )
            if has_high_weight or buy_score >= 8:
                if trend_dir != -1:
                    final_buy = True

        if sell_score >= SELL_THRESHOLD:
            has_high_weight_sell = any(
                r in ['MACD背离', 'W底/M顶', '头肩形态', 'MACD金叉/死叉', 'MA金叉/死叉', '放量突破', '地量地价/天量天价', '趋势线突破', '形态(DTW)']
                for r in triggered_sell_rules
            )
            if has_high_weight_sell or sell_score >= 8:
                if trend_dir != 1:
                    final_sell = True

        if final_buy and final_sell:
            if buy_score > sell_score + 2:
                final_sell = False
            elif sell_score > buy_score + 2:
                final_buy = False
            else:
                final_buy = False
                final_sell = False

        if final_buy:
            buy_reasons = []
            for r in triggered_buy_rules:
                if r.startswith('CDL-'):
                    buy_reasons.append(f"看涨{r[4:]}")
                elif r.startswith('DTW-'):
                    buy_reasons.append(f"{r[4:]}匹配")
                else:
                    buy_reasons.append(r)
            if is_yangxian:
                buy_reasons.append("阳线确认")
            if vol_enough:
                buy_reasons.append(f"放量{vol_ratio_actual:.1f}倍")
            if trend_dir == 1:
                buy_reasons.append("上升趋势中")
            buy_reasons.insert(0, f"[综合评分:{buy_score}]")

            raw_b_candidates.append({
                'index': i,
                'date': date_str,
                'close': current_close,
                'type': 'B',
                'reasons': buy_reasons,
                'score': buy_score,
                'rules': triggered_buy_rules,
            })

        if final_sell:
            sell_reasons = []
            for r in triggered_sell_rules:
                if r.startswith('CDL-'):
                    sell_reasons.append(f"看跌{r[4:]}")
                elif r.startswith('DTW-'):
                    sell_reasons.append(f"{r[4:]}匹配")
                else:
                    sell_reasons.append(r)
            if is_yinxian:
                sell_reasons.append("阴线确认")
            if vol_enough:
                sell_reasons.append(f"放量{vol_ratio_actual:.1f}倍")
            if trend_dir == -1:
                sell_reasons.append("下降趋势中")
            sell_reasons.insert(0, f"[综合评分:{sell_score}]")

            raw_s_candidates.append({
                'index': i,
                'date': date_str,
                'close': current_close,
                'type': 'S',
                'reasons': sell_reasons,
                'score': sell_score,
                'rules': triggered_sell_rules,
            })

    valid_signals = []
    expecting = 'B'
    b_idx = 0
    s_idx = 0

    while True:
        if expecting == 'B':
            found = False
            while b_idx < len(raw_b_candidates):
                if not valid_signals or raw_b_candidates[b_idx]['index'] > valid_signals[-1]['index']:
                    valid_signals.append(raw_b_candidates[b_idx])
                    b_idx += 1
                    expecting = 'S'
                    found = True
                    break
                b_idx += 1
            if not found:
                break
        else:
            found = False
            while s_idx < len(raw_s_candidates):
                if raw_s_candidates[s_idx]['index'] > valid_signals[-1]['index']:
                    valid_signals.append(raw_s_candidates[s_idx])
                    s_idx += 1
                    expecting = 'B'
                    found = True
                    break
                s_idx += 1
            if not found:
                break

    MIN_HOLD_DAYS = 3
    filtered_signals = []
    for idx, sig in enumerate(valid_signals):
        if filtered_signals and (sig['index'] - filtered_signals[-1]['index']) < MIN_HOLD_DAYS:
            continue
        filtered_signals.append(sig)
    valid_signals = filtered_signals

    paired_signals = []
    all_signals = []

    for k in range(len(valid_signals)):
        sig = valid_signals[k]

        if sig['type'] == 'B':
            b_return_rate = None
            next_s = None
            for j in range(k + 1, len(valid_signals)):
                if valid_signals[j]['type'] == 'S':
                    next_s = valid_signals[j]
                    break
            if next_s:
                b_return_rate = round(
                    (next_s['close'] - sig['close']) / sig['close'] * 100, 2
                )

            signal_entry = {
                'date': sig['date'],
                'type': 'B',
                'close': sig['close'],
                'reasons': sig['reasons'],
                'return_rate_pct': b_return_rate,
                'score': sig.get('score', 0),
                'rules': sig.get('rules', []),
            }
            if next_s:
                signal_entry['paired_sell_date'] = next_s['date']
                signal_entry['paired_sell_close'] = next_s['close']
                paired_signals.append({
                    'buy_date': sig['date'],
                    'buy_close': sig['close'],
                    'sell_date': next_s['date'],
                    'sell_close': next_s['close'],
                    'buy_return_rate_pct': b_return_rate,
                    'buy_reasons': sig['reasons'],
                    'sell_reasons': next_s['reasons'],
                    'buy_score': sig.get('score', 0),
                    'sell_score': next_s.get('score', 0),
                })

            all_signals.append(signal_entry)

            buy_doc = {
                'stock_code': stock_code,
                'date': sig['date'],
                'close': sig['close'],
                'buy': True,
                'reasons': sig['reasons'],
                'return_rate_pct': b_return_rate,
                'paired_sell_date': next_s['date'] if next_s else None,
                'paired_sell_close': next_s['close'] if next_s else None,
                'score': sig.get('score', 0),
                'triggered_rules': sig.get('rules', []),
            }
            buy_signals_col.replace_one(
                {'stock_code': stock_code, 'date': sig['date']},
                buy_doc,
                upsert=True,
            )

        elif sig['type'] == 'S':
            s_return_rate = None
            next_b = None
            for j in range(k + 1, len(valid_signals)):
                if valid_signals[j]['type'] == 'B':
                    next_b = valid_signals[j]
                    break
            if next_b:
                s_return_rate = round(
                    (sig['close'] - next_b['close']) / sig['close'] * 100, 2
                )

            signal_entry = {
                'date': sig['date'],
                'type': 'S',
                'close': sig['close'],
                'reasons': sig['reasons'],
                'return_rate_pct': s_return_rate,
                'score': sig.get('score', 0),
                'rules': sig.get('rules', []),
            }
            if next_b:
                signal_entry['next_buy_date'] = next_b['date']
                signal_entry['next_buy_close'] = next_b['close']

            all_signals.append(signal_entry)

            sell_doc = {
                'stock_code': stock_code,
                'date': sig['date'],
                'sell': True,
                'close': sig['close'],
                'reasons': sig['reasons'],
                'return_rate_pct': s_return_rate,
                'next_buy_date': next_b['date'] if next_b else None,
                'next_buy_close': next_b['close'] if next_b else None,
                'score': sig.get('score', 0),
                'triggered_rules': sig.get('rules', []),
            }
            sell_signals_col.replace_one(
                {'stock_code': stock_code, 'date': sig['date']},
                sell_doc,
                upsert=True,
            )

    today_buy = False
    today_sell = False
    today_reasons = []
    today_score = 0
    today_rules = []
    if valid_signals:
        last = valid_signals[-1]
        last_date = klines[-1]['date']
        if len(last_date) > 10:
            last_date = last_date[:10]
        if last['date'] == last_date:
            if last['type'] == 'B':
                today_buy = True
                today_reasons = last['reasons']
                today_score = last.get('score', 0)
                today_rules = last.get('rules', [])
            elif last['type'] == 'S':
                today_sell = True
                today_reasons = last['reasons']
                today_score = last.get('score', 0)
                today_rules = last.get('rules', [])

    rule_stats = {}
    for rule_name in RULE_WEIGHTS:
        b_count = sum(1 for s in all_signals if s['type'] == 'B' and rule_name in s.get('rules', []))
        s_count = sum(1 for s in all_signals if s['type'] == 'S' and rule_name in s.get('rules', []))
        rule_stats[rule_name] = {'buy_count': b_count, 'sell_count': s_count}

    return {
        'stock_code': stock_code,
        'today_buy': today_buy,
        'today_sell': today_sell,
        'today_reasons': today_reasons,
        'today_score': today_score,
        'today_rules': today_rules,
        'paired_signals': paired_signals,
        'all_signals': all_signals,
        'rule_stats': rule_stats,
    }


scan_unified_signals = scan_enhanced_signals


def scan_dynamic_weight_signals(stock_code, klines, start_date=None, end_date=None, precomputed_weights=None):
    import dynamic_weight_engine as dwe
    from scraper import StockScraper

    df = _klines_to_df(klines)
    min_data_needed = 200
    if len(df) < min_data_needed:
        print(f"[动态权重] {stock_code} 数据量不足({len(df)}条)，尝试补充历史数据...")
        try:
            from models import KlineData
            first_date = klines[0]['date'] if klines else None
            extra_klines = StockScraper.get_kline_data(stock_code, period='day',
                                                        count=800, start_date=None)
            if extra_klines and len(extra_klines) > len(klines):
                KlineData.delete(stock_code, period='day')
                KlineData.add_many(stock_code, extra_klines, period='day')
                df = _klines_to_df(extra_klines)
                print(f"[动态权重] 补充完成，数据量: {len(df)}条")
        except Exception as e:
            print(f"[动态权重] 补充数据失败: {e}")

    if len(df) < min_data_needed:
        return {
            'stock_code': stock_code,
            'today_buy': False,
            'today_sell': False,
            'today_reasons': [f'数据不足，需要至少{min_data_needed}条日线数据（当前{len(df)}条），请点击"抓取数据"补充历史数据'],
            'today_score': 0,
            'today_rules': [],
            'paired_signals': [],
            'all_signals': [],
            'rule_stats': {},
        }

    print(f"[动态权重] 开始分析 {stock_code}, 数据量: {len(df)}")

    if precomputed_weights:
        buy_rules_dict = dwe.get_all_rules()[0]
        sell_rules_dict = dwe.get_all_rules()[1]
        _, _, mandatory_sell_dict, buy_restriction_dict, _ = dwe.get_all_rules_extended()
        active_buy = {n: buy_rules_dict[n] for n in precomputed_weights.get('buy_rules', {}) if n in buy_rules_dict}
        active_sell = {n: sell_rules_dict[n] for n in precomputed_weights.get('sell_rules', {}) if n in sell_rules_dict}
        buy_w = precomputed_weights.get('buy_weights', {})
        sell_w = precomputed_weights.get('sell_weights', {})

        signals, reasons = dwe.generate_signals_with_weights(
            df, active_buy, buy_w, active_sell, sell_w,
            buy_details=precomputed_weights.get('buy_details'),
            sell_details=precomputed_weights.get('sell_details'),
            mandatory_sell_rules=mandatory_sell_dict,
            buy_restriction_rules=buy_restriction_dict,
            sell_trigger_count=2,
        )

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

        buy_return_map = {}
        for p in paired_signals:
            buy_return_map[p['buy_date']] = p['return_pct']

        summary = {
            'total_signals': len(signals),
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'paired_count': len(paired_signals),
            'depth_used': 0,
            'active_buy_rules': list(active_buy.keys()),
            'active_sell_rules': list(active_sell.keys()),
            'buy_weights': buy_w,
            'sell_weights': sell_w,
        }
        opt = {'depth_used': 0, 'all_buy_stats': {}, 'all_sell_stats': {}}
    else:
        system_result = dwe.run_system(
            df,
            max_depth=200,
            step=10,
        )

        summary = system_result['summary']
        opt = system_result['optimization']
        paired_signals = system_result['paired_signals']
        all_reasons = system_result['reasons']
        signals = system_result['signals']
        reasons = all_reasons

        buy_return_map = {}
        for p in paired_signals:
            buy_return_map[p['buy_date']] = p['return_pct']

    all_signals = []
    for i, (sig_idx, sig_type) in enumerate(signals):
        row = df.iloc[sig_idx]
        reason = reasons[i] if i < len(reasons) else {}
        sig_date = str(row.get('date', sig_idx))

        return_rate_pct = None
        if sig_type == 'B' and sig_date in buy_return_map:
            return_rate_pct = buy_return_map[sig_date]
        elif sig_type == 'S':
            for p in paired_signals:
                if p.get('sell_date') == sig_date:
                    return_rate_pct = p['return_pct']
                    break

        signal_entry = {
            'date': sig_date,
            'type': sig_type,
            'close': float(row['close']),
            'reasons': reason.get('buy_triggered' if sig_type == 'B' else 'sell_triggered', []),
            'score': round(reason.get('buy_score' if sig_type == 'B' else 'sell_score', 0), 4),
            'rules': reason.get('buy_triggered' if sig_type == 'B' else 'sell_triggered', []),
            'buy_score': round(reason.get('buy_score', 0), 4),
            'sell_score': round(reason.get('sell_score', 0), 4),
            'return_rate_pct': return_rate_pct,
            'confidence': reason.get('confidence', 0),
            'level': reason.get('level', ''),
            'sell_reason_type': reason.get('sell_reason_type', ''),
            'mandatory_sell_triggered': reason.get('mandatory_sell_triggered', []),
            'buy_restriction_triggered': reason.get('buy_restriction_triggered', []),
        }
        all_signals.append(signal_entry)

    today_buy = False
    today_sell = False
    today_reasons = []
    today_score = 0
    today_rules = []
    today_buy_score = 0
    today_sell_score = 0

    if all_signals:
        last = all_signals[-1]
        last_date = klines[-1]['date']
        if len(last_date) > 10:
            last_date = last_date[:10]
        if last['date'] == last_date:
            if last['type'] == 'B':
                today_buy = True
                today_reasons = last['reasons']
                today_score = last['score']
                today_rules = last['rules']
                today_buy_score = last.get('buy_score', 0)
                today_sell_score = last.get('sell_score', 0)
            elif last['type'] == 'S':
                today_sell = True
                today_reasons = last['reasons']
                today_score = last['score']
                today_rules = last['rules']
                today_buy_score = last.get('buy_score', 0)
                today_sell_score = last.get('sell_score', 0)

    rule_stats = {}
    for name in summary.get('active_buy_rules', []):
        s = opt.get('all_buy_stats', {}).get(name, {})
        rule_stats[f"买-{name}"] = {'weight': round(summary['buy_weights'].get(name, 0), 4),
                                      'win_rate': round(s.get('win_rate', 0), 4),
                                      'avg_return': round(s.get('avg_return', 0), 4),
                                      'count': s.get('count', 0)}
    for name in summary.get('active_sell_rules', []):
        s = opt.get('all_sell_stats', {}).get(name, {})
        rule_stats[f"卖-{name}"] = {'weight': round(summary['sell_weights'].get(name, 0), 4),
                                     'win_rate': round(s.get('win_rate', 0), 4),
                                     'avg_return': round(s.get('avg_return', 0), 4),
                                     'count': s.get('count', 0)}

    result = {
        'stock_code': stock_code,
        'today_buy': today_buy,
        'today_sell': today_sell,
        'today_reasons': today_reasons,
        'today_score': today_score,
        'today_rules': today_rules,
        'today_buy_score': today_buy_score,
        'today_sell_score': today_sell_score,
        'paired_signals': paired_signals,
        'all_signals': all_signals,
        'rule_stats': rule_stats,
        'summary': summary,
        'depth_used': opt.get('depth_used', 200),
    }
    return result
