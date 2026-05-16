import numpy as np
import talib
from datetime import datetime
from db import db
from sell_rule_generator import get_or_generate_sell_rules, _compute_pattern as sell_compute_pattern, get_bearish_pattern_cn, load_sell_config
from sell_dtw import SellDTWDetector
from signal_rule_generator import get_or_generate_rules, _compute_pattern as buy_compute_pattern, get_pattern_cn
from signal_dtw import DTWDetector
from signal_config_loader import load_signal_config


buy_signals_col = db['buy_signals']
sell_signals_col = db['sell_signals']


def clear_sell_signals(stock_code):
    sell_signals_col.delete_many({'stock_code': stock_code})


def scan_sell_signals(stock_code, klines, start_date=None, end_date=None):
    sell_config = load_sell_config()
    sell_cfg = sell_config['sell_engine']
    dtw_top_cfg = sell_config['dtw_top_templates']

    buy_config = load_signal_config()
    buy_cfg = buy_config['signal_engine']
    dtw_bottom_cfg = buy_config['dtw_templates']

    if len(klines) < 30:
        return {
            'stock_code': stock_code,
            'today_sell': False,
            'today_reasons': [],
            'paired_signals': [],
            'all_signals': [],
        }

    sell_rule = get_or_generate_sell_rules(stock_code, klines)
    if not sell_rule:
        return {
            'stock_code': stock_code,
            'today_sell': False,
            'today_reasons': [],
            'paired_signals': [],
            'all_signals': [],
        }

    buy_rule = get_or_generate_rules(stock_code, klines)
    if not buy_rule:
        return {
            'stock_code': stock_code,
            'today_sell': False,
            'today_reasons': [],
            'paired_signals': [],
            'all_signals': [],
        }

    top10_bearish = sell_rule.get('top10_bearish', [])
    dtw_top_patterns = sell_rule.get('dtw_top_patterns', [])
    dtw_threshold = sell_rule.get('dtw_threshold', sell_config['sell_engine']['dtw_default_threshold'])
    rule_config = sell_rule.get('config_snapshot', {})

    top10_bullish = buy_rule.get('top10_bullish', [])
    dtw_bottom_patterns = buy_rule.get('dtw_patterns', [])
    dtw_bottom_threshold = buy_rule.get('dtw_threshold', buy_config['signal_engine']['dtw_default_threshold'])
    buy_rule_config = buy_rule.get('config_snapshot', {})

    volume_ratio = rule_config.get('volume_ratio', sell_cfg['volume_ratio'])
    volume_ma_period = rule_config.get('volume_ma_period', sell_cfg['volume_ma_period'])
    dtw_window = rule_config.get('dtw_window', sell_cfg['dtw_window'])

    buy_volume_ratio = buy_rule_config.get('volume_ratio', buy_cfg['volume_ratio'])
    buy_volume_ma_period = buy_rule_config.get('volume_ma_period', buy_cfg['volume_ma_period'])
    buy_dtw_window = buy_rule_config.get('dtw_window', buy_cfg['dtw_window'])

    dtw_top_templates = {}
    for name in dtw_top_patterns:
        if name in dtw_top_cfg:
            dtw_top_templates[name] = dtw_top_cfg[name]
    sell_detector = SellDTWDetector(dtw_top_templates, window=dtw_window, default_threshold=dtw_threshold)

    dtw_bottom_templates = {}
    for name in dtw_bottom_patterns:
        if name in dtw_bottom_cfg:
            dtw_bottom_templates[name] = dtw_bottom_cfg[name]
    buy_detector = DTWDetector(dtw_bottom_templates, window=buy_dtw_window, default_threshold=dtw_bottom_threshold)

    opens = np.array([k['open'] for k in klines], dtype=float)
    highs = np.array([k['high'] for k in klines], dtype=float)
    lows = np.array([k['low'] for k in klines], dtype=float)
    closes = np.array([k['close'] for k in klines], dtype=float)
    volumes = np.array([k['volume'] for k in klines], dtype=float)

    bearish_pattern_results = {}
    for p in top10_bearish:
        result = sell_compute_pattern(p['name'], opens, highs, lows, closes)
        if result is not None:
            bearish_pattern_results[p['name']] = result

    bullish_pattern_results = {}
    for p in top10_bullish:
        result = buy_compute_pattern(p['name'], opens, highs, lows, closes)
        if result is not None:
            bullish_pattern_results[p['name']] = result

    total_days = len(klines)

    raw_b_candidates = []
    raw_s_candidates = []

    for i in range(total_days):
        date_str = klines[i]['date']
        if len(date_str) > 10:
            date_str = date_str[:10]

        current_close = float(closes[i])

        triggered_bullish = []
        for p in top10_bullish:
            pname = p['name']
            if pname in bullish_pattern_results:
                result_arr = bullish_pattern_results[pname]
                if i < len(result_arr) and result_arr[i] == 100:
                    triggered_bullish.append(pname)

        dtw_bottom_matched = []
        if i >= buy_dtw_window - 1:
            window_closes = closes[max(0, i - buy_dtw_window + 1):i + 1]
            matches = buy_detector.match(window_closes.tolist(), threshold=dtw_bottom_threshold)
            for m in matches:
                dtw_bottom_matched.append(m['pattern'])

        is_yangxian = closes[i] > opens[i]

        buy_vol_enough = False
        buy_vol_ratio_actual = 0
        if i >= buy_volume_ma_period:
            avg_vol = np.mean(volumes[max(0, i - buy_volume_ma_period):i])
            if avg_vol > 0:
                buy_vol_ratio_actual = volumes[i] / avg_vol
                if buy_vol_ratio_actual >= buy_volume_ratio:
                    buy_vol_enough = True
        elif i > 0:
            avg_vol = np.mean(volumes[:i])
            if avg_vol > 0:
                buy_vol_ratio_actual = volumes[i] / avg_vol
                if buy_vol_ratio_actual >= buy_volume_ratio:
                    buy_vol_enough = True

        buy_pattern_or_dtw = len(triggered_bullish) > 0 or len(dtw_bottom_matched) > 0
        buy_condition_met = bool(buy_pattern_or_dtw and is_yangxian and buy_vol_enough)

        if buy_condition_met:
            buy_reasons = []
            for pname in triggered_bullish:
                p_info = next((p for p in top10_bullish if p['name'] == pname), None)
                if p_info:
                    cn = get_pattern_cn(pname)
                    wr = f"{p_info['win_rate'] * 100:.0f}%"
                    buy_reasons.append(f"形态：{cn}（胜率{wr}）")
            for pattern_name in dtw_bottom_matched:
                pattern_cn = 'V型底' if pattern_name == 'V' else 'W底'
                buy_reasons.append(f"DTW匹配{pattern_cn}")
            if is_yangxian:
                buy_reasons.append("右侧阳线确认")
            if buy_vol_enough:
                buy_reasons.append(f"成交量放大{buy_vol_ratio_actual:.1f}倍")

            raw_b_candidates.append({
                'index': i,
                'date': date_str,
                'close': current_close,
                'type': 'B',
                'reasons': buy_reasons,
            })

        triggered_bearish = []
        sell_reasons = []
        for p in top10_bearish:
            pname = p['name']
            if pname in bearish_pattern_results:
                result_arr = bearish_pattern_results[pname]
                if i < len(result_arr) and result_arr[i] == -100:
                    cn = get_bearish_pattern_cn(pname)
                    wr = f"{p['win_rate'] * 100:.0f}%"
                    triggered_bearish.append(pname)
                    sell_reasons.append(f"看跌形态：{cn}（胜率{wr}）")

        dtw_top_matched = []
        if i >= dtw_window - 1:
            window_closes = closes[max(0, i - dtw_window + 1):i + 1]
            matches = sell_detector.match(window_closes.tolist(), threshold=dtw_threshold)
            for m in matches:
                dtw_top_matched.append(m['pattern'])
                pattern_cn = '倒V顶' if m['pattern'] == 'INVERSE_V' else 'M头'
                sell_reasons.append(f"DTW匹配{pattern_cn}（距离{m['distance']:.3f}）")

        is_yinxian = closes[i] < opens[i]
        if is_yinxian:
            sell_reasons.append("右侧阴线确认")

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

        pattern_or_dtw = len(triggered_bearish) > 0 or len(dtw_top_matched) > 0
        sell_condition_met = pattern_or_dtw and is_yinxian and vol_enough

        if sell_condition_met:
            raw_s_candidates.append({
                'index': i,
                'date': date_str,
                'close': current_close,
                'type': 'S',
                'reasons': sell_reasons,
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
                b_return_rate = round((next_s['close'] - sig['close']) / sig['close'] * 100, 2)

            signal_entry = {
                'date': sig['date'],
                'type': 'B',
                'close': sig['close'],
                'reasons': sig['reasons'],
                'return_rate_pct': b_return_rate,
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
            }
            buy_signals_col.replace_one(
                {'stock_code': stock_code, 'date': sig['date']},
                buy_doc,
                upsert=True
            )

        elif sig['type'] == 'S':
            s_return_rate = None
            next_b = None
            for j in range(k + 1, len(valid_signals)):
                if valid_signals[j]['type'] == 'B':
                    next_b = valid_signals[j]
                    break
            if next_b:
                s_return_rate = round((sig['close'] - next_b['close']) / sig['close'] * 100, 2)

            signal_entry = {
                'date': sig['date'],
                'type': 'S',
                'close': sig['close'],
                'reasons': sig['reasons'],
                'return_rate_pct': s_return_rate,
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
            }
            sell_signals_col.replace_one(
                {'stock_code': stock_code, 'date': sig['date']},
                sell_doc,
                upsert=True
            )

    today_sell = False
    today_reasons = []
    if valid_signals and valid_signals[-1]['type'] == 'S':
        today_sell = True
        today_reasons = valid_signals[-1]['reasons']

    return {
        'stock_code': stock_code,
        'today_sell': today_sell,
        'today_reasons': today_reasons,
        'paired_signals': paired_signals,
        'all_signals': all_signals,
    }


def _get_sell_signals_from_db(stock_code, start_date=None, end_date=None):
    query = {'stock_code': stock_code, 'sell': True}
    if start_date:
        query['date'] = {'$gte': start_date}
    if end_date:
        if 'date' not in query:
            query['date'] = {}
        query['date']['$lte'] = end_date
    docs = list(sell_signals_col.find(query, {'_id': 0}).sort('date', 1))
    return [{
        'date': d['date'],
        'close': d['close'],
        'sell': True,
        'reasons': d.get('reasons', []),
        'return_rate_pct': d.get('return_rate_pct'),
        'next_buy_date': d.get('next_buy_date'),
        'next_buy_close': d.get('next_buy_close'),
    } for d in docs]
