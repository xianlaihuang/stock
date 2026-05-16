import numpy as np
import talib
from datetime import datetime
from db import db
from signal_rule_generator import get_or_generate_rules, _compute_pattern, get_pattern_cn
from signal_dtw import DTWDetector
from signal_config_loader import load_signal_config


buy_signals_col = db['buy_signals']


def scan_signals(stock_code, klines, start_date=None, end_date=None):
    config = load_signal_config()
    cfg = config['signal_engine']
    dtw_cfg = config['dtw_templates']

    if len(klines) < 30:
        return {'stock_code': stock_code, 'today_buy': False, 'today_reasons': [], 'signals': []}

    rule = get_or_generate_rules(stock_code, klines)
    if not rule:
        return {'stock_code': stock_code, 'today_buy': False, 'today_reasons': [], 'signals': []}

    top10 = rule['top10_bullish']
    dtw_patterns = rule.get('dtw_patterns', [])
    dtw_threshold = rule.get('dtw_threshold', config['signal_engine']['dtw_default_threshold'])
    rule_config = rule.get('config_snapshot', {})

    hold_period = rule_config.get('hold_period', cfg['hold_period'])
    volume_ratio = rule_config.get('volume_ratio', cfg['volume_ratio'])
    volume_ma_period = rule_config.get('volume_ma_period', cfg['volume_ma_period'])
    dtw_window = rule_config.get('dtw_window', cfg['dtw_window'])

    dtw_templates = {}
    for name in dtw_patterns:
        if name in dtw_cfg:
            dtw_templates[name] = dtw_cfg[name]
    detector = DTWDetector(dtw_templates, window=dtw_window, default_threshold=dtw_threshold)

    opens = np.array([k['open'] for k in klines], dtype=float)
    highs = np.array([k['high'] for k in klines], dtype=float)
    lows = np.array([k['low'] for k in klines], dtype=float)
    closes = np.array([k['close'] for k in klines], dtype=float)
    volumes = np.array([k['volume'] for k in klines], dtype=float)

    pattern_results = {}
    for p in top10:
        result = _compute_pattern(p['name'], opens, highs, lows, closes)
        if result is not None:
            pattern_results[p['name']] = result

    signals = []
    total_days = len(klines)

    for i in range(total_days):
        date_str = klines[i]['date']
        if len(date_str) > 10:
            date_str = date_str[:10]

        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue

        reasons = []

        triggered_patterns = []
        for p in top10:
            pname = p['name']
            if pname in pattern_results:
                result_arr = pattern_results[pname]
                if i < len(result_arr) and result_arr[i] == 100:
                    cn = get_pattern_cn(pname)
                    wr = f"{p['win_rate'] * 100:.0f}%"
                    triggered_patterns.append(pname)
                    reasons.append(f"形态：{cn}（胜率{wr}）")

        dtw_matched = []
        if i >= dtw_window - 1:
            window_closes = closes[max(0, i - dtw_window + 1):i + 1]
            matches = detector.match(window_closes.tolist(), threshold=dtw_threshold)
            for m in matches:
                dtw_matched.append(m['pattern'])
                pattern_cn = 'V型底' if m['pattern'] == 'V' else 'W底'
                reasons.append(f"DTW匹配{pattern_cn}（距离{m['distance']:.3f}）")

        is_yangxian = closes[i] > opens[i]
        if is_yangxian:
            reasons.append("右侧阳线确认")

        vol_enough = False
        vol_ratio_actual = 0
        if i >= volume_ma_period:
            avg_vol = np.mean(volumes[max(0, i - volume_ma_period):i])
            if avg_vol > 0:
                vol_ratio_actual = volumes[i] / avg_vol
                if vol_ratio_actual >= volume_ratio:
                    vol_enough = True
                    reasons.append(f"成交量放大{vol_ratio_actual:.1f}倍")
        elif i > 0:
            avg_vol = np.mean(volumes[:i])
            if avg_vol > 0:
                vol_ratio_actual = volumes[i] / avg_vol
                if vol_ratio_actual >= volume_ratio:
                    vol_enough = True
                    reasons.append(f"成交量放大{vol_ratio_actual:.1f}倍")

        pattern_or_dtw = len(triggered_patterns) > 0 or len(dtw_matched) > 0
        buy = bool(pattern_or_dtw and is_yangxian and vol_enough)

        signal_doc = {
            'stock_code': stock_code,
            'date': date_str,
            'close': float(closes[i]),
            'buy': buy,
            'reasons': reasons,
            'triggered_patterns': triggered_patterns,
            'dtw_matched': dtw_matched,
            'is_yangxian': bool(is_yangxian),
            'vol_ratio': round(float(vol_ratio_actual), 2),
        }

        if buy:
            buy_signals_col.replace_one(
                {'stock_code': stock_code, 'date': date_str},
                signal_doc,
                upsert=True
            )

        if start_date or end_date:
            signals.append({
                'date': date_str,
                'close': float(closes[i]),
                'buy': buy,
                'reasons': reasons
            })

    last_signal = None
    if klines:
        last_date = klines[-1]['date']
        if len(last_date) > 10:
            last_date = last_date[:10]
        last_idx = len(klines) - 1

        last_reasons = []
        last_triggered = []
        for p in top10:
            pname = p['name']
            if pname in pattern_results:
                result_arr = pattern_results[pname]
                if last_idx < len(result_arr) and result_arr[last_idx] == 100:
                    cn = get_pattern_cn(pname)
                    wr = f"{p['win_rate'] * 100:.0f}%"
                    last_triggered.append(pname)
                    last_reasons.append(f"形态：{cn}（胜率{wr}）")

        last_dtw = []
        if last_idx >= dtw_window - 1:
            window_closes = closes[max(0, last_idx - dtw_window + 1):last_idx + 1]
            matches = detector.match(window_closes.tolist(), threshold=dtw_threshold)
            for m in matches:
                last_dtw.append(m['pattern'])
                pattern_cn = 'V型底' if m['pattern'] == 'V' else 'W底'
                last_reasons.append(f"DTW匹配{pattern_cn}（距离{m['distance']:.3f}）")

        last_yangxian = closes[last_idx] > opens[last_idx]
        if last_yangxian:
            last_reasons.append("右侧阳线确认")

        last_vol_enough = False
        last_vol_ratio = 0
        if last_idx >= volume_ma_period:
            avg_vol = np.mean(volumes[max(0, last_idx - volume_ma_period):last_idx])
            if avg_vol > 0:
                last_vol_ratio = volumes[last_idx] / avg_vol
                if last_vol_ratio >= volume_ratio:
                    last_vol_enough = True
                    last_reasons.append(f"成交量放大{last_vol_ratio:.1f}倍")

        last_pattern_or_dtw = len(last_triggered) > 0 or len(last_dtw) > 0
        today_buy = bool(last_pattern_or_dtw and last_yangxian and last_vol_enough)

        last_signal = {
            'today_buy': today_buy,
            'today_reasons': last_reasons if today_buy else []
        }

    if not signals and klines:
        signals = _get_signals_from_db(stock_code, start_date, end_date)

    result = {
        'stock_code': stock_code,
        'today_buy': last_signal['today_buy'] if last_signal else False,
        'today_reasons': last_signal['today_reasons'] if last_signal else [],
        'signals': signals
    }

    return result


def _get_signals_from_db(stock_code, start_date=None, end_date=None):
    query = {'stock_code': stock_code, 'buy': True}
    if start_date:
        query['date'] = {'$gte': start_date}
    if end_date:
        if 'date' in query:
            query['date']['$lte'] = end_date
        else:
            query['date'] = {'$lte': end_date}
    docs = list(buy_signals_col.find(query, {'_id': 0}).sort('date', 1))
    return [{'date': d['date'], 'close': d['close'], 'buy': True, 'reasons': d['reasons']} for d in docs]


def clear_signals(stock_code):
    buy_signals_col.delete_many({'stock_code': stock_code})
