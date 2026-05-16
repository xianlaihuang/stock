import talib
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
from db import db
from signal_dtw import DTWDetector
from sell_dtw import SellDTWDetector


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'unified_config.yaml')

_config_cache = None
_config_mtime = None


def load_unified_config():
    global _config_cache, _config_mtime
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        mtime = None
    if _config_cache is not None and _config_mtime == mtime:
        return _config_cache
    with open(CONFIG_PATH, 'r') as f:
        _config_cache = yaml.safe_load(f)
    _config_mtime = mtime
    return _config_cache


stock_rules_col = db['stock_rules']
buy_signals_col = db['buy_signals']
sell_signals_col = db['sell_signals']

stock_rules_col.create_index('stock_code', unique=True)
buy_signals_col.create_index([('stock_code', 1), ('date', 1)], unique=True)
sell_signals_col.create_index([('stock_code', 1), ('date', 1)], unique=True)


def _get_pattern_func(name):
    return getattr(talib, name, None)


def _compute_pattern(name, o, h, l, c):
    func = _get_pattern_func(name)
    if func is None:
        return None
    try:
        return func(o, h, l, c)
    except Exception:
        return None


def _compute_all_pattern_results(pattern_names, opens, highs, lows, closes):
    results = {}
    for pname in pattern_names:
        result = _compute_pattern(pname, opens, highs, lows, closes)
        if result is not None:
            results[pname] = result
    return results


def _find_alternating_bs_signals(raw_b_indices, raw_s_indices, closes, dates):
    valid_signals = []
    expecting = 'B'
    b_idx = 0
    s_idx = 0

    while True:
        if expecting == 'B':
            found = False
            while b_idx < len(raw_b_indices):
                idx = raw_b_indices[b_idx]
                if not valid_signals or idx > valid_signals[-1]['index']:
                    valid_signals.append({
                        'index': idx,
                        'date': dates[idx],
                        'close': float(closes[idx]),
                        'type': 'B',
                    })
                    b_idx += 1
                    expecting = 'S'
                    found = True
                    break
                b_idx += 1
            if not found:
                break
        else:
            found = False
            while s_idx < len(raw_s_indices):
                idx = raw_s_indices[s_idx]
                if idx > valid_signals[-1]['index']:
                    valid_signals.append({
                        'index': idx,
                        'date': dates[idx],
                        'close': float(closes[idx]),
                        'type': 'S',
                    })
                    s_idx += 1
                    expecting = 'B'
                    found = True
                    break
                s_idx += 1
            if not found:
                break

    return valid_signals


def _compute_return_rates(valid_signals):
    for k in range(len(valid_signals)):
        sig = valid_signals[k]
        if sig['type'] == 'B':
            next_s = None
            for j in range(k + 1, len(valid_signals)):
                if valid_signals[j]['type'] == 'S':
                    next_s = valid_signals[j]
                    break
            if next_s:
                sig['return_rate_pct'] = round(
                    (next_s['close'] - sig['close']) / sig['close'] * 100, 2
                )
            else:
                sig['return_rate_pct'] = None
        elif sig['type'] == 'S':
            next_b = None
            for j in range(k + 1, len(valid_signals)):
                if valid_signals[j]['type'] == 'B':
                    next_b = valid_signals[j]
                    break
            if next_b:
                sig['return_rate_pct'] = round(
                    (sig['close'] - next_b['close']) / sig['close'] * 100, 2
                )
            else:
                sig['return_rate_pct'] = None
    return valid_signals


def generate_unified_rules(stock_code, klines):
    config = load_unified_config()
    cfg = config['engine']
    top_n = cfg['top_n_patterns']
    min_occ = cfg['min_occurrences']
    dtw_bottom_cfg = config['dtw_bottom_templates']
    dtw_top_cfg = config['dtw_top_templates']
    pattern_names = config['all_cdl_patterns']
    names_cn = config.get('CDL_NAMES_CN', {})

    if len(klines) < 60:
        return None

    opens = np.array([k['open'] for k in klines], dtype=float)
    highs = np.array([k['high'] for k in klines], dtype=float)
    lows = np.array([k['low'] for k in klines], dtype=float)
    closes = np.array([k['close'] for k in klines], dtype=float)
    volumes = np.array([k['volume'] for k in klines], dtype=float)
    dates = [k['date'][:10] if len(k['date']) > 10 else k['date'] for k in klines]

    pattern_results = _compute_all_pattern_results(pattern_names, opens, highs, lows, closes)

    volume_ma_period = cfg['volume_ma_period']
    volume_ratio = cfg['volume_ratio']

    raw_b_indices = []
    raw_s_indices = []

    for i in range(len(klines)):
        has_bullish = False
        for pname, result_arr in pattern_results.items():
            if i < len(result_arr) and result_arr[i] == 100:
                has_bullish = True
                break

        is_yangxian = closes[i] > opens[i]
        vol_enough = False
        if i >= volume_ma_period:
            avg_vol = np.mean(volumes[max(0, i - volume_ma_period):i])
            if avg_vol > 0 and volumes[i] / avg_vol >= volume_ratio:
                vol_enough = True
        elif i > 0:
            avg_vol = np.mean(volumes[:i])
            if avg_vol > 0 and volumes[i] / avg_vol >= volume_ratio:
                vol_enough = True

        if has_bullish and is_yangxian and vol_enough:
            raw_b_indices.append(i)

        has_bearish = False
        for pname, result_arr in pattern_results.items():
            if i < len(result_arr) and result_arr[i] == -100:
                has_bearish = True
                break

        is_yinxian = closes[i] < opens[i]
        vol_enough_s = False
        if i >= volume_ma_period:
            avg_vol = np.mean(volumes[max(0, i - volume_ma_period):i])
            if avg_vol > 0 and volumes[i] / avg_vol >= volume_ratio:
                vol_enough_s = True
        elif i > 0:
            avg_vol = np.mean(volumes[:i])
            if avg_vol > 0 and volumes[i] / avg_vol >= volume_ratio:
                vol_enough_s = True

        if has_bearish and is_yinxian and vol_enough_s:
            raw_s_indices.append(i)

    valid_signals = _find_alternating_bs_signals(raw_b_indices, raw_s_indices, closes, dates)
    valid_signals = _compute_return_rates(valid_signals)

    b_signal_map = {}
    s_signal_map = {}
    for sig in valid_signals:
        if sig['type'] == 'B':
            b_signal_map[sig['index']] = sig
        else:
            s_signal_map[sig['index']] = sig

    bullish_stats = []
    for pname, result_arr in pattern_results.items():
        total = 0
        return_sum = 0.0
        for i in range(len(result_arr)):
            if result_arr[i] == 100 and i in b_signal_map:
                total += 1
                ret = b_signal_map[i].get('return_rate_pct')
                if ret is not None:
                    return_sum += ret
        if total >= min_occ:
            avg_return = return_sum / total if total > 0 else 0
            bullish_stats.append({
                'name': pname,
                'cn_name': names_cn.get(pname, pname),
                'avg_return_rate': round(avg_return, 2),
                'total': total,
                'return_sum': round(return_sum, 2),
            })

    bearish_stats = []
    for pname, result_arr in pattern_results.items():
        total = 0
        return_sum = 0.0
        for i in range(len(result_arr)):
            if result_arr[i] == -100 and i in s_signal_map:
                total += 1
                ret = s_signal_map[i].get('return_rate_pct')
                if ret is not None:
                    return_sum += ret
        if total >= min_occ:
            avg_return = return_sum / total if total > 0 else 0
            bearish_stats.append({
                'name': pname,
                'cn_name': names_cn.get(pname, pname),
                'avg_return_rate': round(avg_return, 2),
                'total': total,
                'return_sum': round(return_sum, 2),
            })

    bullish_stats.sort(key=lambda x: x['avg_return_rate'], reverse=True)
    bearish_stats.sort(key=lambda x: x['avg_return_rate'], reverse=True)

    top10_bullish = bullish_stats[:top_n]
    top10_bearish = bearish_stats[:top_n]

    dtw_bottom_templates = dict(dtw_bottom_cfg)
    buy_detector = DTWDetector(
        dtw_bottom_templates,
        window=cfg['dtw_window'],
        default_threshold=cfg['dtw_default_threshold'],
    )
    dtw_bottom_threshold, dtw_bottom_patterns = buy_detector.compute_dynamic_threshold(
        closes.tolist(),
        percentile=cfg['dtw_percentile'],
        min_matches=cfg['dtw_min_matches'],
    )

    dtw_top_templates = dict(dtw_top_cfg)
    sell_detector = SellDTWDetector(
        dtw_top_templates,
        window=cfg['dtw_window'],
        default_threshold=cfg['dtw_default_threshold'],
    )
    dtw_top_threshold, dtw_top_patterns = sell_detector.compute_dynamic_threshold(
        closes.tolist(),
        percentile=cfg['dtw_percentile'],
        min_matches=cfg['dtw_min_matches'],
    )

    now = datetime.now()
    expire = now + timedelta(days=cfg['rule_expire_days'])

    rule_doc = {
        'stock_code': stock_code,
        'top10_bullish': top10_bullish,
        'top10_bearish': top10_bearish,
        'dtw_bottom_patterns': dtw_bottom_patterns,
        'dtw_bottom_threshold': dtw_bottom_threshold,
        'dtw_top_patterns': dtw_top_patterns,
        'dtw_top_threshold': dtw_top_threshold,
        'generated_date': now.strftime('%Y-%m-%d'),
        'expire_date': expire.strftime('%Y-%m-%d'),
        'config_snapshot': {
            'volume_ratio': cfg['volume_ratio'],
            'volume_ma_period': cfg['volume_ma_period'],
            'dtw_window': cfg['dtw_window'],
        },
    }

    stock_rules_col.replace_one(
        {'stock_code': stock_code},
        rule_doc,
        upsert=True,
    )

    return rule_doc


def get_unified_rules(stock_code):
    rule = stock_rules_col.find_one({'stock_code': stock_code}, {'_id': 0})
    if rule:
        expire_str = rule.get('expire_date', '')
        try:
            expire_date = datetime.strptime(expire_str, '%Y-%m-%d')
            if expire_date > datetime.now():
                return rule
        except ValueError:
            pass
    return None


def get_or_generate_unified_rules(stock_code, klines):
    rule = get_unified_rules(stock_code)
    if rule:
        return rule
    return generate_unified_rules(stock_code, klines)
