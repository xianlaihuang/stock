import talib
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
from db import db
from sell_dtw import SellDTWDetector


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'sell_config.yaml')


def load_sell_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


stock_sell_rules_col = db['stock_sell_rules']
sell_signals_col = db['sell_signals']

stock_sell_rules_col.create_index('stock_code', unique=True)
sell_signals_col.create_index([('stock_code', 1), ('date', 1)], unique=True)


BEARISH_PATTERN_NAMES_CN = {
    'CDLSHOOTINGSTAR': '射击之星',
    'CDLHANGINGMAN': '上吊线',
    'CDLEVENINGSTAR': '黄昏之星',
    'CDLEVENINGDOJISTAR': '十字黄昏之星',
    'CDLDARKCLOUDCOVER': '乌云盖顶',
    'CDL3BLACKCROWS': '三只乌鸦',
    'CDL2CROWS': '两只乌鸦',
    'CDLBELTHOLD': '大阴烛',
    'CDLONNECK': '待入头',
    'CDLIDENTICAL3CROWS': '三胞胎乌鸦',
    'CDLUPSIDEGAP2CROWS': '上跳两只乌鸦',
    'CDLADVANCEBLOCK': '前进受阻',
    'CDLSTALLEDPATTERN': '停顿形态',
    'CDLGRAVESTONEDOJI': '墓碑十字',
    'CDLHIGHWAVE': '高浪线',
    'CDLHARAMI': '孕线',
    'CDLHARAMICROSS': '十字孕线',
    'CDLDOJISTAR': '十字星',
    'CDLKICKING': '跳空缺口',
    'CDLKICKINGBYLENGTH': '长跳空缺口',
    'CDLCOUNTERATTACK': '反击线',
    'CDLBREAKAWAY': '脱离形态',
    'CDLMATCHINGLOW': '匹配低点',
    'CDL3INSIDE': '三内部下跌',
    'CDL3OUTSIDE': '三外部下跌',
    'CDLRISEFALL3METHODS': '下降三法',
    'CDL3STARSINSOUTH': '南方三星',
    'CDL3LINESTRIKE': '三线打击',
    'CDLXSIDEGAP3METHODS': '跳空三法',
    'CDLTASUKIGAP': '缺口',
    'CDLUNIQUE3RIVER': '独特三河',
    'CDLTRICKS': '捉腰带',
    'CDLSPINNINGTOP': '纺锤顶',
    'CDLGAPSIDEBYWHITELINE': '并列白线',
}


def get_bearish_pattern_cn(name):
    return BEARISH_PATTERN_NAMES_CN.get(name, name)


def _get_pattern_func(name):
    return getattr(talib, name, None)


def _compute_pattern(name, o, h, l, c):
    func = _get_pattern_func(name)
    if func is None:
        return None
    try:
        result = func(o, h, l, c)
        return result
    except Exception:
        return None


def generate_sell_rules(stock_code, klines):
    config = load_sell_config()
    cfg = config['sell_engine']
    hold_period = cfg['hold_period']
    drop_threshold = cfg.get('drop_threshold_pct', -0.02)
    top_n = cfg['top_n_patterns']
    min_occ = cfg['min_occurrences']
    dtw_cfg = config['dtw_top_templates']

    if len(klines) < hold_period + 30:
        return None

    opens = np.array([k['open'] for k in klines], dtype=float)
    highs = np.array([k['high'] for k in klines], dtype=float)
    lows = np.array([k['low'] for k in klines], dtype=float)
    closes = np.array([k['close'] for k in klines], dtype=float)

    pattern_names = config['bearish_patterns']
    pattern_stats = []

    for pname in pattern_names:
        result = _compute_pattern(pname, opens, highs, lows, closes)
        if result is None:
            continue
        wins = 0
        total = 0
        for i in range(len(result)):
            if i + 1 + hold_period >= len(closes):
                break
            if result[i] == -100:
                total += 1
                sell_price = closes[i]
                future_lows = lows[i + 1:i + 1 + hold_period]
                min_low = float(np.min(future_lows))
                drop_ratio = (min_low - sell_price) / sell_price
                if drop_ratio < drop_threshold:
                    wins += 1
        if total >= min_occ:
            win_rate = wins / total
            pattern_stats.append({
                'name': pname,
                'cn_name': get_bearish_pattern_cn(pname),
                'win_rate': round(win_rate, 4),
                'total': total,
                'wins': wins
            })

    pattern_stats.sort(key=lambda x: x['win_rate'], reverse=True)
    top10 = pattern_stats[:top_n]

    dtw_templates = {}
    for name, tmpl in dtw_cfg.items():
        dtw_templates[name] = tmpl

    detector = SellDTWDetector(dtw_templates, window=cfg['dtw_window'], default_threshold=cfg['dtw_default_threshold'])
    dtw_threshold, effective_patterns = detector.compute_dynamic_threshold(
        closes.tolist(),
        percentile=cfg['dtw_percentile'],
        min_matches=cfg['dtw_min_matches']
    )

    now = datetime.now()
    expire = now + timedelta(days=cfg['rule_expire_days'])

    rule_doc = {
        'stock_code': stock_code,
        'top10_bearish': top10,
        'dtw_top_patterns': effective_patterns,
        'dtw_threshold': dtw_threshold,
        'generated_date': now.strftime('%Y-%m-%d'),
        'expire_date': expire.strftime('%Y-%m-%d'),
        'config_snapshot': {
            'hold_period': hold_period,
            'drop_threshold_pct': drop_threshold,
            'volume_ratio': cfg['volume_ratio'],
            'volume_ma_period': cfg['volume_ma_period'],
            'dtw_window': cfg['dtw_window'],
        }
    }

    stock_sell_rules_col.replace_one(
        {'stock_code': stock_code},
        rule_doc,
        upsert=True
    )

    return rule_doc


def get_sell_rules(stock_code):
    rule = stock_sell_rules_col.find_one({'stock_code': stock_code}, {'_id': 0})
    if rule:
        expire_str = rule.get('expire_date', '')
        try:
            expire_date = datetime.strptime(expire_str, '%Y-%m-%d')
            if expire_date > datetime.now():
                return rule
        except ValueError:
            pass
    return None


def get_or_generate_sell_rules(stock_code, klines):
    rule = get_sell_rules(stock_code)
    if rule:
        return rule
    return generate_sell_rules(stock_code, klines)
