from flask import request, jsonify
from app import app
from models import KlineData
from unified_rule_generator import generate_unified_rules, get_unified_rules
from unified_scanner import scan_unified_signals, clear_all_signals, scan_dynamic_weight_signals
import dynamic_weight_engine as dwe


_weight_cache = {}


@app.route('/api/calculate_weights', methods=['POST'])
def calculate_weights():
    data = request.json or {}
    stock_code = data.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 30:
        return jsonify({'success': False, 'message': f'日线数据不足（当前{len(klines) if klines else 0}条，至少需要30条），请先抓取数据'}), 400

    df = dwe._klines_to_df(klines) if hasattr(dwe, '_klines_to_df') else _klines_to_df_local(klines)

    if len(df) < 200:
        from scraper import StockScraper
        try:
            extra_klines = StockScraper.get_kline_data(stock_code, period='day', count=800, start_date=None)
            if extra_klines and len(extra_klines) > len(klines):
                KlineData.delete(stock_code, period='day')
                KlineData.add_many(stock_code, extra_klines, period='day')
                df = _klines_to_df_local(extra_klines)
        except Exception as e:
            print(f"[公式计算] 补充数据失败: {e}")

    if len(df) < 200:
        return jsonify({'success': False, 'message': f'数据不足，需要至少200条日线数据（当前{len(df)}条），请先抓取数据'}), 400

    result = dwe.calculate_weights(df)

    _weight_cache[stock_code] = {
        'buy_weights': result['buy_weights'],
        'sell_weights': result['sell_weights'],
        'buy_rules': {name: True for name in result['buy_rules']},
        'sell_rules': {name: True for name in result['sell_rules']},
        'buy_details': result['buy_details'],
        'sell_details': result['sell_details'],
    }

    buy_list = []
    for name, detail in result['buy_details'].items():
        buy_list.append({
            'name': name,
            'raw_weight': detail['raw_weight'],
            'final_weight': detail['final_weight'],
            'normalized_weight': detail['normalized_weight'],
            'win_rate': detail['win_rate'],
            'avg_return': detail['avg_return'],
            'count': detail['count'],
            'profit_factor': detail.get('profit_factor', 0),
            'avg_hold': detail.get('avg_hold', 0),
            'penalty_multiplier': detail.get('penalty_multiplier', 1.0),
            'penalized': detail['penalized'],
            'penalty_reason': detail['penalty_reason'],
            'level': detail['level'],
        })

    sell_list = []
    for name, detail in result['sell_details'].items():
        sell_list.append({
            'name': name,
            'raw_weight': detail['raw_weight'],
            'final_weight': detail['final_weight'],
            'normalized_weight': detail['normalized_weight'],
            'win_rate': detail['win_rate'],
            'avg_return': detail['avg_return'],
            'count': detail['count'],
            'profit_factor': detail.get('profit_factor', 0),
            'avg_hold': detail.get('avg_hold', 0),
            'penalty_multiplier': detail.get('penalty_multiplier', 1.0),
            'penalized': detail['penalized'],
            'penalty_reason': detail['penalty_reason'],
            'level': detail['level'],
        })

    buy_list.sort(key=lambda x: -x['normalized_weight'])
    sell_list.sort(key=lambda x: -x['normalized_weight'])

    return jsonify({
        'success': True,
        'stock_code': stock_code,
        'buy_weights': result['buy_weights'],
        'sell_weights': result['sell_weights'],
        'buy_details': buy_list,
        'sell_details': sell_list,
        'depth_used': result['depth_used'],
        'found_strict': result['found_strict'],
        'message': f'公式计算完成，深度{result["depth_used"]}，买入{len(buy_list)}规则，卖出{len(sell_list)}规则'
    })


def _klines_to_df_local(klines):
    import pandas as pd
    df = pd.DataFrame(klines)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def get_cached_weights(stock_code):
    return _weight_cache.get(stock_code, None)


@app.route('/api/generate_formula', methods=['POST'])
def generate_formula():
    data = request.json or {}
    stock_code = data.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 30:
        return jsonify({'success': False, 'message': f'日线数据不足（当前{len(klines) if klines else 0}条，至少需要30条），请先抓取数据'}), 400

    rule = generate_unified_rules(stock_code, klines)
    if not rule:
        return jsonify({'success': False, 'message': '公式生成失败，数据量不足'}), 500

    return jsonify({
        'success': True,
        'stock_code': stock_code,
        'rules': {
            'top10_bullish': rule['top10_bullish'],
            'top10_bearish': rule['top10_bearish'],
            'dtw_bottom_patterns': rule['dtw_bottom_patterns'],
            'dtw_bottom_threshold': rule['dtw_bottom_threshold'],
            'dtw_top_patterns': rule['dtw_top_patterns'],
            'dtw_top_threshold': rule['dtw_top_threshold'],
            'generated_date': rule['generated_date'],
            'expire_date': rule['expire_date'],
        },
        'message': f"公式已生成，有效期至{rule['expire_date']}"
    })


@app.route('/api/analyze', methods=['GET'])
def analyze_signals():
    stock_code = request.args.get('stock_code')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 30:
        return jsonify({'success': False, 'message': f'日线数据不足（当前{len(klines) if klines else 0}条，至少需要30条），请先抓取数据'}), 400

    cached = get_cached_weights(stock_code)
    if cached:
        result = scan_dynamic_weight_signals(
            stock_code, klines,
            start_date=start_date, end_date=end_date,
            precomputed_weights=cached
        )
    else:
        result = scan_dynamic_weight_signals(stock_code, klines, start_date=start_date, end_date=end_date)

    return jsonify({
        'success': True,
        'stock_code': result['stock_code'],
        'today_buy': result['today_buy'],
        'today_sell': result['today_sell'],
        'today_reasons': result['today_reasons'],
        'today_score': result.get('today_score', 0),
        'today_rules': result.get('today_rules', []),
        'today_buy_score': result.get('today_buy_score', 0),
        'today_sell_score': result.get('today_sell_score', 0),
        'rule_stats': result.get('rule_stats', {}),
        'paired_signals': result['paired_signals'],
        'all_signals': result['all_signals'],
        'depth_used': result.get('depth_used', 200),
        'summary': {
            'total_signals': result.get('summary', {}).get('total_signals', 0),
            'buy_count': result.get('summary', {}).get('buy_count', 0),
            'sell_count': result.get('summary', {}).get('sell_count', 0),
            'paired_count': result.get('summary', {}).get('paired_count', 0),
            'active_buy_rules': result.get('summary', {}).get('active_buy_rules', []),
            'active_sell_rules': result.get('summary', {}).get('active_sell_rules', []),
            'buy_weights': result.get('summary', {}).get('buy_weights', {}),
            'sell_weights': result.get('summary', {}).get('sell_weights', {}),
        },
    })


@app.route('/api/formula_status', methods=['GET'])
def formula_status():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    rule = get_unified_rules(stock_code)
    if not rule:
        return jsonify({'success': True, 'exists': False, 'message': '未生成公式'})

    return jsonify({
        'success': True,
        'exists': True,
        'rules': {
            'top10_bullish': rule['top10_bullish'],
            'top10_bearish': rule['top10_bearish'],
            'dtw_bottom_patterns': rule.get('dtw_bottom_patterns', []),
            'dtw_bottom_threshold': rule.get('dtw_bottom_threshold', 0.3),
            'dtw_top_patterns': rule.get('dtw_top_patterns', []),
            'dtw_top_threshold': rule.get('dtw_top_threshold', 0.3),
            'generated_date': rule['generated_date'],
            'expire_date': rule['expire_date'],
        }
    })


@app.route('/api/clear_signals', methods=['POST'])
def clear_all_signals_route():
    data = request.json or {}
    stock_code = data.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400
    clear_all_signals(stock_code)
    return jsonify({'success': True, 'message': '信号已清除'})
