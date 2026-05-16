from flask import request, jsonify
from app import app
from models import KlineData
from sell_rule_generator import generate_sell_rules, get_sell_rules
from sell_scanner import scan_sell_signals, clear_sell_signals


@app.route('/api/generate_sell_formula', methods=['POST'])
def generate_sell_formula():
    data = request.json or {}
    stock_code = data.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 30:
        return jsonify({'success': False, 'message': f'日线数据不足（当前{len(klines) if klines else 0}条，至少需要30条），请先抓取数据'}), 400

    rule = generate_sell_rules(stock_code, klines)
    if not rule:
        return jsonify({'success': False, 'message': '卖出规则生成失败，数据量不足'}), 500

    return jsonify({
        'success': True,
        'stock_code': stock_code,
        'rules': {
            'top10_bearish': rule['top10_bearish'],
            'dtw_top_patterns': rule['dtw_top_patterns'],
            'dtw_threshold': rule['dtw_threshold'],
            'generated_date': rule['generated_date'],
            'expire_date': rule['expire_date'],
        },
        'message': f"卖出公式已生成，有效期至{rule['expire_date']}"
    })


@app.route('/api/sell_analyze', methods=['GET'])
def analyze_sell_signals():
    stock_code = request.args.get('stock_code')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 30:
        return jsonify({'success': False, 'message': f'日线数据不足（当前{len(klines) if klines else 0}条，至少需要30条），请先抓取数据'}), 400

    result = scan_sell_signals(stock_code, klines, start_date=start_date, end_date=end_date)

    return jsonify({
        'success': True,
        'stock_code': result['stock_code'],
        'today_sell': result['today_sell'],
        'today_reasons': result['today_reasons'],
        'paired_signals': result['paired_signals'],
        'all_signals': result['all_signals']
    })


@app.route('/api/sell_formula_status', methods=['GET'])
def sell_formula_status():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    rule = get_sell_rules(stock_code)
    if not rule:
        return jsonify({'success': True, 'exists': False, 'message': '未生成卖出公式'})

    return jsonify({
        'success': True,
        'exists': True,
        'rules': {
            'top10_bearish': rule['top10_bearish'],
            'dtw_top_patterns': rule['dtw_top_patterns'],
            'dtw_threshold': rule['dtw_threshold'],
            'generated_date': rule['generated_date'],
            'expire_date': rule['expire_date'],
        }
    })


@app.route('/api/clear_sell_signals', methods=['POST'])
def clear_all_sell_signals():
    data = request.json or {}
    stock_code = data.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400
    clear_sell_signals(stock_code)
    return jsonify({'success': True, 'message': '卖出信号已清除'})
