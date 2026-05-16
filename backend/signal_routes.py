from flask import request, jsonify
from app import app
from models import KlineData
from signal_rule_generator import generate_rules, get_rules
from signal_scanner import scan_signals, clear_signals


@app.route('/api/generate_formula', methods=['POST'])
def generate_formula():
    data = request.json or {}
    stock_code = data.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 30:
        return jsonify({'success': False, 'message': f'日线数据不足（当前{len(klines) if klines else 0}条，至少需要30条），请先抓取数据'}), 400

    rule = generate_rules(stock_code, klines)
    if not rule:
        return jsonify({'success': False, 'message': '规则生成失败，数据量不足'}), 500

    return jsonify({
        'success': True,
        'stock_code': stock_code,
        'rules': {
            'top10_bullish': rule['top10_bullish'],
            'dtw_patterns': rule['dtw_patterns'],
            'dtw_threshold': rule['dtw_threshold'],
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

    result = scan_signals(stock_code, klines, start_date=start_date, end_date=end_date)

    return jsonify({
        'success': True,
        'stock_code': result['stock_code'],
        'today_buy': result['today_buy'],
        'today_reasons': result['today_reasons'],
        'signals': result['signals']
    })


@app.route('/api/signals', methods=['GET'])
def get_signals():
    stock_code = request.args.get('stock_code')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 30:
        return jsonify({'success': False, 'message': '日线数据不足，请先抓取数据'}), 400

    result = scan_signals(stock_code, klines, start_date=start_date, end_date=end_date)

    return jsonify({
        'success': True,
        'stock_code': result['stock_code'],
        'today_buy': result['today_buy'],
        'today_reasons': result['today_reasons'],
        'signals': result['signals']
    })


@app.route('/api/formula_status', methods=['GET'])
def formula_status():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    rule = get_rules(stock_code)
    if not rule:
        return jsonify({'success': True, 'exists': False, 'message': '未生成公式'})

    return jsonify({
        'success': True,
        'exists': True,
        'rules': {
            'top10_bullish': rule['top10_bullish'],
            'dtw_patterns': rule['dtw_patterns'],
            'dtw_threshold': rule['dtw_threshold'],
            'generated_date': rule['generated_date'],
            'expire_date': rule['expire_date'],
        }
    })


@app.route('/api/clear_signals', methods=['POST'])
def clear_all_signals():
    data = request.json or {}
    stock_code = data.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400
    clear_signals(stock_code)
    return jsonify({'success': True, 'message': '信号已清除'})
