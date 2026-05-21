import time

from flask import request, jsonify
from app import app
from models import KlineData, Stock
import engine_v2 as ev2
import analysis_run_log as arl


def _v4_get_conditions():
    from v4_aggressive.engine import get_conditions
    return get_conditions()


def _klines_to_df(klines):
    import pandas as pd
    df = pd.DataFrame(klines)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _today_from_paired(sigs):
    today_buy = False
    today_sell = False
    today_reasons = []
    today_score = 0
    today_buy_score = 0
    today_sell_score = 0
    if sigs:
        last = sigs[-1]
        if last.get('type') == 'B':
            today_buy = True
            today_reasons = last.get('reasons', [])
            today_buy_score = last.get('confidence', 0)
        elif last.get('type') == 'S':
            today_sell = True
            today_reasons = last.get('reasons', [])
            today_sell_score = last.get('confidence', 0)
        today_score = max(today_buy_score, today_sell_score)
    return {
        'today_buy': today_buy,
        'today_sell': today_sell,
        'today_reasons': today_reasons,
        'today_score': today_score,
        'today_buy_score': today_buy_score,
        'today_sell_score': today_sell_score,
        'today_rules': today_reasons,
    }


def _engine_block_from_result(result, depth_used):
    ps = result.get('paired_signals') or []
    t = _today_from_paired(ps)
    summ = result.get('summary') or {}
    if not summ.get('total_signals') and ps is not None:
        summ = {
            'total_signals': len(ps),
            'buy_count': sum(1 for s in ps if s.get('type') == 'B'),
            'sell_count': sum(1 for s in ps if s.get('type') == 'S'),
        }
    return {
        'paired_signals': ps,
        'all_signals': result.get('all_signals', ps),
        'conditions': result.get('conditions', ev2.get_conditions()),
        'rule_stats': result.get('rule_stats', {}),
        'prd_metrics': result.get('prd_metrics', {}),
        'summary': summ,
        'depth_used': depth_used,
        'portfolio_sim': result.get('portfolio_sim') or ev2._portfolio_sim_from_paired(ps),
        **t,
    }


def _run_calculate_weights_for_stock(stock_code, trigger='manual'):
    """公式计算权重并写日志；返回 (payload_dict, http_status)。"""
    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 200:
        msg = f'数据不足（当前{len(klines) if klines else 0}条，至少需要200条），请先抓取数据'
        arl.save_run_log(
            stock_code, 'calculate_weights', success=False,
            duration_ms=0, message=msg, trigger=trigger,
        )
        return {'success': False, 'message': msg, 'stock_code': stock_code}, 400

    df = _klines_to_df(klines)
    t0 = time.perf_counter()
    try:
        result = ev2.calculate_weights_v2(df)
    except Exception as e:
        arl.save_run_log(
            stock_code, 'calculate_weights', success=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            message=str(e), trigger=trigger, depth_used=len(df),
        )
        return {'success': False, 'message': str(e), 'stock_code': stock_code}, 500

    ev2.save_weights(stock_code, result)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    total_return = round(result['total_return'] * 100, 2)
    win_rate = round(result['win_rate'] * 100, 1)
    message = (
        f'V2公式计算完成，迭代{result["iteration_count"]}次，'
        f'sum收益率{result["total_return"]*100:.2f}%'
    )
    arl.save_run_log(
        stock_code, 'calculate_weights', success=True,
        duration_ms=duration_ms, message=message, trigger=trigger, depth_used=len(df),
        weights_opt={
            'total_return_pct': total_return,
            'win_rate_pct': win_rate,
            'iteration_count': result['iteration_count'],
        },
    )

    buy_list = []
    for name, detail in result['buy_details'].items():
        buy_list.append({'name': name, **detail})
    sell_list = []
    for name, detail in result['sell_details'].items():
        sell_list.append({'name': name, **detail})
    buy_list.sort(key=lambda x: -x.get('normalized_weight', 0))
    sell_list.sort(key=lambda x: -x.get('normalized_weight', 0))

    payload = {
        'success': True,
        'stock_code': stock_code,
        'buy_weights': result['buy_weights'],
        'sell_weights': result['sell_weights'],
        'buy_details': buy_list,
        'sell_details': sell_list,
        'iteration_count': result['iteration_count'],
        'total_return': total_return,
        'win_rate': win_rate,
        'message': message,
        'duration_ms': duration_ms,
        'weights_return_pct': total_return,
        'prd_utility_objective': ev2.V2_PRD_UTILITY_OBJECTIVE,
    }
    if 'prd_utility' in result:
        payload['prd_utility'] = result['prd_utility']
    if 'prd_max_drawdown' in result:
        payload['prd_max_drawdown'] = result['prd_max_drawdown']
    return payload, 200


@app.route('/api/v2/calculate_weights', methods=['POST'])
def v2_calculate_weights():
    data = request.json or {}
    stock_code = data.get('stock_code')
    trigger = data.get('trigger', 'manual')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400
    payload, status = _run_calculate_weights_for_stock(stock_code, trigger=trigger)
    return jsonify(payload), status


def _run_analyze_for_stock(stock_code, *, start_date=None, end_date=None, trigger='manual'):
    """运行 V2/V3/V4 分析并写日志；返回 (payload_dict, http_status)。"""
    klines = KlineData.get(stock_code, period='day')
    if not klines or len(klines) < 30:
        msg = f'数据不足（当前{len(klines) if klines else 0}条），请先抓取数据'
        arl.save_run_log(
            stock_code, 'analyze', success=False, duration_ms=0,
            message=msg, trigger=trigger,
        )
        return {'success': False, 'message': msg, 'stock_code': stock_code}, 400

    df = _klines_to_df(klines)
    t0 = time.perf_counter()
    cached = ev2.load_weights(stock_code)
    precomputed = None
    if cached:
        precomputed = {
            'buy_weights': cached['buy_weights'],
            'sell_weights': cached['sell_weights'],
        }

    try:
        dual = ev2.analyze_signals_dual(
            df, precomputed_weights=precomputed,
            start_date=start_date, end_date=end_date,
        )
    except Exception as e:
        arl.save_run_log(
            stock_code, 'analyze', success=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            message=str(e), trigger=trigger, depth_used=len(df),
            start_date=start_date, end_date=end_date,
        )
        return {'success': False, 'message': str(e), 'stock_code': stock_code}, 500

    ev2.save_signals(stock_code, dual)
    v2 = dual['v2']
    v3 = dual['v3']
    v4 = dual.get('v4')
    depth = len(df)
    block2 = _engine_block_from_result(v2, depth)
    block3 = _engine_block_from_result(v3, depth)
    block4 = _engine_block_from_result(v4, depth) if v4 else None
    duration_ms = int((time.perf_counter() - t0) * 1000)

    arl.save_run_log(
        stock_code, 'analyze', success=True,
        duration_ms=duration_ms,
        message='运行分析完成', trigger=trigger,
        v2_portfolio=block2.get('portfolio_sim'),
        v3_portfolio=block3.get('portfolio_sim'),
        v4_portfolio=block4.get('portfolio_sim') if block4 else None,
        depth_used=depth, start_date=start_date, end_date=end_date,
    )

    payload = {
        'success': True,
        'stock_code': stock_code,
        'today_buy': block2['today_buy'],
        'today_sell': block2['today_sell'],
        'today_reasons': block2['today_reasons'],
        'today_score': block2['today_score'],
        'today_buy_score': block2['today_buy_score'],
        'today_sell_score': block2['today_sell_score'],
        'paired_signals': block2['paired_signals'],
        'all_signals': block2['all_signals'],
        'conditions': block2['conditions'],
        'today_rules': block2['today_rules'],
        'rule_stats': block2['rule_stats'],
        'depth_used': block2['depth_used'],
        'summary': block2['summary'],
        'prd_metrics': block2['prd_metrics'],
        'portfolio_sim': block2['portfolio_sim'],
        'v2': block2,
        'v3': block3,
        'v4': block4,
        'duration_ms': duration_ms,
        'v2_return_pct': (block2.get('portfolio_sim') or {}).get('total_return_pct'),
        'v3_return_pct': (block3.get('portfolio_sim') or {}).get('total_return_pct'),
        'v4_return_pct': (block4.get('portfolio_sim') or {}).get('total_return_pct') if block4 else None,
    }
    return payload, 200


@app.route('/api/v2/analyze', methods=['GET'])
def v2_analyze_signals():
    stock_code = request.args.get('stock_code')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    trigger = request.args.get('trigger', 'manual')

    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    payload, status = _run_analyze_for_stock(
        stock_code, start_date=start_date, end_date=end_date, trigger=trigger,
    )
    return jsonify(payload), status


@app.route('/api/v2/weights', methods=['GET'])
def v2_get_weights():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    cached = ev2.load_weights(stock_code)
    if not cached:
        return jsonify({'success': True, 'exists': False, 'message': '未生成V2公式'})

    buy_list = []
    for name, detail in cached['buy_details'].items():
        buy_list.append({'name': name, **detail})

    sell_list = []
    for name, detail in cached['sell_details'].items():
        sell_list.append({'name': name, **detail})

    buy_list.sort(key=lambda x: -x.get('normalized_weight', 0))
    sell_list.sort(key=lambda x: -x.get('normalized_weight', 0))

    return jsonify({
        'success': True,
        'exists': True,
        'stock_code': stock_code,
        'buy_weights': cached['buy_weights'],
        'sell_weights': cached['sell_weights'],
        'buy_details': buy_list,
        'sell_details': sell_list,
        'iteration_count': cached['iteration_count'],
        'total_return': round(cached['total_return'] * 100, 2),
        'win_rate': round(cached['win_rate'] * 100, 1),
    })


@app.route('/api/v2/weights', methods=['DELETE'])
def v2_delete_weights():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    ev2.delete_weights(stock_code)
    return jsonify({'success': True, 'message': 'V2权重已删除'})


@app.route('/api/v2/signals', methods=['DELETE'])
def v2_delete_signals():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    ev2.delete_signals(stock_code)
    return jsonify({'success': True, 'message': 'V2信号已删除'})


@app.route('/api/v2/signals', methods=['GET'])
def v2_get_signals():
    stock_code = request.args.get('stock_code')
    if not stock_code:
        return jsonify({'success': False, 'message': 'stock_code is required'}), 400

    cached = ev2.load_signals(stock_code)
    if not cached:
        return jsonify({'success': True, 'exists': False})

    klines = KlineData.get(stock_code, period='day')
    depth_used = len(klines) if klines else 0

    sigs = cached.get('signals', [])
    sigs_v3 = cached.get('signals_v3')
    sigs_v4 = cached.get('signals_v4')

    if (sigs_v3 is None or sigs_v4 is None) and klines and len(klines) >= 30:
        try:
            df = _klines_to_df(klines)
            cw = ev2.load_weights(stock_code)
            precomputed = None
            if cw:
                precomputed = {
                    'buy_weights': cw['buy_weights'],
                    'sell_weights': cw['sell_weights'],
                }
            dual = ev2.analyze_signals_dual(df, precomputed_weights=precomputed)
            ev2.save_signals(stock_code, dual)
            cached = ev2.load_signals(stock_code) or {}
            sigs = cached.get('signals', sigs)
            sigs_v3 = cached.get('signals_v3')
            sigs_v4 = cached.get('signals_v4')
        except Exception:
            sigs_v3 = sigs_v3 or []
            sigs_v4 = sigs_v4 or []

    if sigs_v3 is None:
        sigs_v3 = []
    if sigs_v4 is None:
        sigs_v4 = []

    port2 = cached.get('portfolio_v2') or ev2._portfolio_sim_from_paired(sigs)
    port3 = cached.get('portfolio_v3') or ev2._portfolio_sim_from_paired(sigs_v3)
    port4 = cached.get('portfolio_v4') or ev2._portfolio_sim_from_paired(sigs_v4)

    pseudo_v2 = {
        'paired_signals': sigs,
        'all_signals': sigs,
        'prd_metrics': {},
        'summary': {
            'total_signals': len(sigs),
            'buy_count': sum(1 for s in sigs if s.get('type') == 'B'),
            'sell_count': sum(1 for s in sigs if s.get('type') == 'S'),
        },
        'portfolio_sim': port2,
        'conditions': ev2.get_conditions(),
        'rule_stats': {},
    }

    pseudo_v3 = {
        'paired_signals': sigs_v3,
        'all_signals': sigs_v3,
        'prd_metrics': {},
        'summary': {
            'total_signals': len(sigs_v3),
            'buy_count': sum(1 for s in sigs_v3 if s.get('type') == 'B'),
            'sell_count': sum(1 for s in sigs_v3 if s.get('type') == 'S'),
        },
        'portfolio_sim': port3,
        'conditions': ev2.get_conditions(),
        'rule_stats': {},
    }

    block2 = _engine_block_from_result(pseudo_v2, depth_used)
    block3 = _engine_block_from_result(pseudo_v3, depth_used)
    pseudo_v4 = {
        'paired_signals': sigs_v4,
        'all_signals': sigs_v4,
        'prd_metrics': {},
        'summary': {
            'total_signals': len(sigs_v4),
            'buy_count': sum(1 for s in sigs_v4 if s.get('type') == 'B'),
            'sell_count': sum(1 for s in sigs_v4 if s.get('type') == 'S'),
        },
        'portfolio_sim': port4,
        'conditions': _v4_get_conditions(),
        'rule_stats': {},
    }
    block4 = _engine_block_from_result(pseudo_v4, depth_used)

    buy_count = sum(1 for s in sigs if s.get('type') == 'B')
    sell_count = sum(1 for s in sigs if s.get('type') == 'S')

    payload = {
        'success': True,
        'exists': True,
        'stock_code': stock_code,
        'today_buy': block2['today_buy'],
        'today_sell': block2['today_sell'],
        'today_reasons': block2['today_reasons'],
        'today_score': block2['today_score'],
        'today_buy_score': block2['today_buy_score'],
        'today_sell_score': block2['today_sell_score'],
        'paired_signals': block2['paired_signals'],
        'all_signals': block2['all_signals'],
        'today_rules': block2['today_rules'],
        'rule_stats': block2['rule_stats'],
        'depth_used': block2['depth_used'],
        'summary': {'total_signals': len(sigs), 'buy_count': buy_count, 'sell_count': sell_count},
        'conditions': ev2.get_conditions(),
        'prd_metrics': block2.get('prd_metrics', {}),
        'portfolio_sim': block2['portfolio_sim'],
        'v2': block2,
        'v3': block3,
    }
    if sigs_v4 is not None:
        payload['v4'] = block4
    return jsonify(payload)


@app.route('/api/v2/conditions', methods=['GET'])
def v2_get_conditions():
    stock_code = request.args.get('stock_code')
    conditions = ev2.get_conditions()
    return jsonify({
        'success': True,
        'conditions': conditions,
    })


@app.route('/api/v2/run_logs', methods=['GET'])
def v2_run_logs_list():
    stock_code = request.args.get('stock_code')
    run_type = request.args.get('run_type')
    try:
        limit = int(request.args.get('limit', 100))
    except (TypeError, ValueError):
        limit = 100
    logs = arl.list_run_logs(stock_code=stock_code, limit=limit, run_type=run_type)
    summary = arl.stock_run_summary(stock_code) if stock_code else None
    return jsonify({'success': True, 'logs': logs, 'summary': summary})


@app.route('/api/v2/run_logs/pool', methods=['GET'])
def v2_run_logs_pool():
    """股票池：最近 N 次运行分析、相对上一执行对比、历次时间线、池级汇总。"""
    stocks = Stock.get_all()
    codes = [s.get('code') for s in stocks if s.get('code')]
    try:
        history_per = int(request.args.get('history_per_stock', 10))
    except (TypeError, ValueError):
        history_per = 10
    history_per = max(1, min(history_per, 30))
    try:
        timeline_limit = int(request.args.get('timeline_limit', 200))
    except (TypeError, ValueError):
        timeline_limit = 200
    rows = arl.list_pool_dashboard_rows(codes, history_per_stock=history_per)
    aggregate = arl.pool_analyze_aggregate(rows)
    timeline = arl.list_pool_analyze_timeline(codes, limit=timeline_limit)
    return jsonify({
        'success': True,
        'stocks': stocks,
        'rows': rows,
        'aggregate': aggregate,
        'timeline': timeline,
    })


def _pool_run_one_stock(code, *, run_weights=True, run_analyze=True, trigger='pool_batch'):
    """单只股票：默认先公式计算再运行分析。"""
    row = {
        'stock_code': code,
        'weights_ok': False,
        'analyze_ok': False,
        'success': False,
        'message': '',
    }
    if run_weights:
        wpayload, wstatus = _run_calculate_weights_for_stock(code, trigger=trigger)
        row['weights_ok'] = wpayload.get('success', False)
        row['weights_return_pct'] = wpayload.get('weights_return_pct')
        row['weights_message'] = wpayload.get('message', '')
        row['weights_duration_ms'] = wpayload.get('duration_ms')
        if not row['weights_ok']:
            row['message'] = wpayload.get('message', '公式计算失败')
            row['http_status'] = wstatus
            return row
    if run_analyze:
        if not run_weights and not ev2.load_weights(code):
            row['message'] = '未计算权重，已跳过分析'
            row['skipped'] = True
            return row
        apayload, astatus = _run_analyze_for_stock(code, trigger=trigger)
        row['analyze_ok'] = apayload.get('success', False)
        row['v2_return_pct'] = apayload.get('v2_return_pct')
        row['v3_return_pct'] = apayload.get('v3_return_pct')
        row['v4_return_pct'] = apayload.get('v4_return_pct')
        row['analyze_duration_ms'] = apayload.get('duration_ms')
        row['http_status'] = astatus
        if not row['analyze_ok']:
            row['message'] = apayload.get('message', '运行分析失败')
            return row
    row['success'] = (row['weights_ok'] or not run_weights) and (row['analyze_ok'] or not run_analyze)
    if row['success']:
        parts = []
        if run_weights:
            parts.append('公式完成')
        if run_analyze:
            parts.append('分析完成')
        row['message'] = '、'.join(parts)
    return row


@app.route('/api/v2/pool/run', methods=['POST'])
@app.route('/api/v2/pool/analyze', methods=['POST'])
def v2_pool_run():
    """股票池批量：默认公式计算 + 运行分析（顺序执行，逐只写日志）。"""
    data = request.json or {}
    codes = data.get('stock_codes')
    run_weights = data.get('run_weights', True)
    if run_weights is None:
        run_weights = True
    run_analyze = data.get('run_analyze', True)
    if run_analyze is None:
        run_analyze = True
    run_weights = bool(run_weights)
    run_analyze = bool(run_analyze)
    trigger = data.get('trigger', 'pool_batch')

    if codes:
        codes = [str(c).strip() for c in codes if str(c).strip()]
    else:
        stocks = Stock.get_all()
        codes = [s.get('code') for s in stocks if s.get('code')]

    if not codes:
        return jsonify({'success': False, 'message': '股票池为空'}), 400
    if not run_weights and not run_analyze:
        return jsonify({'success': False, 'message': '请至少选择公式计算或运行分析'}), 400

    results = []
    ok = 0
    fail = 0
    skip = 0
    for code in codes:
        row = _pool_run_one_stock(
            code, run_weights=run_weights, run_analyze=run_analyze, trigger=trigger,
        )
        if row.get('skipped'):
            skip += 1
        elif row.get('success'):
            ok += 1
        else:
            fail += 1
        results.append(row)

    return jsonify({
        'success': True,
        'message': f'批量完成：成功 {ok}，失败 {fail}，跳过 {skip}',
        'summary': {
            'total': len(codes),
            'ok': ok,
            'fail': fail,
            'skip': skip,
            'run_weights': run_weights,
            'run_analyze': run_analyze,
        },
        'results': results,
    })


@app.route('/api/v2/run_logs', methods=['DELETE'])
def v2_run_logs_delete():
    stock_code = request.args.get('stock_code')
    before_days = request.args.get('before_days')
    try:
        before_days = int(before_days) if before_days is not None else None
    except (TypeError, ValueError):
        before_days = None
    deleted = arl.delete_run_logs(stock_code=stock_code, before_days=before_days)
    return jsonify({'success': True, 'deleted': deleted})
