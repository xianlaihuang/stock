"""智能分析 V2/V3/V4 运行日志（MongoDB analysis_run_log）。"""
from datetime import datetime

from db import db, stock_collection

run_log_collection = db['analysis_run_log']
run_log_collection.create_index([('stock_code', 1), ('created_at', -1)])
run_log_collection.create_index('created_at')


def _stock_name(code):
    try:
        doc = stock_collection.find_one({'code': code}, {'name': 1})
        return (doc or {}).get('name') or ''
    except Exception:
        return ''


def stock_names_map(codes):
    """一次查询股票名称，供批量任务复用。"""
    codes = list({str(c).strip() for c in (codes or []) if str(c).strip()})
    if not codes:
        return {}
    try:
        cursor = stock_collection.find(
            {'code': {'$in': codes}}, {'code': 1, 'name': 1, '_id': 0},
        )
        return {d['code']: (d.get('name') or '') for d in cursor}
    except Exception:
        return {}


def _portfolio_snapshot(port):
    if not port:
        return {}
    return {
        'initial_capital': float(port.get('initial_capital') or 1_000_000),
        'final_equity': port.get('final_equity'),
        'total_return_pct': port.get('total_return_pct'),
        'win_rate_pct': port.get('win_rate_pct'),
        'closed_trades': port.get('closed_trades'),
    }


def save_run_log(
    stock_code,
    run_type,
    *,
    success=True,
    duration_ms=0,
    message='',
    trigger='manual',
    v2_portfolio=None,
    v3_portfolio=None,
    v4_portfolio=None,
    depth_used=None,
    weights_opt=None,
    start_date=None,
    end_date=None,
    stock_name=None,
):
    doc = {
        'stock_code': stock_code,
        'stock_name': stock_name if stock_name is not None else _stock_name(stock_code),
        'run_type': run_type,
        'trigger': trigger,
        'success': bool(success),
        'duration_ms': int(duration_ms or 0),
        'message': message or '',
        'v2': _portfolio_snapshot(v2_portfolio),
        'v3': _portfolio_snapshot(v3_portfolio),
        'v4': _portfolio_snapshot(v4_portfolio),
        'depth_used': depth_used,
        'weights_opt': weights_opt or None,
        'start_date': start_date,
        'end_date': end_date,
        'created_at': datetime.now(),
    }
    result = run_log_collection.insert_one(doc)
    doc['_id'] = str(result.inserted_id)
    return _serialize_doc(doc)


def list_run_logs(stock_code=None, limit=100, run_type=None):
    q = {}
    if stock_code:
        q['stock_code'] = stock_code
    if run_type:
        q['run_type'] = run_type
    cursor = run_log_collection.find(q).sort('created_at', -1).limit(max(1, min(int(limit or 100), 500)))
    return [_serialize_doc(d) for d in cursor]


def delete_run_logs(stock_code=None, before_days=None):
    q = {}
    if stock_code:
        q['stock_code'] = stock_code
    if before_days is not None:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=int(before_days))
        q['created_at'] = {'$lt': cutoff}
    result = run_log_collection.delete_many(q)
    return result.deleted_count


def _pct_from_log(log, engine):
    if not log:
        return None
    v = (log.get(engine) or {}).get('total_return_pct')
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compare_analyze_logs(newer_log, older_log):
    """
    对比两次「运行分析」：newer 为最近一次，older 为上一次。
    返回各引擎收益差（newer − older），用于观察规则/权重调整后的效果。
    """
    if not newer_log or not older_log:
        return None
    if newer_log.get('success') is False or older_log.get('success') is False:
        return None
    out = {}
    for eng in ('v2', 'v3', 'v4'):
        n = _pct_from_log(newer_log, eng)
        o = _pct_from_log(older_log, eng)
        if n is not None and o is not None:
            out[f'{eng}_delta_pct'] = round(n - o, 2)
            out[f'{eng}_older_pct'] = o
            out[f'{eng}_newer_pct'] = n
    return out if out else None


def _analyze_count_by_codes(stock_codes):
    codes = [str(c).strip() for c in (stock_codes or []) if str(c).strip()]
    if not codes:
        return {}
    pipeline = [
        {'$match': {'run_type': 'analyze', 'stock_code': {'$in': codes}}},
        {'$group': {'_id': '$stock_code', 'n': {'$sum': 1}}},
    ]
    return {row['_id']: int(row['n']) for row in run_log_collection.aggregate(pipeline)}


def list_pool_dashboard_rows(stock_codes, history_per_stock=10):
    """
    股票池看板：每只股票各自查询最近 N 次运行分析（时间倒序），避免全局 limit 挤占其它股票条数。
    """
    codes = [str(c).strip() for c in (stock_codes or []) if str(c).strip()]
    if not codes:
        return []

    per = max(1, min(int(history_per_stock or 10), 30))
    total_counts = _analyze_count_by_codes(codes)
    out = []
    for code in sorted(codes):
        hist = [
            _serialize_doc(doc)
            for doc in run_log_collection.find(
                {'run_type': 'analyze', 'stock_code': code},
            ).sort('created_at', -1).limit(per)
        ]
        latest = hist[0] if hist else None
        previous = hist[1] if len(hist) > 1 else None
        out.append({
            'stock_code': code,
            'stock_name': (latest or {}).get('stock_name') or _stock_name(code),
            'latest': latest,
            'previous': previous,
            'compare': compare_analyze_logs(latest, previous),
            'history': hist,
            'has_log': latest is not None,
            'analyze_run_total': total_counts.get(code, 0),
            'analyze_run_shown': len(hist),
        })
    return out


def list_pool_analyze_timeline(stock_codes, limit=200):
    """股票池历次运行分析（时间倒序），每条附带相对上一执行的收益差。"""
    codes = [str(c).strip() for c in (stock_codes or []) if str(c).strip()]
    if not codes:
        return []
    lim = max(1, min(int(limit or 200), 500))
    cursor = run_log_collection.find(
        {'run_type': 'analyze', 'stock_code': {'$in': codes}},
    ).sort('created_at', -1).limit(lim)

    rows = []
    newer_by_code = {}
    for doc in cursor:
        log = _serialize_doc(doc)
        code = log.get('stock_code')
        newer = newer_by_code.get(code)
        log['compare_vs_newer'] = compare_analyze_logs(newer, log) if newer else None
        newer_by_code[code] = log
        rows.append(log)
    return rows


def list_latest_analyze_by_codes(stock_codes):
    """兼容旧接口：仅返回最近一次。"""
    rows = list_pool_dashboard_rows(stock_codes, history_per_stock=1)
    for row in rows:
        row.pop('history', None)
        row.pop('compare', None)
        row.pop('previous', None)
        row.pop('analyze_run_total', None)
        row.pop('analyze_run_shown', None)
    return rows


def pool_analyze_aggregate(rows):
    """对 list_pool_dashboard_rows 结果做池级汇总（含相对上一执行的变化统计）。"""
    v2_rets, v3_rets, v4_rets = [], [], []
    v2_deltas, v3_deltas, v4_deltas = [], [], []
    with_log = 0
    ok_count = 0
    with_compare = 0
    v4_improved = v4_worse = 0
    for row in rows or []:
        lat = row.get('latest')
        if not lat:
            continue
        with_log += 1
        if lat.get('success') is False:
            continue
        ok_count += 1
        for key, bucket in (('v2', v2_rets), ('v3', v3_rets), ('v4', v4_rets)):
            v = (lat.get(key) or {}).get('total_return_pct')
            if v is not None:
                try:
                    bucket.append(float(v))
                except (TypeError, ValueError):
                    pass
        cmp_ = row.get('compare')
        if cmp_:
            with_compare += 1
            for eng, dbucket in (('v2', v2_deltas), ('v3', v3_deltas), ('v4', v4_deltas)):
                d = cmp_.get(f'{eng}_delta_pct')
                if d is not None:
                    dbucket.append(float(d))
            d4 = cmp_.get('v4_delta_pct')
            if d4 is not None:
                if d4 > 0:
                    v4_improved += 1
                elif d4 < 0:
                    v4_worse += 1

    def _avg(arr):
        return round(sum(arr) / len(arr), 2) if arr else None

    return {
        'stock_count': len(rows or []),
        'with_analyze_log': with_log,
        'with_success_log': ok_count,
        'with_compare_pair': with_compare,
        'avg_v2_return_pct': _avg(v2_rets),
        'avg_v3_return_pct': _avg(v3_rets),
        'avg_v4_return_pct': _avg(v4_rets),
        'avg_v2_delta_pct': _avg(v2_deltas),
        'avg_v3_delta_pct': _avg(v3_deltas),
        'avg_v4_delta_pct': _avg(v4_deltas),
        'v4_improved_count': v4_improved,
        'v4_worse_count': v4_worse,
    }


def stock_run_summary(stock_code, limit=5):
    """每只股票最近几次运行的 V2/V3/V4 收益摘要，便于对比。"""
    logs = list_run_logs(stock_code=stock_code, limit=limit, run_type='analyze')
    if not logs:
        return None
    latest = logs[0]
    prev = logs[1] if len(logs) > 1 else None
    return {
        'stock_code': stock_code,
        'stock_name': latest.get('stock_name', ''),
        'latest': latest,
        'previous': prev,
        'run_count': len(logs),
    }


def _serialize_doc(doc):
    if not doc:
        return doc
    out = dict(doc)
    oid = out.pop('_id', None)
    if oid is not None:
        out['id'] = str(oid)
    ca = out.get('created_at')
    if isinstance(ca, datetime):
        out['created_at'] = ca.isoformat(timespec='seconds')
    return out
