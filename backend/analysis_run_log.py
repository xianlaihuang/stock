"""智能分析 V2/V3 运行日志（MongoDB analysis_run_log）。"""
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
    depth_used=None,
    weights_opt=None,
    start_date=None,
    end_date=None,
):
    doc = {
        'stock_code': stock_code,
        'stock_name': _stock_name(stock_code),
        'run_type': run_type,
        'trigger': trigger,
        'success': bool(success),
        'duration_ms': int(duration_ms or 0),
        'message': message or '',
        'v2': _portfolio_snapshot(v2_portfolio),
        'v3': _portfolio_snapshot(v3_portfolio),
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


def stock_run_summary(stock_code, limit=5):
    """每只股票最近几次运行的 V2/V3 收益摘要，便于对比。"""
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
