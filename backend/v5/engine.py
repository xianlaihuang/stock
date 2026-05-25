"""
V5 独立 B/S 引擎（占位实现）。

与 V2/V3/V4 无规则代码依赖；具体买卖逻辑后续在 v5/ 包内完善。
"""
from v5.rules import get_conditions, V5_VERSION


def _empty_today():
    return {
        'today_buy': False,
        'today_sell': False,
        'today_reasons': [],
        'today_score': 0,
        'today_buy_score': 0,
        'today_sell_score': 0,
    }


def analyze_signals_v5(df, start_date=None, end_date=None, **kwargs):
    """
    运行 V5 分析。

    当前为骨架：返回空信号列表与规则说明，供前后端联调与后续规则填充。
    kwargs 预留扩展（如 lite、自定义参数），暂不使用。
    """
    del start_date, end_date, kwargs

    if df is None or len(df) < 1:
        paired_signals = []
    else:
        paired_signals = []

    today = _empty_today()
    conditions = get_conditions()

    return {
        'paired_signals': paired_signals,
        'all_signals': paired_signals,
        'prd_metrics': {},
        **today,
        'conditions': conditions,
        'today_rules': today['today_reasons'],
        'rule_stats': {},
        'depth_used': len(df) if df is not None else 0,
        'summary': {
            'total_signals': len(paired_signals),
            'buy_count': sum(1 for s in paired_signals if s.get('type') == 'B'),
            'sell_count': sum(1 for s in paired_signals if s.get('type') == 'S'),
        },
        'engine_mode': 'v5',
        'engine_version': V5_VERSION,
    }
