"""V5 规则定义（占位，后续在此独立演进）。"""

V5_VERSION = '0.1.0-scaffold'


def get_conditions():
    """返回前端规则面板所需的结构化说明。"""
    return {
        'buy_necessary': [
            {
                'name': '待定义',
                'description': 'V5 买入必要条件尚未实现，后续在 v5/rules.py 与 v5/engine.py 中完善。',
            },
        ],
        'buy_sufficient': [
            {
                'name': '待定义',
                'description': 'V5 买入充分条件尚未实现。',
            },
        ],
        'sell_necessary': [
            {
                'name': '待定义',
                'description': 'V5 卖出必要条件尚未实现。',
            },
        ],
        'sell_sufficient': [
            {
                'name': '待定义',
                'description': 'V5 卖出充分条件尚未实现。',
            },
        ],
        'optimization': [
            {
                'name': '独立引擎',
                'description': f'V5 为全新规则体系（当前版本 {V5_VERSION}），不复用 V2/V3/V4 打标逻辑与权重公式。',
            },
        ],
    }
