import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import dynamic_weight_engine as dwe_v1
import dynamic_weight_engine_v2 as dwe_v2
from models import KlineData


def klines_to_df(klines):
    df = pd.DataFrame(klines)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def test_stock(code, name):
    print(f"\n{'='*80}")
    print(f"  测试股票: {name}({code})")
    print(f"{'='*80}")

    klines = KlineData.get(code, period='day')
    if not klines or len(klines) < 100:
        print(f"  数据不足: {len(klines) if klines else 0}条")
        return None

    df = klines_to_df(klines)
    print(f"  数据量: {len(df)}条, 日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    buy_rules, sell_rules = dwe_v1.get_all_rules()
    print(f"  买入规则: {len(buy_rules)}, 卖出规则: {len(sell_rules)}")

    # V1 算法
    print(f"\n  --- V1 算法 (标记值计数 + 固定5天回测) ---")
    try:
        v1_result = dwe_v1.calculate_weights(df)
        v1_buy_w = v1_result['buy_weights']
        v1_sell_w = v1_result['sell_weights']

        v1_signals, _ = dwe_v1.generate_signals_with_weights(
            df, buy_rules, v1_buy_w, sell_rules, v1_sell_w,
            min_interval=3, ma_period=20, start_offset=60
        )

        v1_eval = _eval_v1_signals(v1_signals, df)
        print(f"  V1 信号数: {len(v1_signals)}")
        print(f"  V1 胜率: {v1_eval['win_rate']*100:.1f}%")
        print(f"  V1 平均收益: {v1_eval['avg_return']*100:.2f}%")
        print(f"  V1 盈亏比: {v1_eval['profit_factor']:.2f}")
    except Exception as e:
        print(f"  V1 错误: {e}")
        v1_eval = None

    # V2 算法
    print(f"\n  --- V2 算法 (渐进惩罚 + 动态阈值 + 信号质量过滤) ---")
    try:
        v2_result = dwe_v2.calculate_weights_v2(df)
        v2_buy_w = v2_result['buy_weights']
        v2_sell_w = v2_result['sell_weights']

        v2_signals = dwe_v2.generate_signals_v2(
            df, buy_rules, v2_buy_w, sell_rules, v2_sell_w,
            min_interval=3, ma_period=20, start_offset=60,
            confidence_medium=0.55, confidence_strong=0.70,
            min_score_threshold=0.05,
            buy_details=v2_result.get('buy_details'),
            sell_details=v2_result.get('sell_details'),
            min_triggered_win_rate=0.50
        )

        v2_eval = dwe_v2.evaluate_signals(v2_signals, df)
        print(f"  V2 信号数: {v2_eval['count']} (强:{v2_eval['strong_count']}, 中:{v2_eval['medium_count']})")
        print(f"  V2 总胜率: {v2_eval['win_rate']*100:.1f}%")
        print(f"  V2 平均收益: {v2_eval['avg_return']*100:.2f}%")
        print(f"  V2 盈亏比: {v2_eval['profit_factor']:.2f}")
        print(f"  V2 强信号胜率: {v2_eval['strong_wr']*100:.1f}%")
        print(f"  V2 中信号胜率: {v2_eval['medium_wr']*100:.1f}%")

        print(f"\n  --- V2 买入规则权重 Top5 ---")
        buy_sorted = sorted(v2_result['buy_details'].items(),
                           key=lambda x: -x[1].get('normalized_weight', 0))
        for name, detail in buy_sorted[:5]:
            wr = detail.get('win_rate', 0) * 100
            ret = detail.get('avg_return', 0) * 100
            nw = detail.get('normalized_weight', 0) * 100
            pf = detail.get('profit_factor', 0)
            ah = detail.get('avg_hold', 0)
            pm = detail.get('penalty_multiplier', 1.0)
            pen = f'⚠×{pm:.2f}' if detail.get('penalized') else '✓'
            print(f"    {name}: 权重{nw:.1f}% 胜率{wr:.0f}% 收益{ret:.1f}% 盈亏比{pf:.2f} 持有{ah:.0f}天 {pen}")

        print(f"\n  --- V2 卖出规则权重 Top5 ---")
        sell_sorted = sorted(v2_result['sell_details'].items(),
                            key=lambda x: -x[1].get('normalized_weight', 0))
        for name, detail in sell_sorted[:5]:
            wr = detail.get('win_rate', 0) * 100
            ret = detail.get('avg_return', 0) * 100
            nw = detail.get('normalized_weight', 0) * 100
            pf = detail.get('profit_factor', 0)
            ah = detail.get('avg_hold', 0)
            pm = detail.get('penalty_multiplier', 1.0)
            pen = f'⚠×{pm:.2f}' if detail.get('penalized') else '✓'
            print(f"    {name}: 权重{nw:.1f}% 胜率{wr:.0f}% 收益{ret:.1f}% 盈亏比{pf:.2f} 持有{ah:.0f}天 {pen}")

    except Exception as e:
        import traceback
        print(f"  V2 错误: {e}")
        traceback.print_exc()
        v2_eval = None

    return {
        'code': code,
        'name': name,
        'v1': v1_eval,
        'v2': v2_eval,
    }


def _eval_v1_signals(signals, df, hold_days=5):
    close = df['close'].values.astype(float)
    n = len(close)
    returns = []
    for idx, sig_type in signals:
        if idx + hold_days >= n:
            continue
        entry = close[idx]
        if entry <= 0:
            continue
        exit_p = close[idx + hold_days]
        ret = (exit_p - entry) / entry
        if sig_type == 'S':
            ret = -ret
        returns.append(ret)

    if not returns:
        return {'win_rate': 0, 'avg_return': 0, 'count': 0, 'profit_factor': 0}

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    tp = sum(wins) if wins else 0
    tl = abs(sum(losses)) if losses else 0.001

    return {
        'win_rate': len(wins) / len(returns),
        'avg_return': np.mean(returns),
        'count': len(returns),
        'profit_factor': tp / tl,
    }


if __name__ == '__main__':
    stocks = [
        ('600519', '贵州茅台'),
        ('000001', '平安银行'),
        ('002347', '泰尔股份'),
    ]

    results = []
    for code, name in stocks:
        r = test_stock(code, name)
        if r:
            results.append(r)

    print(f"\n\n{'='*80}")
    print(f"  汇总对比")
    print(f"{'='*80}")
    print(f"{'股票':<12} | {'V1胜率':>8} | {'V1收益':>8} | {'V1盈亏比':>8} | {'V2胜率':>8} | {'V2收益':>8} | {'V2盈亏比':>8} | {'V2强信号胜率':>10}")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for r in results:
        v1 = r.get('v1') or {}
        v2 = r.get('v2') or {}
        v1_wr = f"{v1.get('win_rate', 0)*100:.1f}%"
        v1_ret = f"{v1.get('avg_return', 0)*100:.2f}%"
        v1_pf = f"{v1.get('profit_factor', 0):.2f}"
        v2_wr = f"{v2.get('win_rate', 0)*100:.1f}%"
        v2_ret = f"{v2.get('avg_return', 0)*100:.2f}%"
        v2_pf = f"{v2.get('profit_factor', 0):.2f}"
        v2_swr = f"{v2.get('strong_wr', 0)*100:.1f}%"
        print(f"{r['name']:<12} | {v1_wr:>8} | {v1_ret:>8} | {v1_pf:>8} | {v2_wr:>8} | {v2_ret:>8} | {v2_pf:>8} | {v2_swr:>10}")

    v1_avg_wr = np.mean([r['v1']['win_rate'] for r in results if r.get('v1')]) * 100 if results else 0
    v2_avg_wr = np.mean([r['v2']['win_rate'] for r in results if r.get('v2')]) * 100 if results else 0
    v1_avg_ret = np.mean([r['v1']['avg_return'] for r in results if r.get('v1')]) * 100 if results else 0
    v2_avg_ret = np.mean([r['v2']['avg_return'] for r in results if r.get('v2')]) * 100 if results else 0
    print(f"\n  平均胜率: V1={v1_avg_wr:.1f}% → V2={v2_avg_wr:.1f}% (提升{v2_avg_wr-v1_avg_wr:+.1f}%)")
    print(f"  平均收益: V1={v1_avg_ret:.2f}% → V2={v2_avg_ret:.2f}% (提升{v2_avg_ret-v1_avg_ret:+.2f}%)")
