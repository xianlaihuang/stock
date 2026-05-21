#!/usr/bin/env python3
"""股票池 V4 改动前后对比：legacy(含黄昏星/十字/射击之星+守MA5) vs slim(已去掉)。"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_v2 as ev2
from models import Stock, KlineData
from routes_v2 import _klines_to_df
from v4_aggressive.engine import analyze_signals_v4_aggressive

MIN_KLINES = 60
OUT_CSV = os.path.join(os.path.dirname(__file__), 'pool_return_compare_v4_ab.csv')


def _precomputed(code):
    c = ev2.load_weights(code)
    if not c:
        return None
    return {
        'buy_weights': c.get('buy_weights', {}),
        'sell_weights': c.get('sell_weights', {}),
    }


def _port(ps):
    return ev2._portfolio_sim_from_paired(ps or [])


def _run_stock(code, name, pre):
    klines = KlineData.get(code, period='day')
    if not klines or len(klines) < MIN_KLINES:
        return None
    df = _klines_to_df(klines)
    t0 = time.perf_counter()
    try:
        r_old = analyze_signals_v4_aggressive(df, pre, bearish_candle_mode='legacy')
        r_new = analyze_signals_v4_aggressive(df, pre, bearish_candle_mode='slim')
    except Exception as e:
        return {'code': code, 'name': name, 'error': str(e)}
    elapsed = time.perf_counter() - t0
    p_old = _port(r_old.get('paired_signals'))
    p_new = _port(r_new.get('paired_signals'))
    old_ret = p_old.get('total_return_pct') or 0
    new_ret = p_new.get('total_return_pct') or 0
    return {
        'code': code,
        'name': name,
        'bars': len(df),
        'v4_legacy_ret': old_ret,
        'v4_legacy_trades': p_old.get('closed_trades'),
        'v4_legacy_win': p_old.get('win_rate_pct'),
        'v4_slim_ret': new_ret,
        'v4_slim_trades': p_new.get('closed_trades'),
        'v4_slim_win': p_new.get('win_rate_pct'),
        'delta_slim_minus_legacy': round(new_ret - old_ret, 2),
        'delta_trades': (p_new.get('closed_trades') or 0) - (p_old.get('closed_trades') or 0),
        'seconds': round(elapsed, 1),
    }


def main():
    stocks = Stock.get_all() or []
    if not stocks:
        print('股票池为空')
        return 1

    rows = []
    print(f'V4 A/B：legacy(改动前) vs slim(改动后) · 共 {len(stocks)} 只\n')
    for i, s in enumerate(stocks):
        code = s.get('code')
        name = s.get('name', '')
        if not code:
            continue
        print(f'[{i+1}/{len(stocks)}] {code} {name}', flush=True)
        row = _run_stock(code, name, _precomputed(code))
        if row is None:
            print('  跳过: K线不足')
            continue
        if 'error' in row:
            print(f'  失败: {row["error"]}')
        else:
            d = row['delta_slim_minus_legacy']
            print(
                f'  改动前 legacy {row["v4_legacy_ret"]:+.2f}% ({row["v4_legacy_trades"]}笔) → '
                f'改动后 slim {row["v4_slim_ret"]:+.2f}% ({row["v4_slim_trades"]}笔) · '
                f'Δ {d:+.2f}% · 笔数Δ {row["delta_trades"]:+d}'
            )
        rows.append(row)

    ok = [r for r in rows if r and 'error' not in r]
    ok.sort(key=lambda r: r.get('delta_slim_minus_legacy') or 0, reverse=True)

    fields = [
        'code', 'name', 'bars',
        'v4_legacy_ret', 'v4_legacy_trades', 'v4_legacy_win',
        'v4_slim_ret', 'v4_slim_trades', 'v4_slim_win',
        'delta_slim_minus_legacy', 'delta_trades', 'seconds',
    ]
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in ok:
            w.writerow(r)

    n = len(ok)
    avg_old = round(sum(r['v4_legacy_ret'] for r in ok) / n, 2) if n else 0
    avg_new = round(sum(r['v4_slim_ret'] for r in ok) / n, 2) if n else 0
    avg_d = round(sum(r['delta_slim_minus_legacy'] for r in ok) / n, 2) if n else 0
    slim_better = sum(1 for r in ok if (r['delta_slim_minus_legacy'] or 0) > 0)
    legacy_better = sum(1 for r in ok if (r['delta_slim_minus_legacy'] or 0) < 0)
    same = n - slim_better - legacy_better

    print('\n' + '=' * 72)
    print('【V4 改动前后汇总】100万初始 · 全仓复利 · 仅差异=看跌K线形态规则')
    print('=' * 72)
    print(f'有效: {n} 只')
    print(f'平均收益  改动前 legacy: {avg_old}%  |  改动后 slim: {avg_new}%  |  平均Δ: {avg_d}%')
    print(f'slim 更高: {slim_better} 只 | legacy 更高: {legacy_better} 只 | 持平: {same} 只')

    print('\n【去掉三种形态后收益提升最多 TOP 5】')
    for r in ok[:5]:
        print(
            f"  {r['code']} {r['name']:<8}  {r['v4_legacy_ret']:+7.2f}% → {r['v4_slim_ret']:+7.2f}%  "
            f"Δ {r['delta_slim_minus_legacy']:+.2f}%  (笔 {r['v4_legacy_trades']}→{r['v4_slim_trades']})"
        )

    print('\n【去掉三种形态后收益下降最多 TOP 5】')
    for r in sorted(ok, key=lambda x: x.get('delta_slim_minus_legacy') or 0)[:5]:
        print(
            f"  {r['code']} {r['name']:<8}  {r['v4_legacy_ret']:+7.2f}% → {r['v4_slim_ret']:+7.2f}%  "
            f"Δ {r['delta_slim_minus_legacy']:+.2f}%"
        )

    print(f'\nCSV: {OUT_CSV}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
