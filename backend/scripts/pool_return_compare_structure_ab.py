#!/usr/bin/env python3
"""股票池 V4 结构识别改动前后对比：legacy(W/M/V旧) vs curve(W/M/V顶+ATR摆动，与V底一致)。"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_v2 as ev2
from models import Stock, KlineData
from routes_v2 import _klines_to_df
from v4_aggressive.engine import analyze_signals_v4_aggressive
from v4_aggressive.v4_structure_curves import reset_structure_registry_cache

MIN_KLINES = 60
OUT_CSV = os.path.join(os.path.dirname(__file__), 'pool_return_compare_structure_ab.csv')


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
        reset_structure_registry_cache()
        r_old = analyze_signals_v4_aggressive(
            df, pre, structure_mode='legacy', bearish_candle_mode='slim',
        )
        reset_structure_registry_cache()
        r_new = analyze_signals_v4_aggressive(
            df, pre, structure_mode='curve', bearish_candle_mode='slim',
        )
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
        'legacy_ret': old_ret,
        'legacy_trades': p_old.get('closed_trades'),
        'legacy_win': p_old.get('win_rate_pct'),
        'curve_ret': new_ret,
        'curve_trades': p_new.get('closed_trades'),
        'curve_win': p_new.get('win_rate_pct'),
        'delta_curve_minus_legacy': round(new_ret - old_ret, 2),
        'delta_trades': (p_new.get('closed_trades') or 0) - (p_old.get('closed_trades') or 0),
        'seconds': round(elapsed, 1),
    }


def main():
    stocks = Stock.get_all() or []
    if not stocks:
        print('股票池为空')
        return 1

    rows = []
    print('V4 结构 A/B：legacy(旧W/M/V) vs curve(与V底同逻辑)\n')
    print('说明：看跌K线均为 slim；仅 W底/M顶/V反右侧/V反顶部 识别方式不同\n')

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
            print(
                f'  改动前 {row["legacy_ret"]:+.2f}% ({row["legacy_trades"]}笔) → '
                f'改动后 {row["curve_ret"]:+.2f}% ({row["curve_trades"]}笔) · '
                f'Δ {row["delta_curve_minus_legacy"]:+.2f}%'
            )
        rows.append(row)

    ok = [r for r in rows if r and 'error' not in r]
    n = len(ok)
    avg_old = round(sum(r['legacy_ret'] for r in ok) / n, 2) if n else 0
    avg_new = round(sum(r['curve_ret'] for r in ok) / n, 2) if n else 0
    avg_d = round(sum(r['delta_curve_minus_legacy'] for r in ok) / n, 2) if n else 0
    curve_better = sum(1 for r in ok if (r['delta_curve_minus_legacy'] or 0) > 0)
    legacy_better = sum(1 for r in ok if (r['delta_curve_minus_legacy'] or 0) < 0)

    ok.sort(key=lambda r: r.get('delta_curve_minus_legacy') or 0, reverse=True)
    fields = [
        'code', 'name', 'bars',
        'legacy_ret', 'legacy_trades', 'legacy_win',
        'curve_ret', 'curve_trades', 'curve_win',
        'delta_curve_minus_legacy', 'delta_trades', 'seconds',
    ]
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in ok:
            w.writerow(r)

    print('\n' + '=' * 72)
    print('【汇总】')
    print(f'有效 {n} 只 | 平均 legacy {avg_old}% → curve {avg_new}% | 池均Δ {avg_d}%')
    print(f'curve 更高 {curve_better} 只 | legacy 更高 {legacy_better} 只')

    print('\n【curve 提升最多 TOP 5】')
    for r in ok[:5]:
        print(
            f"  {r['code']} {r['name']:<8}  {r['legacy_ret']:+7.2f}% → {r['curve_ret']:+7.2f}%  "
            f"Δ {r['delta_curve_minus_legacy']:+.2f}%"
        )

    print('\n【curve 下降最多 TOP 5】')
    for r in sorted(ok, key=lambda x: x.get('delta_curve_minus_legacy') or 0)[:5]:
        print(
            f"  {r['code']} {r['name']:<8}  {r['legacy_ret']:+7.2f}% → {r['curve_ret']:+7.2f}%  "
            f"Δ {r['delta_curve_minus_legacy']:+.2f}%"
        )

    print(f'\nCSV: {OUT_CSV}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
