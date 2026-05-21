#!/usr/bin/env python3
"""对比 002979 的 V3 稳定版 vs 推测异常版交易差异。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper import StockScraper
import engine_v2 as ev2
from v4_aggressive.engine import analyze_signals_v4_aggressive, _mandatory_buy_rules_v4
from routes_v2 import _klines_to_df

CODE = '002979'
TARGET_RET = 25.41
TARGET_N = 30


def trades_from_ps(ps):
    out = []
    b = None
    brules = None
    for row in ps:
        if row['type'] == 'B':
            b = row.get('date')
            brules = tuple(sorted((row.get('rules') or row.get('reasons') or [])))
        elif row['type'] == 'S' and b:
            key = (b, row.get('date'), round(float(row.get('return_pct') or 0), 2))
            out.append({
                'buy': b,
                'sell': row.get('date'),
                'ret': row.get('return_pct'),
                'sell_reason': row.get('sell_reason_type', ''),
                'buy_rules': brules,
                'key': key,
            })
            b = None
    return out


def port(ps):
    return ev2._portfolio_sim_from_paired(ps)


def main():
    kl = StockScraper.get_kline_data(CODE, period='day', count=800)
    if not kl or len(kl) < 60:
        print('K线不足')
        return 1
    df = _klines_to_df(kl)
    pre = None
    c = ev2.load_weights(CODE)
    if c:
        pre = {'buy_weights': c.get('buy_weights', {}), 'sell_weights': c.get('sell_weights', {})}

    mb_v3 = ev2._mandatory_buy_rules_v3(df, 60)
    mb_v4 = _mandatory_buy_rules_v4(df, 60)

    scenarios = {
        'V3_当前稳定(1.32%/25笔)': ev2.analyze_signals_v2(
            df, pre, engine_mode='v3', mandatory_buy_rules_override=mb_v3,
        ),
        'V4_当前(-0.67%/28笔)': analyze_signals_v4_aggressive(df, pre),
        'V3误挂V4必买_推测': ev2.analyze_signals_v2(
            df, pre, engine_mode='v3', mandatory_buy_rules_override=mb_v4,
        ),
        'V4引擎当V3跑_推测': ev2.analyze_signals_v2(
            df, pre, engine_mode='v4', mandatory_buy_rules_override=mb_v4,
        ) if False else None,
    }
    scenarios = {k: v for k, v in scenarios.items() if v is not None}

    print('=' * 72)
    print(f'{CODE} 交易 diff | K线 {len(df)} 根')
    print('=' * 72)
    print('\n【场景收益】')
    for name, r in scenarios.items():
        p = port(r['paired_signals'])
        print(f"  {name}: {p['total_return_pct']:+.2f}% / {p['closed_trades']} 笔")

    print(f'\n【目标】5/17 异常日志: +{TARGET_RET}% / {TARGET_N} 笔\n')

    best_name = None
    best_d = 1e9
    for name, r in scenarios.items():
        p = port(r['paired_signals'])
        d = abs(p['total_return_pct'] - TARGET_RET) + abs(p['closed_trades'] - TARGET_N) * 2
        if d < best_d:
            best_d = d
            best_name = name
    print(f'最接近异常日志的场景: {best_name} (distance={best_d:.2f})\n')

    base_name = 'V3_当前稳定(1.32%/25笔)'
    alt_candidates = [n for n in scenarios if n != base_name]

    base = trades_from_ps(scenarios[base_name]['paired_signals'])
    base_keys = {x['key'] for x in base}

    print(f'基准: {base_name} — {len(base)} 笔\n')

    for alt_name in alt_candidates:
        alt = trades_from_ps(scenarios[alt_name]['paired_signals'])
        alt_keys = {x['key'] for x in alt}
        only_alt = [x for x in alt if x['key'] not in base_keys]
        only_base = [x for x in base if x['key'] not in alt_keys]
        same = [x for x in alt if x['key'] in base_keys]

        print('-' * 72)
        print(f'diff: {base_name}')
        print(f'  vs  {alt_name}')
        print(f'  相同 {len(same)} 笔 | 仅基准 {len(only_base)} 笔 | 仅对比 {len(only_alt)} 笔')

        if only_alt:
            print(f'\n  ▶ 仅【{alt_name}】有 ({len(only_alt)} 笔) — 异常多出来的来源:')
            for x in only_alt:
                ag = [r for r in (x.get('buy_rules') or ()) if '激进' in r or 'V4' in r or 'V反' in r]
                print(f"    {x['buy']} → {x['sell']}  {x['ret']:+.2f}%  | 卖:{x['sell_reason'] or '-'}")
                if ag:
                    print(f"       买入: {', '.join(ag)}")

        if only_base:
            print(f'\n  ▶ 仅【{base_name}】有 ({len(only_base)} 笔) — 当前多保留:')
            for x in only_base:
                print(f"    {x['buy']} → {x['sell']}  {x['ret']:+.2f}%  | 卖:{x['sell_reason'] or '-'}")

    # 复利贡献：仅 alt 独有笔的收益乘积影响
    if best_name and best_name != base_name:
        only_alt = [x for x in trades_from_ps(scenarios[best_name]['paired_signals'])
                    if x['key'] not in base_keys]
        if only_alt:
            eq = 1.0
            for x in only_alt:
                eq *= (1.0 + float(x['ret']) / 100.0)
            print('\n' + '=' * 72)
            print(f'【收益差拆解】相对稳定 V3，{best_name} 多出的 {len(only_alt)} 笔')
            print(f'  若单独复利这些多出来的笔: ×{eq:.4f} ({(eq-1)*100:+.2f}%)')
            rets = [float(x['ret']) for x in only_alt]
            print(f'  单笔合计: {sum(rets):+.2f}%')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
