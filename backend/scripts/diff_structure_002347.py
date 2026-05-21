#!/usr/bin/env python3
"""002347 泰尔：V4 结构 legacy vs curve 交易与形态事件对比。"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_v2 as ev2
from models import KlineData
from routes_v2 import _klines_to_df
from v4_aggressive.engine import analyze_signals_v4_aggressive
from v4_aggressive.v4_structure_curves import reset_structure_registry_cache

CODE = '002347'


def trades_from_ps(ps):
    out = []
    b = None
    brules = None
    for row in ps:
        if row['type'] == 'B':
            b = row.get('date')
            brules = list(row.get('rules') or row.get('reasons') or [])
        elif row['type'] == 'S' and b:
            out.append({
                'buy': b,
                'sell': row.get('date'),
                'ret': float(row.get('return_pct') or 0),
                'sell_reason': row.get('sell_reason_type', '') or '',
                'buy_rules': brules,
                'key': (b, row.get('date'), round(float(row.get('return_pct') or 0), 2)),
            })
            b = None
    return out


def count_events(name, df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)
    open_ = df['open'].values if 'open' in df.columns else close
    from v4_aggressive.strategy_vw import (
        detect_w_right_bottom_events,
        detect_w_right_bottom_events_legacy,
        detect_v_right_bottom_events,
        detect_v_right_bottom_events_legacy,
    )
    fns = {
        'w_leg': lambda: detect_w_right_bottom_events_legacy(close, high, low, n),
        'w_cur': lambda: detect_w_right_bottom_events(close, high, low, n, open_=open_),
        'v_leg': lambda: detect_v_right_bottom_events_legacy(close, high, low, n),
        'v_cur': lambda: detect_v_right_bottom_events(close, high, low, n, open_=open_),
    }
    evs = fns[name]()
    return len(evs), evs


def main():
    klines = KlineData.get(CODE, period='day')
    df = _klines_to_df(klines)
    pre = ev2.load_weights(CODE)
    pw = {'buy_weights': pre['buy_weights'], 'sell_weights': pre['sell_weights']} if pre else None

    reset_structure_registry_cache()
    r_leg = analyze_signals_v4_aggressive(df, pw, structure_mode='legacy', bearish_candle_mode='slim')
    reset_structure_registry_cache()
    r_cur = analyze_signals_v4_aggressive(df, pw, structure_mode='curve', bearish_candle_mode='slim')

    p_leg = ev2._portfolio_sim_from_paired(r_leg['paired_signals'])
    p_cur = ev2._portfolio_sim_from_paired(r_cur['paired_signals'])
    t_leg = trades_from_ps(r_leg['paired_signals'])
    t_cur = trades_from_ps(r_cur['paired_signals'])
    k_leg = {t['key'] for t in t_leg}
    k_cur = {t['key'] for t in t_cur}
    only_leg = [t for t in t_leg if t['key'] not in k_cur]
    only_cur = [t for t in t_cur if t['key'] not in k_leg]
    same = [t for t in t_cur if t['key'] in k_leg]

    from v4_aggressive.strategy_vw import (
        detect_w_right_bottom_events,
        detect_w_right_bottom_events_legacy,
        detect_v_right_bottom_events,
        detect_v_right_bottom_events_legacy,
    )
    from v4_aggressive.v4_structure_curves import get_structure_registry

    open_ = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    n = len(c)
    reg = get_structure_registry(open_, h, l, c, n)

    print('=' * 72)
    print(f'{CODE} 泰尔股份 | K线 {n} 根')
    print('=' * 72)
    print(f"\n收益: legacy {p_leg['total_return_pct']:+.2f}% ({p_leg['closed_trades']}笔)")
    print(f"      curve  {p_cur['total_return_pct']:+.2f}% ({p_cur['closed_trades']}笔)")
    print(f"      Δ {p_cur['total_return_pct'] - p_leg['total_return_pct']:+.2f}%")

    nw_l, _ = count_events('w_leg', df)
    nw_c, _ = count_events('w_cur', df)
    nv_l, _ = count_events('v_leg', df)
    nv_c, _ = count_events('v_cur', df)

    print('\n【形态事件数】必买触发源')
    print(f'  W底右侧  legacy {nw_l} → curve {nw_c}')
    print(f'  V反右侧  legacy {nv_l} → curve {nv_c}')
    print(f'  registry: V形态 {len(reg.v_patterns)} W形态 {len(reg.w_patterns)} '
          f'M跌破日 {len(reg.m_break_bars)} V顶跌破日 {len(reg.v_top_break_bars)}')

    def rule_cnt(ps, side='B'):
        c = Counter()
        for row in ps:
            if row['type'] != side:
                continue
            for r in row.get('rules') or row.get('reasons') or []:
                c[str(r).split('·')[0]] += 1
        return c

    print('\n【买入规则次数】legacy vs curve')
    bl, bc = rule_cnt(r_leg['paired_signals'], 'B'), rule_cnt(r_cur['paired_signals'], 'B')
    for name in sorted(set(bl) | set(bc)):
        print(f'  {name}: {bl.get(name,0)} → {bc.get(name,0)}')

    print('\n【卖出原因次数】')
    sl, sc = Counter(), Counter()
    for row in r_leg['paired_signals']:
        if row['type'] == 'S':
            sl[row.get('sell_reason_type') or '计数/规则'] += 1
    for row in r_cur['paired_signals']:
        if row['type'] == 'S':
            sc[row.get('sell_reason_type') or '计数/规则'] += 1
    for name in sorted(set(sl) | set(sc)):
        print(f'  {name}: {sl.get(name,0)} → {sc.get(name,0)}')

    print(f'\n【交易重合】相同 {len(same)} | 仅legacy {len(only_leg)} | 仅curve {len(only_cur)}')

    if only_leg:
        print('\n▶ 仅 legacy 有的交易（curve 少掉的主要利润来源）:')
        only_leg.sort(key=lambda x: -x['ret'])
        for t in only_leg:
            rules = ', '.join(t['buy_rules'][:4])
            print(f"  {t['buy']} → {t['sell']}  {t['ret']:+.2f}%  卖:{t['sell_reason'] or '-'}  买:[{rules}]")
        eq = 1.0
        for t in only_leg:
            eq *= 1 + t['ret'] / 100
        print(f'  若仅从复利角度少掉这 {len(only_leg)} 笔: 乘数 ×{eq:.4f} ({(eq-1)*100:+.2f}%)')

    if only_cur:
        print('\n▶ 仅 curve 有的交易:')
        only_cur.sort(key=lambda x: -x['ret'])
        for t in only_cur[:15]:
            rules = ', '.join(t['buy_rules'][:4])
            print(f"  {t['buy']} → {t['sell']}  {t['ret']:+.2f}%  卖:{t['sell_reason'] or '-'}  买:[{rules}]")

    # 相同交易但收益不同（应极少）
    diff_ret = []
    cur_map = {t['key']: t for t in t_cur}
    for t in t_leg:
        if t['key'] in cur_map and abs(t['ret'] - cur_map[t['key']]['ret']) > 0.01:
            diff_ret.append((t, cur_map[t['key']]))
    if diff_ret:
        print('\n▶ 同日买卖但收益率不同:')
        for a, b in diff_ret:
            print(f"  {a['key']} legacy {a['ret']:+.2f}% curve {b['ret']:+.2f}%")

    # 大亏/大赚对比
    print('\n【legacy 单笔>30%】')
    for t in sorted(t_leg, key=lambda x: -x['ret']):
        if t['ret'] > 30:
            in_c = '✓curve也有' if t['key'] in k_cur else '✗curve无'
            print(f"  {t['ret']:+.1f}% {t['buy']}→{t['sell']} {in_c}  {t['sell_reason']}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
