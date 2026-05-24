"""
V4 B/S 信号：按触发的规则名输出 K 线浮框用的结构上下文（日期、价格、组成缺口的两根 K 线等）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v4_aggressive import dynamic_weight_engine as dwe
from v4_aggressive.strategy_vw import detect_w_right_bottom_events, detect_v_right_bottom_events


def _bar_date(df, i):
    return dwe.bar_date_str_from_df(df, int(i))


def _bar_line(df, i, role, price=None):
    d = _bar_date(df, i)
    if d is None:
        return f'{role}: 第{i}棒'
    px = f' @{round(float(price), 2)}' if price is not None else ''
    return f'{role}: {d}{px}'


def _historical_prior_high_lines(df, idx, buy_triggered=None):
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    buy_high = float(high[idx])
    buy_body = dwe.bar_body_top_at(open_, close, idx)
    pi, pw, prior_body = dwe.find_historical_prior_high_for_buy(
        df, idx, buy_triggered=buy_triggered,
    )
    if pi is None:
        return ['历史前高: 买入日前未找到 high 高于买入日最高价的合格前高']
    upside = (prior_body - buy_body) / buy_body * 100.0 if buy_body > 0 else None
    lines = [
        _bar_line(df, idx, '买入日最高价', buy_high),
        f'买入日实体上沿: {buy_body:.2f}',
        _bar_line(df, pi, '历史前高(实体上沿)', prior_body),
        f'历史前高最高价: {pw:.2f}',
    ]
    if upside is not None:
        lines.append(
            f'实体上沿→历史前高实体上沿空间: {upside:.2f}%（门槛≥3%，仅回看买入日前）'
        )
    return lines


def _next_prior_high_lines(df, idx, buy_triggered=None):
    return _historical_prior_high_lines(df, idx, buy_triggered=buy_triggered)


def _gap_lines(df, idx):
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    gap = dwe.find_recent_unfilled_upper_gap(close, high, low, idx, lookback=200, min_bars_after=5)
    if gap is None:
        return ['未找到未回补的上方缺口']
    n = gap['n']
    nm1 = gap['n_minus_1']
    return [
        _bar_line(df, nm1, '缺口左棒(上沿=最低)', gap['gap_top']),
        _bar_line(df, n, '缺口右棒(下沿=最高)', gap['gap_bottom']),
        f'缺口条件: 左棒最低 {gap["gap_top"]:.2f} > 右棒最高 {gap["gap_bottom"]:.2f}',
        f'自 {_bar_date(df, n)} 起至信号日最高价未回补左棒最低(上沿)',
    ]


def _hs_top_lines(df, idx):
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    n = len(close)
    from v4_aggressive.v4_structure_curves import get_structure_registry

    reg = get_structure_registry(open_, high, low, close, n)
    idx = int(idx)
    for p in reg.hs_top_patterns:
        if p.break_idx is None or int(p.break_idx) != idx:
            continue
        return [
            _bar_line(df, p.left_shoulder_idx, '左肩(摆动高点)', p.left_shoulder_price),
            _bar_line(df, p.head_idx, '头部(摆动高点)', p.head_price),
            _bar_line(df, p.right_shoulder_idx, '右肩(摆动高点)', p.right_shoulder_price),
            f'颈线(两肩间low最小): {p.neckline_price:.2f}',
            _bar_line(df, p.break_idx, '首次跌破颈线日收盘', close[p.break_idx]),
        ]
    return ['头肩顶: 该日非 registry 记录的首次颈线跌破日']


def _m_top_lines(df, idx):
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    n = len(close)
    highs = []
    for i in range(8, n):
        if high[i] == high[max(0, i - 8) : min(n, i + 4)].max():
            highs.append(i)
    filtered = [highs[0]] if highs else []
    for i in range(1, len(highs)):
        if highs[i] - filtered[-1] >= 5:
            filtered.append(highs[i])
    for i in range(len(filtered) - 1):
        idx1, idx2 = filtered[i], filtered[i + 1]
        if idx2 - idx1 < 10 or idx2 - idx1 > 60:
            continue
        p1, p2 = high[idx1], high[idx2]
        if abs(p1 - p2) / max(p1, p2) > 0.02:
            continue
        between_low = close[idx1 : idx2 + 1].min()
        retrace = (max(p1, p2) - between_low) / max(p1, p2)
        if retrace < 0.03:
            continue
        top_price = max(p1, p2)
        for j in range(idx2 + 1, min(idx2 + 30, n)):
            if j == idx and close[j] < top_price * 0.98:
                return [
                    _bar_line(df, idx1, 'M左顶', p1),
                    _bar_line(df, idx2, 'M右顶', p2),
                    f'M颈线(两顶间最低收盘): {between_low:.2f}',
                    f'跌破线: 顶部×98% = {top_price * 0.98:.2f}',
                    _bar_line(df, j, '跌破日收盘', close[j]),
                ]
    return ['M顶结构: 信号日未匹配到标准双顶跌破序列']


def _w_right_lines(df, idx):
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    n = len(close)
    for ev in detect_w_right_bottom_events(close, high, low, n):
        if int(ev['entry']) != int(idx):
            continue
        idx1, idx2 = ev.get('idx1'), ev.get('idx2')
        lines = []
        if idx1 is not None:
            lines.append(_bar_line(df, idx1, 'W左底'))
        if idx2 is not None:
            lines.append(_bar_line(df, idx2, 'W右底'))
        lines.extend([
            _bar_line(df, idx, 'W右侧确认日(必买日)'),
            f'瓶口(两底间high最大): {ev["neck"]:.2f}',
            f'止损参考: {ev["stop_ref"]:.2f}',
        ])
        return lines
    return ['W底右侧: 该日无匹配的 W 右侧事件']


def _v_right_lines(df, idx):
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    n = len(close)

    from v4_aggressive.v4_structure_curves import get_structure_registry

    reg = get_structure_registry(open_, high, low, close, n)
    for p in reg.v_patterns:
        if p.parent is not None:
            continue
        if p.right_entry_idx is None or int(p.right_entry_idx) != int(idx):
            continue
        return [
            _bar_line(df, p.left_peak_idx, 'V起点(左肩摆动高点)', p.neck_price),
            _bar_line(df, p.bottom_idx, 'V底(摆动低点)', p.bottom_price),
            f'左侧跌深(均价曲线): {p.drop_pct * 100:.2f}%',
            _bar_line(df, p.right_entry_idx, 'V右侧确认日(必买日)'),
            f'瓶口(high区间最大): {p.neck_price:.2f}',
        ]

    for ev in detect_v_right_bottom_events(close, high, low, n, open_=open_):
        if int(ev['entry']) != int(idx):
            continue
        low_idx = ev.get('low_idx')
        lines = [
            _bar_line(df, idx, 'V右侧确认日'),
            f'瓶口: {ev["neck"]:.2f}',
            f'V底参考价: {ev["stop_ref"]:.2f}',
        ]
        if low_idx is not None:
            lines.insert(0, _bar_line(df, low_idx, 'V底'))
        return lines
    return ['V反右侧: 该日无匹配的 V 右侧事件']


def _pressure_lines(df, idx):
    high = df['high'].values.astype(float) if 'high' in df.columns else df['close'].values.astype(float)
    pivot_i = None
    pivot_p = -1.0
    for i in range(idx - 25, max(10, idx - 220), -1):
        if i < 5 or i + 5 > len(high):
            continue
        hh = float(high[i])
        if hh < float(np.max(high[i - 5 : i + 6])) - 1e-9:
            continue
        zhi = (hh + 2.0) if hh >= 30.0 else hh * 1.02
        if i + 1 < idx:
            mx_after = float(np.max(high[i + 1 : idx]))
            if mx_after >= zhi * (1.0 - 1e-9):
                continue
        pivot_i = i
        pivot_p = hh
        break
    if pivot_i is None:
        return ['压力位: 未定位到历史局部高点']
    if pivot_p >= 30.0:
        zone_lo, zone_hi = pivot_p - 2.0, pivot_p + 2.0
        band = f'±2元 ({zone_lo:.2f}~{zone_hi:.2f})'
    else:
        zone_lo, zone_hi = pivot_p * 0.98, pivot_p * 1.02
        band = f'±2% ({zone_lo:.2f}~{zone_hi:.2f})'
    return [
        _bar_line(df, pivot_i, '压力来源(历史局部高点)', pivot_p),
        f'压力带: {band}',
        _bar_line(df, idx, '信号日最高价', float(high[idx])),
    ]


def _v_aggr_lines(df, idx, rule_name):
    from v4_aggressive import v4_v_rev_rules as v4r

    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    n = len(close)
    import talib
    ma5 = talib.MA(close, timeperiod=5)
    ma20 = talib.MA(close, timeperiod=20)
    for ev in v4r.detect_v4_aggressive_bottom_events(close, open_, high, low, n, ma5, ma20):
        if int(ev['entry']) != int(idx) or ev.get('_rule') != rule_name:
            continue
        lines = [_bar_line(df, idx, '激进底买入日')]
        if ev.get('neck') is not None:
            lines.append(f'V左区参考高价: {ev["neck"]:.2f}')
        if ev.get('v_bottom_low') is not None:
            lines.append(f'V底区间最低: {ev["v_bottom_low"]:.2f}')
        if ev.get('_signal_bar') is not None:
            lines.insert(0, _bar_line(df, ev['_signal_bar'], '形态信号棒'))
        return lines
    return [f'{rule_name}: 该日无匹配事件']


def _rule_context_lines(df, idx, rule_name):
    idx = int(idx)
    if rule_name in dwe.BREAK_PRIOR_HIGH_BUY_RULE_NAMES or rule_name == '放量突破高点':
        return _next_prior_high_lines(df, idx)
    if rule_name == '历史缺口阻力':
        return _gap_lines(df, idx)
    if rule_name == 'M顶跌破':
        return _m_top_lines(df, idx)
    if rule_name == '头肩顶跌破':
        return _hs_top_lines(df, idx)
    if rule_name == 'W底右侧':
        return ['V4 已禁用 W 结构，不产生 W底右侧 信号']
    if rule_name in ('V反右侧', 'V反底部'):
        return _v_right_lines(df, idx)
    if rule_name in ('接近前高巨量阴增', '高位放量阴线看跌', '高开低走长阴压区必卖'):
        lines = _pressure_lines(df, idx)
        if rule_name == '高开低走长阴压区必卖':
            close = df['close'].values.astype(float)
            open_ = df['open'].values.astype(float) if 'open' in df.columns else close
            high = df['high'].values.astype(float) if 'high' in df.columns else close
            low = df['low'].values.astype(float) if 'low' in df.columns else close
            o, c, h, l = float(open_[idx]), float(close[idx]), float(high[idx]), float(low[idx])
            pc = float(close[idx - 1]) if idx >= 1 else c
            body = o - c
            rng = max(h - l, 1e-9)
            lines.extend([
                f'前收: {pc:.2f} → 开 {o:.2f}（跳空高开 {(o/pc-1)*100:.2f}%）',
                f'长阴: 收 {c:.2f}，实体/振幅 {body/rng*100:.1f}%，开盘→收回落 {body/max(o-l,1e-9)*100:.1f}%',
            ])
        return lines
    if rule_name in ('V反激进底-金针', 'V反激进底-止跌', 'V反激进底-大低开阳'):
        return _v_aggr_lines(df, idx, rule_name)
    if rule_name == 'V4V反吞没':
        lines = _v_right_lines(df, idx)
        if len(lines) > 1:
            lines.insert(0, '吞没信号日通常为确认日前一根')
        return lines
    if rule_name == 'N字形突破':
        from v4_aggressive.strategy_vw import find_v_right_context_before
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float) if 'high' in df.columns else close
        low = df['low'].values.astype(float) if 'low' in df.columns else close
        open_ = df['open'].values.astype(float) if 'open' in df.columns else close
        n = len(close)
        ctx = find_v_right_context_before(close, high, low, n, idx, max_span=60, open_=open_)
        if ctx is None:
            return ['N字形: 前60根内无 V反右侧 背景（V4 不与 W 绑定）']
        entry = ctx.get('entry')
        neck = ctx.get('neck')
        return [
            _bar_line(df, entry, 'V反右侧日'),
            f'瓶口(V最高点): {neck}',
            f'突破要求: high > 瓶口×{1.001:.3f}',
            _bar_line(df, idx, 'N字突破日'),
        ]
    return None


def _v_pattern_snapshot(df, p, *, note=None):
    """V 形结构快照：供浮框固定展示「从哪天开始算」。"""
    snap = {
        'kind': 'V',
        'left_peak_date': _bar_date(df, p.left_peak_idx),
        'left_peak_price': round(float(p.neck_price), 2),
        'bottom_date': _bar_date(df, p.bottom_idx),
        'bottom_price': round(float(p.bottom_price), 2),
        'drop_pct_avg': round(float(p.drop_pct) * 100, 2),
        'neck_high': round(float(p.neck_price), 2),
    }
    if p.right_entry_idx is not None:
        snap['right_entry_date'] = _bar_date(df, p.right_entry_idx)
        snap['right_entry_price'] = round(float(df['close'].iloc[int(p.right_entry_idx)]), 2)
    if note:
        snap['note'] = note
    return snap


def build_v_structure_snapshot(df, confirm_bar_idx, rule_names=None):
    """
    在确认棒（实际 B 点）查找 V 形；跌深从 left_peak 至 bottom 的均价曲线计算。
    优先：right_entry 等于 confirm_bar_idx 的外层 V。
    """
    confirm_bar_idx = int(confirm_bar_idx)
    close = df['close'].values.astype(float)
    open_ = df['open'].values.astype(float) if 'open' in df.columns else close
    high = df['high'].values.astype(float) if 'high' in df.columns else close
    low = df['low'].values.astype(float) if 'low' in df.columns else close
    n = len(close)

    from v4_aggressive.v4_structure_curves import get_structure_registry

    reg = get_structure_registry(open_, high, low, close, n)
    for p in reg.v_patterns:
        if p.parent is not None:
            continue
        if p.right_entry_idx is not None and int(p.right_entry_idx) == confirm_bar_idx:
            return _v_pattern_snapshot(df, p)

    names = [str(r).split('·')[0] for r in (rule_names or [])]
    if any(x.startswith('V反') or x.startswith('V4V') for x in names):
        best = None
        best_dist = 10 ** 9
        for p in reg.v_patterns:
            if p.parent is not None:
                continue
            if p.bottom_idx <= confirm_bar_idx <= p.bottom_idx + 15:
                d = confirm_bar_idx - p.bottom_idx
                if d < best_dist:
                    best_dist = d
                    best = p
        if best is not None:
            return _v_pattern_snapshot(df, best, note='买入日在V底后15根内，关联最近V左区')
    return None


def build_w_structure_snapshot(df, confirm_bar_idx):
    """V4 已禁用 W 结构。"""
    del df, confirm_bar_idx
    return None


def build_signal_rule_details(df, confirm_bar_idx, rule_names, *, buy_side=True):
    """
    返回 [{rule, lines: [str, ...]}, ...]，供前端 K 线 tooltip 展示。
    confirm_bar_idx 必须为实际 B/S 确认棒（非 pending 信号日）。
    """
    del buy_side
    if confirm_bar_idx is None or not rule_names:
        return []
    out = []
    seen = set()
    for raw in rule_names:
        name = str(raw).split('·')[0].strip()
        if name in seen:
            continue
        seen.add(name)
        lines = _rule_context_lines(df, confirm_bar_idx, name)
        if lines:
            out.append({'rule': name, 'lines': lines})
    return out


def append_engine_sell_details(row, reasons):
    """引擎层卖出（非规则表）补充到 rule_details。"""
    rt = reasons.get('sell_reason_type') or ''
    if not rt:
        return row
    lines = []
    if reasons.get('v_bottom_stop') is not None:
        lines.append(f'V底止损线: {reasons["v_bottom_stop"]}')
    if rt == dwe.V4_BEARISH_MA5_BREAK_SELL_REASON and reasons.get('bearish_pattern_kind'):
        lines.append(f'待破MA5的形态: {reasons["bearish_pattern_kind"]}')
    if not lines:
        return row
    details = list(row.get('rule_details') or [])
    details.append({'rule': rt, 'lines': lines})
    row['rule_details'] = details
    return row


def enrich_extreme_with_next_prior_high(
    df, signal_bar_idx, annotation: dict, anchor_bar_idx=None,
):
    """兼容旧名；V4 不再往后扫描前高。"""
    return annotation or {}
