#!/usr/bin/env python3
"""
002979 V 左肩因果验证图：bar t 仅使用 t 及以前数据，不偷看右侧/未来。

生成独立 HTML（ECharts），可在浏览器打开验证。
用法:
  python scripts/v_geometry_demo_002979.py
  python scripts/v_geometry_demo_002979.py --code 002347 --open
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from routes_v2 import _klines_to_df
from scraper import StockScraper
from v4_aggressive.v4_structure_curves import V_DROP_MIN, _build_avg
from v4_aggressive.v_left_geometry import scan_v_left_causal


def _dates(df: pd.DataFrame) -> list:
    if 'date' in df.columns:
        return [str(d)[:10] for d in df['date'].tolist()]
    return [str(i) for i in range(len(df))]


def _ohlc_rows(df: pd.DataFrame) -> list:
    rows = []
    for _, r in df.iterrows():
        rows.append([
            float(r['open']),
            float(r['close']),
            float(r['low']),
            float(r['high']),
        ])
    return rows


def _collect_patterns(df: pd.DataFrame):
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    avg = _build_avg(o, h, l, c)
    import talib
    atr = talib.ATR(h, l, c, timeperiod=14)

    dates = _dates(df)
    geos = scan_v_left_causal(o, h, l, c, avg, atr, drop_min=V_DROP_MIN)
    items = []

    for geo in geos:
        bi = geo.bottom_idx
        eff_lp = geo.effective_peak_idx
        if geo.kind == 'v_left_event_fake':
            kind = 'event_fake'
        elif geo.kind == 'v_left':
            kind = 'v_left_only'
        else:
            kind = 'v_left_weak'

        items.append({
            'kind': kind,
            'effective_peak_idx': eff_lp,
            'bottom_idx': bi,
            'confirm_idx': geo.confirm_idx,
            'left_peak_date': dates[eff_lp],
            'bottom_date': dates[bi],
            'confirm_date': dates[geo.confirm_idx] if geo.confirm_idx >= 0 else '—',
            'drop_pct': round(geo.drop_pct * 100, 2),
            'geometry': {
                'bar_span': geo.bar_span,
                'path_efficiency': round(geo.path_efficiency, 3),
                'bottom_rejection': round(geo.bottom_rejection, 3),
                'atr_path': round(geo.atr_path, 2),
                'geo_kind': geo.kind,
                'tags': geo.tags + ['因果扫描'],
                'reasons': geo.reject_reasons,
                'summary': geo.summary_cn(),
            },
        })
    return items, dates, _ohlc_rows(df)


def _build_html(code: str, name: str, dates: list, ohlc: list, patterns: list) -> str:
    kind_meta = {
        'v_left_only': {'color': '#f39c12', 'label': 'V左(因果)', 'symbol': 'triangle', 'size': 36, 'at': 'bottom'},
        'event_fake': {'color': '#e74c3c', 'label': 'event假V', 'symbol': 'diamond', 'size': 40, 'at': 'bottom'},
        'v_left_weak': {'color': '#666', 'label': '左肩弱', 'symbol': 'emptyCircle', 'size': 22, 'at': 'bottom'},
    }

    mark_points = []
    mark_lines = []
    table_rows = []

    for i, p in enumerate(patterns):
        km = kind_meta.get(p['kind'], kind_meta['v_left_weak'])
        g = p['geometry']
        bi_d = p['bottom_date']
        lp_d = p['left_peak_date']
        eff_i = p.get('effective_peak_idx', p['bottom_idx'])

        tip = (
            f"<b>{km['label']}</b><br/>"
            f"左肩: {lp_d} → V底: {bi_d}<br/>"
            f"确认锁定: {p.get('confirm_date', '—')}<br/>"
            f"跌深(high): {p['drop_pct']}% · ATR路{g['atr_path']}<br/>"
            f"η={g['path_efficiency']} 探底={g['bottom_rejection']*100:.0f}%<br/>"
            f"{g['summary']}"
        )

        lbl = {'v_left_only': 'V左', 'event_fake': '假', 'v_left_weak': '?'}.get(p['kind'], '?')
        coord_idx = p['bottom_idx']
        coord_price = ohlc[coord_idx][2]  # low at bottom
        mark_points.append({
            'name': km['label'],
            'coord': [bi_d, coord_price],
            'value': bi_d,
            'symbol': km['symbol'],
            'symbolSize': km['size'],
            'symbolRotate': 180 if km['symbol'] == 'triangle' else 0,
            'itemStyle': {'color': km['color']},
            'label': {'show': True, 'formatter': lbl, 'color': '#fff', 'fontSize': 10, 'fontWeight': 'bold', 'position': 'bottom'},
            'tooltip': {'formatter': tip},
        })

        if p['kind'] in ('v_left_only', 'event_fake'):
            mark_lines.append([
                {'coord': [lp_d, ohlc[eff_i][3]], 'lineStyle': {'color': km['color'], 'type': 'dashed', 'width': 1.5}},
                {'coord': [bi_d, ohlc[p['bottom_idx']][2]]},
            ])

        table_rows.append(
            f"<tr><td>{i+1}</td><td>{lp_d}</td><td>{bi_d}</td><td>{p.get('confirm_date','—')}</td>"
            f"<td style='color:{km['color']};font-weight:600'>{km['label']}</td>"
            f"<td>{p['drop_pct']}</td><td>{g['atr_path']}</td><td>{g['path_efficiency']}</td>"
            f"<td>{g['bottom_rejection']*100:.0f}%</td>"
            f"<td>{', '.join(g['tags']) or '—'}</td></tr>"
        )

    n_left = sum(1 for p in patterns if p['kind'] == 'v_left_only')
    n_fake = sum(1 for p in patterns if p['kind'] == 'event_fake')

    # 默认 zoom 到最近约 18 个月
    n_bars = len(dates)
    zoom_start = max(0, int(100 - 280 / max(n_bars, 1) * 100))

    payload = json.dumps({
        'dates': dates,
        'ohlc': ohlc,
        'markPoints': mark_points,
        'markLines': mark_lines,
        'zoomStart': zoom_start,
    }, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{code} {name} · V左肩几何验证</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif; background: #1a1a2e; color: #e0e0e0; }}
  header {{ padding: 16px 20px; background: #16213e; border-bottom: 1px solid #333; }}
  header h1 {{ margin: 0 0 8px; font-size: 18px; color: #fff; }}
  header p {{ margin: 4px 0; font-size: 13px; color: #aaa; line-height: 1.5; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  #chart {{ width: 100%; height: 62vh; min-height: 420px; }}
  .panel {{ padding: 12px 20px 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ border: 1px solid #333; padding: 6px 8px; text-align: left; }}
  th {{ background: #16213e; color: #8ab4f8; }}
  tr:nth-child(even) {{ background: #1f1f35; }}
  .hint {{ color: #f39c12; font-size: 12px; margin-bottom: 10px; }}
</style>
</head>
<body>
<header>
  <h1>{code} {name} · V 左肩因果验证</h1>
  <p>
    <strong>大原则</strong>：V 左 = 左肩 bar → V 底 bar 的左段（几何只用 V 底当日及以前数据）。
    在 V 底后首次出现 <em>ATR 级反弹</em> 的 bar 上 <strong>确认锁定</strong>；锁定后标注不随后续更深底而删除或改写。
    不要求 V 右走完。
  </p>
  <div class="legend">
    <span><i class="dot" style="background:#f39c12"></i> V左 已锁定 ({n_left})</span>
    <span><i class="dot" style="background:#e74c3c"></i> event假V ({n_fake})</span>
    <span>虚线：左肩 high → V 底 low</span>
  </div>
</header>
<div id="chart"></div>
<div class="panel">
  <h3 style="font-size:14px;margin:0 0 8px;">已锁定 V 左（左肩 → V底 → 确认日）</h3>
  <table>
    <thead><tr>
      <th>#</th><th>左肩</th><th>V底</th><th>确认锁定</th><th>判定</th>
      <th>跌深%</th><th>ATR路</th><th>η</th><th>探底%</th><th>标签</th>
    </tr></thead>
    <tbody>{''.join(table_rows) if table_rows else '<tr><td colspan="9" style="text-align:center;color:#666">无 V 候选</td></tr>'}</tbody>
  </table>
</div>
<script>
const DATA = {payload};
const chart = echarts.init(document.getElementById('chart'), 'dark');
chart.setOption({{
  backgroundColor: '#1a1a2e',
  animation: false,
  tooltip: {{
    trigger: 'axis',
    axisPointer: {{ type: 'cross' }},
    backgroundColor: 'rgba(22,33,62,0.95)',
    borderColor: '#444',
    textStyle: {{ color: '#eee', fontSize: 12 }}
  }},
  grid: {{ left: 56, right: 24, top: 48, bottom: 72 }},
  dataZoom: [
    {{ type: 'inside', start: DATA.zoomStart, end: 100 }},
    {{ type: 'slider', start: DATA.zoomStart, end: 100, height: 22, bottom: 8, borderColor: '#444', fillerColor: 'rgba(74,144,217,0.15)' }}
  ],
  xAxis: {{
    type: 'category', data: DATA.dates, boundaryGap: true,
    axisLine: {{ lineStyle: {{ color: '#555' }} }},
    axisLabel: {{ color: '#999', fontSize: 10, rotate: 45 }}
  }},
  yAxis: {{
    scale: true, splitLine: {{ lineStyle: {{ color: '#2a2a40' }} }},
    axisLabel: {{ color: '#999' }}
  }},
  series: [{{
    type: 'candlestick',
    data: DATA.ohlc,
    itemStyle: {{
      color: '#ef5350', color0: '#26a69a',
      borderColor: '#ef5350', borderColor0: '#26a69a'
    }},
    markPoint: {{
      data: DATA.markPoints,
      tooltip: {{ trigger: 'item' }}
    }},
    markLine: {{
      symbol: ['none', 'none'],
      data: DATA.markLines,
      silent: true
    }}
  }}]
}});
window.addEventListener('resize', () => chart.resize());
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', default='002979')
    ap.add_argument('--count', type=int, default=800)
    ap.add_argument('--open', action='store_true', help='生成后在浏览器打开')
    args = ap.parse_args()

    code = args.code.strip()
    kl = StockScraper.get_kline_data(code, period='day', count=args.count)
    if not kl or len(kl) < 60:
        print(f'{code} K线不足')
        return 1
    df = _klines_to_df(kl)
    patterns, dates, ohlc = _collect_patterns(df)

    name = code
    try:
        from models import Stock
        doc = Stock.get_by_code(code)
        if doc and doc.get('name'):
            name = doc['name']
    except Exception:
        pass

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'v_geometry_{code}.html')
    html = _build_html(code, name, dates, ohlc, patterns)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'已生成: {out_path}')
    n_left = sum(1 for p in patterns if p['kind'] == 'v_left_only')
    n_fake = sum(1 for p in patterns if p['kind'] == 'event_fake')
    print(f'K线 {len(df)} 根 | V左(因果) {n_left} | event假V {n_fake}')
    for p in patterns:
        if p['bottom_date'] >= '2024-01-15' and p['bottom_date'] <= '2024-03-05':
            print(f"  {p['bottom_date']}: {p['kind']} peak={p['left_peak_date']} drop={p['drop_pct']}%")

    if args.open:
        webbrowser.open('file://' + os.path.abspath(out_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
