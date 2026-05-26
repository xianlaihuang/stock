#!/usr/bin/env python3
"""
V5 几何验证图：V 左 + 压力位（锁定 + 持仓期 MA20）。

用法:
  python scripts/v5_geometry_demo.py --code 002347
  python scripts/v5_geometry_demo.py --code 002347 --open
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
import http.server
import socket
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_v2 as ev2
from routes_v2 import _klines_to_df
from scraper import StockScraper
from v5.annotations import scan_v5_annotations


def _build_html(code: str, name: str, ann: dict) -> str:
    dates = ann['dates']
    ohlc = ann['ohlc']
    mark_points = ann['mark_points']
    mark_lines = ann['mark_lines']
    ma_line = ann['ma_line']
    n_left = ann['n_v_left']
    n_pl = ann['n_pressure_locked']
    n_pd = ann['n_pressure_dynamic']
    n_hold = ann['n_holding']
    table_v = ann['table_v']
    table_p = ann['table_p']
    n_bars = len(dates)
    zoom_start = max(0, int(100 - 280 / max(n_bars, 1) * 100))

    payload = json.dumps({
        'dates': dates,
        'ohlc': ohlc,
        'markPoints': mark_points,
        'markLines': mark_lines,
        'maLine': ma_line,
        'zoomStart': zoom_start,
    }, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>{code} {name} · V5 验证</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script>
if (typeof echarts === 'undefined') {{
  document.write('<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js"><\\/script>');
}}
</script>
<style>
  body {{ margin:0; font-family: -apple-system, "PingFang SC", sans-serif; background:#1a1a2e; color:#e0e0e0; }}
  header {{ padding:16px 20px; background:#16213e; border-bottom:1px solid #333; }}
  header h1 {{ margin:0 0 8px; font-size:18px; color:#fff; }}
  header p {{ margin:4px 0; font-size:13px; color:#aaa; line-height:1.5; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:10px; font-size:12px; }}
  .dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; }}
  #chart {{ width:100%; height:58vh; min-height:400px; }}
  .panel {{ padding:12px 20px 24px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin-bottom:16px; }}
  th, td {{ border:1px solid #333; padding:6px 8px; text-align:left; }}
  th {{ background:#16213e; color:#8ab4f8; }}
  tr:nth-child(even) {{ background:#1f1f35; }}
</style>
</head>
<body>
<header>
  <h1>{code} {name} · V5 验证（V左 + 压力位）</h1>
  <p>因果 + 确认锁定；压力位含 <b>锁定</b>（摆动/V左肩/缺口）与 <b>持仓期 MA20</b>（有仓才算）。已锁定标注不溯及修改。</p>
  <div class="legend">
    <span><i class="dot" style="background:#f39c12"></i> V左 ({n_left})</span>
    <span><i class="dot" style="background:#9b59b6"></i> 压力位·锁定 ({n_pl})</span>
    <span style="color:#3498db">— MA20压力轨 ({n_pd} 点 / {n_hold} 段持仓)</span>
  </div>
</header>
<div id="chart"></div>
<div class="panel">
  <h3 style="font-size:14px">V 左</h3>
  <table><thead><tr><th>#</th><th>左肩</th><th>V底</th><th>确认</th><th>跌深%</th></tr></thead>
  <tbody>{table_v}</tbody></table>
  <h3 style="font-size:14px">压力位（锁定）</h3>
  <table><thead><tr><th>#</th><th>锚点</th><th>确认</th><th>P</th><th>压力带</th><th>来源</th></tr></thead>
  <tbody>{table_p}</tbody></table>
</div>
<script>
const DATA = {payload};
const errEl = document.createElement('div');
errEl.style.cssText = 'color:#f66;padding:12px 20px;display:none';
document.body.insertBefore(errEl, document.body.firstChild);
try {{
  if (typeof echarts === 'undefined') throw new Error('ECharts 未加载，请检查网络或使用 --serve 本地打开');
  const chart = echarts.init(document.getElementById('chart'), 'dark');
  chart.setOption({{
  backgroundColor:'#1a1a2e', animation:false,
  tooltip: {{ trigger:'axis', axisPointer:{{type:'cross'}} }},
  legend: {{ data:['K线','MA20压力'], textStyle:{{color:'#aaa'}}, top:8 }},
  grid: {{ left:56, right:24, top:48, bottom:72 }},
  dataZoom: [
    {{ type:'inside', start:DATA.zoomStart, end:100 }},
    {{ type:'slider', start:DATA.zoomStart, end:100, height:22, bottom:8 }}
  ],
  xAxis: {{ type:'category', data:DATA.dates, axisLabel:{{rotate:45, fontSize:10, color:'#999'}} }},
  yAxis: {{ scale:true, splitLine:{{lineStyle:{{color:'#2a2a40'}}}}, axisLabel:{{color:'#999'}} }},
  series: [
    {{
      name:'K线', type:'candlestick', data:DATA.ohlc,
      itemStyle: {{ color:'#ef5350', color0:'#26a69a', borderColor:'#ef5350', borderColor0:'#26a69a' }},
      markPoint: {{ data: DATA.markPoints, tooltip:{{trigger:'item'}} }},
      markLine: {{ symbol:['none','none'], data: DATA.markLines, silent:true }}
    }},
    {{
      name:'MA20压力', type:'line', data: DATA.maLine, connectNulls:false,
      lineStyle: {{ color:'#3498db', width:1.5 }}, symbol:'none', z:5
    }}
  ]
  }});
  window.addEventListener('resize', () => chart.resize());
}} catch (e) {{
  errEl.style.display = 'block';
  errEl.textContent = '图表加载失败: ' + e.message;
  console.error(e);
}}
</script>
</body>
</html>"""


def _collect(code: str, count: int):
    kl = StockScraper.get_kline_data(code, period='day', count=count)
    if not kl or len(kl) < 60:
        return None
    df = _klines_to_df(kl)

    cw = ev2.load_weights(code)
    paired = []
    if cw:
        try:
            dual = ev2.analyze_signals_dual(df, cw)
            paired = dual.get('v4', {}).get('paired_signals') or dual.get('v2', {}).get('paired_signals') or []
        except Exception:
            paired = []

    ann = scan_v5_annotations(df, paired)
    dates = ann['dates']
    ohlc = []
    for _, r in df.iterrows():
        ohlc.append([float(r['open']), float(r['close']), float(r['low']), float(r['high'])])

    mark_points = []
    mark_lines = []

    for g in ann['v_lefts']:
        if g.kind != 'v_left':
            continue
        bi, lp = g.bottom_idx, g.effective_peak_idx
        mark_points.append({
            'coord': [dates[bi], ohlc[bi][2]],
            'symbol': 'triangle', 'symbolSize': 34, 'symbolRotate': 180,
            'itemStyle': {'color': '#f39c12'},
            'label': {'show': True, 'formatter': 'V左', 'color': '#fff', 'fontSize': 10, 'position': 'bottom'},
            'tooltip': {'formatter': f"V左<br/>{dates[lp]}→{dates[bi]}<br/>确认{dates[g.confirm_idx]}"},
        })
        mark_lines.append([
            {'coord': [dates[lp], ohlc[lp][3]], 'lineStyle': {'color': '#f39c12', 'type': 'dashed'}},
            {'coord': [dates[bi], ohlc[bi][2]]},
        ])

    for i, p in enumerate(ann['pressure_locked'][-40:]):
        d0 = dates[p.anchor_idx]
        mark_points.append({
            'coord': [d0, p.P],
            'symbol': 'pin', 'symbolSize': 28,
            'itemStyle': {'color': '#9b59b6'},
            'label': {'show': True, 'formatter': '压', 'color': '#fff', 'fontSize': 9},
            'tooltip': {'formatter': f"压力位(锁定)<br/>P={p.P:.2f}<br/>[{p.zone_lo:.2f},{p.zone_hi:.2f}]<br/>{p.source}"},
        })
        mark_lines.append({
            'yAxis': round(p.P, 4),
            'lineStyle': {'color': 'rgba(155,89,182,0.45)', 'type': 'dashed', 'width': 1},
            'label': {'formatter': f'{p.P:.2f}', 'color': '#9b59b6', 'fontSize': 8},
        })

    ma_line = [None] * len(dates)
    for p in ann['pressure_dynamic']:
        ma_line[p.bar_idx] = round(p.P, 3)

    table_v = ''
    vi = 0
    for g in ann['v_lefts']:
        if g.kind != 'v_left':
            continue
        vi += 1
        table_v += (
            f"<tr><td>{vi}</td><td>{dates[g.effective_peak_idx]}</td>"
            f"<td>{dates[g.bottom_idx]}</td><td>{dates[g.confirm_idx]}</td>"
            f"<td>{g.drop_pct*100:.1f}</td></tr>"
        )
    if not table_v:
        table_v = '<tr><td colspan="5" style="text-align:center;color:#666">无</td></tr>'

    table_p = ''
    for i, p in enumerate(ann['pressure_locked'][:80], 1):
        table_p += (
            f"<tr><td>{i}</td><td>{dates[p.anchor_idx]}</td><td>{dates[p.confirm_idx]}</td>"
            f"<td>{p.P:.2f}</td><td>[{p.zone_lo:.2f},{p.zone_hi:.2f}]</td><td>{p.source}</td></tr>"
        )
    if not table_p:
        table_p = '<tr><td colspan="6" style="text-align:center;color:#666">无</td></tr>'

    return {
        'dates': dates,
        'ohlc': ohlc,
        'mark_points': mark_points,
        'mark_lines': mark_lines,
        'ma_line': ma_line,
        'n_v_left': ann['counts']['v_left'],
        'n_pressure_locked': ann['counts']['pressure_locked'],
        'n_pressure_dynamic': ann['counts']['pressure_dynamic'],
        'n_holding': ann['counts']['holding_segments'],
        'table_v': table_v,
        'table_p': table_p,
        'ann': ann,
    }


def _serve_and_open(html_path: str):
    out_dir = os.path.dirname(os.path.abspath(html_path))
    fname = os.path.basename(html_path)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=out_dir, **kw)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    httpd = http.server.HTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{port}/{fname}'
    print(f'本地服务: {url}')
    webbrowser.open(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', default='002347')
    ap.add_argument('--count', type=int, default=800)
    ap.add_argument('--open', action='store_true', help='生成后浏览器打开（走本地 http）')
    args = ap.parse_args()

    data = _collect(args.code.strip(), args.count)
    if data is None:
        print('K线不足')
        return 1

    name = args.code
    try:
        from models import Stock
        doc = Stock.get_by_code(args.code.strip())
        if doc and doc.get('name'):
            name = doc['name']
    except Exception:
        pass

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'v5_geometry_{args.code.strip()}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(_build_html(args.code.strip(), name, data))

    c = data['ann']['counts']
    print(f'已生成: {out_path}')
    print(f"K线 {len(data['dates'])} | V左 {c['v_left']} | 压力位锁定 {c['pressure_locked']} | 动态MA {c['pressure_dynamic']} | 持仓段 {c['holding_segments']}")

    if args.open:
        _serve_and_open(out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
