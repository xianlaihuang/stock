import requests
import json

BASE = 'http://localhost:5001/api'
results = []

def test(name, func):
    try:
        result = func()
        results.append({'name': name, 'result': result, 'status': 'PASS' if result else 'FAIL'})
        status_icon = 'PASS' if result else 'FAIL'
        print(f'[{status_icon}] {name}')
    except Exception as e:
        results.append({'name': name, 'result': str(e), 'status': 'ERROR'})
        print(f'[ERROR] {name}: {e}')

def test_scrape():
    r = requests.post(f'{BASE}/stocks/002347/scrape')
    d = r.json()
    return d.get('success', False)

def test_minute():
    r = requests.get(f'{BASE}/stocks/002347/kline?period=minute')
    d = r.json()
    data = d.get('data', [])
    return len(data) > 0

def test_minute_date():
    r = requests.get(f'{BASE}/stocks/002347/kline?period=minute&date=2026-05-08')
    d = r.json()
    data = d.get('data', [])
    if len(data) > 0:
        return all(k['date'].startswith('2026-05-08') for k in data)
    return False

def test_date_format():
    r = requests.get(f'{BASE}/stocks/002347/kline?period=minute')
    d = r.json()
    data = d.get('data', [])
    if data:
        dates = set(k['date'].split(' ')[0] for k in data)
        return len(dates) >= 1
    return False

def test_minute_by_date():
    r = requests.get(f'{BASE}/stocks/002347/kline?period=day')
    d = r.json()
    days = d.get('data', [])
    if days:
        last_day = days[-1]['date'].split(' ')[0]
        r2 = requests.get(f'{BASE}/stocks/002347/kline?period=minute&date={last_day}')
        d2 = r2.json()
        return len(d2.get('data', [])) > 0
    return False

def test_financial():
    r = requests.get(f'{BASE}/stocks/002347/financial')
    d = r.json()
    data = d.get('data', [])
    return len(data) > 0

def test_financial_indicators():
    r = requests.get(f'{BASE}/stocks/002347/financial')
    d = r.json()
    data = d.get('data', [])
    if not data:
        return False
    item = data[0]
    required = ['eps', 'bps', 'revenue', 'net_profit', 'roe', 'gross_margin',
                'revenue_yoy', 'net_profit_yoy', 'deducted_net_profit_yoy',
                'asset_liability_ratio', 'current_ratio', 'quick_ratio',
                'cash_per_share', 'capital_reserve_per_share', 'undistributed_per_share']
    missing = [k for k in required if k not in item or item[k] is None]
    if missing:
        print(f'  Missing indicators: {missing}')
        return False
    return True

def test_financial_types():
    r = requests.get(f'{BASE}/stocks/002347/financial')
    d = r.json()
    data = d.get('data', [])
    types = set(item.get('report_type', '') for item in data)
    has_annual = any('年报' in item.get('report_date_name', '') for item in data)
    has_quarterly = any(t in types for t in ['一季报', '中报', '三季报'])
    return has_annual and has_quarterly

def test_kline_data():
    r = requests.get(f'{BASE}/stocks/002347/kline?period=day')
    d = r.json()
    return d.get('success', False) and len(d.get('data', [])) > 0

def test_duplicate_add():
    r = requests.post(f'{BASE}/stocks', json={'code': '002347'})
    return r.status_code == 409

def test_delete_api():
    r = requests.delete(f'{BASE}/stocks/999999')
    return r.status_code in [200, 404]

test('TC-SCRAP-001 Scrape data success', test_scrape)
test('TC-MIN-001 Minute line data not empty', test_minute)
test('TC-MIN-002 Filter minute by date 2026-05-08', test_minute_date)
test('TC-MIN-003 Minute date format correct', test_date_format)
test('TC-DBL-001 Get minute by date API (double-click)', test_minute_by_date)
test('TC-FIN-001 Financial data not empty', test_financial)
test('TC-FIN-002 Financial indicators complete', test_financial_indicators)
test('TC-FIN-003 Financial data has annual and quarterly', test_financial_types)
test('TC-BAK-001 K-line data fetch (fallback)', test_kline_data)
test('TC-ADD-002 Duplicate add returns 409', test_duplicate_add)
test('TC-DEL-001 Delete stock API available', test_delete_api)

print()
passed = sum(1 for r in results if r['status'] == 'PASS')
failed = sum(1 for r in results if r['status'] == 'FAIL')
errored = sum(1 for r in results if r['status'] == 'ERROR')
total = len(results)
print(f'Results: {passed}/{total} passed, {failed} failed, {errored} errors')
