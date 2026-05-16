from flask import Flask, request, jsonify
from app import app
from models import Stock, KlineData, FinancialData
from scraper import StockScraper

@app.route('/api/stocks', methods=['GET'])
def get_all_stocks():
    stocks = Stock.get_all()
    return jsonify({'success': True, 'data': stocks})

@app.route('/api/stocks/<code>', methods=['GET'])
def get_stock(code):
    stock = Stock.get(code)
    if stock:
        stock.pop('_id', None)
        return jsonify({'success': True, 'data': stock})
    return jsonify({'success': False, 'message': 'Stock not found'}), 404

scraping_status = {}

def _aggregate_period_from_daily(daily_klines, period):
    from collections import defaultdict
    import datetime as dt
    if not daily_klines:
        return []
    buckets = defaultdict(list)
    for k in daily_klines:
        d = k['date'].split(' ')[0]
        try:
            date_obj = dt.datetime.strptime(d, '%Y-%m-%d')
            if period == 'month':
                key = date_obj.strftime('%Y-%m')
            elif period == 'year':
                key = str(date_obj.year)
            else:
                continue
            buckets[key].append(k)
        except:
            continue
    result = []
    for key in sorted(buckets.keys()):
        group = buckets[key]
        first = group[0]
        last = group[-1]
        result.append({
            'date': last['date'],
            'open': first['open'],
            'close': last['close'],
            'high': max(k['high'] for k in group),
            'low': min(k['low'] for k in group),
            'volume': sum(k['volume'] for k in group),
            'amount': sum(k.get('amount', 0) for k in group),
        })
    return result

def _scrape_stock_data_internal(code):
    periods = ['day', 'minute', 'week', 'month', 'year']
    new_kline_count = 0
    day_klines = None
    for period in periods:
        if period == 'minute':
            KlineData.delete(code, period='minute')
            kline_data = StockScraper.get_kline_data(code, period='minute')
            if kline_data:
                new_kline_count += len(kline_data)
                KlineData.add_many(code, kline_data, period='minute')
            continue

        existing_klines = KlineData.get(code, period=period)
        last_kline_date = None
        if existing_klines:
            last_kline_date = existing_klines[-1]['date']
        
        day_count = 800 if period == 'day' and not last_kline_date else None
        kline_data = StockScraper.get_kline_data(code, period=period, start_date=last_kline_date, count=day_count or 120)

        if not kline_data and period in ('month', 'year'):
            all_day_klines = KlineData.get(code, period='day')
            if all_day_klines:
                print(f"[聚合] {period}线API无数据，从{len(all_day_klines)}条日线聚合生成...")
                kline_data = _aggregate_period_from_daily(all_day_klines, period)
                print(f"[聚合] {period}线聚合完成: {len(kline_data)}条")

        if kline_data:
            new_kline_count += len(kline_data)
            KlineData.add_many(code, kline_data, period=period)
            if period == 'day':
                day_klines = kline_data
    
    financial_data = StockScraper.get_financial_data(code)
    new_financial_count = 0
    if financial_data:
        existing_financial_dates = set(f['report_date'] for f in FinancialData.get(code))
        for item in financial_data:
            if item['report_date'] not in existing_financial_dates:
                FinancialData.add(code, item)
                new_financial_count += 1
    
    Stock.update(code, {})
    
    return new_kline_count, new_financial_count


def _sync_one_period_kline(code, period):
    """仅增量抓取单周期 K 线并写入库（用于盘中 sync=1，避免整表重抓）。"""
    if period == 'minute':
        try:
            kline_data = StockScraper.get_kline_data(code, period='minute')
            if not kline_data:
                return 0
            KlineData.delete(code, period='minute')
            KlineData.add_many(code, kline_data, period='minute')
            return len(kline_data)
        except Exception as e:
            print(f"[sync minute] {code}: {e}")
            return 0
    existing = KlineData.get(code, period=period)
    last_kline_date = None
    if existing:
        last_kline_date = existing[-1]['date']
    day_count = 800 if period == 'day' and not last_kline_date else None
    try:
        kline_data = StockScraper.get_kline_data(
            code, period=period, start_date=last_kline_date, count=day_count or 120
        )
    except Exception as e:
        print(f"[sync {period}] {code}: {e}")
        return 0
    if not kline_data and period in ('month', 'year'):
        all_day_klines = KlineData.get(code, period='day')
        if all_day_klines:
            kline_data = _aggregate_period_from_daily(all_day_klines, period)
    if kline_data:
        KlineData.add_many(code, kline_data, period=period)
        return len(kline_data)
    return 0


@app.route('/api/stocks', methods=['POST'])
def add_stock():
    data = request.json
    code = data.get('code')
    if not code:
        return jsonify({'success': False, 'message': 'Stock code is required'}), 400
    
    if Stock.get(code):
        return jsonify({'success': False, 'message': 'Stock already exists'}), 409
    
    name = StockScraper.get_stock_name(code)
    if Stock.add(code, name):
        return jsonify({'success': True, 'message': '股票添加成功，请点击"抓取数据"按钮获取行情数据', 'data': {'code': code, 'name': name}})
    return jsonify({'success': False, 'message': 'Failed to add stock'}), 500

@app.route('/api/stocks/<code>', methods=['DELETE'])
def delete_stock(code):
    if Stock.delete(code):
        return jsonify({'success': True, 'message': 'Stock deleted successfully'})
    return jsonify({'success': False, 'message': 'Stock not found'}), 404

@app.route('/api/stocks/<code>/scrape', methods=['POST'])
def scrape_stock_data(code):
    global scraping_status
    stock = Stock.get(code)
    if not stock:
        return jsonify({'success': False, 'message': 'Stock not found'}), 404
    
    if scraping_status.get(code):
        return jsonify({'success': False, 'message': '正在抓取中，请稍候'})
    
    scraping_status[code] = True
    
    try:
        new_kline_count, new_financial_count = _scrape_stock_data_internal(code)
        
        message = f'Data scraped successfully'
        if new_kline_count > 0 or new_financial_count > 0:
            message += f' (新增K线: {new_kline_count}, 新增财务: {new_financial_count})'
        else:
            message += ' (暂无新数据)'
        
        return jsonify({'success': True, 'message': message})
    finally:
        scraping_status[code] = False

@app.route('/api/stocks/<code>/scraping-status', methods=['GET'])
def get_scraping_status(code):
    return jsonify({'success': True, 'scraping': scraping_status.get(code, False)})

@app.route('/api/stocks/<code>/kline', methods=['GET'])
def get_kline_data(code):
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    period = request.args.get('period', 'day')
    date = request.args.get('date')
    after_q = request.args.get('after')
    after_exclusive = str(after_q).strip() if after_q else None
    do_sync = str(request.args.get('sync', '')).lower() in ('1', 'true', 'yes')
    if do_sync:
        try:
            _sync_one_period_kline(code, period)
        except Exception as e:
            print(f"[kline sync] {code} {period}: {e}")
    if after_exclusive:
        klines = KlineData.get(code, None, None, period, after_exclusive=after_exclusive)
    else:
        klines = KlineData.get(code, start_date, end_date, period)
    if date and period == 'minute':
        klines = [k for k in klines if str(k.get('date', '')).startswith(date)]

    # 指定交易日时：必须先走历史源。若先拉「当日分时」再按日期过滤，会得到空（接口只含最近交易日）。
    if period == 'minute' and date and (not klines or len(klines) == 0):
        try:
            from tdx_source import fetch_historical_minute_for_date
            htdx = fetch_historical_minute_for_date(code, date)
            if htdx:
                klines = htdx
        except Exception as e:
            print(f"Error fetching historical minute tdx: {e}")
    if period == 'minute' and date and (not klines or len(klines) == 0):
        try:
            from scraper import StockScraper
            hist_klines = StockScraper.get_historical_minute_from_sina(code, date)
            if hist_klines:
                klines = hist_klines
        except Exception as e:
            print(f"Error fetching historical minute data: {e}")

    if period == 'minute' and (not klines or len(klines) == 0):
        try:
            from scraper import StockScraper
            from datetime import datetime, timedelta, timezone
            cn_today = datetime.now(timezone(timedelta(hours=8))).date()
            d0 = None
            if date:
                try:
                    d0 = datetime.strptime(str(date).strip()[:10], '%Y-%m-%d').date()
                except ValueError:
                    d0 = None
            use_live = (not date) or (d0 is not None and d0 == cn_today)
            if use_live:
                fresh_klines = StockScraper.get_kline_data(code, period='minute')
                if fresh_klines:
                    if date:
                        pref = str(date).strip()[:10]
                        fresh_klines = [k for k in fresh_klines if str(k.get('date', '')).startswith(pref)]
                    if fresh_klines:
                        klines = fresh_klines
        except Exception as e:
            print(f"Error fetching minute data on-the-fly: {e}")
    if period in ('month', 'year') and (not klines or len(klines) == 0):
        day_klines = KlineData.get(code, period='day')
        if day_klines:
            print(f"[聚合] {period}线无数据，从{len(day_klines)}条日线聚合...")
            klines = _aggregate_period_from_daily(day_klines, period)
            print(f"[聚合] {period}线聚合完成: {len(klines)}条")
            if klines:
                KlineData.add_many(code, klines, period=period)
    if after_exclusive:
        ae = after_exclusive
        klines = [k for k in (klines or []) if str(k.get('date', '')).strip() > ae]
    return jsonify({'success': True, 'data': klines})

@app.route('/api/stocks/<code>/financial', methods=['GET'])
def get_financial_data(code):
    data = FinancialData.get(code)
    return jsonify({'success': True, 'data': data})

@app.route('/api/analysis/<code>', methods=['POST'])
def analyze_stock(code):
    data = request.json
    parameters = data.get('parameters', {})
    
    klines = KlineData.get(code)
    if not klines:
        return jsonify({'success': False, 'message': 'No kline data available'}), 404
    
    analysis_result = core_analysis(klines, parameters)
    return jsonify({'success': True, 'data': analysis_result})

def core_analysis(klines, parameters):
    if not klines:
        return {'signal': 'hold', 'confidence': 0, 'reason': 'No data'}
    
    latest = klines[-1]
    prev = klines[-2] if len(klines) > 1 else latest
    
    close_change = (latest['close'] - prev['close']) / prev['close'] * 100
    
    if close_change > 3:
        return {
            'signal': 'buy',
            'confidence': min(close_change / 5, 1),
            'reason': f'Price increased by {close_change:.2f}%',
            'current_price': latest['close'],
            'volume': latest['volume']
        }
    elif close_change < -3:
        return {
            'signal': 'sell',
            'confidence': min(abs(close_change) / 5, 1),
            'reason': f'Price decreased by {abs(close_change):.2f}%',
            'current_price': latest['close'],
            'volume': latest['volume']
        }
    else:
        return {
            'signal': 'hold',
            'confidence': 0.5,
            'reason': 'Price stable',
            'current_price': latest['close'],
            'volume': latest['volume']
        }

@app.route('/api/trading/buy', methods=['POST'])
def buy_stock():
    data = request.json
    code = data.get('code')
    quantity = data.get('quantity', 100)
    
    result = {
        'success': False,
        'message': 'Trading API not implemented',
        'code': code,
        'quantity': quantity,
        'order_type': 'buy',
        'status': 'pending',
        'estimated_price': 0
    }
    
    klines = KlineData.get(code)
    if klines:
        result['estimated_price'] = klines[-1]['close']
        result['success'] = True
        result['message'] = 'Order submitted successfully (simulated)'
    
    return jsonify(result)

@app.route('/api/trading/sell', methods=['POST'])
def sell_stock():
    data = request.json
    code = data.get('code')
    quantity = data.get('quantity', 100)
    
    result = {
        'success': False,
        'message': 'Trading API not implemented',
        'code': code,
        'quantity': quantity,
        'order_type': 'sell',
        'status': 'pending',
        'estimated_price': 0
    }
    
    klines = KlineData.get(code)
    if klines:
        result['estimated_price'] = klines[-1]['close']
        result['success'] = True
        result['message'] = 'Order submitted successfully (simulated)'
    
    return jsonify(result)

@app.route('/api/trading/orders', methods=['GET'])
def get_orders():
    orders = []
    return jsonify({'success': True, 'data': orders})