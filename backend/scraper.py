import re
import requests
import json
import time
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup


class StockScraper:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1'
    }

    @staticmethod
    def get_stock_name(code):
        stock_names = {
            '600519': '贵州茅台',
            '000858': '五粮液',
            '601318': '中国平安',
            '600036': '招商银行',
            '000001': '平安银行',
            '601398': '工商银行',
            '600000': '浦发银行',
            '601899': '紫金矿业',
            '000333': '美的集团',
            '002594': '比亚迪',
            '600030': '中信证券',
            '000651': '格力电器',
            '600585': '海螺水泥',
            '601628': '中国人寿',
            '600028': '中国石化',
            '601857': '中国石油',
            '000002': '万科A',
            '600104': '上汽集团',
            '002304': '洋河股份',
            '600276': '恒瑞医药'
        }
        
        if code in stock_names:
            return stock_names[code]
        
        name = StockScraper._get_stock_name_from_eastmoney(code)
        if name and name != f'股票{code}':
            return name
        
        name = StockScraper._get_stock_name_from_tencent(code)
        if name and name != f'股票{code}':
            return name
        
        return f'股票{code}'

    @staticmethod
    def _get_stock_name_from_eastmoney(code):
        if code.startswith('6'):
            market = '1'
        else:
            market = '0'
        url = f'https://push2.eastmoney.com/api/qt/stock/get?secid={market}.{code}&fields=f57,f58'
        try:
            response = requests.get(url, headers=StockScraper.headers, timeout=10)
            response.encoding = 'utf-8'
            data = response.json()
            if data.get('data'):
                name = data['data'].get('f58', '')
                if name and name != 'null':
                    return name.strip()
        except Exception as e:
            print(f"Error getting stock name from eastmoney: {e}")
        return None

    @staticmethod
    def _get_stock_name_from_tencent(code):
        market = 'sh' if code.startswith('6') else 'sz'
        url = f'http://qt.gtimg.cn/q={market}{code}'
        try:
            response = requests.get(url, headers=StockScraper.headers, timeout=10)
            response.encoding = 'gbk'
            text = response.text.strip()
            if text and '~' in text:
                parts = text.split('~')
                if len(parts) > 1:
                    name = parts[1]
                    if name:
                        return name.strip()
        except Exception as e:
            print(f"Error getting stock name from tencent: {e}")
        return None

    @staticmethod
    def _get_em_market_code(code):
        pure_code = code.replace('.SZ', '').replace('.SH', '')
        if pure_code.startswith('6'):
            return '1', pure_code
        return '0', pure_code

    @staticmethod
    def get_kline_data(code, period='day', count=120, start_date=None):
        if period == 'minute':
            try:
                from tdx_source import fetch_minute_today_trends
                klines = fetch_minute_today_trends(code)
                if klines and len(klines) > 0:
                    print(f"[OK] Got {len(klines)} minute klines for {code} from tdx")
                    return klines
            except Exception as e:
                print(f"[FAIL] tdx minute for {code}: {e}")
            klines = StockScraper.get_minute_from_eastmoney_trends(code, start_date=start_date)
            if klines and len(klines) > 0:
                print(f"[OK] Got {len(klines)} minute klines for {code} from eastmoney-trends2")
                return klines
            print(f"[FAIL] eastmoney-trends2 returned empty for {code} (period: minute)")

        if period == 'day':
            try:
                from tdx_source import fetch_daily_kline
                want = max(int(count or 120), 800)
                tdx = fetch_daily_kline(code, count=want)
                if tdx and len(tdx) > 0:
                    if start_date:
                        tdx = [k for k in tdx if k['date'] > start_date]
                    if tdx:
                        print(f"[OK] Got {len(tdx)} klines for {code} (period: day) from tdx")
                        return tdx
            except Exception as e:
                print(f"[FAIL] tdx daily for {code}: {e}")

        sources = [
            ('eastmoney', StockScraper.get_kline_from_eastmoney),
            ('sina', StockScraper.get_kline_from_sina),
            ('tencent', StockScraper.get_kline_from_tencent),
        ]
        
        for source_name, source_func in sources:
            try:
                klines = source_func(code, count, start_date, period)
                if klines and len(klines) > 0:
                    print(f"[OK] Got {len(klines)} klines for {code} (period: {period}) from {source_name}")
                    return klines
                else:
                    print(f"[FAIL] {source_name} returned empty for {code} (period: {period})")
            except Exception as e:
                print(f"[ERROR] {source_name} failed for {code} (period: {period}): {e}")
        
        print(f"[ALL FAILED] All APIs failed for {code} (period: {period}), returning empty list")
        return []

    @staticmethod
    def _validate_kline_date(raw_date):
        today = datetime.now().date()
        try:
            if ' ' in raw_date:
                date_part = raw_date.split(' ')[0]
                parsed_date = datetime.strptime(date_part, '%Y-%m-%d').date()
            else:
                parsed_date = datetime.strptime(raw_date, '%Y-%m-%d').date()
            
            if parsed_date > today:
                return None
            if parsed_date.year < 2000:
                return None
            return raw_date
        except ValueError:
            return None

    @staticmethod
    def get_kline_from_eastmoney(code, count=120, start_date=None, period='day'):
        market, pure_code = StockScraper._get_em_market_code(code)
        
        period_map = {
            'minute': 1,
            'day': 101,
            'week': 102,
            'month': 103,
            'year': 104
        }
        klt = period_map.get(period, 101)
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                if retry > 0:
                    time.sleep(2)
                
                url = f'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{pure_code}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt={klt}&fqt=1&end=20500101&lmt={count * 4 if period == "year" else count}'
                headers = dict(StockScraper.headers)
                headers['Accept-Encoding'] = 'gzip, deflate'
                headers['Referer'] = 'https://quote.eastmoney.com/'
                headers['Origin'] = 'https://quote.eastmoney.com'
                
                response = requests.get(url, headers=headers, timeout=15)
                response.encoding = 'utf-8'
                
                if not response or not response.text:
                    continue
                    
                data = response.json()
                if data and data.get('data') and data['data'].get('klines'):
                    klines_data = data['data']['klines']
                    klines = []
                    
                    for item in klines_data:
                        parts = item.split(',')
                        if len(parts) >= 6:
                            raw_date = parts[0].strip()
                            validated_date = StockScraper._validate_kline_date(raw_date)
                            if validated_date is None:
                                continue
                            
                            kline = {
                                'date': validated_date,
                                'open': float(parts[1]),
                                'close': float(parts[2]),
                                'high': float(parts[3]),
                                'low': float(parts[4]),
                                'volume': int(float(parts[5])),
                                'amount': float(parts[6]) if len(parts) > 6 else 0
                            }
                            if start_date:
                                if kline['date'] > start_date:
                                    klines.append(kline)
                            else:
                                klines.append(kline)
                    
                    if period == 'year' and klines:
                        klines = StockScraper._filter_yearly_data(klines)
                    
                    return klines
            except requests.exceptions.RequestException as e:
                if retry == max_retries - 1:
                    print(f"Request error getting kline from eastmoney after {max_retries} retries: {e}")
            except Exception as e:
                if retry == max_retries - 1:
                    print(f"Error getting kline from eastmoney after {max_retries} retries: {e}")
        
        return None

    @staticmethod
    def get_kline_from_sina(code, count=120, start_date=None, period='day'):
        pure_code = code.replace('.SZ', '').replace('.SH', '')
        market = 'sh' if pure_code.startswith('6') else 'sz'
        period_map = {
            'minute': 5,
            'day': 240,
            'week': 1200,
            'month': 5200,
            'year': 5200
        }
        scale = period_map.get(period, 240)
        
        try:
            url = f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={market}{pure_code}&scale={scale}&ma=no&datalen={count}'
            response = requests.get(url, headers=StockScraper.headers, timeout=15)
            response.encoding = 'utf-8'
            
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                klines = []
                for item in data:
                    raw_date = item.get('day', '').strip()
                    validated_date = StockScraper._validate_kline_date(raw_date)
                    if validated_date is None:
                        continue
                    
                    kline = {
                        'date': validated_date,
                        'open': float(item['open']),
                        'close': float(item['close']),
                        'high': float(item['high']),
                        'low': float(item['low']),
                        'volume': int(float(item['volume'])),
                        'amount': float(item.get('amount', 0))
                    }
                    if start_date:
                        if kline['date'] > start_date:
                            klines.append(kline)
                    else:
                        klines.append(kline)
                return klines
        except Exception as e:
            print(f"Error getting kline from sina: {e}")
        
        return None

    @staticmethod
    def get_minute_from_eastmoney_trends(code, start_date=None):
        market, pure_code = StockScraper._get_em_market_code(code)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Referer': 'https://quote.eastmoney.com/',
        }

        url = f'https://push2his.eastmoney.com/api/qt/stock/trends2/get?secid={market}.{pure_code}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58&iscr=0'
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            data = resp.json()
            if not data or not data.get('data'):
                return None
            trends = data['data'].get('trends', [])
            if not trends:
                return None
            pre_close = data['data'].get('prePrice', data['data'].get('preClose', 0))
            all_klines = []
            day_open = None
            day_high = -float('inf')
            day_low = float('inf')
            for item in trends:
                parts = item.split(',')
                if len(parts) < 7:
                    continue
                try:
                    time_str = parts[0].strip()
                    if time_str and not re.match(r'\d{4}-\d{2}-\d{2}', time_str):
                        cn_today = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
                        time_str = f'{cn_today} {time_str}'
                    price = float(parts[1])
                    vol = int(float(parts[5]))
                    amount = float(parts[6]) if len(parts) > 6 else 0
                    if day_open is None:
                        day_open = price
                    day_high = max(day_high, price)
                    day_low = min(day_low, price)
                    kline = {
                        'date': time_str,
                        'open': day_open,
                        'close': price,
                        'high': day_high,
                        'low': day_low,
                        'volume': vol,
                        'amount': amount,
                        'pre_close': pre_close,
                    }
                    all_klines.append(kline)
                except (ValueError, IndexError):
                    continue
            return all_klines if all_klines else None
        except Exception as e:
            print(f"Error getting minute trends for {code}: {e}")
            return None

    @staticmethod
    def get_historical_minute_from_sina(code, target_date):
        pure_code = code.replace('.SZ', '').replace('.SH', '')
        market = 'sh' if pure_code.startswith('6') else 'sz'
        try:
            url = f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={market}{pure_code}&scale=5&ma=no&datalen=2000'
            resp = requests.get(url, headers=StockScraper.headers, timeout=15)
            resp.encoding = 'utf-8'
            data = resp.json()
            if not isinstance(data, list):
                return None
            day_klines = []
            for item in data:
                raw_date = item.get('day', '').strip()
                if not raw_date.startswith(target_date):
                    continue
                kline = {
                    'date': raw_date,
                    'open': float(item['open']),
                    'close': float(item['close']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'volume': int(float(item['volume'])),
                    'amount': 0,
                }
                day_klines.append(kline)
            if not day_klines:
                return None
            pre_close = None
            for item in data:
                raw_date = item.get('day', '').strip()
                if raw_date < target_date and raw_date.split(' ')[0] < target_date:
                    pre_close = float(item['close'])
            if pre_close is None and len(data) > 0:
                for item in data:
                    raw_date = item.get('day', '').strip()
                    d = raw_date.split(' ')[0]
                    if d < target_date:
                        pre_close = float(item['close'])
            for k in day_klines:
                k['pre_close'] = pre_close if pre_close else day_klines[0]['open']
            return day_klines
        except Exception as e:
            print(f"Error getting historical minute from sina for {code} on {target_date}: {e}")
            return None

    @staticmethod
    def get_kline_from_tencent(code, count=120, start_date=None, period='day'):
        pure_code = code.replace('.SZ', '').replace('.SH', '')
        market = 'sh' if pure_code.startswith('6') else 'sz'
        
        period_type_map = {
            'minute': 'm5',
            'day': 'day',
            'week': 'week',
            'month': 'month',
        }
        kline_type = period_type_map.get(period, 'day')
        
        if period == 'year':
            kline_type = 'month'
        
        try:
            url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{pure_code},{kline_type},,{count},1'
            headers = dict(StockScraper.headers)
            headers['Referer'] = 'http://gu.qq.com/'
            
            response = requests.get(url, headers=headers, timeout=15)
            response.encoding = 'utf-8'
            
            if not response or not response.text:
                return None
            
            text = response.text
            if text.startswith('\ufeff'):
                text = text[1:]
            
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return None
            
            if not data or data.get('code') != 0:
                return None
            
            raw_data = data.get('data', {})
            stock_data = raw_data.get(f'{market}{pure_code}', {}) if isinstance(raw_data, dict) else {}
            if not stock_data:
                if isinstance(raw_data, dict):
                    for k, v in raw_data.items():
                        if isinstance(v, dict):
                            stock_data = v
                            break
                if not stock_data:
                    return None
            
            # 腾讯API返回的数据格式: qfqday/day/qfqweek/week/qfqmonth/month
            klines_data = None
            for key in ['qfqday', 'day', 'qfqweek', 'week', 'qfqmonth', 'month', 'm5']:
                if key in stock_data:
                    klines_data = stock_data[key]
                    break
            
            if not klines_data or not isinstance(klines_data, list):
                return None
            
            klines = []
            for item in klines_data:
                if isinstance(item, list) and len(item) >= 6:
                    raw_date = str(item[0]).strip()
                    validated_date = StockScraper._validate_kline_date(raw_date)
                    if validated_date is None:
                        continue
                    
                    kline = {
                        'date': validated_date,
                        'open': float(item[1]),
                        'close': float(item[2]),
                        'high': float(item[3]),
                        'low': float(item[4]),
                        'volume': int(float(item[5])),
                        'amount': 0
                    }
                    if start_date:
                        if kline['date'] > start_date:
                            klines.append(kline)
                    else:
                        klines.append(kline)
            
            if period == 'year' and klines:
                klines = StockScraper._filter_yearly_data(klines)
            
            return klines
        except Exception as e:
            print(f"Error getting kline from tencent: {e}")
        
        return None

    @staticmethod
    def _filter_yearly_data(klines):
        yearly_data = {}
        for kline in klines:
            date_str = kline['date']
            date_part = date_str.split(' ')[0] if ' ' in date_str else date_str
            
            if '-' in date_part:
                try:
                    year = date_part.split('-')[0]
                    month = int(date_part.split('-')[1])
                    if month in [12]:
                        yearly_data[year] = kline
                except (ValueError, IndexError):
                    continue
        
        years_in_data = set()
        for kline in klines:
            date_str = kline['date']
            date_part = date_str.split(' ')[0] if ' ' in date_str else date_str
            if '-' in date_part:
                try:
                    year = date_part.split('-')[0]
                    years_in_data.add(year)
                except (ValueError, IndexError):
                    continue
        
        for year in years_in_data:
            if year not in yearly_data:
                year_data = []
                for kline in klines:
                    date_str = kline['date']
                    date_part = date_str.split(' ')[0] if ' ' in date_str else date_str
                    if date_part.startswith(year):
                        year_data.append(kline)
                
                if year_data:
                    yearly_data[year] = max(year_data, key=lambda x: x['date'])
        
        return sorted(yearly_data.values(), key=lambda x: x['date'])

    @staticmethod
    def get_financial_data(code):
        # 降级策略：东方财富 → 腾讯
        result = StockScraper._get_financial_from_eastmoney(code)
        if result and len(result) > 0:
            return result
        
        result = StockScraper._get_financial_from_tencent(code)
        if result and len(result) > 0:
            return result
        
        print(f"[ALL FAILED] All financial APIs failed for {code}, returning empty list")
        return []

    @staticmethod
    def _get_financial_from_eastmoney(code):
        url = f'https://datacenter-web.eastmoney.com/api/data/v1/get?' \
              f'sortColumns=NOTICE_DATE&sortTypes=-1&pageSize=20&pageNumber=1' \
              f'&reportName=RPT_F10_FINANCE_MAINFINADATA&columns=ALL' \
              f'&filter=(SECURITY_CODE=%22{code}%22)'
        
        try:
            headers = dict(StockScraper.headers)
            headers['Referer'] = 'https://data.eastmoney.com/'
            response = requests.get(url, headers=headers, timeout=15)
            response.encoding = 'utf-8'
            
            if not response or not response.text or response.text.strip() == '':
                return None
            
            text = response.text
            if text.startswith('\ufeff'):
                text = text[1:]
            
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"JSON decode error for financial data from eastmoney: {e}")
                return None
            
            if not data or not data.get('result') or not data['result'].get('data'):
                return None
            
            items = data['result']['data']
            financials = []
            for item in items:
                report_date = item.get('REPORT_DATE', '')
                if report_date:
                    if ' ' in report_date:
                        report_date = report_date.split(' ')[0]
                    elif 'T' in report_date:
                        report_date = report_date.split('T')[0]
                
                financial = {
                    'report_date': report_date,
                    'report_type': item.get('REPORT_TYPE', ''),
                    'report_date_name': item.get('REPORT_DATE_NAME', ''),
                    'eps': float(item.get('EPSJB', 0) or 0),
                    'eps_deducted': float(item.get('KCFJCXSYJLR', 0) or 0) / 100000000 if item.get('KCFJCXSYJLR') else 0,
                    'bps': float(item.get('BPS', 0) or 0),
                    'revenue': float(item.get('TOTALOPERATEREVE', 0) or 0),
                    'gross_profit': float(item.get('MLR', 0) or 0),
                    'net_profit': float(item.get('PARENTNETPROFIT', 0) or 0),
                    'deducted_net_profit': float(item.get('KCFJCXSYJLR', 0) or 0),
                    'roe': float(item.get('ROEJQ', 0) or 0),
                    'gross_margin': float(item.get('XSMLL', 0) or 0),
                    'net_margin': float(item.get('ZZCJLL', 0) or 0),
                    'revenue_yoy': float(item.get('TOTALOPERATEREVETZ', 0) or 0),
                    'net_profit_yoy': float(item.get('PARENTNETPROFITTZ', 0) or 0),
                    'deducted_net_profit_yoy': float(item.get('KCFJCXSYJLRTZ', 0) or 0),
                    'revenue_qoq': float(item.get('YYZSRGDHBZC', 0) or 0),
                    'net_profit_qoq': float(item.get('NETPROFITRPHBZC', 0) or 0),
                    'asset_liability_ratio': float(item.get('ZCFZL', 0) or 0),
                    'current_ratio': float(item.get('LD', 0) or 0),
                    'quick_ratio': float(item.get('SD', 0) or 0),
                    'cash_per_share': float(item.get('MGJYXJJE', 0) or 0),
                    'capital_reserve_per_share': float(item.get('MGZBGJ', 0) or 0),
                    'undistributed_per_share': float(item.get('MGWFPLR', 0) or 0),
                }
                financials.append(financial)
            return financials
        except Exception as e:
            print(f"Error getting financial data from eastmoney: {e}")
        
        return None

    @staticmethod
    def _get_financial_from_tencent(code):
        market = 'sh' if code.startswith('6') else 'sz'
        url = f'http://web.ifzq.gtimg.cn/appstock/app/finance/getFinanceInfo?code={market}{code}'
        
        try:
            headers = dict(StockScraper.headers)
            headers['Referer'] = 'http://gu.qq.com/'
            response = requests.get(url, headers=headers, timeout=15)
            response.encoding = 'utf-8'
            
            if not response or not response.text or response.text.strip() == '':
                return None
            
            text = response.text
            if text.startswith('\ufeff'):
                text = text[1:]
            
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return None
            
            if not data or data.get('code') != 0:
                return None
            
            financial_data = data.get('data', {}).get('data', [])
            if not financial_data:
                return None
            
            financials = []
            for item in financial_data:
                if isinstance(item, dict):
                    financial = {
                        'report_date': item.get('REPORTDATE', item.get('reportDate', '')),
                        'eps': float(item.get('BASIC_EPS', item.get('eps', '0'))),
                        'pe': float(item.get('PE', item.get('pe', '0'))),
                        'pb': float(item.get('PB', item.get('pb', '0'))),
                        'roe': float(item.get('ROE', item.get('roe', '0'))),
                        'revenue': float(item.get('TOTAL_OPERATE_INCOME', item.get('revenue', '0'))),
                        'profit': float(item.get('NET_PROFIT', item.get('profit', '0'))),
                        'gross_margin': float(item.get('GROSS_PROFIT_MARGIN', item.get('grossMargin', '0')))
                    }
                    financials.append(financial)
            return financials
        except Exception as e:
            print(f"Error getting financial data from tencent: {e}")
        
        return None
