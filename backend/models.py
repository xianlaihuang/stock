from db import stock_collection, kline_collection, financial_collection
from datetime import datetime

class Stock:
    @staticmethod
    def add(code, name):
        stock = {
            'code': code,
            'name': name,
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        try:
            stock_collection.insert_one(stock)
            return True
        except Exception as e:
            return False

    @staticmethod
    def get(code):
        return stock_collection.find_one({'code': code})

    @staticmethod
    def get_all():
        return list(stock_collection.find({}, {'_id': 0}))

    @staticmethod
    def delete(code):
        result = stock_collection.delete_one({'code': code})
        kline_collection.delete_many({'code': code})
        financial_collection.delete_many({'code': code})
        return result.deleted_count > 0

    @staticmethod
    def update(code, data):
        data['last_updated'] = datetime.now()
        result = stock_collection.update_one({'code': code}, {'$set': data})
        return result.modified_count > 0

class KlineData:
    @staticmethod
    def add_many(code, klines, period='day'):
        print(f"[DEBUG] Adding {len(klines)} klines for {code}, period {period}")
        for kline in klines:
            kline['code'] = code
            kline['period'] = period
        try:
            result = kline_collection.insert_many(klines, ordered=False)
            print(f"[DEBUG] Inserted {len(result.inserted_ids)} klines")
            return True
        except Exception as e:
            print(f"[DEBUG] Error adding klines: {e}")
            return False

    @staticmethod
    def get(code, start_date=None, end_date=None, period='day', after_exclusive=None):
        query = {'code': code, 'period': period}
        ae = str(after_exclusive).strip() if after_exclusive else ''
        if ae:
            query['date'] = {'$gt': ae}
        else:
            if start_date:
                query['date'] = {'$gte': start_date}
            if end_date:
                query['date'] = query.get('date', {})
                query['date']['$lte'] = end_date
        return list(kline_collection.find(query, {'_id': 0}).sort('date', 1))

    @staticmethod
    def delete(code, period=None):
        query = {'code': code}
        if period:
            query['period'] = period
        result = kline_collection.delete_many(query)
        return result.deleted_count

    @staticmethod
    def delete_for_calendar_days(code, period, day_prefixes):
        """按交易日前缀删除（分时 date 形如 YYYY-MM-DD HH:MM:SS）。"""
        deleted = 0
        for day in day_prefixes or []:
            d = str(day).strip()[:10]
            if len(d) < 10:
                continue
            r = kline_collection.delete_many({
                'code': code,
                'period': period,
                'date': {'$regex': f'^{d}'},
            })
            deleted += r.deleted_count
        return deleted

class FinancialData:
    @staticmethod
    def add(code, data):
        data['code'] = code
        data['created_at'] = datetime.now()
        try:
            financial_collection.replace_one(
                {'code': code, 'report_date': data['report_date']},
                data,
                upsert=True
            )
            return True
        except Exception as e:
            return False

    @staticmethod
    def get(code):
        return list(financial_collection.find({'code': code}, {'_id': 0}).sort('report_date', -1))

    @staticmethod
    def delete(code):
        result = financial_collection.delete_many({'code': code})
        return result.deleted_count