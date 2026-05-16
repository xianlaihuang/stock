from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv('MONGO_URI'))
db = client[os.getenv('DB_NAME')]

stock_collection = db['stocks']
kline_collection = db['kline_data']
financial_collection = db['financial_data']

stock_collection.create_index('code', unique=True)
kline_collection.create_index([('code', 1), ('period', 1), ('date', 1)], unique=True)
financial_collection.create_index([('code', 1), ('report_date', 1)], unique=True)