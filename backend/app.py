from flask import Flask, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

@app.route('/')
@app.route('/index.html')
def index():
    return send_from_directory('../frontend', 'index.html')

from routes import *
from unified_routes import *
from routes_v2 import *

if __name__ == '__main__':
    # debug + reloader：改后端 .py 后自动重载（无需手停服务）
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=True, threaded=True)