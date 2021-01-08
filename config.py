import os
import logging.config
import json
from pathlib import Path

# ------- DO NOT EDIT ---------
from typing import Set

ROOT_PATH = Path(__file__).resolve().parent
# ------- Logger Configuration ---------
log_config = ROOT_PATH / 'logging.json'
if os.path.exists(log_config):
    with open(log_config) as f:
        log_config_obj = json.load(f)
    logging.config.dictConfig(log_config_obj)
# ------- Basic Configuration ---------
DEBUG = True
CACHE_PATH = ROOT_PATH / 'cache'
OUT_PATH = ROOT_PATH / 'out'
DATA_CENTER_URL = 'http://127.0.0.1/gf'
DATA_CENTER_AUTH_URL = 'http://127.0.0.1/auth/token'
DATA_CENTER_AUTH_USERNAME = 'admin'
DATA_CENTER_AUTH_PASSWORD = 'admin'


# RADAR
RADAR_DATA_PATH = r''
RADAR_TYPE_QPEF_PATH =['QPEZIP','QPFZIP']
host_list = ['127.0.0.1:9200']
POINT = [{'area': 'JSQ', 'name': '监视区', 'lon': 118.1, 'lat': 24.46, 'dis': 10000, 'threshold': 30}]

# ------- Basic Configuration ---------

