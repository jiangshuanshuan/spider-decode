import logging
import os

import re
from pathlib import Path

from requests_html import HTMLSession
import demjson

MODULE_NAME = os.path.split(__file__)[-1].split(".")[0]
logger = logging.getLogger(f'module.utils.{MODULE_NAME}')
_session = HTMLSession()

# 中国天气网网站爬取云图、雷达


def run():
    radar()


def radar():
    #https://pi.weather.com.cn/i/product/pic/m/sevp_aoc_rdcp_sldas_ebref_achn_l88_pi_20201223063600001.png
    url = 'http://d1.weather.com.cn/radar/JC_RADAR_CHN_JB.html'
    text_res = _session.get(url=url).text
    json_str_res = re.findall(r'\((.*?)\)', text_res)[0]
    json_res = demjson.decode(json_str_res)
    radar_list = json_res['radars']
    os_path = os.getcwd() + '/zgtwq_radar/ebref_achn/'
    if not Path(os_path).exists():
        Path(os_path).mkdir(parents=True)
    for r in radar_list:
        htp = f"https://pi.weather.com.cn/i/product/pic/m/sevp_aoc_rdcp_sldas_{r['fn']}_l88_pi_{r['ft']}.png"
        with open(os_path + r['ft'] + '.png', 'wb') as f:
            cloud_byte = _session.get(htp).content
            f.write(cloud_byte)


def cloud():
   # https://pi.weather.com.cn/i/product/pic/m/sevp_nsmc_wxbl_fy4a_etcc_achn_lno_py_20201223065300000.jpg
    url = 'http://d1.weather.com.cn/satellite2015/JC_YT_DL_WXZXCSYT_4A.html'
    text_res = _session.get(url=url).text
    json_str_res = re.findall(r'\((.*?)\)', text_res)[0]
    json_res = demjson.decode(json_str_res)
    radar_list = json_res['radars']
    os_path = os.getcwd() + '/zgtwq_cloud/wxbl_fy4a_etcc_achn/'
    if not Path(os_path).exists():
        Path(os_path).mkdir(parents=True)
    for r in radar_list:
        htp = f"https://pi.weather.com.cn/i/product/pic/m/sevp_nsmc_{r['fn']}_lno_py_{r['ft']}.jpg"
        with open(os_path + r['ft'] + '.jpg', 'wb') as f:
            cloud_byte = _session.get(htp).content
            f.write(cloud_byte)


if __name__ == '__main__':
    run()
