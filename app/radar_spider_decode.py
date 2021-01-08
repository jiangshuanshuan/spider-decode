import logging
import os

import requests
import re


MODULE_NAME = os.path.split(__file__)[-1].split(".")[0]
logger = logging.getLogger(f'module.utils.{MODULE_NAME}')
_session = requests.Session()
# 中央气象台网站爬取云图、雷达


def run():
    radar()


def radar():
    url = 'http://www.nmc.cn/publish/radar/tian-jin/tian-jin.htm'
    res = _session.get(url=url).content
    htp_list = re.findall(r'data-img="(.*?)" data-time=', str(res))
    os_path = os.getcwd() + '/radar/'
    if not os.path.exists(os_path):
        os.mkdir(os_path)
    for htp in htp_list:
        cloud_name = re.findall(r'RDCP/(.*?)\?', htp)[0]
        print(cloud_name)
        with open(os_path + cloud_name, 'wb') as f:
            cloud_byte = _session.get(htp).content
            f.write(cloud_byte)


def cloud():
    url = 'http://www.nmc.cn/publish/satellite/FY4A-true-color.htm'
    res = _session.get(url=url).content
    htp_list = re.findall(r'data-img="(.*?)" data-time=', str(res))
    os_path = os.getcwd() + '/cloud/'
    if not os.path.exists(os_path):
        os.mkdir(os_path)
    for htp in htp_list:
        cloud_name = re.findall(r'medium/(.*?)\?', htp)[0]
        print(cloud_name)
        with open(os_path + cloud_name, 'wb') as f:
            cloud_byte = _session.get(htp).content
            f.write(cloud_byte)


if __name__ == '__main__':
    run()
