import os
import logging
import requests
import config
from requests.adapters import HTTPAdapter
from pathlib import Path
# typing
from typing import Union
from typing import Dict

MODULE_NAME = os.path.split(__file__)[-1].split(".")[0]
logger = logging.getLogger(f'module.utils.{MODULE_NAME}')

_session = requests.Session()
_session.mount('http://', HTTPAdapter(max_retries=2))
_session.mount('https://', HTTPAdapter(max_retries=2))

_HEADER_JAR: Dict[str, str] = {}


def grid_upload(file: Union[str, Path],
                data_code: str,
                element: str,
                time_fore: str,
                level: str,
                period: str,
                force: bool = False,
                del_after: bool = False
                ):
    """上传格点数据文件到数据中台
    :param file: 文件路径
    :param data_code: 资料代码 自定
    :param element: 预报元素
    :param time_fore: 起报时间 yyyyMMddHHmmss
    :param level: 多层为all 地面为single 其他单层为数字
    :param period: 预报时效周期 HHHMM  三位小时两位分钟
    :param force: 强制上传 不管服务器上有没有对应文件
    :param del_after: 上传完成后删除本地文件
    :return: None
    """
    base_url = config.DATA_CENTER_URL
    file = Path(file)
    time_fore = time_fore.ljust(14, '0')
    period = period.ljust(5, '0')
    params = {'timeCode': time_fore, 'level': level, 'period': period}
    if force is not True:
        check_url = base_url + f'/api/v1/grid/exist/{data_code}/{element}'
        resp = _session.get(check_url, params=params, headers=tokenize())
        if resp.status_code != 200:
            raise IOError(f'{resp.text}')
        resp = resp.json()
        if resp['code'] != 0:
            raise IOError(resp)
        if resp['result'] is True:
            logger.info(f'相同参数的文件服务器已存在，且未开启强制上传，跳过上传 {data_code}/{element} {params}')
            return
    upload_url = base_url + f'/api/v1/grid/upload/{data_code}/{element}'
    logger.debug(f'上传文件 {str(file)} 到 {upload_url} {params}')
    with open(file, 'rb') as f:
        files = {'file': f}
        resp = _session.post(upload_url, files=files, params=params, headers=tokenize())
        if resp.status_code != 200:
            raise IOError(f'{resp.text}')
        resp = resp.json()
        if resp['code'] != 0:
            raise IOError(f'上传api错误 {resp}')
        logger.debug(resp['msg'])
    if del_after:
        os.remove(file)
    return


def tokenize(header: Dict[str, str] = None) -> Dict[str, str]:
    if header is None:
        header = {}
    if 'Authorization' not in _HEADER_JAR:
        resp = _session.post(config.DATA_CENTER_AUTH_URL,
                             params={
                                 'username': config.DATA_CENTER_AUTH_USERNAME,
                                 'password': config.DATA_CENTER_AUTH_PASSWORD
                             })
        if resp.status_code != 200:
            raise IOError(f'{resp.text}')
        resp = resp.json()
        if 'token' not in resp:
            raise IOError(f'{resp}')
        token = 'Bearer ' + resp['token']
        logger.debug(f"{token}")
        _HEADER_JAR['Authorization'] = token
    header.update(_HEADER_JAR)
    return header
