#!/usr/bin/env python3
# coding=utf8
import config
import logging
import fire
import os
import sys
from importlib import import_module
# typing
from typing import List
from typing import Union

logger = logging.getLogger('module.manage')

app_dir = os.path.join(config.ROOT_PATH, 'app')
app_names_exist: List[str] = list(map(lambda x: x[:-3],
                                      filter(lambda x: not x.startswith('_'), os.listdir(app_dir))))


def run(app_name, *args, **kwargs) -> None:
    """
    执行指定模块的计算工作
    :param app_name: 模块名
    :param args:
    :return: none
    """
    app_suggest: List[str] = []
    if app_name not in app_names_exist:
        for exist_name in app_names_exist:
            if exist_name.startswith(app_name):
                app_suggest.append(exist_name)
        if len(app_suggest) != 0:
            logger.error('{} 模块不存在，是否是如下相似模块{}'.format(app_name, app_suggest))
        else:
            logger.error('{} 模块不存在，请检查模块名'.format(app_name))
        sys.exit(-1)
    app = import_module('app.{}'.format(app_name))
    logger.info('[manage][run] 开始执行模块 {}'.format(app_name))
    if hasattr(app, 'run') and callable(app.run):
        app.run(*args, **kwargs)
    else:
        raise ValueError(f'{app_name}没有run函数，不是标准的模块')
    logger.info('[manage][run] 模块 {} 运行完成'.format(app_name))


def runs(*app_names, ignore_err='OFF') -> None:
    """
    执行指定模块的计算工作 不能定义参数
    :param app_names: [] 模块名
    :param ignore_err: OFF|ON 是否忽略运行错误
    :return: none
    """
    app_not_exist: List[str] = []
    app_suggest: List[List[str]] = []
    if len(app_names) == 1 and app_names[0] == 'all':
        logger.info('[manage][runs] 运行所有模块')
        app_names = app_names_exist
    for name in app_names:
        if name not in app_names_exist:
            app_not_exist.append(name)
            suggest = []
            for exist_name in app_names_exist:
                if exist_name.startswith(name):
                    suggest.append(exist_name)
            app_suggest.append(suggest)
    if len(app_not_exist) != 0:
        for i, name in enumerate(app_not_exist):
            if len(app_suggest[i]) != 0:
                logger.error('{} 模块不存在，是否是如下相似模块{}'.format(name, app_suggest[i]))
            else:
                logger.error('{} 模块不存在，请检查模块名'.format(name))
        logger.error('运行停止')
        sys.exit(-1)

    if ignore_err == 'ON':
        logger.info('ignore_err=ON 开启忽略错误模式')
    n_app = len(app_names)
    for i, name in enumerate(app_names):
        try:
            app = import_module('app.{}'.format(name))
            logger.info('[manage][runs] {}/{} 开始执行模块 {}'.format(i + 1, n_app, name))
            if hasattr(app, 'run') and callable(app.run):
                app.run()
            else:
                raise ValueError(f'{name}没有run函数，不是标准的模块')
            logger.info('[manage][runs] {}/{} 模块 {} 运行完成'.format(i + 1, n_app, name))
        except Exception as e:
            if ignore_err == 'OFF':
                raise e
            logger.exception(e)
            pass
    logger.info('所有模块运行完成')


# def draw_from_file(file: str, time: Union[str, float] = None, level: Union[str, float] = None):
#     """
#     绘制cache中的指定文件 manage.py draw [filename] [time] [level]
#     :param file: 输出文件名
#     :param time: 绘图时次
#     :param level: 绘图层次
#     :return: None
#     """
#     import xgridio
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import cartopy.crs as ccrs
#     from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#     from matplotlib import cm
#
#     time_input_is_num: bool = False
#     level_input_is_num: bool = False
#     if isinstance(time, (int, float)):
#         time_input_is_num = True
#     if isinstance(level, (int, float)):
#         level_input_is_num = True
#
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'TimesNewRoman', 'Microsoft YaHei']
#
#     data = xgridio.xgrid_read_xr(file, uncompress=True)
#     if time is None or level is None:
#         raise KeyError(f'time跟level可选值 {data.time.values} {data.level.values}')
#     time = np.array(time).astype(data.time.dtype)
#     level = np.array(level).astype(data.level.dtype)
#
#     if np.issubdtype(str, time.dtype) and time_input_is_num:
#         time = str(time).rjust(len(str(data.time.values[0])), '0')
#     if np.issubdtype(str, level.dtype) and level_input_is_num:
#         level = str(level).rjust(len(str(data.level.values[0])), '0')
#
#     try:
#         data = data.sel(time=time, level=level)
#     except KeyError:
#         raise KeyError(f'time跟level可选值 {data.time.values} {data.level.values}  你选择了[{time}] [{level}]')
#
#     llon = round(data.longitude.values.min())
#     ulon = round(data.longitude.values.max())
#     llat = round(data.latitude.values.min())
#     ulat = round(data.latitude.values.max())
#     logger.debug(f'plot region: {(llon, ulon, llat, ulat)}')
#
#     fig = plt.figure(dpi=150)
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#     ax.set_extent((llon, ulon, llat, ulat), crs=ccrs.PlateCarree())
#     ax.coastlines(linewidth=0.5)
#
#     gridlines_style = {'draw_labels': True, 'linestyle': '--', 'alpha': 0.7}
#
#     gl = ax.gridlines(
#         ylocs=np.linspace(llat, ulat, 5),
#         xlocs=np.linspace(llon, ulon, 5),
#         **gridlines_style
#     )
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER
#     cf = ax.contourf(data.longitude.values, data.latitude.values, data.values,
#                      transform=ccrs.PlateCarree(),
#                      cmap=cm.get_cmap('RdBu_r'))
#     fig.colorbar(cf, orientation='horizontal', shrink=0.8)
#     plt.show()
#
#
# def draw_ani(file: str, level: Union[str, float] = None):
#     import json
#     from PIL import Image
#     import numpy as np
#     import pandas as pd
#     import xgridio
#
#     data = xgridio.xgrid_read_xr(file)
#     if level is not None:
#         level = np.array(level).astype(data.level.dtype)
#         try:
#             data = data.sel(level=level)
#         except KeyError:
#             raise KeyError(f'level可选值 {data.level.values}')
#
#     time_list = pd.to_datetime(data.time, format='%Y%m%d%H')
#     data = data.assign_coords(time=time_list).origin
#
#     frames = []
#     if data.level.ndim == 0 and data.time.ndim == 0:
#         im = Image.fromarray(data.values, 'L')
#         frames.append(im)
#     elif data.level.ndim == 0:
#         for t in data.time:
#             im = Image.fromarray(data.sel(time=t).values, 'L')
#             frames.append(im)
#     elif data.time.ndim == 0:
#         for lev in data.level:
#             im = Image.fromarray(data.sel(level=lev).values, 'L')
#             frames.append(im)
#     else:
#         for lev in data.level:
#             for t in data.time:
#                 im = Image.fromarray(data.sel(time=t, level=lev).values, 'L')
#                 frames.append(im)
#     exif = json.dumps(data.attrs).encode('utf8')
#     frames[0].save('t.webp', save_all=True, append_images=frames[1:], duration=80,
#                    quality=100, allow_mixed=True, method=6, exif=exif)
#     frames[0].save('t.png', save_all=True, append_images=frames[1:], duration=80, optimize=True, exif=exif)


if __name__ == '__main__':
    # draw_ani(r'C:\Users\mlog\Desktop\r.zip')
    fire.Fire({
        'run': run,
        'runs': runs,
        # 'draw': draw_from_file,
        # 'ani': draw_ani,
    })