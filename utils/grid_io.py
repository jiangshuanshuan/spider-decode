import os
import logging
import tarfile

import config
import json
import struct
import zipfile
import utils
import xarray as xr
import numpy as np
from pathlib import Path
from utils import net_io
# typing
from typing import Any
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Callable
from typing import Optional
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial.qhull import Delaunay

from datetime import datetime, timezone, timedelta
import time

MODULE_NAME = os.path.split(__file__)[-1].split(".")[0]
logger = logging.getLogger(f'module.utils.{MODULE_NAME}')
_INTERP_SWITCH: int = 1


def latest_folder_by_name(data_folder: Union[Path, str]) -> Path:
    """通过数据目录里的文件夹名获取最新的文件夹"""
    folders = list(Path(data_folder).glob('2*'))
    folders.sort(key=lambda x: str(x.name))
    return folders[-1]


def gfs_to_cache(source_folder: Union[Path, str],
                 func_filter: Callable[[Path], bool] = None
                 ) -> Path:
    """将指定目录中的指定文件夹下的gfs文件复制到缓存  返回目标文件夹Path对象"""
    files = list(Path(source_folder).glob('gfs.*'))
    if func_filter is not None:
        files = [x for x in files if func_filter(x)]

    dest_folder = config.CACHE_PATH / 'fragments' / 'source'
    if not dest_folder.exists():
        dest_folder.mkdir(parents=True)

    meta_file = dest_folder / 'meta.json'
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
            source_changed = (meta['source'] != str(source_folder))
    else:
        source_changed = True

    dest_remain = list(dest_folder.glob('gfs.*'))
    if source_changed:
        logger.info('数据源变化 清除残留数据...')
        for remain in dest_remain:
            os.remove(remain)

    for i, source in enumerate(files):
        dest_name = dest_folder / source.name
        if dest_name.exists():
            continue
        logger.info(f'文件复制中... {i + 1}/{len(files)}')
        logger.debug(f'复制文件 {str(source)}')
        utils.copyfile(source, dest_name)
    with open(meta_file, 'w') as f:
        json.dump({'source': str(source_folder)}, f)
    logger.info('文件复制完成')
    return dest_folder


def gfs_to_nc(folder: Union[Path, str],
              var_name: str,
              type_of_level: str,
              layer_range: Tuple[float, float] = None,
              filter_by_keys: Dict[str, str] = None,
              func_filter: Callable[[Path, Dict], bool] = None
              ) -> Tuple[List[Path], Dict[str, np.ndarray]]:
    """将目录里的gfs文件根据配置剪裁成对应的nc文件"""
    key_filter = {'typeOfLevel': type_of_level, 'name': var_name}
    lev_max: Optional[np.ndarray] = None
    lev_min: Optional[np.ndarray] = None
    level: Optional[np.ndarray] = None
    value_key: Optional[str] = None
    time_fore: Optional[str] = None
    meta: Optional[dict] = None

    if filter_by_keys is not None:
        key_filter.update(filter_by_keys)

    cache = config.CACHE_PATH / 'fragments' / var_name.replace(' ', '_') / type_of_level
    if not cache.exists():
        cache.mkdir(parents=True)
    meta_file = cache / 'meta.json'

    source_meta_file = folder / 'meta.json'
    if not source_meta_file.exists():
        raise ValueError(f'{source_meta_file} 不存在')
    with open(source_meta_file) as f:
        source_meta = json.load(f)

    source_updated = True
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        if meta['source'] == source_meta['source']:
            source_updated = False

    if source_updated:
        logger.info('数据源变化,清理上次的裁取文件')
        for file in cache.glob('*'):
            os.remove(file)

    data_path = Path(folder).glob('gfs.*')
    files = [x for x in data_path if not str(x).endswith('.idx')]
    if func_filter is not None:
        files = [x for x in data_path if func_filter(x, meta)]

    select_value = {
        'latitude': slice(config.GFS_LIMIT_REGION['north'], config.GFS_LIMIT_REGION['south']),
        'longitude': slice(config.GFS_LIMIT_REGION['west'], config.GFS_LIMIT_REGION['east']),
    }
    if layer_range is not None:
        select_value[type_of_level] = slice(*layer_range)

    out_file_list = []
    for i, file in enumerate(files):
        save_path = cache / f'{file.name[-3:]}.nc'
        if save_path.exists() and not source_updated:
            out_file_list.append(save_path)
            continue
        logger.info(f'GFS文件预裁取中...[{var_name}] {type_of_level}  {i + 1}/{len(files)}')
        dat = xr.open_dataset(file, engine='cfgrib', backend_kwargs={'filter_by_keys': key_filter}).sel(**select_value)
        dat = dat.rename({type_of_level: 'level'})
        dat.to_netcdf(save_path)

        value_key = list(dat.keys())[0]
        if lev_max is None or lev_min is None:
            lev_max = dat[value_key].max(['latitude', 'longitude']).values
            lev_min = dat[value_key].min(['latitude', 'longitude']).values
            level = dat[value_key].level.values
            time_fore = dat[value_key].time.dt.strftime("%Y%m%d%H").values.tolist()
        else:
            lev_max = np.max([lev_max, dat[value_key].max(['latitude', 'longitude'])], axis=0)
            lev_min = np.min([lev_min, dat[value_key].min(['latitude', 'longitude'])], axis=0)
        out_file_list.append(save_path)

    if lev_max is not None:
        lev_max = lev_max.reshape(-1)
        lev_min = lev_min.reshape(-1)
        level = level.reshape(-1)

        meta = {
            'max': lev_max.tolist(),
            'min': lev_min.tolist(),
            'level': level.tolist(),
            'key': value_key,
            'time_fore': time_fore,
            'source': source_meta['source'],
            'type_of_level': type_of_level
        }

    with meta_file.open(mode='w', encoding='utf8') as f:
        json.dump(meta, f)
    logger.info('GFS文件预裁取完成')
    return out_file_list, meta


_type_mapping_to_xgrid = {
    'float32': ('Float32', 999999),
    'uint8': ('Int8', 255),
}
_type_mapping_to_np = {
    'Float32': ('float32', 8, 999999),
    'Int8': ('uint8', 1, 255),
}


def xgrid_writer(data: Union[xr.DataArray, xr.Dataset],
                 save_path: Union[str, Path],
                 dtype: str = 'uint8',
                 undef: float = None,
                 meta: Dict[str, Any] = None,
                 data_processor: Callable[[np.ndarray, Dict[str, Any]], np.ndarray] = None
                 ) -> Path:
    """将格点数据打包成指定zip压缩的指定二进制格式"""
    if isinstance(data, xr.Dataset):
        data: xr.DataArray = data[list(data.keys())[0]]

    if data.ndim not in {4, 3, 2}:
        raise ValueError('data should be reshaped as (t,z,y,x) , (z,y,x) , (t,y,x) or (y,x)')

    if data.ndim == 2:
        data = data.transpose('latitude', 'longitude')

    elif data.ndim == 3:
        if 'level' in data.dims:
            data = data.transpose('level', 'latitude', 'longitude')
        elif 'time' in data.dims:
            data = data.transpose('time', 'latitude', 'longitude')
        else:
            raise ValueError('data must be include [level] or [time]')

    elif data.ndim == 4:
        data = data.transpose('time', 'level', 'latitude', 'longitude')

    data = data.sortby('latitude')

    time_list = data.time.dt.strftime("%Y%m%d%H%M%S").values.reshape(-1).tolist()
    lon = data.longitude.values.tolist()
    lat = data.latitude.values.tolist()
    level_list = data.level.values.reshape(-1).tolist()
    file_name = Path(save_path)
    logger.debug(f'写入zip [{str(file_name)}]')

    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True)

    data_type, default_undef = _type_mapping_to_xgrid[dtype]
    if undef is None:
        undef = default_undef

    undef_mask = np.isnan(data.values)

    with zipfile.ZipFile(file_name, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zfile:
        dfile_name = '.'.join(file_name.name.split('.')[:-1]) + '.dat'
        with zfile.open(dfile_name, mode='w') as raw:
            header = {
                'xSize': data.longitude.shape[0],
                'xStart': lon[0],
                'xEnd': lon[-1],
                'xDelta': lon[1] - lon[0],

                'ySize': data.latitude.shape[0],
                'yStart': lat[0],
                'yEnd': lat[-1],
                'yDelta': lat[1] - lat[0],

                'levels': len(level_list),
                'levelList': level_list,

                'times': len(time_list),
                'timeList': time_list,

                'undef': undef,
                'dataScale': 1,
                'dataOffset': 0,
                'units': '',
                'littleEndian': True,
                'dataType': data_type,
                'unsigned': dtype.startswith('u')
            }
            if meta is not None:
                header.update(meta)
            target_array: np.ndarray = data.values
            target_array[undef_mask] = undef
            if data_processor is not None:
                target_array = data_processor(target_array, header)

            header = json.dumps(header, ensure_ascii=False).encode('utf-8')
            header_len = len(header)
            raw.write(struct.pack('i', header_len))
            raw.write(header)
            raw.write(target_array.astype(dtype).tobytes())
    logger.debug('zip写入完成')
    return file_name


def nc_to_xr(files_nc: List[Union[Path, str]],
             meta: Dict[str, Any],
             func_filter: Callable[[Path, Dict], bool] = None
             ) -> xr.DataArray:
    key: str = meta.pop('key')
    levels = meta.pop('level')
    vmax = xr.DataArray(meta['max'], coords={'level': levels}, dims=('level',))
    vmin = xr.DataArray(meta['min'], coords={'level': levels}, dims=('level',))
    vmax_min = vmax - vmin

    if func_filter is not None:
        files_nc = [Path(f) for f in files_nc if func_filter(Path(f), meta)]
    dat_group = []
    for i, file in enumerate(files_nc):
        logger.info(f'zip文件预处理中... {key} {i + 1}/{len(files_nc)}')
        dataset = xr.open_dataset(file)
        dat = dataset[key].assign_coords({'time': dataset.valid_time})
        dat_group.append(dat)

    if len(dat_group) != 0:
        dat: xr.DataArray = xr.concat(dat_group, 'time')
        dat = dat.sortby('time')
        dat = (dat - vmin) / vmax_min * 254

        dat.attrs['key'] = key
        dat.attrs['vmax'] = vmax.values.tolist()
        dat.attrs['vmin'] = vmin.values.tolist()
        dat.attrs['time_fore'] = meta['time_fore']
        return dat

    raise ValueError('没有文件被输出')


def xr_to_xgrid(data: xr.DataArray, split: str, var_type: str, data_code: str) -> List[Path]:
    """xr转为xgrid并上传
    :param data: 数据xarray
    :param split: enum['time', 'level', 'none']
    :param var_type: 变量类型  high或surface
    :param data_code: 用于上传的资料代码
    """
    save_path = config.OUT_PATH / var_type / data.attrs['key']

    out_group = []
    if split == 'none':
        save_file = save_path / f'{data.attrs["key"]}.zip'
        zfile = xgrid_writer(data, save_file, 'uint8', meta=data.attrs)
        net_io.grid_upload(zfile, data_code, f"{data.attrs['key']}_{var_type}",
                           data.attrs['time_fore'], 'all', '00000')
        out_group.append(zfile)

    elif split == 'time':
        for t in data.time:
            dat_frag = data.sel(time=t)
            t = str((t - data.time[0]).dt.seconds.values // 3600).rjust(3, '0')
            save_file = save_path / 'time' / f'{t}.zip'
            zfile = xgrid_writer(dat_frag, save_file, 'uint8', meta=data.attrs)
            if int(t) != 0:
                net_io.grid_upload(zfile, data_code, f"{data.attrs['key']}_{var_type}",
                                   data.attrs['time_fore'], 'all', t)
            out_group.append(zfile)

    elif split == 'level':
        for i_lev, lev in enumerate(data.level):
            dat_frag = data.sel(level=lev)
            save_file = save_path / 'lev' / f'{str(lev.values)}.zip'
            zfile = xgrid_writer(dat_frag, save_file, 'uint8', meta=data.attrs)
            net_io.grid_upload(zfile, data_code, f"{data.attrs['key']}_{var_type}",
                               data.attrs['time_fore'], str(lev.values), '00000')
            out_group.append(zfile)

    else:
        raise ValueError(f'split={split} is not valid')

    return out_group


def xgrid_reader(file: Union[str, Path], scaled: bool = False) -> xr.DataArray:
    file = Path(file)
    with zipfile.ZipFile(file, "r") as z_file:
        d_file = z_file.filelist[0]
        with z_file.open(d_file) as raw:
            header_len = raw.read(4)
            header_len = struct.unpack('I', header_len)[0]
            header = raw.read(header_len)
            header = json.loads(header.decode('utf-8'))
            num_x = header['xSize']
            num_y = header['ySize']
            num_z = header['levels']
            num_t = header['times']
            time_list = header['timeList']
            lev_list = header['levelList']
            dat_max = header.get('max', None)
            dat_min = header.get('min', None)

            logger.debug(f'shape: {(num_t, num_z, num_y, num_x)}')
            logger.debug('levels: {}'.format(header['levelList']))
            logger.debug('times: {}'.format(header['timeList']))

            dtype = _type_mapping_to_np[header['dataType']]
            data = np.ndarray(
                shape=(num_t, num_z, num_y, num_x),
                dtype=dtype[0],
                buffer=raw.read(num_x * num_y * num_z * num_t * dtype[1])
            )

    lat = np.linspace(header['yStart'], header['yEnd'], header['ySize'])
    lon = np.linspace(header['xStart'], header['xEnd'], header['xSize'])
    undef_mask = (data == dtype[2])

    dat_max_min = None
    if dat_max is not None and dat_min is not None:
        dat_max = xr.DataArray(dat_max, coords={'level': lev_list}, dims=['level'])
        dat_min = xr.DataArray(dat_min, coords={'level': lev_list}, dims=['level'])
        dat_max_min = dat_max - dat_min

    dat = xr.DataArray(data,
                       coords={'lat': lat, 'lon': lon, 'time': time_list, 'level': lev_list},
                       dims=['time', 'level', 'lat', 'lon']
                       )

    if dat_max_min is not None and scaled is True:
        dat = dat.astype('float32') / 254 * dat_max_min + dat_min
        dat[undef_mask] = np.nan

    dat.attrs.update(header)
    return dat


def _interp(orig_val: np.ndarray, orig_lat: np.ndarray, orig_lon: np.ndarray,
            lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
        格点插值
    :param orig_val: 等距离文件中的格点数组
    :param orig_lat: 等距离文件的lat数组
    :param orig_lon: 等距离文件的lon数组
    :param lat: 所需要的等经纬度的lat数组
    :param lon: 所需要的等经纬度的lon数组
    :return:
    """
    if orig_val.ndim != 2:
        raise ValueError('interp only receive 2-dim variable')
    if lat.ndim == 1 or lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    if _INTERP_SWITCH == 1:
        tri = Delaunay(np.asarray([orig_lon.ravel(), orig_lat.ravel()]).T)
        interpolator = LinearNDInterpolator(tri, orig_val.ravel())
        val = interpolator((lon, lat))
    elif _INTERP_SWITCH == 2:
        val = griddata(np.asarray([orig_lon.ravel(), orig_lat.ravel()]).T, orig_val.ravel(),
                       (lon, lat))
    else:
        raise ValueError('INTERP_SWITCH only can be 1 or 2')

    return val


def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """
    try:
        t = tarfile.open(fname)
        t.extractall(path=dirs)
        return True
    except Exception as e:
        print(e)
        return False


def unzip(path, folder_abs):
    '''
    基本格式：zipfile.ZipFile(filename[,mode[,compression[,allowZip64]]])
    mode：可选 r,w,a 代表不同的打开文件的方式；r 只读；w 重写；a 添加
    compression：指出这个 zipfile 用什么压缩方法，默认是 ZIP_STORED，另一种选择是 ZIP_DEFLATED；
    allowZip64：bool型变量，当设置为True时可以创建大于 2G 的 zip 文件，默认值 True；

    '''
    zip_file = zipfile.ZipFile(path)
    zip_list = zip_file.namelist()  # 得到压缩包里所有文件

    for f in zip_list:
        zip_file.extract(f, folder_abs)  # 循环解压文件到指定目录

    zip_file.close()  # 关闭文件，必须有，释放内存


def read_csv(path):
    """
        读取csv 通过typeOfLevel列过滤数组
    :param path: csv路径
    :return:
    """
    data = pd.read_csv(path, encoding='gbk')
    dl = data.values.tolist();
    return dl


def datetime_to_long(time_start: datetime):
    """
    datetime格式转long 毫秒
    :param time_start:
    :return:
    """
    return int(time.mktime(time_start.timetuple()) * 1000.0 + time_start.microsecond / 1000.0)