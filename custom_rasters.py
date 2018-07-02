from pprint import pprint

import numpy as np
from burito.raster import Raster
import multiprocessing as mp
import buzzard as buzz

g_io_pool = mp.pool.ThreadPool(12)
g_cpu_pool = mp.pool.ThreadPool(12)
g_gpu_pool = mp.pool.ThreadPool(1)

def _Raster(**kwargs):

    if 'fp' in kwargs:
        kwargs['footprint'] = kwargs['fp']
        del kwargs['fp']
    if 'sr' in kwargs:
        kwargs['srs'] = kwargs['sr']
        del kwargs['sr']
    if 'band_count' in kwargs:
        kwargs['nbands'] = kwargs['band_count']
        del kwargs['band_count']

    # pprint(kwargs)
    return Raster(**kwargs)


def derive_raster(raster, **kwargs):
    if 'fp' not in kwargs:
        kwargs['fp'] = raster.fp
    fp = kwargs['fp']

    if 'dtype' not in kwargs:
        kwargs['dtype'] = raster.dtype
    kwargs['band_count'] = len(raster)
    kwargs['nodata'] = raster.nodata
    if 'sr' not in kwargs:
        kwargs['sr'] = raster.wkt_origin

    if 'computation_function' not in kwargs:
        # resample = not kwargs['fp'].same_grid(raster.fp)
        kwargs['primitives'] = {'src': raster.get_multi_data_queue}
        primitive_fp = raster.fp
        primitive_nodata = raster.nodata

        def make_primitive_fps(fp):
            return {
                'src': (primitive_fp & fp).dilate(4),
            }

        def compute_data(fp, parrs, pfps, _):
            arr = buzz.Raster._remap(
                pfps[0], fp, parrs[0], nodata=primitive_nodata,
            )
            return arr

        kwargs['computation_function'] = compute_data
        kwargs['to_collect_of_to_compute'] = make_primitive_fps

        counts = np.ceil(fp.rsize / 1000)
        kwargs['computation_fps'] = fp.tile_count(*counts, boundary_effect='shrink').flatten()

    if 'cache_dir' in kwargs and kwargs['cache_dir'] is not None:
        kwargs['cached'] = True
        # kwargs['overwrite'] =
        # kwargs['cache_dir'] =
        if 'cache_fps' not in kwargs or 'cache_fps' is None:
            kwargs['cache_fps'] = kwargs['fp'].tile(
                (512, 512),
                boundary_effect='shrink',
            )
    else:
        kwargs['cached'] = False
        # kwargs['overwrite'] =

    kwargs['io_pool'] = g_io_pool
    kwargs['computation_pool'] = g_cpu_pool
    kwargs['merge_pool'] = None
    kwargs['merge_function'] = None

    return _Raster(**kwargs)


def wrap_buzzard_raster(buzz_raster, **kwargs):
    if 'fp' not in kwargs:
        kwargs['fp'] = buzz_raster.fp
    fp = kwargs['fp']
    resample = not kwargs['fp'].same_grid(buzz_raster.fp)

    if 'dtype' not in kwargs:
        kwargs['dtype'] = buzz_raster.dtype
    kwargs['band_count'] = len(buzz_raster)
    kwargs['nodata'] = buzz_raster.nodata
    if 'sr' not in kwargs:
        kwargs['sr'] = buzz_raster.wkt_origin

    path = buzz_raster.path
    def compute_data(fp, *args):
        if resample:
            ds = buzz.DataSource(allow_interpolation=True)
        else:
            ds = buzz.DataSource(allow_interpolation=True)
        with ds.open_araster(path).close as r:
            print(f'Reading {fp.rarea:12,}px of {path.split("/")[-1]:10} ')
            arr = r.get_data(fp, band=-1)
            print(f'   Read {fp.rarea:12,}px of {path.split("/")[-1]:10} ')
        return arr

    kwargs['computation_function'] = compute_data
    counts = np.ceil(fp.rsize / 1000)
    kwargs['computation_fps'] = fp.tile_count(*counts, boundary_effect='shrink').flatten()

    if 'cache_dir' in kwargs and kwargs['cache_dir'] is not None:
        kwargs['cached'] = True
        # kwargs['overwrite'] =
        # kwargs['cache_dir'] =
        if 'cache_fps' not in kwargs or 'cache_fps' is None:
            kwargs['cache_fps'] = kwargs['fp'].tile(
                (512, 512),
                boundary_effect='shrink',
            )
    else:
        kwargs['cached'] = False
        # kwargs['overwrite'] =

    kwargs['io_pool'] = g_io_pool
    kwargs['computation_pool'] = g_cpu_pool if resample else g_io_pool
    kwargs['primitives'] = {}
    kwargs['to_collect_of_to_compute'] = None
    kwargs['merge_pool'] = None
    kwargs['merge_function'] = None

    return _Raster(**kwargs)
