from pprint import pprint

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

    pprint(kwargs)
    return Raster(**kwargs)


def wrap_buzzard_raster(buzz_raster, **kwargs):
    if 'fp' not in kwargs:
        kwargs['fp'] = buzz_raster.fp

    resample = not kwargs['fp'].same_grid(buzz_raster.fp)

    if 'dtype' not in kwargs:
        kwargs['dtype'] = buzz_raster.dtype
    kwargs['band_count'] = len(buzz_raster)
    kwargs['nodata'] = buzz_raster.nodata
    if 'sr' not in kwargs:
        kwargs['sr'] = buzz_raster.wkt_origin

    # thread_storage = threading.local()
    path = buzz_raster.path
    def compute_data(compute_fp, *args):
        if resample:
            ds = buzz.DataSource(allow_interpolation=True)
        else:
            ds = buzz.DataSource(allow_interpolation=True)
        with ds.open_araster(path).close as r:
            print('Reading', path)
            arr = r.get_data(compute_fp, band=-1)
            print('   Read', path)
        return arr

    kwargs['computation_function'] = compute_data


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
    kwargs['computation_fps'] = None
    kwargs['merge_pool'] = None
    kwargs['merge_function'] = None

    return _Raster(**kwargs)
