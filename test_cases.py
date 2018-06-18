from Raster import raster_factory
import buzzard as buzz
import queue
import numpy as np
import multiprocessing as mp
import multiprocessing.pool
import gc

import scipy.ndimage as ndi

def main():
    g_cpu_pool = mp.pool.ThreadPool()
    g_io_pool = mp.pool.ThreadPool()
    full_fp = buzz.Footprint(tl=(0, 0), size=(12691, 12691), rsize=(126910, 126910))
    print(full_fp, full_fp.scale, full_fp.rarea*4/(1024**3))
    print()
    print(full_fp._significant_min)

    def compute_data(fp, *args):
        return np.random.rand(fp.shape[0], fp.shape[1])

    def filter_data(fp, *data):
        array, = data
        assert np.array_equal(fp.shape, array.shape)
        return ndi.gaussian_filter(array, 0.5)


    random_raster = raster_factory(footprint=full_fp,
                         dtype='float32',
                         nbands=1,
                         nodata=-99999,
                         srs=None, # dsm.wkt_origin
                         computation_function=compute_data,
                         cached=False,
                         cache_dir=None,
                         cache_fps=None,
                         io_pool=g_io_pool,
                         computation_pool=g_cpu_pool,
                         primitives={},
                         to_collect_of_to_compute=None,
                         to_compute_fps=None,
                         merge_pool=None,
                         merge_function=None
                        )

    
    filter_raster = raster_factory(footprint=full_fp,
        dtype='float32',
        nbands=1,
        nodata=-99999,
        srs=None,
        computation_function=filter_data,
        cached=False,
        cache_dir=None,
        cache_fps=None,
        io_pool=g_io_pool,
        computation_pool=g_cpu_pool,
        primitives={"random": random_raster.get_multi_data_queue},
        to_collect_of_to_compute=lambda fp: {"random": fp},
        to_compute_fps=None,
        merge_pool=None,
        merge_function=None
        )


    tile_count = np.ceil(full_fp.rsize / 1000)
    tiles = full_fp.tile_count(*tile_count)

    a = filter_raster.get_multi_data([tiles[10, 18], tiles[3,50]])

    for i in range(2):
        next(a)
        gc.collect()

if __name__ == "__main__":
    main()
