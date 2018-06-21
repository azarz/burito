from Raster import raster_factory
import buzzard as buzz
import queue
import numpy as np
import multiprocessing as mp
import multiprocessing.pool
import gc
import time
import sys

import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def main():
    g_cpu_pool = mp.pool.Pool()
    g_io_pool = mp.pool.ThreadPool()
    g_merge_pool = mp.pool.ThreadPool()
    full_fp = buzz.Footprint(tl=(0, 0), size=(12691, 12691), rsize=(126910, 126910))

    def compute_data(fp, *args):
        return np.random.rand(fp.shape[0], fp.shape[1])

    def gaussian_filter(fp, *data):
        array, = data
        assert np.array_equal(fp.shape, array.shape)
        return ndi.gaussian_filter(array, 0.5)

    def median_filter(fp, *data):
        array, = data
        return ndi.median_filter(array, size=5)

    def sobel_filter(fp, *data):
        array, = data
        return ndi.sobel(array)

    def summ(fp, *data):
        array1, array2 = data
        return array1 + array2

    def dot(fp, *data):
        array1, array2 = data

        return (array1 + array2)**0.5
 
    random_raster = raster_factory(footprint=full_fp,
                         dtype='float32',
                         computation_function=compute_data,
                         io_pool=g_io_pool,
                         computation_pool=g_cpu_pool,
                         to_collect_of_to_compute=None,
                         merge_pool=g_merge_pool
                        )

    
    g_filter_raster = raster_factory(footprint=full_fp,
        dtype='float32',
        computation_function=gaussian_filter,
        io_pool=g_io_pool,
        computation_pool=g_cpu_pool,
        primitives={"random": random_raster.get_multi_data_queue},
        to_collect_of_to_compute=lambda fp: {"random": fp},
        merge_pool=g_merge_pool
        )

    m_filter_raster = raster_factory(footprint=full_fp,
        dtype='float32',
        computation_function=median_filter,
        io_pool=g_io_pool,
        computation_pool=g_cpu_pool,
        primitives={"random": random_raster.get_multi_data_queue},
        to_collect_of_to_compute=lambda fp: {"random": fp},
        merge_pool=g_merge_pool
        )

    summed_raster = raster_factory(footprint=full_fp,
        dtype='float32',
        computation_function=summ,
        io_pool=g_io_pool,
        computation_pool=g_cpu_pool,
        primitives={"gaussian": g_filter_raster.get_multi_data_queue, "median":m_filter_raster.get_multi_data_queue},
        to_collect_of_to_compute=lambda fp: {"gaussian": fp, "median": fp},
        merge_pool=g_merge_pool
        )


    sobel_raster = raster_factory(footprint=full_fp,
        dtype='float32',
        computation_function=sobel_filter,
        io_pool=g_io_pool,
        computation_pool=g_cpu_pool,
        primitives={"gaussian": g_filter_raster.get_multi_data_queue},
        to_collect_of_to_compute=lambda fp: {"gaussian": fp},
        merge_pool=g_merge_pool
        )


    dot_raster = sobel_raster = raster_factory(footprint=full_fp,
        dtype='float32',
        computation_function=dot,
        io_pool=g_io_pool,
        computation_pool=g_cpu_pool,
        primitives={"summ": summed_raster.get_multi_data_queue, "sobel": sobel_raster.get_multi_data_queue},
        to_collect_of_to_compute=lambda fp: {"summ": fp, "sobel": fp},
        merge_pool=g_merge_pool
        )

    tile_count = np.ceil(full_fp.rsize / 1000)
    tiles = full_fp.tile_count(50,50)

    print(tile_count)

    # for array, fp in zip(dot_raster.get_multi_data(tiles.flat), tiles.flat):
    #     plt.imshow(array)
    #     plt.show()
    #     gc.collect()

    a = dot_raster.get_multi_data(tiles.flat)
    # time.sleep(5)
    del a
    input()

if __name__ == "__main__":
    main()
