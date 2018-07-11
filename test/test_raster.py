"""
Tests of the burito library
"""

import numpy as np
import multiprocessing as mp
import multiprocessing.pool
import buzzard as buzz
import pandas as pd

from burito.raster import Raster


class RasterStateOberver():
    def __init__(self):
        self.df = pd.DataFrame()

    def callback(self, string, raster, **kwargs):
        self.df = self.df.append(dict(zip(['op_tag', 'nbands'], [string, raster.nbands])), ignore_index=True)



def test_simple_raster():
    computation_pool = mp.pool.ThreadPool(1)
    io_pool = mp.pool.ThreadPool(1)
    footprint = buzz.Footprint(tl=(0, 0), size=(10, 10), rsize=(10, 10))

    obs = RasterStateOberver()

    def compute_data(fp, *args):
        return np.zeros(fp.shape)

    simple_raster = Raster(
        footprint=footprint,
        computation_function=compute_data,
        computation_pool=computation_pool,
        io_pool=io_pool,
        debug_callback=obs.callback
    )

    array = simple_raster.get_data(footprint)

    assert np.all(array == 0)
    print(obs.df)


def test_complicated_raster_dependencies():
    computation_pool = mp.pool.ThreadPool(6)
    io_pool = mp.pool.ThreadPool(6)
    footprint = buzz.Footprint(tl=(0, 0), size=(10, 10), rsize=(10, 10))

    def ones(fp, *args):
        return np.ones(fp.shape)

    def twos(fp, *args):
        return 2*np.ones(fp.shape)

    def arange(fp, *args):
        return np.arange(fp.shape[0]*fp.shape[1]).reshape(fp.shape)

    def summ(fp, data, *args):
        summed = data[0] + data[1] + data[2]
        assert np.array_equal(summed.shape, fp.shape)
        return summed

    def diff(fp,  data, *args):
        diffed = data[0] - data[1] - data[2]
        assert np.array_equal(diffed.shape, fp.shape)
        return diffed

    def product(fp, data, *args):
        producted = data[0] * data[1] * data[2]
        assert np.array_equal(producted.shape, fp.shape)
        return producted

    ones_raster = Raster(
        footprint=footprint,
        computation_function=ones,
        computation_pool=computation_pool,
        io_pool=io_pool
    )

    twos_raster = Raster(
        footprint=footprint,
        computation_function=twos,
        computation_pool=computation_pool,
        io_pool=io_pool
    )

    arange_raster = Raster(
        footprint=footprint,
        computation_function=arange,
        computation_pool=computation_pool,
        io_pool=io_pool
    )

    primitives = {
        "ones": ones_raster.get_multi_data_queue,
        "twos": twos_raster.get_multi_data_queue,
        "arange": arange_raster.get_multi_data_queue
    }

    def to_collect_of_to_compute(fp):
        return {"ones": fp, "twos": fp, "arange": fp}

    summ_raster = Raster(
        footprint=footprint,
        computation_function=summ,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives=primitives,
        to_collect_of_to_compute=to_collect_of_to_compute
    )

    diff_raster = Raster(
        footprint=footprint,
        computation_function=diff,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives=primitives,
        to_collect_of_to_compute=to_collect_of_to_compute
    )

    product_raster = Raster(
        footprint=footprint,
        computation_function=product,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives=primitives,
        to_collect_of_to_compute=to_collect_of_to_compute
    )

    summ_array = summ_raster.get_data(footprint)
    diff_array = diff_raster.get_data(footprint)
    prod_array = product_raster.get_data(footprint)
    prod2 = product_raster.get_data(footprint)

    assert summ_array[0, 1] == 4
    assert diff_array[0, 1] == -2
    assert prod_array[0, 1] == 2
    assert np.array_equal(prod_array, prod2)


def test_simple_cached():
    computation_pool = mp.pool.ThreadPool(1)
    io_pool = mp.pool.ThreadPool(1)
    footprint = buzz.Footprint(tl=(0, 0), size=(10, 10), rsize=(10, 10))

    obs = RasterStateOberver()

    def compute_data(fp, *args):
        return np.zeros(fp.shape)

    simple_raster = Raster(
        footprint=footprint,
        computation_function=compute_data,
        computation_pool=computation_pool,
        io_pool=io_pool,
        cached=True,
        cache_dir='./test_cache/',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink'),
        overwrite=True,
        debug_callback=obs.callback,
    )

    array = simple_raster.get_data(footprint)

    assert np.all(array == 0)
    # print(hej)



def test_concurrent_cached():
    computation_pool = mp.pool.ThreadPool(1)
    io_pool = mp.pool.ThreadPool(1)
    footprint = buzz.Footprint(tl=(0, 0), size=(10, 10), rsize=(10, 10))



    def compute_data(fp, *args):
        return np.zeros(fp.shape)

    simple_raster = Raster(
        footprint=footprint,
        computation_function=compute_data,
        computation_pool=computation_pool,
        io_pool=io_pool,
        cached=True,
        cache_dir='./test_cache/0/',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink'),
        overwrite=True
    )

    dependent_raster_1 = Raster(
        footprint=footprint,
        computation_function=compute_data,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives={"prim": simple_raster.get_multi_data_queue},
        to_collect_of_to_compute=lambda fp: {"prim": fp},
        cached=True,
        cache_dir='./test_cache/1/',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink'),
        overwrite=True
    )

    dependent_raster_2 = Raster(
        footprint=footprint,
        computation_function=compute_data,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives={"prim": simple_raster.get_multi_data_queue},
        to_collect_of_to_compute=lambda fp: {"prim": fp},
        cached=True,
        cache_dir='./test_cache/2/',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink'),
        overwrite=True
    )

    arrays1 = dependent_raster_1.get_multi_data(footprint.tile_count(5, 5).flat)
    arrays2 = dependent_raster_2.get_multi_data(reversed(list(footprint.tile_count(5, 5).flat)))

    assert np.all(next(arrays1) == 0)
    assert np.all(next(arrays2) == 0)


def test_complicated_cached_dependencies():

    computation_pool = mp.pool.ThreadPool(6)
    io_pool = mp.pool.ThreadPool(6)
    footprint = buzz.Footprint(tl=(0, 0), size=(10, 10), rsize=(10, 10))

    def ones(fp, *args):
        return np.ones(fp.shape)

    def twos(fp, *args):
        return 2*np.ones(fp.shape)

    def arange(fp, *args):
        return np.arange(fp.shape[0]*fp.shape[1]).reshape(fp.shape)

    def summ(fp, data, *args):
        summed = data[0] + data[1] + data[2]
        assert np.array_equal(summed.shape, fp.shape)
        return summed

    def diff(fp,  data, *args):
        diffed = data[0] - data[1] - data[2]
        assert np.array_equal(diffed.shape, fp.shape)
        return diffed

    def product(fp, data, *args):
        producted = data[0] * data[1] * data[2]
        assert np.array_equal(producted.shape, fp.shape)
        return producted

    ones_raster = Raster(
        footprint=footprint,
        computation_function=ones,
        computation_pool=computation_pool,
        io_pool=io_pool,
        cached=True,
        cache_dir='./test_cache/compl/1',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink')
    )

    twos_raster = Raster(
        footprint=footprint,
        computation_function=twos,
        computation_pool=computation_pool,
        io_pool=io_pool,
        cached=True,
        cache_dir='./test_cache/compl/2',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink')
    )

    arange_raster = Raster(
        footprint=footprint,
        computation_function=arange,
        computation_pool=computation_pool,
        io_pool=io_pool,
        cached=True,
        cache_dir='./test_cache/compl/3',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink')
    )

    primitives = {
        "ones": ones_raster.get_multi_data_queue,
        "twos": twos_raster.get_multi_data_queue,
        "arange": arange_raster.get_multi_data_queue
    }

    def to_collect_of_to_compute(fp):
        return {"ones": fp, "twos": fp, "arange": fp}

    summ_raster = Raster(
        footprint=footprint,
        computation_function=summ,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives=primitives,
        to_collect_of_to_compute=to_collect_of_to_compute,
        cached=True,
        cache_dir='./test_cache/compl/4',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink')
    )

    diff_raster = Raster(
        footprint=footprint,
        computation_function=diff,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives=primitives,
        to_collect_of_to_compute=to_collect_of_to_compute,
        cached=True,
        cache_dir='./test_cache/compl/5',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink')
    )

    product_raster = Raster(
        footprint=footprint,
        computation_function=product,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives=primitives,
        to_collect_of_to_compute=to_collect_of_to_compute,
        cached=True,
        cache_dir='./test_cache/compl/6',
        cache_fps=footprint.tile_count(3, 3, boundary_effect='shrink')
    )

    summ_array = summ_raster.get_data(footprint)
    diff_array = diff_raster.get_data(footprint)
    prod_array = product_raster.get_data(footprint)
    prod2 = product_raster.get_data(footprint)

    assert summ_array[0, 1] == 4
    assert diff_array[0, 1] == -2
    assert prod_array[0, 1] == 2
    assert np.array_equal(prod_array, prod2)


if __name__ == '__main__':
    test_simple_raster()
    # test_complicated_raster_dependencies()

    # test_simple_cached()
    # test_concurrent_cached()
    # test_complicated_cached_dependencies()
