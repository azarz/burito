"""
Tests of the rammuth library
"""

import numpy as np
import multiprocessing as mp
import multiprocessing.pool
import buzzard as buzz

from rammuth.raster import raster_factory


def test_simple_raster():
    computation_pool = mp.pool.ThreadPool(1)
    io_pool = mp.pool.ThreadPool(1)
    footprint = buzz.Footprint(tl=(0, 0), size=(10, 10), rsize=(10, 10))

    def compute_data(fp, *data):
        return np.zeros(fp.shape)

    simple_raster = raster_factory(
        footprint=footprint,
        computation_function=compute_data,
        computation_pool=computation_pool,
        io_pool=io_pool
    )

    array = simple_raster.get_data(footprint)

    assert np.all(array == 0)


def test_complicated_raster_dependencies():
    computation_pool = mp.pool.ThreadPool(6)
    io_pool = mp.pool.ThreadPool(6)
    footprint = buzz.Footprint(tl=(0, 0), size=(10, 10), rsize=(10, 10))
    
    def ones(fp, *data):
        return np.ones(fp.shape)

    def twos(fp, *data):
        return 2*np.ones(fp.shape)

    def arange(fp, *data):
        return np.arange(fp.shape[0]*fp.shape[1]).reshape(fp.shape)

    def summ(fp, *data):
        summed = data[0] + data[1] + data[2]
        assert np.array_equal(summed.shape, fp.shape)
        return summed

    def diff(fp, *data):
        diffed = data[0] - data[1] - data[2]
        assert np.array_equal(diffed.shape, fp.shape)
        return diffed

    def product(fp, *data):
        producted = data[0] * data[1] * data[2]
        assert np.array_equal(producted.shape, fp.shape)
        return producted

    ones_raster = raster_factory(
        footprint=footprint,
        computation_function=ones,
        computation_pool=computation_pool,
        io_pool=io_pool
    )

    twos_raster = raster_factory(
        footprint=footprint,
        computation_function=twos,
        computation_pool=computation_pool,
        io_pool=io_pool
    )

    arange_raster = raster_factory(
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

    summ_raster = raster_factory(
        footprint=footprint,
        computation_function=summ,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives=primitives,
        to_collect_of_to_compute=to_collect_of_to_compute
    )

    diff_raster = raster_factory(
        footprint=footprint,
        computation_function=diff,
        computation_pool=computation_pool,
        io_pool=io_pool,
        primitives=primitives,
        to_collect_of_to_compute=to_collect_of_to_compute
    )

    product_raster = raster_factory(
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

    assert summ_array[0, 1] == 4
    assert diff_array[0, 1] == -2
    assert prod_array[0, 1] == 2

if __name__ == '__main__':
    test_complicated_raster_dependencies()
