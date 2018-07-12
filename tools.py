"""
Multi-threaded, back pressure management, caching
"""

from pathlib import Path
import multiprocessing as mp
import multiprocessing.pool
import os
import threading
import time
import datetime
from collections import defaultdict
import glob
import sys
import queue
import itertools
import collections
import uuid

import numpy as np
import networkx as nx
import buzzard as buzz
from buzzard import _tools
import rtree.index
from osgeo import gdal, osr
from buzzard._tools import conv
import names



class DebugCtxMngmnt(object):
    def __init__(self, functions, raster):
        self._functions = functions
        self.raster = raster

    def __call__(self, string, **kwargs):
        self.string = string
        self.kwargs = kwargs
        return self

    def __enter__(self):
        if self._functions is not None:
            for function in self._functions:
                function(self.string + "::before", self.raster, **self.kwargs)

    def __exit__(self, _, __, ___):
        if self._functions is not None:
            for function in self._functions:
                function(self.string + "::after", self.raster, **self.kwargs)


class GetDataWithPrimitive(object):
    """Used to retrieve the context of a get_multi_data_queue function"""
    def __init__(self, obj, function):
        self._primitive = obj
        self._function = function

    def __call__(self, fp_iterable, band=-1, queue_size=5):
        return self._function(fp_iterable, band, queue_size)

    @property
    def primitive(self):
        """
        Returns the primitive raster
        """
        return self._primitive



class Singleton(type):
    """
    implementation of the singleton design pattern
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonCounter(defaultdict, metaclass=Singleton):
    """
    the threadpool task counter is a defaultdict(int) applying the singleton pattern
    """
    def __init__(self, *args):
        if args:
            super().__init__(*args)
        else:
            super().__init__(int)





name_dict = SingletonCounter()

global thread_pool_task_counter
thread_pool_task_counter = SingletonCounter()

global qryids
qryids = collections.defaultdict(lambda: f'{len(qryids):02d}')
global queids
queids = collections.defaultdict(lambda: f'{len(queids):02d}')

def get_uname():
    def int_to_roman(input):
        """ Convert an integer to a Roman numeral. """
        ints = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
        nums = ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
        result = []
        for i, _ in enumerate(ints):
            count = int(input / ints[i])
            result.append(nums[i] * count)
            input -= ints[i] * count
        return ''.join(result)

    name = names.get_first_name()
    name_dict[name] += 1

    if name_dict[name] == 1:
        return name
    else:
        return name + ' ' + int_to_roman(name_dict[name])


_debug_lock = threading.RLock()

def qeinfo(q):
    with _debug_lock:
        return f'{queids[q] if q is not None else None}'

def qrinfo(query):
    with _debug_lock:
        q = query.produced()
        return f'qr:{qryids[query]} in:{[qeinfo(q) for q in query.collected.values()]!s:6} out:{qeinfo(query.produced())}'


def is_tiling_valid(fp, tiles):
    tiles = list(tiles)
    assert isinstance(tiles[0], buzz.Footprint)

    if any(not tile.same_grid(fp) for tile in tiles):
        print("not same grid")
        return False

    idx = rtree.index.Index()
    bound_inset = np.r_[
        1 / 4,
        1 / 4,
        -1 / 4,
        -1 / 4,
    ]

    tls = fp.spatial_to_raster([tile.tl for tile in tiles])
    rsizes = np.array([tile.rsize for tile in tiles])

    if np.any(tls < 0):
        print("tiles out of fp")
        return False

    if np.any(tls[:, 0] + rsizes[:, 0] > fp.rw):
        print("tiles out of fp")
        return False

    if np.any(tls[:, 1] + rsizes[:, 1] > fp.rh):
        print("tiles out of fp")
        return False

    if np.prod(rsizes, axis=1).sum() != fp.rarea:
        print("tile area wrong")
        print(rsizes.shape)
        print(np.prod(rsizes, axis=1).sum())
        print(np.prod(rsizes, axis=1).shape)
        print(fp.rarea)
        return False

    for i, (tl, rsize) in enumerate(zip(tls, rsizes)):
        bounds = (*tl, *(tl + rsize))
        bounds += bound_inset

        if len(list(idx.intersection(bounds))) > 0:
            print("tiles overlap")
            return False

        else:
            idx.insert(i, bounds)

    return True