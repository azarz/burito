"""
Multi-threaded, back pressure management, caching
"""

from pathlib import Path
import multiprocessing as mp
import multiprocessing.pool
import os
import threading
import time
from collections import defaultdict
import glob
import shutil
import sys
import queue

import numpy as np
import networkx as nx
import buzzard as buzz

from Query import Query
from SingletonCounter import SingletonCounter
from checksum import checksum, checksum_file
from Raster import Raster

threadPoolTaskCounter = SingletonCounter()

def _get_graph_uid(fp, _type):
    return hash(repr(fp) + _type)


raster = Raster()

def _pressure_ratio_test():
    query = Query(queue.Queue(5))
    assert raster._pressure_ratio(query) == 0

def len_test():
    assert len(raster) == raster._num_bands

def _collect_data(self, to_collect):
    """
    collects data from primitives
    in: {
       "prim_1": [to_collect_p1_1, ..., to_collect_p1_n],
       ...,
       'prim_p": [to_collect_pp_1, ..., to_collect_pp_n]
    }
    out: {"prim_1": queue_1, "prim_2": queue_2, ..., "prim_p": queue_p}
    """
    print(self.__class__.__name__, " collecting ", threading.currentThread().getName())
    results = {}
    for primitive in self._primitives.keys():
        results[primitive] = self._primitives[primitive](to_collect[primitive])
    return results

def _to_compute_of_to_produce(self, fp):
    to_compute_list = []
    for computation_tile in self._computation_tiles.flat:
        if fp.share_area(computation_tile):
            to_compute_list.append(computation_tile)

    return to_compute_list


def get_multi_data_queue(self, fp_iterable, queue_size=5):
    """
    returns a queue (could be generator) from a fp_iterable
    """
    fp_iterable = list(fp_iterable)
    q = queue.Queue(queue_size)
    query = Query(q)
    to_produce = [(fp, "sleeping") for fp in fp_iterable]

    query.to_produce += to_produce

    self._queries.append(query)
    self._new_queries.append(query)

    return q


def get_multi_data(self, fp_iterable, queue_size=5):
    """
    returns a  generator from a fp_iterable
    """
    fp_iterable = list(fp_iterable)
    out_queue = self.get_multi_data_queue(fp_iterable, queue_size)

    def out_generator():
        """
        output generator from the queue. next() waits the .get() to end
        """
        for fp in fp_iterable:
            assert fp.same_grid(self.fp)
            result = out_queue.get()
            assert np.array_equal(fp.shape, result.shape[0:2])
            yield result

    return out_generator()


def get_data(self, fp):
    """
    returns a np array
    """
    return next(self.get_multi_data([fp]))