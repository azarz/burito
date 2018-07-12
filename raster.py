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

from burito.uids_of_paths import md5
from burito.query import Query
from burito.singleton_counter import SingletonCounter
from burito.get_data_with_primitive import GetDataWithPrimitive


thread_pool_task_counter = SingletonCounter()

qryids = collections.defaultdict(lambda: f'{len(qryids):02d}')
queids = collections.defaultdict(lambda: f'{len(queids):02d}')
_debug_lock = threading.RLock()



class DebugCtxMngmnt(object):
    def __init__(self, function, raster):
        self._function = function
        self.raster = raster

    def __call__(self, string, **kwargs):
        self.string = string
        self.kwargs = kwargs
        return self

    def __enter__(self):
        if self._function is not None:
            self._function(self.string + "::before", self.raster, **self.kwargs)

    def __exit__(self, _, __, ___):
        if self._function is not None:
            self._function(self.string + "::after", self.raster, **self.kwargs)




def qeinfo(q):
    with _debug_lock:
        return f'{queids[q] if q is not None else None}'

def qrinfo(query):
    with _debug_lock:
        q = query.produced()
        return f'qr:{qryids[query]} in:{[qeinfo(q) for q in query.collected.values()]!s:6} out:{qeinfo(query.produced())}'



name_dict = SingletonCounter()

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



class Raster(object):
    def __init__(self,
                 footprint=None,
                 dtype='float32',
                 nbands=1,
                 nodata=None,
                 srs=None,
                 computation_function=None,
                 cached=False,
                 overwrite=False,
                 cache_dir=None,
                 cache_fps=None,
                 io_pool=None,
                 computation_pool=None,
                 primitives=None,
                 to_collect_of_to_compute=None,
                 max_computation_size=None,
                 computation_fps=None,
                 merge_pool=None,
                 merge_function=None,
                 debug_callback=None):
        """
        creates a raster from arguments
        """
        if footprint is None:
            raise ValueError("footprint must be provided")
        if cached:
            if cache_dir is None:
                raise ValueError("cache_dir must be provided when cached")
            backend_raster = BackendCachedRaster(footprint,
                                                 dtype,
                                                 nbands,
                                                 nodata,
                                                 srs,
                                                 computation_function,
                                                 overwrite,
                                                 cache_dir,
                                                 np.asarray(cache_fps),
                                                 io_pool,
                                                 computation_pool,
                                                 primitives,
                                                 to_collect_of_to_compute,
                                                 computation_fps,
                                                 merge_pool,
                                                 merge_function,
                                                 debug_callback
                                                )
        else:
            backend_raster = BackendRaster(footprint,
                                           dtype,
                                           nbands,
                                           nodata,
                                           srs,
                                           computation_function,
                                           io_pool,
                                           computation_pool,
                                           primitives,
                                           to_collect_of_to_compute,
                                           max_computation_size,
                                           merge_pool,
                                           merge_function,
                                           debug_callback
                                          )

        self._backend = backend_raster

        self._scheduler_thread = threading.Thread(target=self._backend._scheduler, daemon=True)
        self._scheduler_thread.start()

    def __del__(self):
        self._backend._stop_scheduler = True



    @property
    def fp(self):
        """
        returns the raster footprint
        """
        return self._backend.fp

    @property
    def nbands(self):
        """
        returns the raster's number of bands'
        """
        return self._backend.nbands

    @property
    def nodata(self):
        """
        returns the raster nodata
        """
        return self._backend.nodata

    @property
    def wkt_origin(self):
        """
        returns the raster wkt origin
        """
        return self._backend.wkt_origin

    @property
    def dtype(self):
        """
        returns the raster dtype
        """
        return self._backend.dtype

    @property
    def pxsizex(self):
        """
        returns the raster 1D pixel size
        """
        return self._backend.pxsizex

    @property
    def primitives(self):
        """
        Returns dictionnary of raster primitives
        """
        return self._backend.primitives


    def __len__(self):
        return len(self._backend)



    @property
    def get_multi_data_queue(self):
        """
        returns a queue from a fp_iterable and can manage the context of the method
        """

        def _get_multi_data_queue(fp_iterable, band=-1, queue_size=5):
            """
            returns a queue from a fp_iterable
            """

            # Normalize and check band parameter
            bands, is_flat = _tools.normalize_band_parameter(band, len(self), None)

            fp_iterable = list(fp_iterable)
            for fp in fp_iterable:
                assert fp.same_grid(self.fp)
            q = queue.Queue(queue_size)
            query = Query(q, bands, is_flat)
            # to_produce = [(fp, "sleeping", str(uuid.uuid4())) for fp in fp_iterable]
            to_produce = [(fp, "sleeping", get_uname()) for fp in fp_iterable]

            query.to_produce += to_produce

            # self._backend._queries.append(query)
            self._backend._new_queries.append(query)

            return q

        return GetDataWithPrimitive(self, _get_multi_data_queue)


    def get_multi_data(self, fp_iterable, band=-1, queue_size=5):
        """
        returns a  generator from a fp_iterable
        """
        fp_iterable = list(fp_iterable)
        out_queue = self.get_multi_data_queue(fp_iterable, band, queue_size)

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


    def get_data(self, fp, band=-1):
        """
        returns a np array
        """
        return next(self.get_multi_data([fp], band))



class BackendRaster(object):
    """
    class defining the raster behaviour
    """
    def __init__(self,
                 footprint,
                 dtype,
                 nbands,
                 nodata,
                 srs,
                 computation_function,
                 io_pool,
                 computation_pool,
                 primitives,
                 to_collect_of_to_compute,
                 max_computation_size,
                 merge_pool,
                 merge_function,
                 debug_callback):

        self._debug_watcher = DebugCtxMngmnt(debug_callback, self)

        self._full_fp = footprint
        if computation_function is None:
            raise ValueError("computation function must be provided")
        self._compute_data = computation_function
        self._dtype = dtype
        self._num_bands = nbands
        self._nodata = nodata
        self._wkt_origin = srs

        if io_pool is None:
            self._io_pool = mp.pool.ThreadPool()
        elif isinstance(io_pool, int):
            self._io_pool = mp.pool.ThreadPool(io_pool)
        else:
            self._io_pool = io_pool

        if computation_pool is None:
            self._computation_pool = mp.pool.ThreadPool()
        elif isinstance(computation_pool, int):
            self._computation_pool = mp.pool.ThreadPool(computation_pool)
        else:
            self._computation_pool = computation_pool

        if merge_pool is None:
            self._merge_pool = mp.pool.ThreadPool()
        elif isinstance(merge_pool, int):
            self._merge_pool = mp.pool.ThreadPool(merge_pool)
        else:
            self._merge_pool = merge_pool

        if primitives is None:
            primitives = {}

        self._primitive_functions = primitives
        self._primitive_rasters = {
            key: (primitives[key].primitive if hasattr(primitives[key], "primitive") else None)
            for key in primitives
        }


        if primitives.keys() and to_collect_of_to_compute is None:
            raise ValueError("must provide to_collect_of_to_compute when having primitives")
        self._to_collect_of_to_compute = to_collect_of_to_compute

        self._max_computation_size = max_computation_size

        self._queries = []
        self._new_queries = []

        self._graph = nx.DiGraph()

        def default_merge_data(out_fp, in_fps, in_arrays):
            """
            Default merge function: burning
            """
            out_data = np.full(
                tuple(out_fp.shape) + (self._num_bands,),
                self.nodata or 0,
                dtype=self.dtype
            )
            for to_burn_fp, to_burn_data in zip(in_fps, in_arrays):
                out_data[to_burn_fp.slice_in(out_fp, clip=True)] = to_burn_data[out_fp.slice_in(to_burn_fp, clip=True)]
            return out_data

        if merge_function is None:
            self._merge_data = default_merge_data
        else:
            self._merge_data = merge_function

        # Used to track the number of pending tasks
        self._num_pending = defaultdict(int)

        self._stop_scheduler = False


    def _pressure_ratio(self, query):
        """
        defines a pressure ration of a query: lesser values -> emptier query
        """
        if query.produced() is None:
            return -1
        num = query.produced().qsize() + self._num_pending[id(query)]
        den = query.produced().maxsize
        return num/den
        # return np.random.rand()

    @property
    def fp(self):
        """
        returns the raster footprint
        """
        return self._full_fp

    @property
    def nbands(self):
        """
        returns the raster's number of bands'
        """
        return self._num_bands

    @property
    def nodata(self):
        """
        returns the raster nodata
        """
        return self._nodata

    @property
    def wkt_origin(self):
        """
        returns the raster wkt origin
        """
        return self._wkt_origin

    @property
    def dtype(self):
        """
        returns the raster dtype
        """
        return self._dtype

    @property
    def pxsizex(self):
        """
        returns the raster 1D pixel size
        """
        return self._full_fp.pxsizex

    @property
    def primitives(self):
        """
        Returns dictionnary of raster primitives
        """
        return self._primitive_rasters.copy()


    def __len__(self):
        return int(self._num_bands)


    def _scheduler(self):
        header = '{!s:25}'.format('<prim[{}] {} {}>'.format(
            ','.join(self._primitive_functions.keys()),
            self._num_bands,
            self.dtype,
        ))
        # header = f"{list(self._primitive_functions.keys())!s:20} {self._num_bands!s:3} {self.dtype!s:10}"
        self.h = header
        print(header, "scheduler in")
        # list of available and produced to_collect footprints
        available_to_produce = set()
        put_counter = collections.Counter()
        get_counter = collections.Counter()

        while True:
            # time.sleep(0.05)
            skip = False

            if self._stop_scheduler:
                print("going to sleep")
                return

            assert len(set(map(id, self._queries))) == len(self._queries)
            assert len(set(map(id, self._new_queries))) == len(self._new_queries)

            # Consuming the new queries
            while self._new_queries:
                with self._debug_watcher("scheduler::new_query"):
                    query = self._new_queries.pop(0)

                    # if cached raster, checking the cache
                    if isinstance(self, BackendCachedRaster):
                        # adding the queries to check
                        unique_to_read_fps = set()
                        for to_produce in query.to_produce:
                            unique_to_read_fps |= set(self._to_read_of_to_produce(to_produce[0]))
                        query.to_check = list(unique_to_read_fps)

                    self._queries.append(query)

                    print(self.h, qrinfo(query), f'new query with {len(query.to_produce)} to_produce arrived')
                    skip = True
                    break

            # ordering queries accroding to their pressure
            assert len(set(map(id, self._queries))) == len(self._queries)
            assert len(set(map(id, self._new_queries))) == len(self._new_queries)

            ordered_queries = sorted(self._queries, key=self._pressure_ratio)

            # getting the emptiest query
            for query in ordered_queries:
                if skip:
                    break

                # If the query has been dropped
                if query.produced() is None:
                    with self._debug_watcher("scheduler::cleaning_dropped_query"):
                        print(self.h, qrinfo(query), f'cleaning: dropped by main program')
                        if self._num_pending[id(query)]: # could be false because dropped too early
                            del self._num_pending[id(query)]
                        if query.was_included_in_graph:
                            to_delete_nodes = list(nx.dfs_postorder_nodes(self._graph, source=id(query)))
                            for node_id in to_delete_nodes:
                                node = self._graph.nodes[node_id]
                                node["linked_queries"].remove(query)
                                if not node["linked_queries"]:
                                    self._graph.remove_node(node_id)
                        self._queries.remove(query)

                        skip = True
                        break

                # If there are still fps to check
                if query.to_check and thread_pool_task_counter[id(self._io_pool)] < self._io_pool._processes:
                    assert isinstance(self, BackendCachedRaster)
                    with self._debug_watcher("scheduler::starting_check_fp"):
                        to_check_fp = query.to_check.pop(0)
                        index = self._indices_of_cache_tiles[to_check_fp]

                        if self._cache_checksum_array[index] is None:
                            print(self.h, qrinfo(query), f'checking a cache footprint')
                            query.checking.append((index, self._io_pool.apply_async(self._check_cache_file, (to_check_fp, ))))
                            thread_pool_task_counter[id(self._io_pool)] += 1

                            skip = True
                            break

                # If there are still fps currently being checked
                if query.checking:
                    assert isinstance(self, BackendCachedRaster)
                    for still_checking in query.checking:
                        if still_checking[1].ready():
                            with self._debug_watcher("scheduler::ending_check_fp"):
                                print(self.h, qrinfo(query), f'checked a cache footprint')
                                self._cache_checksum_array[still_checking[0]] = still_checking[1].get()
                                thread_pool_task_counter[id(self._io_pool)] -= 1
                                query.checking.remove(still_checking)

                                skip = True
                                break

                # If there are still fps to check or currently being checked, skipping query
                if query.to_check or query.checking:
                    assert isinstance(self, BackendCachedRaster)
                    continue

                if not query.was_included_in_graph:
                    with self._debug_watcher("scheduler::updating_graph"):
                        self._update_graph_from_query(query)
                        query.was_included_in_graph = True
                        print(self.h, qrinfo(query), f'new query with {list(len(p) for p in query.to_collect.values())} to_collect was added to graph')
                        skip = True
                        break

                # If all to_produced was consumed: query ended
                if not query.to_produce:
                    with self._debug_watcher("scheduler::cleaning_ended_query"):
                        print(self.h, qrinfo(query), f'cleaning: treated all produce')
                        del self._num_pending[id(query)]
                        self._graph.remove_node(id(query))
                        self._queries.remove(query)

                        skip = True
                        break

                isolates_not_query = [
                    node_id for node_id in list(nx.isolates(self._graph))
                    if node_id not in list(map(id, self._queries))
                ]
                # checking if the graph was correctly cleaned
                assert len(isolates_not_query) == 0, isolates_not_query

                # if the emptiest query is full, waiting
                if query.produced().full():
                    continue

                # detecting which produce footprints are available
                # while there is space
                while query.produced().qsize() + self._num_pending[id(query)] < query.produced().maxsize and query.to_produce[-1][1] == "sleeping":
                    with self._debug_watcher("scheduler::produce_sleeping_to_pending"):
                        # getting the first sleeping to_produce
                        first_sleeping_i = [to_produce[1] for to_produce in query.to_produce].index('sleeping')
                        to_produce_available = query.to_produce[first_sleeping_i][0]

                        # getting its id in the graph
                        to_produce_available_id = query.to_produce[first_sleeping_i][2]

                        available_to_produce.add(to_produce_available_id)

                        to_produce_index = query.to_produce.index((to_produce_available, "sleeping", to_produce_available_id))
                        query.to_produce[to_produce_index] = (to_produce_available, "pending", to_produce_available_id)

                        self._num_pending[id(query)] += 1

                        assert query.produced().qsize() + self._num_pending[id(query)] <= query.produced().maxsize

                        skip = True
                        break

                # getting the in_queue of data to discard
                for primitive in query.collected:
                    if not query.collected[primitive].empty() and query.to_collect[primitive][0] in query.to_discard[primitive]:
                        with self._debug_watcher("scheduler::discard"):
                            query.collected[primitive].get(block=False)
                            skip = True
                            break

                # iterating through the graph
                for index, to_produce in enumerate(query.to_produce):
                    if skip:
                        break

                    if to_produce[1] == "sleeping":
                        continue

                    # beginning at to_produce
                    first_node_id = to_produce[2]

                    # going as deep as possible
                    depth_node_ids = iter(nx.dfs_postorder_nodes(self._graph, source=first_node_id))

                    while True:
                        try:
                            node_id = next(depth_node_ids)
                        except StopIteration:
                            break

                        node = self._graph.nodes[node_id]

                        # If there are out edges, not stopping (unless it is a compute node)
                        if len(self._graph.out_edges(node_id)) > 0:
                            continue

                        # Skipping the nodes not linked to available (pending) to_produce
                        if available_to_produce.isdisjoint(node["linked_to_produce"]):
                            continue

                        # if deepest is to_compute, collecting (if possible) and computing
                        if node["type"] == "to_compute" and node["future"] is None:
                            # testing if at least 1 of the collected queues is empty (1 queue per primitive)
                            if any([query.collected[primitive].empty() for primitive in query.collected]):
                                continue

                            # asserting it's the 1st to_compute
                            if query.to_compute.index(node['footprint']) != 0:
                                continue

                            if thread_pool_task_counter[id(node["pool"])] < node["pool"]._processes:
                                with self._debug_watcher("scheduler::compute"):
                                    collected_data = []
                                    primitive_footprints = []

                                    get_counter[query] += 1
                                    print(self.h, qrinfo(query), f'compute data for the {get_counter[query]:02d}th time node_id:({node_id})')

                                    for collected_primitive in query.collected.keys():
                                        collected_data.append(query.collected[collected_primitive].get(block=False))
                                        primitive_footprints.append(query.to_collect[collected_primitive].pop(0))

                                    assert len(collected_data) == len(self._primitive_functions.keys())

                                    node["future"] = self._computation_pool.apply_async(
                                        self._compute_data,
                                        (
                                            node["footprint"],
                                            collected_data,
                                            primitive_footprints,
                                            self
                                        )
                                    )

                                    thread_pool_task_counter[id(self._computation_pool)] += 1
                                    query.to_compute.pop(0)
                                    node["linked_queries"].remove(query)

                                    for linked_query in node["linked_queries"]:
                                        for collected_primitive, primitive_footprint in zip(query.collected.keys(), primitive_footprints):
                                            linked_query.to_discard[collected_primitive].append(primitive_footprint)

                                    skip = True
                                    break

                        # if the deepest is to_produce, updating produced
                        if index == 0 and node["type"] == "to_produce":
                            with self._debug_watcher("scheduler::put_data"):
                                # If the query has not been dropped
                                if query.produced() is not None:
                                    assert not query.produced().full()
                                    if node["is_flat"]:
                                        node["in_data"] = node["in_data"].squeeze(axis=-1)
                                    query.produced().put(node["in_data"].astype(self._dtype), timeout=1e-2)

                                query.to_produce.pop(0)

                                put_counter[query] += 1
                                print(self.h, qrinfo(query), f'    put data for the {put_counter[query]:02d}th time, {len(query.to_produce):02d} left')

                                self._graph.remove_node(node_id)
                                self._num_pending[id(query)] -= 1

                                skip = True
                                break

                        # skipping the ready to_produce that are not at index 0
                        if node["type"] == "to_produce":
                            continue

                        if node["type"] == "to_merge" and node["future"] is None:
                            if thread_pool_task_counter[id(self._merge_pool)] < self._merge_pool._processes:
                                with self._debug_watcher("scheduler::starting_merge"):
                                    node["future"] = self._merge_pool.apply_async(
                                        self._merge_data,
                                        (
                                            node["footprint"],
                                            node["in_fp"],
                                            node["in_data"]
                                        )
                                    )
                                    thread_pool_task_counter[id(self._merge_pool)] += 1
                                    assert thread_pool_task_counter[id(self._merge_pool)] <= self._merge_pool._processes

                                    skip = True
                                    break

                        in_edges = list(self._graph.in_edges(node_id))

                        if node["future"] is None:
                            if thread_pool_task_counter[id(node["pool"])] < node["pool"]._processes:
                                if node["type"] == "to_read":
                                    with self._debug_watcher("scheduler::starting_read"):
                                        assert len(in_edges) == 1
                                        in_edge = in_edges[0]
                                        produce_node = self._graph.nodes[in_edge[0]]
                                        if produce_node["in_data"] is None:
                                            produce_node["in_data"] = np.full(
                                                tuple(produce_node["footprint"].shape) + (len(query.bands),),
                                                self.nodata or 0,
                                                dtype=self.dtype
                                            )
                                        node["future"] = self._io_pool.apply_async(
                                            self._read_cache_data,
                                            (
                                                node["footprint"],
                                                produce_node["footprint"],
                                                produce_node["in_data"],
                                                node["bands"]
                                            )
                                        )
                                else:
                                    assert node["type"] == "to_write"
                                    with self._debug_watcher("scheduler::starting_write"):
                                        node["future"] = node["pool"].apply_async(
                                            self._write_cache_data,
                                            (
                                                node["footprint"],
                                                node["in_data"]
                                            )
                                        )
                                    thread_pool_task_counter[id(node["pool"])] += 1

                                skip = True
                                break

                        elif node["future"].ready():
                            with self._debug_watcher("scheduler::ending_" + node["type"] + "_operation"):
                                in_data = node["future"].get()
                                if node["type"] == "to_write":
                                    self._cache_checksum_array[self._indices_of_cache_tiles[node["footprint"]]] = True
                                if in_data is not None:
                                    in_data = in_data.astype(self._dtype).reshape(tuple(node["footprint"].shape) + (len(node["bands"]),))
                                thread_pool_task_counter[id(node["pool"])] -= 1

                                for in_edge in in_edges:
                                    in_node = self._graph.nodes[in_edge[0]]
                                    if in_node["type"] == "to_merge":
                                        in_node["in_data"].append(in_data)
                                        in_node["in_fp"].append(node["footprint"])
                                    elif in_node["type"] == "to_produce" and node["type"] == "to_read":
                                        pass
                                    else:
                                        in_node["in_data"] = in_data
                                    self._graph.remove_edge(*in_edge)

                                self._graph.remove_node(node_id)

                                skip = True
                                break

            if not skip:
                if not self._queries:
                    time.sleep(0.2)
                else:
                    time.sleep(0.1)



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
        # print(self.h, "collecting")
        results = {}
        for primitive in self._primitive_functions.keys():
            results[primitive] = self._primitive_functions[primitive](to_collect[primitive])
        return results

    def _to_compute_of_to_produce(self, fp):
        count = np.ceil(fp.rsize / self._max_computation_size)
        tiles = fp.tile_count(*count)
        return list(tiles.flat)


    def _update_graph_from_query(self, new_query):
        """
        Updates the dependency graph from the new queries (NO CACHE!)
        """

        # {
        #    "p1": [to_collect_p1_1, ..., to_collect_p1_n],
        #    ...,
        #    "pp": [to_collect_pp_1, ..., to_collect_pp_n]
        # }
        # with p # of primitives and n # of to_compute fps

        # initializing to_collect dictionnary
        new_query.to_collect = {key: [] for key in self._primitive_functions.keys()}
        new_query.to_discard = {key: [] for key in self._primitive_functions.keys()}

        self._graph.add_node(
            id(new_query),
            linked_queries=set([new_query]),
        )
        # time_dict = defaultdict(float)
        # counter = defaultdict(int)
        for to_produce, _, to_produce_uid in new_query.to_produce:
            # start = datetime.datetime.now()
            print(self.h, qrinfo(new_query), f'{"to_produce":>15}', to_produce_uid)
            self._graph.add_node(
                to_produce_uid,
                futures=[],
                footprint=to_produce,
                in_data=None,
                type="to_produce",
                linked_to_produce=set([to_produce_uid]),
                linked_queries=set([new_query]),
                bands=new_query.bands,
                is_flat=new_query.is_flat
            )

            self._graph.add_edge(id(new_query), to_produce_uid)
            to_merge = to_produce

            # to_merge_uid = str(uuid.uuid4())
            to_merge_uid = get_uname()

            print(self.h, qrinfo(new_query), f'    {"to_merge":>15}', to_merge_uid)

            self._graph.add_node(
                to_merge_uid,
                footprint=to_merge,
                future=None,
                futures=[],
                type="to_merge",
                pool=self._merge_pool,
                in_data=[],
                in_fp=[],
                linked_to_produce=set([to_produce_uid]),
                linked_queries=set([new_query]),
                bands=new_query.bands
            )
            self._graph.add_edge(to_produce_uid, to_merge_uid)

            if self._max_computation_size is not None:
                multi_to_compute = self._to_compute_of_to_produce(to_merge)
            else:
                multi_to_compute = [to_produce]
            # time_dict["produce"] += (datetime.datetime.now() - start).total_seconds()

            for to_compute in multi_to_compute:
                # start = datetime.datetime.now()
                # to_compute_uid = str(uuid.uuid4())
                to_compute_uid = get_uname()

                print(self.h, qrinfo(new_query), f'        {"to_compute":>15}', to_compute_uid)

                self._graph.add_node(
                    to_compute_uid,
                    footprint=to_compute,
                    future=None,
                    type="to_compute",
                    pool=self._computation_pool,
                    in_data=None,
                    linked_to_produce=set([to_produce_uid]),
                    linked_queries=set([new_query]),
                    bands=new_query.bands
                )
                new_query.to_compute.append(to_compute)

                self._graph.add_edge(to_merge_uid, to_compute_uid)

                if self._to_collect_of_to_compute is None:
                    continue
                # time_dict["compute1"] += (datetime.datetime.now() - start).total_seconds()
                # start = datetime.datetime.now()
                multi_to_collect = self._to_collect_of_to_compute(to_compute)
                # counter["multi_to_c"] += 1
                # time_dict["compute2"] += (datetime.datetime.now() - start).total_seconds()
                # start = datetime.datetime.now()
                # np.arange(10000)**2**0.5
                # time_dict["compute3"] += (datetime.datetime.now() - start).total_seconds()
                # start = datetime.datetime.now()

                if multi_to_collect.keys() != self._primitive_functions.keys():
                    raise ValueError("to_collect keys do not match primitives")

                for key in multi_to_collect:
                    new_query.to_collect[key].append(multi_to_collect[key])

            # print(time_dict)
            # print(counter)
        new_query.collected = self._collect_data(new_query.to_collect)




class BackendCachedRaster(BackendRaster):
    """
    Cached implementation of raster
    """
    def __init__(self,
                 footprint,
                 dtype,
                 nbands,
                 nodata,
                 srs,
                 computation_function,
                 overwrite,
                 cache_dir,
                 cache_tiles,
                 io_pool,
                 computation_pool,
                 primitives,
                 to_collect_of_to_compute,
                 computation_tiles,
                 merge_pool,
                 merge_function,
                 debug_callback):


        self._cache_dir = cache_dir
        if cache_tiles is None:
            raise ValueError("cache tiles must be provided")
        if not isinstance(cache_tiles, np.ndarray):
            raise ValueError("cache tiles must be in np array")
        self._cache_tiles = cache_tiles

        self._indices_of_cache_tiles = {
            self._cache_tiles[index]: index
            for index in np.ndindex(*self._cache_tiles.shape)
        }

        assert is_tiling_valid(footprint, cache_tiles.flat)

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            if overwrite:
                for path in glob.glob(cache_dir + "/*_[a-f0-9]*.tif"):
                    os.remove(path)

        if computation_tiles is None:
            computation_tiles = cache_tiles

        # Array used to track the state of cahce tiles:
        # None: not yet met
        # False: met, has to be written
        # True: met, already written and valid
        self._cache_checksum_array = np.empty(cache_tiles.shape, dtype=object)

        self._cache_idx = rtree.index.Index()
        cache_fps = list(cache_tiles.flat)
        if not isinstance(cache_fps[0], buzz.Footprint):
            raise ValueError("cache tiles must be footprints")
        pxsizex = min(fp.pxsize[0] for fp in cache_fps)
        bound_inset = np.r_[
            pxsizex / 4,
            pxsizex / 4,
            pxsizex / -4,
            pxsizex / -4,
        ]

        for i, fp in enumerate(cache_fps):
            self._cache_idx.insert(i, fp.bounds + bound_inset)

        # self._cache_priority_dict = {fp: index for index, fp in enumerate(sorted(cache_fps, key=lambda fp: (-fp.tly, fp.tlx)))}
        # self._priority_to_cache_fp_dict = {index: fp for index, fp in enumerate(sorted(cache_fps, key=lambda fp: (-fp.tly, fp.tlx)))}

        self._computation_tiles = computation_tiles

        self._computation_idx = rtree.index.Index()
        computation_fps = list(computation_tiles.flat)
        assert isinstance(computation_fps[0], buzz.Footprint)
        pxsizex = min(fp.pxsize[0] for fp in computation_fps)
        bound_inset = np.r_[
            pxsizex / 4,
            pxsizex / 4,
            pxsizex / -4,
            pxsizex / -4,
        ]

        for i, fp in enumerate(computation_fps):
            self._computation_idx.insert(i, fp.bounds + bound_inset)

        # self._compute_priority_dict = {fp: index for index, fp in enumerate(sorted(computation_fps, key=lambda fp: (-fp.tly, fp.tlx)))}
        # self._priority_to_compute_fp_dict = {index: fp for index, fp in enumerate(sorted(computation_fps, key=lambda fp: (-fp.tly, fp.tlx)))}


        super().__init__(footprint, dtype, nbands, nodata, srs,
                         computation_function,
                         io_pool,
                         computation_pool,
                         primitives,
                         to_collect_of_to_compute,
                         max_computation_size=None,
                         merge_pool=merge_pool,
                         merge_function=merge_function,
                         debug_callback=debug_callback
                        )


    def _check_cache_file(self, footprint):
        cache_tile_paths = self._get_cache_tile_path(footprint)
        result = False
        if cache_tile_paths:
            for cache_path in cache_tile_paths:
                checksum_dot_tif = cache_path.split('_')[-1]
                file_checksum = checksum_dot_tif.split('.')[0]
                if md5(cache_path) == file_checksum:
                    result = True
                else:
                    os.remove(cache_path)

        return result


    def _get_cache_tile_path_prefix(self, cache_tile):
        """
        Returns a string, which is a path to a cache tile from its fp
        """
        tile_index = self._indices_of_cache_tiles[cache_tile]
        path = str(
            Path(self._cache_dir) /
            "fullsize_{:05d}_{:05d}_tilesize_{:05d}_{:05d}_tilepxindex_{:05d}_{:05d}_tileindex_{:05d}_{:05d}".format(
                *self._full_fp.rsize,
                *cache_tile.rsize,
                *self._full_fp.spatial_to_raster(cache_tile.tl),
                tile_index[0],
                tile_index[1]
            )
        )
        return path


    def _get_cache_tile_path(self, cache_tile):
        """
        Returns a string, which is a path to a cache tile from its fp
        """
        prefix = self._get_cache_tile_path_prefix(cache_tile)
        files_paths = glob.glob(prefix + "*_[a-f0-9]*.tif")
        return files_paths


    def _read_cache_data(self, cache_tile, produce_fp, produced_data, bands):
        """
        reads cache data
        """
        # print(self.h, "reading")

        filepaths = self._get_cache_tile_path(cache_tile)

        assert len(filepaths) == 1, len(filepaths)
        filepath = filepaths[0]

        # Open a raster datasource
        options = ()
        gdal_ds = gdal.OpenEx(
            filepath,
            conv.of_of_mode('r') | conv.of_of_str('raster'),
            ['GTiff'],
            options,
        )
        if gdal_ds is None:
            raise ValueError('Could not open `{}` with `{}` (gdal error: `{}`)'.format(
                filepath, 'GTiff', gdal.GetLastErrorMsg()
            ))

        assert produce_fp.same_grid(cache_tile)

        to_read_fp = produce_fp.intersection(cache_tile)

        rtlx, rtly = cache_tile.spatial_to_raster(to_read_fp.tl)

        assert rtlx >= 0 and rtlx < cache_tile.rsizex
        assert rtly >= 0 and rtly < cache_tile.rsizey

        for band in bands:
            a = gdal_ds.GetRasterBand(band).ReadAsArray(
                int(rtlx),
                int(rtly),
                int(to_read_fp.rsizex),
                int(to_read_fp.rsizey),
                buf_obj=produced_data[to_read_fp.slice_in(produce_fp, clip=True) + (band - 1, )]
            )
            if a is None:
                raise ValueError('Could not read array (gdal error: `{}`)'.format(
                    gdal.GetLastErrorMsg()
                ))


    def _write_cache_data(self, cache_tile, data):
        """
        writes cache data
        """
        # print(self.h, "writing ")
        sr = self.wkt_origin
        filepath = os.path.join(self._cache_dir, str(uuid.uuid4()))

        dr = gdal.GetDriverByName("GTiff")
        if os.path.isfile(filepath):
            err = dr.Delete(filepath)
            if err:
                raise Exception('Could not delete %s' % filepath)

        options = ()
        gdal_ds = dr.Create(
            filepath, cache_tile.rsizex, cache_tile.rsizey, self.nbands, conv.gdt_of_any_equiv(self.dtype), options
        )
        if gdal_ds is None:
            raise Exception('Could not create gdal dataset (%s)' % gdal.GetLastErrorMsg())
        if sr is not None:
            gdal_ds.SetProjection(osr.GetUserInputAsWKT(sr))
        gdal_ds.SetGeoTransform(cache_tile.gt)


        if self.nodata is not None:
            for i in range(self.nbands):
                gdal_ds.GetRasterBand(i + 1).SetNoDataValue(self.nodata)

         # band_schema = None
        # band_schema = None

        # Check array shape
        array = np.asarray(data)

        if array.shape[:2] != tuple(cache_tile.shape):
            raise ValueError('Incompatible shape between array:%s and fp:%s' % (
                array.shape, cache_tile.shape
            )) # pragma: no cover

        # Normalize and check array shape
        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        elif array.ndim != 3:
            raise ValueError('Array has shape %d' % array.shape) # pragma: no cover
        if array.shape[-1] != self.nbands:
            raise ValueError('Incompatible band count between array:%d and band:%d' % (
                array.shape[-1], self.nbands
            )) # pragma: no cover

        # Normalize array dtype
        array = array.astype(self.dtype)

        if array.dtype == np.int8:
            array = array.astype('uint8')

        def _blocks_of_footprint(fp, bands):
            for i, band in enumerate(bands):
                yield fp, band, i # Todo use tile_count and gdal block size

        bands = list(range(1, self.nbands + 1))

        for tile, band, dim in _blocks_of_footprint(cache_tile, bands):
            tilearray = array[:, :, dim][tile.slice_in(cache_tile)]
            assert np.array_equal(tilearray.shape[0:2], cache_tile.shape)
            gdalband = gdal_ds.GetRasterBand(band)
            gdalband.WriteArray(tilearray)

        gdal_ds.FlushCache()
        del gdalband
        del gdal_ds

        file_hash = md5(filepath)

        new_file_path = self._get_cache_tile_path_prefix(cache_tile) + "_" + file_hash + ".tif"

        os.rename(filepath, new_file_path)



    def _update_graph_from_query(self, new_query):
        """
        Updates the dependency graph from the new queries
        """

        # [
        #    [to_collect_p1_1, ..., to_collect_p1_n],
        #    ...,
        #    [to_collect_pp_1, ..., to_collect_pp_n]
        # ]
        # with p # of primitives and n # of to_compute fps

        # initializing to_collect dictionnary
        new_query.to_collect = {key: [] for key in self._primitive_functions.keys()}
        new_query.to_discard = {key: [] for key in self._primitive_functions.keys()}

        self._graph.add_node(
            id(new_query),
            linked_queries=set([new_query]),
        )

        for to_produce, _, to_produce_uid in new_query.to_produce:
            print(self.h, qrinfo(new_query), f'{"to_produce":>15}', to_produce_uid)
            self._graph.add_node(
                to_produce_uid,
                footprint=to_produce,
                futures=[],
                in_data=None,
                type="to_produce",
                linked_to_produce=set([to_produce_uid]),
                linked_queries=set([new_query]),
                is_flat=new_query.is_flat,
                bands=new_query.bands
            )
            to_read_tiles = self._to_read_of_to_produce(to_produce)

            self._graph.add_edge(id(new_query), to_produce_uid)

            for to_read in to_read_tiles:
                # to_read_uid = str(uuid.uuid4())
                to_read_uid = get_uname()
                print(self.h, qrinfo(new_query), f'{"to_read":>15}', to_read_uid)

                self._graph.add_node(
                    to_read_uid,
                    footprint=to_read,
                    future=None,
                    type="to_read",
                    pool=self._io_pool,
                    linked_to_produce=set([to_produce_uid]),
                    linked_queries=set([new_query]),
                    bands=new_query.bands
                )
                self._graph.add_edge(to_produce_uid, to_read_uid)

                # if the tile is not written, writing it
                if not self._is_written(to_read):
                    to_write = to_read

                    to_write_uid = str(repr(to_write) + "to_write")
                    print(self.h, qrinfo(new_query), f'{"to_write":>15}', to_write_uid)
                    if to_write_uid in self._graph.nodes():
                        self._graph.nodes[to_write_uid]["linked_to_produce"].add(to_produce_uid)
                        self._graph.nodes[to_write_uid]["linked_queries"].add(new_query)
                    else:
                        self._graph.add_node(
                            to_write_uid,
                            footprint=to_write,
                            future=None,
                            type="to_write",
                            pool=self._io_pool,
                            in_data=None,
                            linked_to_produce=set([to_produce_uid]),
                            linked_queries=set([new_query]),
                            bands=new_query.bands
                        )
                    self._graph.add_edge(to_read_uid, to_write_uid)

                    to_merge = to_write
                    to_merge_uid = str(repr(to_merge) + "to_merge")
                    print(self.h, qrinfo(new_query), f'{"to_merge":>15}', to_merge_uid)
                    if to_merge_uid in self._graph.nodes():
                        self._graph.nodes[to_merge_uid]["linked_to_produce"].add(to_produce_uid)
                        self._graph.nodes[to_merge_uid]["linked_queries"].add(new_query)
                    else:
                        self._graph.add_node(
                            to_merge_uid,
                            footprint=to_merge,
                            future=None,
                            futures=[],
                            type="to_merge",
                            pool=self._merge_pool,
                            in_data=[],
                            in_fp=[],
                            linked_to_produce=set([to_produce_uid]),
                            linked_queries=set([new_query]),
                            bands=new_query.bands
                        )
                    self._graph.add_edge(to_write_uid, to_merge_uid)

                    to_compute_multi = self._to_compute_of_to_write(to_write)

                    for to_compute in to_compute_multi:
                        to_compute_uid = str(repr(to_compute) + "to_compute")
                        print(self.h, qrinfo(new_query), f'{"to_compute":>15}', to_compute_uid)
                        if to_compute not in new_query.to_compute:
                            new_query.to_compute.append(to_compute)
                        if to_compute_uid in self._graph.nodes():
                            self._graph.nodes[to_compute_uid]["linked_to_produce"].add(to_produce_uid)
                            self._graph.nodes[to_compute_uid]["linked_queries"].add(new_query)
                        else:
                            self._graph.add_node(
                                to_compute_uid,
                                footprint=to_compute,
                                future=None,
                                type="to_compute",
                                pool=self._computation_pool,
                                in_data=None,
                                linked_to_produce=set([to_produce_uid]),
                                linked_queries=set([new_query]),
                                bands=new_query.bands
                            )

                            if self._to_collect_of_to_compute is not None:
                                multi_to_collect = self._to_collect_of_to_compute(to_compute)

                                if multi_to_collect.keys() != self._primitive_functions.keys():
                                    raise ValueError("to_collect keys do not match primitives")

                                for key in multi_to_collect:
                                    new_query.to_collect[key].append(multi_to_collect[key])

                        self._graph.add_edge(to_merge_uid, to_compute_uid)

        new_query.collected = self._collect_data(new_query.to_collect)


    def _to_read_of_to_produce(self, fp):
        to_read_list = self._cache_idx.intersection(fp.bounds)
        return [list(self._cache_tiles.flat)[i] for i in to_read_list]

    def _is_written(self, cache_fp):
        return self._cache_checksum_array[self._indices_of_cache_tiles[cache_fp]]

    def _to_compute_of_to_write(self, fp):
        to_compute_list = self._computation_idx.intersection(fp.bounds)
        return [list(self._computation_tiles.flat)[i] for i in to_compute_list]
