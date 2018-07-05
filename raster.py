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
import shutil
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

from burito.query import Query
from burito.singleton_counter import SingletonCounter
from burito.checksum import checksum, checksum_file
from burito.get_data_with_primitive import GetDataWithPrimitive


thread_pool_task_counter = SingletonCounter()

qryids = collections.defaultdict(lambda: f'{len(qryids):02d}')
queids = collections.defaultdict(lambda: f'{len(queids):02d}')
_debug_lock = threading.RLock()

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


def quick_share_area(fp, other):
    idx = rtree.index.Index()

    pxsizex = min(fp.pxsize[0], other.pxsize[0])
    bound_inset = np.r_[
        pxsizex / 4,
        pxsizex / 4,
        -pxsizex / 4,
        -pxsizex / 4,
    ]

    idx.insert(0, fp.bounds + bound_inset)
    idx.insert(1, other.bounds + bound_inset)

    return False



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
                 computation_fps=None,
                 merge_pool=None,
                 merge_function=None):
        """
        creates a raster from arguments
        """
        if footprint is None:
            raise ValueError()
        if cached:
            assert cache_dir != None
            backend_raster = BackendCachedRaster(footprint,
                                                 dtype,
                                                 nbands,
                                                 nodata,
                                                 srs,
                                                 computation_function,
                                                 overwrite,
                                                 cache_dir,
                                                 cache_fps,
                                                 io_pool,
                                                 computation_pool,
                                                 primitives,
                                                 to_collect_of_to_compute,
                                                 computation_fps,
                                                 merge_pool,
                                                 merge_function
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
                                           computation_fps,
                                           merge_pool,
                                           merge_function
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
                 computation_tiles,
                 merge_pool,
                 merge_function):

        self._full_fp = footprint
        if computation_function is None:
            raise ValueError()
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

        if primitives.keys():
            assert to_collect_of_to_compute is not None
        self._to_collect_of_to_compute = to_collect_of_to_compute

        self._computation_tiles = computation_tiles

        self._queries = []
        self._new_queries = []

        self._graph = nx.DiGraph()

        def default_merge_data(out_fp, in_fps, in_arrays):
            """
            Default merge function: burning
            """
            out_data = np.zeros(tuple(out_fp.shape) + (self._num_bands,))
            for to_burn_fp, to_burn_data in zip(in_fps, in_arrays):
                out_data[to_burn_fp.slice_in(out_fp, clip=True)] = to_burn_data[out_fp.slice_in(to_burn_fp, clip=True)]
            return out_data

        if merge_function is None:
            self._merge_data = default_merge_data
        else:
            self._merge_data = merge_function

        # Used to track the number of pending tasks
        self._num_pending = defaultdict(int)

        # Used to track the number of pending tasks
        self._num_pending = defaultdict(int)

        self._stop_scheduler = False

        self._computation_idx = rtree.index.Index()

        if computation_tiles is not None:
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


    def _burn_data(self, produce_fp, produced_data, to_burn_fp, to_burn_data):
        assert tuple(to_burn_fp.shape) == to_burn_data.shape[:2]
        assert to_burn_fp.poly.within(produce_fp.poly)
        produced_data[to_burn_fp.slice_in(produce_fp)] = to_burn_data


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

            self._clean_graph()

            assert len(set(map(id, self._queries))) == len(self._queries)
            assert len(set(map(id, self._new_queries))) == len(self._new_queries)

            # Consuming the new queries
            while self._new_queries:
                query = self._new_queries[0]

                # if cached raster, checking the cache
                if isinstance(self, BackendCachedRaster):
                    # launching the check process
                    if query.cache_checking is None:
                        query.cache_checking = self._io_pool.apply_async(self._check_query, (query,))
                        # self._check_query(query)
                        break
                    # retrieving the results
                    elif not query.cache_checking.ready():
                        break
                    else:
                        query.cache_checking.get()

                query = self._new_queries.pop(0)
                self._queries.append(query)
                self._update_graph_from_query(query)

                print(self.h, qrinfo(query), f'new query with {len(query.to_produce)} to_produce, {list(len(p) for p in query.to_collect.values())} to_collect')

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

                # If all to_produced was consumed: query ended
                if not query.to_produce:
                    print(self.h, qrinfo(query), f'cleaning: treated all produce')
                    self._num_pending[id(query)] = 0
                    self._queries.remove(query)
                    skip = True
                    break

                # If the query has been dropped
                if query.produced() is None:
                    print(self.h, qrinfo(query), f'cleaning: dropped by main program')
                    self._num_pending[id(query)] = 0
                    to_delete_edges = list(nx.dfs_edges(self._graph, source=id(query)))
                    self._graph.remove_edges_from(to_delete_edges)
                    self._graph.remove_node(id(query))
                    self._queries.remove(query)
                    skip = True
                    break

                # if the emptiest query is full, waiting
                if query.produced().full():
                    continue

                # detecting which produce footprints are available
                # while there is space
                while query.produced().qsize() + self._num_pending[id(query)] < query.produced().maxsize and query.to_produce[-1][1] == "sleeping":
                    # getting the first sleeping to_produce

                    first_sleeping_i = [to_produce[1] for to_produce in query.to_produce].index('sleeping')

                    to_produce_available = query.to_produce[first_sleeping_i][0]

                    # getting its id in the graph
                    to_produce_available_id = query.to_produce[first_sleeping_i][2]

                    available_to_produce.add(to_produce_available_id)

                    to_produce_index = query.to_produce.index((to_produce_available, "sleeping", to_produce_available_id))
                    query.to_produce[to_produce_index] = (to_produce_available, "pending", to_produce_available_id)

                    self._num_pending[id(query)] += 1

                    skip = True
                    break


                for primitive in query.collected:
                    if not query.collected[primitive].empty() and query.to_collect[primitive][0] in query.to_discard[primitive]:
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
                    depth_node_ids = nx.dfs_postorder_nodes(self._graph.copy(), source=first_node_id)

                    for node_id in depth_node_ids:
                        node = self._graph.nodes[node_id]

                        # If there are out edges, not stopping (unless it is a compute node)
                        if len(self._graph.out_edges(node_id)) > 0 and node["type"] != "to_compute":
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
                                collected_data = []
                                primitive_footprints = []

                                get_counter[query] += 1
                                print(self.h, qrinfo(query), f'compute data for the {get_counter[query]:02d}th time node_id:({node_id})')

                                for collected_primitive in query.collected.keys():
                                    collected_data.append(query.collected[collected_primitive].get(block=False))
                                    primitive_footprints.append(query.to_collect[collected_primitive].pop(0))

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
                                    for collected_primitive in query.collected.keys():
                                        linked_query.to_discard[collected_primitive].append(next(primitive_footprints))

                                skip = True
                                break

                        # if the deepest is to_produce, updating produced
                        if index == 0 and node["type"] == "to_produce":
                            not_ready_list = [future for future in node["futures"] if not future.ready()]
                            if not not_ready_list:

                                # If the query has not been dropped
                                if query.produced() is not None:
                                    if node["is_flat"]:
                                        node["data"] = node["data"].squeeze(axis=-1)
                                    query.produced().put(node["data"].astype(self._dtype), timeout=1e-2)


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
                            if thread_pool_task_counter[id(node["pool"])] < node["pool"]._processes:
                                node["future"] = node["pool"].apply_async(
                                    node["function"],
                                    (
                                        node["footprint"],
                                        node["in_fp"],
                                        node["in_data"]
                                    )
                                )
                                thread_pool_task_counter[id(node["pool"])] += 1
                                skip = True
                                break

                        in_edges = list(self._graph.in_edges(node_id))

                        if node["future"] is None:
                            if thread_pool_task_counter[id(node["pool"])] < node["pool"]._processes:
                                if node["type"] == "to_read":
                                    node["future"] = node["pool"].apply_async(
                                        node["function"],
                                        (
                                            node["cache_fp"],
                                            node["produce_fp"],
                                            node["bands"]
                                        )
                                    )
                                else:
                                    node["future"] = node["pool"].apply_async(
                                        node["function"],
                                        (
                                            node["footprint"],
                                            node["in_data"]
                                        )
                                    )
                                thread_pool_task_counter[id(node["pool"])] += 1
                                skip = True
                                break

                        elif node["future"].ready():
                            in_data = node["future"].get()
                            if in_data is not None:
                                in_data = in_data.astype(self._dtype).reshape(tuple(node["footprint"].shape) + (len(node["bands"]),))
                            thread_pool_task_counter[id(node["pool"])] -= 1

                            for in_edge in in_edges:
                                in_node = self._graph.nodes[in_edge[0]]
                                if in_node["type"] == "to_produce":
                                    if in_node["data"] is None:
                                        in_node["data"] = np.zeros(tuple(in_node["footprint"].shape) + (len(query.bands),))
                                    in_node["futures"].append(self._io_pool.apply_async(
                                        self._burn_data,
                                        (
                                            in_node["footprint"],
                                            in_node["data"],
                                            node["footprint"],
                                            in_data
                                        )
                                    ))
                                elif in_node["type"] == "to_merge":
                                    in_node["in_data"].append(in_data)
                                    in_node["in_fp"].append(node["footprint"])
                                else:
                                    in_node["in_data"] = in_data
                                self._graph.remove_edge(*in_edge)
                            skip = True
                            break

            if not skip:
                if not self._queries:
                    time.sleep(0.2)
                else:
                    time.sleep(0.1)


    def _clean_graph(self):
        """
        removes the graph's orphans
        """
        # Used to keep duplicates in to_produce
        to_remove = list(nx.isolates(self._graph))
        self._graph.remove_nodes_from(to_remove)



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
        to_compute_list = self._computation_idx.intersection(fp.bounds)
        return [list(self._computation_tiles.flat)[i] for i in to_compute_list]


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
            id(new_query)
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
                data=None,
                type="to_produce",
                linked_to_produce=set([to_produce_uid]),
                bands=new_query.bands,
                is_flat=new_query.is_flat
            )

            self._graph.add_edge(id(new_query), to_produce_uid)

            if self._computation_tiles is not None:
                to_merge = to_produce

                # to_merge_uid = str(uuid.uuid4())
                to_merge_uid = get_uname()

                print(self.h, qrinfo(new_query), f'    {"to_merge":>15}', to_merge_uid)
                if to_merge_uid in self._graph.nodes():
                    self._graph.nodes[to_merge_uid]["linked_to_produce"].add(to_produce_uid)
                else:
                    self._graph.add_node(
                        to_merge_uid,
                        footprint=to_merge,
                        future=None,
                        futures=[],
                        type="to_merge",
                        pool=self._merge_pool,
                        function=self._merge_data,
                        in_data=[],
                        in_fp=[],
                        linked_to_produce=set([to_produce_uid]),
                        bands=new_query.bands
                    )
                self._graph.add_edge(to_produce_uid, to_merge_uid)

                multi_to_compute = self._to_compute_of_to_produce(to_merge)
            else:
                multi_to_compute = [to_produce]
                to_merge_uid = None
            # time_dict["produce"] += (datetime.datetime.now() - start).total_seconds()

            for to_compute in multi_to_compute:
                # start = datetime.datetime.now()
                # to_compute_uid = str(uuid.uuid4())
                to_compute_uid = get_uname()

                print(self.h, qrinfo(new_query), f'        {"to_compute":>15}', to_compute_uid)

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
                        function=self._compute_data,
                        in_data=None,
                        linked_to_produce=set([to_produce_uid]),
                        linked_queries=set([new_query]),
                        bands=new_query.bands
                    )
                    new_query.to_compute.append(to_compute)
                if to_merge_uid is None:
                    self._graph.add_edge(to_produce_uid, to_compute_uid)
                else:
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

                assert multi_to_collect.keys() == self._primitive_functions.keys()

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
                 merge_function):



        self._cache_dir = cache_dir
        self._cache_tiles = cache_tiles

        assert is_tiling_valid(footprint, cache_tiles.flat)

        if overwrite:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

        if computation_tiles is None:
            computation_tiles = cache_tiles

        # Array used to track the state of cahce tiles:
        # None: not yet met
        # False: met, has to be written
        # True: met, already written and valid
        self._cache_checksum_array = np.empty(cache_tiles.shape, dtype=object)
        self._check_lock = threading.Lock()

        self._cache_idx = rtree.index.Index()
        cache_fps = list(cache_tiles.flat)
        assert isinstance(cache_fps[0], buzz.Footprint)
        pxsizex = min(fp.pxsize[0] for fp in cache_fps)
        bound_inset = np.r_[
            pxsizex / 4,
            pxsizex / 4,
            pxsizex / -4,
            pxsizex / -4,
        ]

        for i, fp in enumerate(cache_fps):
            self._cache_idx.insert(i, fp.bounds + bound_inset)


        super().__init__(footprint, dtype, nbands, nodata, srs,
                         computation_function,
                         io_pool,
                         computation_pool,
                         primitives,
                         to_collect_of_to_compute,
                         computation_tiles,
                         merge_pool,
                         merge_function
                        )





    def _to_check_of_to_produce(self, to_produce_fps):
        intersecting_fps = []
        for to_produce in to_produce_fps:
            intersecting_fps += self._to_read_of_to_produce(to_produce[0])
        intersecting_fps = set(intersecting_fps)

        indices = []
        for intersecting_fp in intersecting_fps:
            indices.append(np.where(self._cache_tiles == intersecting_fp))

        for index in indices:
            with self._check_lock:
                if self._cache_checksum_array[index][0] is not None:
                    indices.remove(index)

        return indices


    def _check_cache_fp(self, index):
        footprint = self._cache_tiles[index][0]
        with self._check_lock:
            self._cache_checksum_array[index] = self._check_cache_file(footprint)

    def _check_cache_file(self, footprint):
        cache_tile_path = self._get_cache_tile_path(footprint)
        if not cache_tile_path:
            return False
        else:
            cache_path = cache_tile_path[0]
            checksum_dot_tif = cache_path.split('_')[-1]
            file_checksum = checksum_dot_tif.split('.')[0]
            if int(file_checksum, base=16) == checksum_file(cache_path):
                return True
        return False

    def _check_query(self, query):
        # print(self.h, " checking query ", threading.currentThread().getName())
        to_produce_fps = query.to_produce
        to_check = self._to_check_of_to_produce(to_produce_fps)
        self._io_pool.map(self._check_cache_fp, to_check)
        # print(self.h, " query checked ", threading.currentThread().getName())



    def _get_cache_tile_path_prefix(self, cache_tile):
        """
        Returns a string, which is a path to a cache tile from its fp
        """
        tile_index = np.where(self._cache_tiles == cache_tile)
        path = str(
            Path(self._cache_dir) /
            "fullsize_{:05d}_{:05d}_tilesize_{:05d}_{:05d}_tilepxindex_{:05d}_{:05d}_tileindex_{:05d}_{:05d}".format(
                *self._full_fp.rsize,
                *cache_tile.rsize,
                *self._full_fp.spatial_to_raster(cache_tile.tl),
                tile_index[0][0],
                tile_index[1][0]
            )
        )
        return path


    def _get_cache_tile_path(self, cache_tile):
        """
        Returns a string, which is a path to a cache tile from its fp
        """
        prefix = self._get_cache_tile_path_prefix(cache_tile)
        file_path = glob.glob(prefix + "_0x*.tif")
        assert len(file_path) <= 1, file_path
        return file_path


    def _read_cache_data(self, cache_tile, produce_fp, bands):
        """
        reads cache data
        """
        # print(self.h, "reading")
        filepath = self._get_cache_tile_path(cache_tile)[0]

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

        samplebands = []
        for i in bands:
            a = gdal_ds.GetRasterBand(i).ReadAsArray(
                int(rtlx), int(rtly), int(to_read_fp.rsizex), int(to_read_fp.rsizey)
            )
            if a is None:
                raise ValueError('Could not read array (gdal error: `{}`)'.format(
                    gdal.GetLastErrorMsg()
                ))
            samplebands.append(a)
        samplebands = np.stack(samplebands, -1)

        assert np.array_equal(samplebands.shape[0:2], to_read_fp.shape)
        return samplebands


    def _write_cache_data(self, cache_tile, data):
        """
        writes cache data
        """
        # print(self.h, "writing ")
        cs = checksum(data)
        filepath = self._get_cache_tile_path_prefix(cache_tile) + "_" + f'{cs:#010x}' + ".tif"
        sr = self.wkt_origin

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

        with self._check_lock:
            self._cache_checksum_array[np.where(self._cache_tiles == cache_tile)] = True


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
            id(new_query)
        )

        for to_produce, _, to_produce_uid in new_query.to_produce:

            self._graph.add_node(
                to_produce_uid,
                footprint=to_produce,
                futures=[],
                data=None,
                type="to_produce",
                in_data=None,
                linked_to_produce=set([to_produce_uid]),
                is_flat=new_query.is_flat,
                bands=new_query.bands
            )
            to_read_tiles = self._to_read_of_to_produce(to_produce)

            self._graph.add_edge(id(new_query), to_produce_uid)

            for to_read in to_read_tiles:
                # to_read_uid = str(uuid.uuid4())
                to_read_uid = get_uname()

                self._graph.add_node(
                    to_read_uid,
                    footprint=to_produce.intersection(to_read),
                    cache_fp=to_read,
                    future=None,
                    type="to_read",
                    pool=self._io_pool,
                    function=self._read_cache_data,
                    produce_fp=to_produce,
                    linked_to_produce=set([to_produce_uid]),
                    bands=new_query.bands
                )
                self._graph.add_edge(to_produce_uid, to_read_uid)

                # if the tile is not written, writing it
                if not self._is_written(to_read):
                    to_write = to_read

                    to_write_uid = str(repr(to_write) + "to_write")
                    if to_write_uid in self._graph.nodes():
                        self._graph.nodes[to_write_uid]["linked_to_produce"].add(to_produce_uid)
                    else:
                        self._graph.add_node(
                            to_write_uid,
                            footprint=to_write,
                            future=None,
                            type="to_write",
                            pool=self._io_pool,
                            function=self._write_cache_data,
                            in_data=None,
                            linked_to_produce=set([to_produce_uid]),
                            bands=new_query.bands
                        )
                    self._graph.add_edge(to_read_uid, to_write_uid)

                    to_merge = to_write
                    to_merge_uid = str(repr(to_merge) + "to_merge")
                    if to_merge_uid in self._graph.nodes():
                        self._graph.nodes[to_merge_uid]["linked_to_produce"].add(to_produce_uid)
                    else:
                        self._graph.add_node(
                            to_merge_uid,
                            footprint=to_merge,
                            future=None,
                            futures=[],
                            type="to_merge",
                            pool=self._merge_pool,
                            function=self._merge_data,
                            in_data=[],
                            in_fp=[],
                            linked_to_produce=set([to_produce_uid]),
                            bands=new_query.bands
                        )
                    self._graph.add_edge(to_write_uid, to_merge_uid)

                    to_compute_multi = self._to_compute_of_to_write(to_write)

                    for to_compute in to_compute_multi:
                        to_compute_uid = str(repr(to_compute) + "to_compute")
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
                                function=self._compute_data,
                                in_data=None,
                                linked_to_produce=set([to_produce_uid]),
                                linked_queries=set([new_query]),
                                bands=new_query.bands
                            )
                            new_query.to_compute.append(to_compute)
                        self._graph.add_edge(to_merge_uid, to_compute_uid)

                        if self._to_collect_of_to_compute is None:
                            continue
                        multi_to_collect = self._to_collect_of_to_compute(to_compute)

                        assert multi_to_collect.keys() == self._primitive_functions.keys()

                        for key in multi_to_collect:
                            new_query.to_collect[key].append(multi_to_collect[key])

        new_query.collected = self._collect_data(new_query.to_collect)


    def _to_read_of_to_produce(self, fp):
        to_read_list = self._cache_idx.intersection(fp.bounds)
        return [list(self._cache_tiles.flat)[i] for i in to_read_list]

    def _is_written(self, cache_fp):
        return self._cache_checksum_array[np.where(self._cache_tiles == cache_fp)]

    def _to_compute_of_to_write(self, fp):
        to_compute_list = self._computation_idx.intersection(fp.bounds)
        return [list(self._computation_tiles.flat)[i] for i in to_compute_list]
