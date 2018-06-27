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

import numpy as np
import networkx as nx
import buzzard as buzz
from buzzard import _tools
import rtree.index

from burito.query import Query
from burito.singleton_counter import SingletonCounter
from burito.checksum import checksum, checksum_file
from burito.get_data_with_primitive import GetDataWithPrimitive


threadPoolTaskCounter = SingletonCounter()

def _get_graph_uid(fp, _type):
    return hash(repr(fp) + _type)




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
    """
    Abstract class defining the raster behaviour
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

        self._thread_storage = threading.local()

        self._graph = nx.DiGraph()

        def default_merge_data(out_fp, out_data, in_fps, in_arrays):
            """
            Defeult merge function: burning
            """
            if len(out_data.shape) == 3 and out_data.shape[2] == 1:
                out_data = out_data.squeeze(axis=-1)
            for to_burn_fp, to_burn_data in zip(in_fps, in_arrays):
                out_data[to_burn_fp.slice_in(out_fp, clip=True)] = to_burn_data[out_fp.slice_in(to_burn_fp, clip=True)]
            return out_data

        if merge_function is None:
            self._merge_data = default_merge_data
        else:
            self._merge_data = merge_function

        # Used to keep duplicates in to_produce
        self._to_produce_in_occurencies_dict = defaultdict(int)
        self._to_produce_out_occurencies_dict = defaultdict(int)
        self._to_produce_available_occurencies_dict = defaultdict(int)

        # Used to track the number of pending tasks
        self._num_pending = defaultdict(int)

        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._scheduler_thread.start()


    def _pressure_ratio(self, query):
        """
        defines a pressure ration of a query: lesser values -> emptier query
        """
        if query.produced() is None:
            return -1
        num = query.produced().qsize() + self._num_pending[id(query)]
        den = query.produced().maxsize
        return num/den

    @property
    def fp(self):
        """
        returns the raster footprint
        """
        return self._full_fp

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
        produced_data[to_burn_fp.slice_in(produce_fp)] = to_burn_data


    def _scheduler(self):
        header = f"{list(self._primitive_functions.keys())!s:25}"
        self.h = header
        print(header, "scheduler in")
        # list of available and produced to_collect footprints
        available_to_produce = set()
        
        while True:

            self._clean_graph()

            # Consuming the new queries
            while self._new_queries:
                print(header, "updating graph ")

                a = datetime.datetime.now()
                new_query = self._new_queries.pop(0)
                b = datetime.datetime.now() - a
                print(header, "popped, query size:", len(new_query.to_produce), b.total_seconds())

                if isinstance(self, CachedRaster):
                    print(header, "check array", self._cache_checksum_array.flatten())
                    a = datetime.datetime.now()
                    print(header, "checking ")
                    self._check_query(new_query)
                    # for i in range(1000000):
                    #     i = (i**2)**0.5
                    b = datetime.datetime.now() - a
                    print(header, "checked ", b.total_seconds())
                    print(header, "check array after check", self._cache_checksum_array.flatten())

                a = datetime.datetime.now()
                self._update_graph_from_query(new_query)
                b = datetime.datetime.now() - a
                print(header, "updating ended ", b.total_seconds())


            # ordering queries accroding to their pressure
            ordered_queries = sorted(self._queries, key=self._pressure_ratio)
            # getting the emptiest query
            for query in ordered_queries:

                # If all to_produced was consumed: query ended
                if not query.to_produce:
                    self._num_pending[query] = 0
                    self._queries.remove(ordered_queries[0])
                    continue

                # If the query has been dropped
                if query.produced() is None:
                    self._num_pending[query] = 0
                    to_delete_edges = nx.dfs_edges(self._graph, source=id(ordered_queries[0]))
                    self._graph.remove_edges_from(to_delete_edges)
                    self._graph.remove_node(id(ordered_queries[0]))
                    self._queries.remove(ordered_queries[0])
                    continue

                # if the emptiest query is full, waiting
                if query.produced().full():
                    continue

                # detecting which produce footprints are available
                # while there is space
                while query.produced().qsize() + self._num_pending[id(query)] < query.produced().maxsize and query.to_produce[-1][1] == "sleeping":
                    # getting the first sleeping to_produce
                    to_produce_available = [to_produce[0] for to_produce in query.to_produce if to_produce[1] == "sleeping"][0]
                    # getting its id in the graph
                    to_produce_available_id = _get_graph_uid(
                        to_produce_available,
                        "to_produce" + str(self._to_produce_available_occurencies_dict[to_produce_available])
                    )

                    available_to_produce.add(to_produce_available_id)
                    query.to_produce[query.to_produce.index((to_produce_available, "sleeping"))] = (to_produce_available, "pending")

                    self._num_pending[id(query)] += 1
                    self._to_produce_available_occurencies_dict[to_produce_available] += 1


                # iterating through the graph
                for index, to_produce in enumerate(query.to_produce):
                    if to_produce[1] == "sleeping":
                        continue

                    # beginning at to_produce
                    first_node_id = _get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_out_occurencies_dict[to_produce[0]]))
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

                        # Skipping the collect
                        if node["type"] == "to_collect":
                            continue

                        # if deepest is to_compute, collecting (if possible) and computing
                        if node["type"] == "to_compute" and node["future"] is None:
                            # testing if at least 1 of the collected queues is empty (1 queue per primitive)
                            if any([query.collected[primitive].empty() for primitive in query.collected]):
                                break
                            collected_data = []
                            primitive_footprints = []

                            for collected_primitive in query.collected.keys():
                                collected_data.append(query.collected[collected_primitive].get(block=False))
                                primitive_footprints.append(query.to_collect[collected_primitive].pop(0))

                            if threadPoolTaskCounter[id(node["pool"])] < node["pool"]._processes:
                                node["future"] = self._computation_pool.apply_async(
                                    self._compute_data,
                                    (
                                        node["footprint"],
                                        collected_data,
                                        primitive_footprints,
                                        self
                                    )
                                )
                                threadPoolTaskCounter[id(self._computation_pool)] += 1

                                compute_out_edges = list(self._graph.out_edges(node))
                                self._graph.remove_edges_from(compute_out_edges)

                            continue

                        # if the deepest is to_produce, updating produced
                        if index == 0 and node["type"] == "to_produce":
                            not_ready_list = [future for future in node["futures"] if not future.ready()]
                            if not not_ready_list:

                                if len(node["data"].shape) == 3 and node["data"].shape[2] == 1:
                                    node["data"] = node["data"].squeeze(axis=-1)
                                # If the query has not been dropped
                                if query.produced() is not None:
                                    if len(node["data"].shape) == 3 and node["data"].shape[2] == 1:
                                        node["data"] = node["data"].squeeze(axis=-1)
                                    query.produced().put(node["data"].astype(self._dtype), timeout=1e-2)
                                query.to_produce.pop(0)

                                self._to_produce_out_occurencies_dict[to_produce[0]] += 1
                                self._graph.remove_node(node_id)

                                self._num_pending[id(query)] -= 1

                            continue

                        # skipping the ready to_produce that are not at index 0
                        if node["type"] == "to_produce":
                            continue

                        if node["type"] == "to_merge" and node["future"] is None:
                            if threadPoolTaskCounter[id(node["pool"])] < node["pool"]._processes:
                                node["future"] = node["pool"].apply_async(
                                    node["function"],
                                    (
                                        node["footprint"],
                                        node["data"],
                                        node["in_fp"],
                                        node["in_data"]
                                    )
                                )
                                threadPoolTaskCounter[id(node["pool"])] += 1
                            continue

                        in_edges = list(self._graph.in_edges(node_id))

                        if node["future"] is None:
                            if threadPoolTaskCounter[id(node["pool"])] < node["pool"]._processes:
                                if node["type"] == "to_read":
                                    node["future"] = node["pool"].apply_async(
                                        node["function"],
                                        (
                                            node["footprint"],
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
                                threadPoolTaskCounter[id(node["pool"])] += 1
                            continue

                        elif node["future"].ready():
                            in_data = node["future"].get()
                            threadPoolTaskCounter[id(node["pool"])] -= 1

                            for in_edge in in_edges:
                                if self._graph.nodes[in_edge[0]]["type"] == "to_produce":
                                    self._graph.nodes[in_edge[0]]["futures"].append(self._io_pool.apply_async(
                                        self._burn_data,
                                        (
                                            self._graph.nodes[in_edge[0]]["footprint"],
                                            self._graph.nodes[in_edge[0]]["data"],
                                            node["footprint"],
                                            in_data
                                        )
                                    ))
                                elif self._graph.nodes[in_edge[0]]["type"] == "to_merge":
                                    self._graph.nodes[in_edge[0]]["in_data"].append(in_data)
                                    self._graph.nodes[in_edge[0]]["in_fp"].append(node["footprint"])
                                else:
                                    self._graph.nodes[in_edge[0]]["in_data"] = in_data
                                self._graph.remove_edge(*in_edge)
                            continue


            if not self._queries:
                time.sleep(1e-1)
                continue
            time.sleep(1e-2)


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
        print(self.h, "collecting")
        results = {}
        for primitive in self._primitive_functions.keys():
            results[primitive] = self._primitive_functions[primitive](to_collect[primitive])
        return results

    def _to_compute_of_to_produce(self, fp):
        to_compute_list = []
        for computation_tile in self._computation_tiles.flat:
            if fp.share_area(computation_tile):
                to_compute_list.append(computation_tile)

        return to_compute_list


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

        self._graph.add_node(
            id(new_query)
        )

        for to_produce in new_query.to_produce:
            to_produce_uid = _get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_in_occurencies_dict[to_produce[0]]))
            self._to_produce_in_occurencies_dict[to_produce[0]] += 1
            self._graph.add_node(
                to_produce_uid,
                futures=[],
                footprint=to_produce[0],
                data=np.zeros(tuple(to_produce[0].shape) + (len(new_query.bands),)),
                type="to_produce",
                linked_to_produce=set([to_produce_uid])
            )

            self._graph.add_edge(id(new_query), to_produce_uid)

            if isinstance(self._computation_tiles, np.ndarray):
                multi_to_compute = self._to_compute_of_to_produce(to_produce[0])

                to_merge = to_produce

                to_merge_uid = _get_graph_uid(to_merge, "to_merge")
                if to_merge_uid in self._graph.nodes():
                    self._graph.nodes[to_merge_uid]["linked_to_produce"].add(to_produce_uid)
                else:
                    self._graph.add_node(
                        to_merge_uid,
                        footprint=to_merge,
                        future=None,
                        futures=[],
                        data=np.zeros(tuple(to_produce.shape) + (self._num_bands,)),
                        type="to_merge",
                        pool=self._merge_pool,
                        function=self._merge_data,
                        in_data=[],
                        in_fp=[],
                        linked_to_produce=set([to_produce_uid])
                    )
                self._graph.add_edge(to_produce_uid, to_merge_uid)

            else:
                multi_to_compute = [to_produce[0]]
                to_merge_uid = None

            for to_compute in multi_to_compute:
                to_compute_uid = _get_graph_uid(to_compute, "to_compute")
                if to_compute_uid in self._graph.nodes():
                    self._graph.nodes[to_compute_uid]["linked_to_produce"].add(to_produce_uid)
                else:
                    self._graph.add_node(
                        to_compute_uid,
                        footprint=to_compute,
                        future=None,
                        type="to_compute",
                        pool=self._computation_pool,
                        function=self._compute_data,
                        in_data=None,
                        linked_to_produce=set([to_produce_uid])
                    )
                if to_merge_uid is None:
                    self._graph.add_edge(to_produce_uid, to_compute_uid)
                else:
                    self._graph.add_edge(to_merge_uid, to_compute_uid)

                if self._to_collect_of_to_compute is None:
                    continue
                multi_to_collect = self._to_collect_of_to_compute(to_compute)

                assert multi_to_collect.keys() == self._primitive_functions.keys()

                for key in multi_to_collect:
                    if multi_to_collect[key] not in new_query.to_collect[key]:
                        new_query.to_collect[key].append(multi_to_collect[key])

                for key in multi_to_collect:
                    to_collect_uid = _get_graph_uid(multi_to_collect[key], "to_collect" + key + str(id(new_query)))
                    if to_collect_uid in self._graph.nodes():
                        self._graph.nodes[to_collect_uid]["linked_to_produce"].add(to_produce_uid)
                    else:
                        self._graph.add_node(
                            to_collect_uid,
                            footprint=multi_to_collect[key],
                            future=None,
                            type="to_collect",
                            primitive=key,
                            linked_to_produce=set([to_produce_uid])
                        )
                    self._graph.add_edge(to_compute_uid, to_collect_uid)

        new_query.collected = self._collect_data(new_query.to_collect)



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
            bands, _ = _tools.normalize_band_parameter(band, len(self), None)

            fp_iterable = list(fp_iterable)
            q = queue.Queue(queue_size)
            query = Query(q, bands)
            to_produce = [(fp, "sleeping") for fp in fp_iterable]

            query.to_produce += to_produce

            self._queries.append(query)
            self._new_queries.append(query)

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





class CachedRaster(Raster):
    """
    Cached implementation of abstract raster
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
                 cache_fps,
                 io_pool,
                 computation_pool,
                 primitives,
                 to_collect_of_to_compute,
                 computation_tiles,
                 merge_pool,
                 merge_function):

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

        self._cache_dir = cache_dir
        self._cache_tiles = cache_fps

        assert is_tiling_valid(footprint, cache_fps.flat)

        if overwrite:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

        # Array used to track the state of cahce tiles:
        # None: not yet met
        # False: met, has to be written
        # True: met, already written and valid
        self._cache_checksum_array = np.empty(cache_fps.shape, dtype=object)

        # Used to keep duplicates in to_read
        self._to_read_in_occurencies_dict = defaultdict(int)


    def _to_check_of_to_produce(self, to_produce_fps):
        intersecting_fps = []
        for to_produce in to_produce_fps:
            intersecting_fps += self._to_read_of_to_produce(to_produce[0])
        intersecting_fps = set(intersecting_fps)

        indices = []
        for intersecting_fp in intersecting_fps:
            indices.append(np.where(self._cache_tiles == intersecting_fp))

        for index in indices:
            if self._cache_checksum_array[index][0] is not None:
                indices.remove(index)

        return indices


    def _check_cache_fp(self, index):
        footprint = self._cache_tiles[index][0]
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
        print(self.h, "before share")
        to_check = self._to_check_of_to_produce(to_produce_fps)
        print(self.h, "after share, before map")
        self._io_pool.map(self._check_cache_fp, to_check)
        print(self.h, "after map")
        # print(self.h, " query checked ", time.clock(), threading.currentThread().getName())



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
        file_path = glob.glob(prefix + "*")
        assert len(file_path) <= 1
        return file_path


    def _read_cache_data(self, cache_tile, produce_fp, bands):
        """
        reads cache data
        """
        print(self.h, "reading ")
        filepath = self._get_cache_tile_path(cache_tile)[0]

        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds
        else:
            ds = self._thread_storage.ds

        with ds.open_araster(filepath).close as src:
            data = src.get_data(band=bands, fp=produce_fp.intersection(cache_tile))

        return data


    def _write_cache_data(self, cache_tile, data):
        """
        writes cache data
        """
        print(self.h, "writing ")
        cs = checksum(data)
        filepath = self._get_cache_tile_path_prefix(cache_tile) + "_" + f'{cs:#010x}' + ".tif"
        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds
        else:
            ds = self._thread_storage.ds

        out_proxy = ds.create_araster(filepath,
                                      cache_tile,
                                      data.dtype,
                                      self._num_bands,
                                      driver="GTiff",
                                      sr=self.wkt_origin
                                     )
        out_proxy.set_data(data, band=-1)
        out_proxy.close()
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

        self._graph.add_node(
            id(new_query)
        )

        for to_produce in new_query.to_produce:
            to_produce_uid = _get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_in_occurencies_dict[to_produce[0]]))
            self._to_produce_in_occurencies_dict[to_produce[0]] += 1
            self._graph.add_node(
                to_produce_uid,
                footprint=to_produce[0],
                futures=[],
                data=np.zeros(tuple(to_produce[0].shape) + (len(new_query.bands),)),
                type="to_produce",
                in_data=None,
                linked_to_produce=set([to_produce_uid])
            )
            to_read_tiles = self._to_read_of_to_produce(to_produce[0])

            self._graph.add_edge(id(new_query), to_produce_uid)

            for to_read in to_read_tiles:
                to_read_uid = _get_graph_uid(to_read, "to_read" + str(self._to_read_in_occurencies_dict[to_read]))
                self._to_read_in_occurencies_dict[to_read] += 1

                self._graph.add_node(
                    to_read_uid,
                    footprint=to_read,
                    future=None,
                    type="to_read",
                    pool=self._io_pool,
                    function=self._read_cache_data,
                    produce_fp=to_produce[0],
                    linked_to_produce=set([to_produce_uid]),
                    bands=new_query.bands
                )
                self._graph.add_edge(to_produce_uid, to_read_uid)

                # if the tile is not written, writing it
                if not self._is_written(to_read):
                    to_write = to_read

                    to_write_uid = _get_graph_uid(to_write, "to_write")
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
                            linked_to_produce=set([to_produce_uid])
                        )
                    self._graph.add_edge(to_read_uid, to_write_uid)

                    to_merge = to_write
                    to_merge_uid = _get_graph_uid(to_merge, "to_merge")
                    if to_merge_uid in self._graph.nodes():
                        self._graph.nodes[to_merge_uid]["linked_to_produce"].add(to_produce_uid)
                    else:
                        self._graph.add_node(
                            to_merge_uid,
                            footprint=to_merge,
                            future=None,
                            futures=[],
                            data=np.zeros(tuple(to_write.shape) + (self._num_bands,)),
                            type="to_merge",
                            pool=self._merge_pool,
                            function=self._merge_data,
                            in_data=[],
                            in_fp=[],
                            linked_to_produce=set([to_produce_uid])
                        )
                    self._graph.add_edge(to_write_uid, to_merge_uid)

                    to_compute_multi = self._to_compute_of_to_write(to_write)

                    for to_compute in to_compute_multi:
                        to_compute_uid = _get_graph_uid(to_compute, "to_compute")
                        if to_compute_uid in self._graph.nodes():
                            self._graph.nodes[to_compute_uid]["linked_to_produce"].add(to_produce_uid)
                        else:
                            self._graph.add_node(
                                to_compute_uid,
                                footprint=to_compute,
                                future=None,
                                type="to_compute",
                                pool=self._computation_pool,
                                function=self._compute_data,
                                in_data=None,
                                linked_to_produce=set([to_produce_uid])
                            )
                        self._graph.add_edge(to_merge_uid, to_compute_uid)

                        if self._to_collect_of_to_compute is None:
                            continue
                        multi_to_collect = self._to_collect_of_to_compute(to_compute)

                        assert multi_to_collect.keys() == self._primitive_functions.keys()

                        for key in multi_to_collect:
                            if multi_to_collect[key] not in new_query.to_collect[key]:
                                new_query.to_collect[key].append(multi_to_collect[key])

                            to_collect_uid = _get_graph_uid(multi_to_collect[key], "to_collect" + key + str(id(new_query)))
                            if to_collect_uid in self._graph.nodes():
                                self._graph.nodes[to_collect_uid]["linked_to_produce"].add(to_produce_uid)
                            else:
                                self._graph.add_node(
                                    to_collect_uid,
                                    footprint=multi_to_collect[key],
                                    future=None,
                                    type="to_collect",
                                    primitive=key,
                                    linked_to_produce=set([to_produce_uid])
                                )
                            self._graph.add_edge(to_compute_uid, to_collect_uid)

        new_query.collected = self._collect_data(new_query.to_collect)


    def _to_read_of_to_produce(self, fp):
        to_read_list = []
        for cache_tile in self._cache_tiles.flat:
            if fp.share_area(cache_tile):
                to_read_list.append(cache_tile)

        return to_read_list

    def _is_written(self, cache_fp):
        return self._cache_checksum_array[np.where(self._cache_tiles == cache_fp)]

    def _to_compute_of_to_write(self, fp):
        to_compute_list = []
        for computation_tile in self._computation_tiles.flat:
            if fp.share_area(computation_tile):
                to_compute_list.append(computation_tile)

        return to_compute_list



def raster_factory(footprint,
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

    if cached:
        assert cache_dir != None
        raster = CachedRaster(footprint,
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
        raster = Raster(footprint,
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

    return raster
