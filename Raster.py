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
import weakref

import numpy as np
import networkx as nx
import buzzard as buzz

from Query import Query
from SingletonCounter import SingletonCounter
from checksum import checksum, checksum_file

threadPoolTaskCounter = SingletonCounter()

def _get_graph_uid(fp, _type):
    return hash(repr(fp) + _type)


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

        self._primitives = primitives
        self._to_collect_of_to_compute = to_collect_of_to_compute

        self._computation_tiles = computation_tiles

        self._queries = []
        self._new_queries = []

        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._scheduler_thread.start()

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
        self._to_produce_collect_occurencies_dict = defaultdict(int)

        # Used to track the number of pending tasks
        self._num_pending = defaultdict(int)


    def _pressure_ratio(self, query):
        """
        defines a pressure ration of a query: lesser values -> emptier query
        """
        num = query.produced.qsize() + self._num_pending[id(query)]
        den = query.produced.maxsize
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



    def __len__(self):
        return int(self._num_bands)


    def _burn_data(self, big_fp, to_fill_data, to_burn_fp, to_burn_data):
        print(self.__class__.__name__, " burning ", threading.currentThread().getName())
        if len(to_fill_data.shape) == 3 and to_fill_data.shape[2] == 1:
            to_fill_data = to_fill_data.squeeze(axis=-1)
        to_fill_data[to_burn_fp.slice_in(big_fp, clip=True)] = to_burn_data[big_fp.slice_in(to_burn_fp, clip=True)]



    def _scheduler(self):
        print(self.__class__.__name__, " scheduler in ", threading.currentThread().getName())
        # list of available and produced to_collect footprints
        to_collect_batch = {key: [] for key in self._primitives.keys()}
        while True:
            time.sleep(1e-2)

            if not self._queries:
                continue

            while self._new_queries:
                print(self.__class__.__name__, " updating graph ", threading.currentThread().getName())
                new_query = self._new_queries.pop(0)
                if isinstance(self, CachedRaster):
                    self._check_query(new_query)
                self._update_graph_from_query(new_query)

            # ordering queries accroding to their pressure
            ordered_queries = sorted(self._queries, key=self._pressure_ratio)
            # getting the emptiest query
            query = ordered_queries[0]

            if not query.to_produce:
                self._num_pending[query] = 0
                self._queries.remove(query)
                continue

            # if the emptiest query is full, waiting
            if query.produced.full():
                continue

            # detecting which footprints to collect from the queue + pending
            # while there is space
            while query.produced.qsize() + self._num_pending[id(query)] < query.produced.maxsize and query.to_produce[-1][1] == "sleeping":
                # getting the first sleeping to_produce
                to_produce_available = [to_produce[0] for to_produce in query.to_produce if to_produce[1] == "sleeping"][0]
                # getting its id in the graph
                to_produce_available_id = _get_graph_uid(
                    to_produce_available,
                    "to_produce" + str(self._to_produce_collect_occurencies_dict[to_produce_available])
                )

                # getting the ids of the depth firsrt search
                depth_node_ids = nx.dfs_postorder_nodes(self._graph.copy(), source=to_produce_available_id)

                for node_id in depth_node_ids:
                    node = self._graph.nodes[node_id]
                    # for each to collect, appending the footprint to the batch
                    if node["type"] == "to_collect" and node["footprint"] not in to_collect_batch[node["primitive"]]:
                        to_collect_batch[node["primitive"]].append(node["footprint"])

                # updating the to_produce status
                query.to_produce[query.to_produce.index((to_produce_available, "sleeping"))] = (to_produce_available, "pending")

                self._num_pending[id(query)] += 1
                self._to_produce_collect_occurencies_dict[to_produce_available] += 1

            # testing if at least 1 of the collected queues is empty (1 queue per primitive)
            one_is_empty = False
            for primitive in self._primitives.keys():
                collected_primitive = query.collected[primitive]
                if collected_primitive.empty():
                    one_is_empty = True

            # if threre are primitives
            if list(self._primitives.keys()):
                prim = list(self._primitives.keys())[0]

                too_many_tasks = threadPoolTaskCounter[id(self._computation_pool)] >= self._computation_pool._processes
                # if they are all not empty and can be collected without saturation
                if not one_is_empty and query.to_collect[prim][0] in to_collect_batch[prim] and not too_many_tasks:
                    # getting all the collected data
                    collected_data = []
                    for collected_primitive in query.collected.keys():
                        collected_data.append(query.collected[collected_primitive].get(block=False))

                    # for each graph edge out of the collected, applying the asyncresult to the out node
                    for prim in self._primitives.keys():
                        try:
                            collect_in_edges = self._graph.copy().in_edges(_get_graph_uid(
                                query.to_collect[prim][0],
                                "to_collect" + prim + str(id(query))
                            ))

                            for edge in collect_in_edges:
                                compute_node = self._graph.nodes[edge[0]]
                                threadPoolTaskCounter[id(self._computation_pool)] += 1
                                compute_node["future"] = self._computation_pool.apply_async(
                                    self._compute_data,
                                    (
                                        self._graph.nodes[edge[0]]["footprint"],
                                        *collected_data
                                    )
                                )
                                compute_out_edges = self._graph.copy().out_edges(edge[0])
                                for to_remove_edge in compute_out_edges:
                                    self._graph.remove_edge(*to_remove_edge)
                        except nx.NetworkXError:
                            pass
                        finally:
                            query.to_collect[prim].pop(0)
                    continue

            # iterating through the graph
            for index, to_produce in enumerate(query.to_produce):
                if to_produce[1] == "sleeping":
                    continue
                else:
                    # beginning at to_produce
                    first_node_id = _get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_out_occurencies_dict[to_produce[0]]))
                    # going as deep as possible
                    depth_node_ids = nx.dfs_postorder_nodes(self._graph.copy(), source=first_node_id)
                    for node_id in depth_node_ids:
                        node = self._graph.nodes[node_id]

                        if len(self._graph.out_edges(node_id)) > 0:
                            continue

                        # should not happen, but not fata if happens
                        if node["type"] == "to_collect":
                            continue

                        # if the deepest is to_produce, updating produced
                        if index == 0 and node["type"] == "to_produce":
                            not_ready_list = [future for future in node["futures"] if not future.ready()]
                            if not not_ready_list:

                                if len(node["data"].shape) == 3 and node["data"].shape[2] == 1:
                                    node["data"] = node["data"].squeeze(axis=-1)
                                query.produced.put(node["data"].astype(self._dtype), timeout=1e-2)
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

                        in_edges = self._graph.copy().in_edges(node_id)

                        if node["future"] is None:
                            if threadPoolTaskCounter[id(node["pool"])] < node["pool"]._processes:
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


            self._clean_graph()



    def _clean_graph(self):
        """
        removes the graph's orphans
        """
        # Used to keep duplicates in to_produce
        to_produce_occurencies_dict = self._to_produce_out_occurencies_dict.copy()

        to_remove = list(nx.isolates(self._graph))

        # not removing the orphan to_produce, they are removed when consumed
        for query in self._queries:
            for to_produce in query.to_produce:
                to_produce_uid = _get_graph_uid(to_produce[0], "to_produce" + str(to_produce_occurencies_dict[to_produce[0]]))
                to_produce_occurencies_dict[to_produce[0]] += 1
                while to_produce_uid in to_remove:
                    to_remove.remove(to_produce_uid)

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
        new_query.to_collect = {key: [] for key in self._primitives.keys()}

        for to_produce in new_query.to_produce:
            to_produce_uid = _get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_in_occurencies_dict[to_produce[0]]))
            self._to_produce_in_occurencies_dict[to_produce[0]] += 1
            self._graph.add_node(
                to_produce_uid,
                futures=[],
                footprint=to_produce[0],
                data=np.zeros(tuple(to_produce[0].shape) + (self._num_bands,)),
                type="to_produce"
            )

            if isinstance(self._computation_tiles, np.ndarray):
                multi_to_compute = self._to_compute_of_to_produce(to_produce[0])

                to_merge = to_produce

                to_merge_uid = _get_graph_uid(to_merge, "to_merge")
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
                    in_fp=[]
                )
                self._graph.add_edge(to_produce_uid, to_merge_uid)

            else:
                multi_to_compute = [to_produce[0]]
                to_merge_uid = None

            for to_compute in multi_to_compute:
                to_compute_uid = _get_graph_uid(to_compute, "to_compute")

                self._graph.add_node(
                    to_compute_uid,
                    footprint=to_compute,
                    future=None,
                    type="to_compute",
                    pool=self._computation_pool,
                    function=self._compute_data,
                    in_data=None
                )
                if to_merge_uid is None:
                    self._graph.add_edge(to_produce_uid, to_compute_uid)
                else:
                    self._graph.add_edge(to_merge_uid, to_compute_uid)

                if self._to_collect_of_to_compute is None:
                    continue
                multi_to_collect = self._to_collect_of_to_compute(to_compute)

                for key in multi_to_collect:
                    if multi_to_collect[key] not in new_query.to_collect[key]:
                        new_query.to_collect[key].append(multi_to_collect[key])

                for key in multi_to_collect:
                    to_collect_uid = _get_graph_uid(multi_to_collect[key], "to_collect" + key + str(id(new_query)))
                    self._graph.add_node(
                        to_collect_uid,
                        footprint=multi_to_collect[key],
                        future=None,
                        type="to_collect",
                        primitive=key
                    )
                    self._graph.add_edge(to_compute_uid, to_collect_uid)

        new_query.collected = self._collect_data(new_query.to_collect)



    def get_multi_data_queue(self, fp_iterable, queue_size=5):
        """
        returns a queue (could be generator) from a fp_iterable
        """
        fp_iterable = list(fp_iterable)
        query = Query(queue_size)
        to_produce = [(fp, "sleeping") for fp in fp_iterable]

        query.to_produce += to_produce

        self._queries.append(query)
        self._new_queries.append(query)

        return query.produced


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


    def _check_cache_fps(self, indices):
        for index in indices:
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
        to_produce_fps = query.to_produce
        to_check = self._to_check_of_to_produce(to_produce_fps)
        self._check_cache_fps(to_check)


    def _get_cache_tile_path_prefix(self, cache_tile):
        """
        Returns a string, which is a path to a cache tile from its fp
        """
        tile_index = np.where(self._cache_tiles == cache_tile)
        path = str(
            Path(self._cache_dir) /
            "{}_{}_{}_{}_{:.0f}_{:.0f}_{}_{}".format(
                *self._full_fp.rsize,
                *cache_tile.rsize,
                cache_tile.tl[0]/cache_tile.pxsizex - self._full_fp.tl[0]/cache_tile.pxsizex,
                self._full_fp.tl[1]/cache_tile.pxsizex - cache_tile.tl[1]/cache_tile.pxsizex,
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


    def _read_cache_data(self, cache_tile, _placeholder=None):
        """
        reads cache data
        """
        print(self.__class__.__name__, " reading ", threading.currentThread().getName())
        filepath = self._get_cache_tile_path(cache_tile)[0]

        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds
        else:
            ds = self._thread_storage.ds

        with ds.open_araster(filepath).close as src:
            data = src.get_data(band=-1, fp=cache_tile)
        return data


    def _write_cache_data(self, cache_tile, data):
        """
        writes cache data
        """
        print(self.__class__.__name__, " writing ", threading.currentThread().getName())
        cs = checksum(data)
        filepath = self._get_cache_tile_path_prefix(cache_tile) + "_" + f'`{cs:#010x}`' + ".tif"
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
        new_query.to_collect = {key: [] for key in self._primitives.keys()}

        for to_produce in new_query.to_produce:
            to_produce_uid = _get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_in_occurencies_dict[to_produce[0]]))
            self._to_produce_in_occurencies_dict[to_produce[0]] += 1
            self._graph.add_node(
                to_produce_uid,
                footprint=to_produce[0],
                futures=[],
                data=np.zeros(tuple(to_produce[0].shape) + (self._num_bands,)),
                type="to_produce",
                in_data=None
            )
            to_read_tiles = self._to_read_of_to_produce(to_produce[0])

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
                    in_data=None
                )
                self._graph.add_edge(to_produce_uid, to_read_uid)

                # if the tile is not written, writing it
                if not self._is_written(to_read):
                    to_write = to_read

                    to_write_uid = _get_graph_uid(to_write, "to_write")
                    if to_write_uid in self._graph.nodes():
                        self._graph.add_edge(to_read_uid, to_write_uid)
                    else:
                        self._graph.add_node(
                            to_write_uid,
                            footprint=to_write,
                            future=None,
                            type="to_write",
                            pool=self._io_pool,
                            function=self._write_cache_data,
                            in_data=None
                        )
                        self._graph.add_edge(to_read_uid, to_write_uid)

                        to_merge = to_write
                        to_merge_uid = _get_graph_uid(to_merge, "to_merge")

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
                            in_fp=[]
                        )
                        self._graph.add_edge(to_write_uid, to_merge_uid)

                        to_compute_multi = self._to_compute_of_to_write(to_write)

                        for to_compute in to_compute_multi:
                            to_compute_uid = _get_graph_uid(to_compute, "to_compute")

                            self._graph.add_node(
                                to_compute_uid,
                                footprint=to_compute,
                                future=None,
                                type="to_compute",
                                pool=self._computation_pool,
                                function=self._compute_data,
                                in_data=None
                            )
                            self._graph.add_edge(to_merge_uid, to_compute_uid)
                            if self._to_collect_of_to_compute is None:
                                continue
                            multi_to_collect = self._to_collect_of_to_compute(to_compute)

                            for key in multi_to_collect:
                                if multi_to_collect[key] not in new_query.to_collect[key]:
                                    new_query.to_collect[key].append(multi_to_collect[key])

                            for key in multi_to_collect:
                                to_collect_uid = _get_graph_uid(multi_to_collect[key], "to_collect" + key + str(id(new_query)))
                                self._graph.add_node(
                                    to_collect_uid,
                                    footprint=multi_to_collect[key],
                                    future=None,
                                    type="to_collect",
                                    primitive=key
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
                   primitives={},
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
