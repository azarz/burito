"""
Multi-threaded, back pressure management, caching
"""

from pathlib import Path
import multiprocessing as mp
import multiprocessing.pool
import os
import queue
import threading
import time
from collections import defaultdict

from keras.models import load_model

import scipy.ndimage as ndi
import numpy as np
import networkx as nx
import buzzard as buzz

from show_many_images import show_many_images
from uids_of_paths import uids_of_paths
from watcher import Watcher

from Query import FullQuery

CATEGORIES = (
    #0        1       2        3        4
    'nolab', 'vege', 'water', 'tapis', 'building',
    #5         6        7           8         9
    'blocks', 'berms', 'vehicles', 'stocks', 'aggregate',
    #10       11       12
    'faces', 'roads', 'bank',
)
INDEX_OF_LABNAME = {}
LABEL_COUNT = len(CATEGORIES)
for i, cat in enumerate(CATEGORIES):
    globals()['INDEX_' + cat.upper()] = i
    INDEX_OF_LABNAME[cat] = i



# DIR_NAMES = uids_of_paths({
#     "ortho": rgb_path,
#     "dsm": dsm_path
# })
DIR_NAMES = {
    frozenset(["ortho"]): "rgb",
    frozenset(["dsm"]): "dsm",
    frozenset(["ortho", "dsm"]): "both"
}

CACHE_DIR = "./.cache"

def output_fp_to_input_fp(fp, scale, rsize):
    """
    Creates a footprint of rsize from fp using dilation and scale
    Used to compute a keras model input footprint from output
    (deduce the big RGB and slopes extents from the smaller output heatmap)
    """
    out = buzz.Footprint(tl=fp.tl, size=fp.size, rsize=fp.size/scale)
    padding = (rsize - out.rsizex) / 2
    assert padding == int(padding)
    out = out.dilate(padding)
    return out


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
                 to_collect_of_to_compute):

        self._full_fp = footprint
        self._compute_data = computation_function
        self._dtype = dtype
        self._num_bands = nbands
        self._nodata = nodata
        self._wkt_origin = srs
        self._io_pool = io_pool
        self._computation_pool = computation_pool
        self._primitives = primitives
        self._to_collect_of_to_compute = to_collect_of_to_compute

        self._queries = []
        self._new_queries = []

        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._scheduler_thread.start()

        # self._computation_pool = mp.pool.ThreadPool()
        # self._io_pool = mp.pool.ThreadPool()
        self._merge_pool = mp.pool.ThreadPool()

        self._thread_storage = threading.local()

        self._graph = nx.DiGraph()

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
        num = query.produce.verbed.qsize() + self._num_pending[id(query)]
        den = query.produce.verbed.maxsize
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

    def _get_graph_uid(self, fp, _type):
        return hash(repr(fp) + _type)

    def _scheduler(self):
        print(self.__class__.__name__, " scheduler in ", threading.currentThread().getName())
        # list of available and produced to_collect footprints
        to_collect_batch = {key: [] for key in self._primitives.keys()}
        while True:
            time.sleep(1e-2)

            if not self._queries:
                continue

            self._update_graph_from_queries()

            # ordering queries accroding to their pressure
            ordered_queries = sorted(self._queries, key=self._pressure_ratio)
            # getting the emptiest query
            query = ordered_queries[0]

            if not query.produce.to_verb:
                self._num_pending[query] = 0
                self._num_put[query] = 0
                self._queries.remove(query)
                continue

            # if the emptiest query is full, waiting
            if query.produce.verbed.full():
                continue

            # detecting which footprints to collect from the queue + pending
            # while there is space
            while query.produce.verbed.qsize() + self._num_pending[id(query)] < query.produce.verbed.maxsize:
                # getting the first sleeping to_produce
                to_produce_available = [to_produce[0] for to_produce in query.produce.to_verb if to_produce[1] == "sleeping"][0]
                # getting its id in the graph
                to_produce_available_id = self._get_graph_uid(
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
                query.produce.to_verb[query.produce.to_verb.index((to_produce_available, "sleeping"))] = (to_produce_available, "pending")

                self._num_pending[id(query)] += 1
                self._to_produce_collect_occurencies_dict[to_produce_available] += 1

            # testing if at least 1 of the collected queues is empty (1 queue per primitive)
            one_is_empty = False
            for primitive in self._primitives.keys():
                collected_primitive = query.collect.verbed[primitive]
                if collected_primitive.empty():
                    one_is_empty = True

            prim = list(self._primitives.keys())[0]

            # if they are all not empty and can be collected without saturation
            if not one_is_empty and query.collect.to_verb[prim][0] in to_collect_batch[prim]:
                # getting all the collected data
                collected_data = []
                for collected_primitive in query.collect.verbed.keys():
                    collected_data.append(query.collect.verbed[collected_primitive].get(block=False))

                # for each graph edge out of the collected, applying the asyncresult to the out node
                for prim in self._primitives.keys():
                    try:
                        collect_in_edges = self._graph.copy().in_edges(self._get_graph_uid(
                            query.collect.to_verb[prim][0],
                            "to_collect" + prim + str(id(query))
                        ))

                        for edge in collect_in_edges:
                            compute_node = self._graph.nodes[edge[0]]
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
                        query.collect.to_verb[prim].pop(0)
                continue

            # iterating through the graph
            for index, to_produce in enumerate(query.produce.to_verb):
                if to_produce[1] == "sleeping":
                    continue
                else:
                    # beginning at to_produce
                    first_node_id = self._get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_out_occurencies_dict[to_produce[0]]))
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
                                query.produce.verbed.put(node["data"].astype(self._dtype), timeout=1e-2)
                                query.produce.to_verb.pop(0)

                                self._to_produce_out_occurencies_dict[to_produce[0]] += 1
                                self._graph.remove_node(node_id)

                                self._num_pending[id(query)] -= 1

                            continue

                        # skipping the ready to_produce that are not at index 0
                        if node["type"] == "to_produce":
                            continue

                        # if the deepest is to_write, writing the data
                        if node["type"] == "to_write" and node["future"] is None:
                            not_ready_list = [future for future in node["futures"] if not future.ready()]
                            if not not_ready_list:
                                if len(node["data"].shape) == 3 and node["data"].shape[2] == 1:
                                    node["data"] = node["data"].squeeze(axis=-1)

                                node["future"] = node["pool"].apply_async(
                                    node["function"],
                                    (
                                        node["footprint"],
                                        node["data"].astype(self._dtype)
                                    )
                                )
                            continue

                        in_edges = self._graph.copy().in_edges(node_id)

                        if node["future"] is None:
                            node["future"] = node["pool"].apply_async(
                                node["function"],
                                (
                                    node["footprint"],
                                    node["in_data"]
                                )
                            )

                        elif node["future"].ready():
                            in_data = node["future"].get()

                            for in_edge in in_edges:
                                if self._graph.nodes[in_edge[0]]["type"] in ("to_produce", "to_write"):
                                    self._graph.nodes[in_edge[0]]["futures"].append(self._merge_pool.apply_async(
                                        self._burn_data,
                                        (
                                            self._graph.nodes[in_edge[0]]["footprint"],
                                            self._graph.nodes[in_edge[0]]["data"],
                                            node["footprint"],
                                            in_data
                                        )
                                    ))
                                else:
                                    self._graph.nodes[in_edge[0]]["in_data"] = in_data
                                self._graph.remove_edge(*in_edge)


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
            for to_produce in query.produce.to_verb:
                to_produce_uid = self._get_graph_uid(to_produce[0], "to_produce" + str(to_produce_occurencies_dict[to_produce[0]]))
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
            results[primitive] = self._primitives[primitive].get_multi_data_queue(to_collect[primitive])
        return results


    def _update_graph_from_queries(self):
        """
        Updates the dependency graph from the new queries (NO CACHE!)
        """

        while self._new_queries:
            print(self.__class__.__name__, " updating graph ", threading.currentThread().getName())
            new_query = self._new_queries.pop(0)

            # {
            #    "p1": [to_collect_p1_1, ..., to_collect_p1_n],
            #    ...,
            #    "pp": [to_collect_pp_1, ..., to_collect_pp_n]
            # }
            # with p # of primitives and n # of to_compute fps

            # initializing to_collect dictionnary
            new_query.collect.to_verb = {key: [] for key in self._primitives.keys()}

            for to_produce in new_query.produce.to_verb:
                to_produce_uid = self._get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_in_occurencies_dict[to_produce[0]]))
                self._to_produce_in_occurencies_dict[to_produce[0]] += 1
                self._graph.add_node(
                    to_produce_uid,
                    futures=[],
                    footprint=to_produce[0],
                    data=np.zeros(tuple(to_produce[0].shape) + (self._num_bands,)),
                    type="to_produce"
                )

                to_compute = to_produce[0]

                to_compute_uid = self._get_graph_uid(to_compute, "to_compute")

                self._graph.add_node(
                    to_compute_uid,
                    footprint=to_compute,
                    future=None,
                    type="to_compute",
                    pool=self._computation_pool,
                    function=self._compute_data,
                    in_data=None
                )
                self._graph.add_edge(to_produce_uid, to_compute_uid)
                multi_to_collect = self._to_collect_of_to_compute(to_compute)

                for key in multi_to_collect:
                    if multi_to_collect[key] not in new_query.collect.to_verb[key]:
                        new_query.collect.to_verb[key].append(multi_to_collect[key])

                for key in multi_to_collect:
                    to_collect_uid = self._get_graph_uid(multi_to_collect[key], "to_collect" + key + str(id(new_query)))
                    self._graph.add_node(
                        to_collect_uid,
                        footprint=multi_to_collect[key],
                        future=None,
                        type="to_collect",
                        primitive=key
                    )
                    self._graph.add_edge(to_compute_uid, to_collect_uid)

            new_query.collect.verbed = self._collect_data(new_query.collect.to_verb)



    def get_multi_data_queue(self, fp_iterable, queue_size=5):
        """
        returns a queue (could be generator) from a fp_iterable
        """
        query = FullQuery(queue_size)
        to_produce = [(fp, "sleeping") for fp in list(fp_iterable)]

        query.produce.to_verb += to_produce

        self._queries.append(query)
        self._new_queries.append(query)

        return query.produce.verbed


    def get_multi_data(self, fp_iterable, queue_size=5):
        """
        returns a  generator from a fp_iterable
        """
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
                 cache_dir,
                 cache_fps,
                 io_pool,
                 computation_pool,
                 primitives,
                 to_collect_of_to_compute,
                 to_compute_fps):

        super().__init__(footprint, dtype, nbands, nodata, srs,
                         computation_function,
                         io_pool,
                         computation_pool,
                         primitives,
                         to_collect_of_to_compute
                        )

        self._computation_tiles = to_compute_fps

        self._cache_dir = cache_dir
        self._cache_tiles = cache_fps

        # Used to keep duplicates in to_read
        self._to_read_in_occurencies_dict = defaultdict(int)




    def _get_cache_tile_path(self, cache_tile):
        """
        Returns a string, which is a path to a cache tile from its fp
        """
        path = str(
            Path(self._cache_dir) /
            "{:.2f}_{:.2f}_{:.2f}_{}".format(*cache_tile.tl, cache_tile.pxsizex, cache_tile.rsizex)
        )
        return path


    def _read_cache_data(self, cache_tile, _placeholder=None):
        """
        reads cache data
        """
        print(self.__class__.__name__, " reading ", threading.currentThread().getName())
        filepath = self._get_cache_tile_path(cache_tile)

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
        filepath = self._get_cache_tile_path(cache_tile)
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
                                      sr=self._primitives[list(self._primitives.keys())[0]].wkt_origin
                                     )
        out_proxy.set_data(data, band=-1)
        out_proxy.close()


    def _update_graph_from_queries(self):
        """
        Updates the dependency graph from the new queries
        """

        while self._new_queries:
            print(self.__class__.__name__, " updating graph ", threading.currentThread().getName())
            new_query = self._new_queries.pop(0)

            # [
            #    [to_collect_p1_1, ..., to_collect_p1_n],
            #    ...,
            #    [to_collect_pp_1, ..., to_collect_pp_n]
            # ]
            # with p # of primitives and n # of to_compute fps

            # initializing to_collect dictionnary
            new_query.collect.to_verb = {key: [] for key in self._primitives.keys()}

            for to_produce in new_query.produce.to_verb:
                to_produce_uid = self._get_graph_uid(to_produce[0], "to_produce" + str(self._to_produce_in_occurencies_dict[to_produce[0]]))
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
                new_query.read.to_verb.append(to_read_tiles)

                for to_read in to_read_tiles:
                    to_read_uid = self._get_graph_uid(to_read, "to_read" + str(self._to_read_in_occurencies_dict[to_read]))
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

                        new_query.write.to_verb.append(to_write)

                        to_write_uid = self._get_graph_uid(to_write, "to_write")
                        if to_write_uid in self._graph.nodes():
                            self._graph.add_edge(to_read_uid, to_write_uid)
                        else:
                            self._graph.add_node(
                                to_write_uid,
                                footprint=to_write,
                                future=None,
                                futures=[],
                                data=np.zeros(tuple(to_write.shape) + (self._num_bands,)),
                                type="to_write",
                                pool=self._io_pool,
                                function=self._write_cache_data,
                                in_data=None
                            )
                            self._graph.add_edge(to_read_uid, to_write_uid)
                            to_compute_multi = self._to_compute_of_to_write(to_write)
                            new_query.compute.to_verb.append(to_compute_multi)

                            for to_compute in to_compute_multi:
                                to_compute_uid = self._get_graph_uid(to_compute, "to_compute")

                                self._graph.add_node(
                                    to_compute_uid,
                                    footprint=to_compute,
                                    future=None,
                                    type="to_compute",
                                    pool=self._computation_pool,
                                    function=self._compute_data,
                                    in_data=None
                                )
                                self._graph.add_edge(to_write_uid, to_compute_uid)
                                multi_to_collect = self._to_collect_of_to_compute(to_compute)

                                for key in multi_to_collect:
                                    if multi_to_collect[key] not in new_query.collect.to_verb[key]:
                                        new_query.collect.to_verb[key].append(multi_to_collect[key])

                                for key in multi_to_collect:
                                    to_collect_uid = self._get_graph_uid(multi_to_collect[key], "to_collect" + key + str(id(new_query)))
                                    self._graph.add_node(
                                        to_collect_uid,
                                        footprint=multi_to_collect[key],
                                        future=None,
                                        type="to_collect",
                                        primitive=key
                                    )
                                    self._graph.add_edge(to_compute_uid, to_collect_uid)

            new_query.collect.verbed = self._collect_data(new_query.collect.to_verb)


    def _to_read_of_to_produce(self, fp):
        to_read_list = []
        for cache_tile in self._cache_tiles.flat:
            if fp.share_area(cache_tile):
                to_read_list.append(cache_tile)

        return to_read_list

    def _is_written(self, cache_fp):
        return os.path.isfile(self._get_cache_tile_path(cache_fp))

    def _to_compute_of_to_write(self, fp):
        to_compute_list = []
        for computation_tile in self._computation_tiles.flat:
            if fp.share_area(computation_tile):
                to_compute_list.append(computation_tile)

        return to_compute_list





class ResampledRaster(CachedRaster):
    """
    resampled raster from buzzard raster
    """

    def __init__(self, raster, scale, cache_dir, cache_fps):

        full_fp = raster.fp.intersection(raster.fp, scale=scale, alignment=(0, 0))

        num_bands = len(raster)
        nodata = raster.nodata
        wkt_origin = raster.wkt_origin
        dtype = raster.dtype

        primitives = {"primitive": raster}

        computation_pool = mp.pool.ThreadPool()
        io_pool = mp.pool.ThreadPool()

        def compute_data(compute_fp, *data): #*prim_footprints?
            """
            resampled raster compted data when collecting. this is a particular case
            """
            print(self.__class__.__name__, " computing ", threading.currentThread().getName())
            if not hasattr(self._thread_storage, "ds"):
                ds = buzz.DataSource(allow_interpolation=True)
                self._thread_storage.ds = ds
            else:
                ds = self._thread_storage.ds

            with ds.open_araster(self._primitives["primitive"].path).close as prim:
                got_data = prim.get_data(compute_fp, band=-1)

            assert len(data) == 1

            return got_data

        def collect_data(to_collect):
            """
            mocks the behaviour of a primitive so the general function works
            """
            print(self.__class__.__name__, " collecting ", threading.currentThread().getName())
            result = queue.Queue()
            for _ in to_collect["primitive"]:
                result.put([])

            return {"primitive": result}

        def to_collect_of_to_compute(fp):
            """
            mocks the behaviour of a tranformation
            """
            return {"primitive": fp}

        super().__init__(full_fp, dtype, num_bands, nodata, wkt_origin,
                         compute_data, cache_dir, cache_fps, io_pool, computation_pool,
                         primitives, to_collect_of_to_compute, cache_fps
                        )

        self._collect_data = collect_data





class Slopes(Raster):
    """
    slopes from a raster (abstract raster)
    """
    def __init__(self, dsm):
        def compute_data(compute_fp, *data):
            """
            computes up and down slopes
            """
            print(self.__class__.__name__, " computing", threading.currentThread().getName())
            arr, = data
            assert arr.shape == tuple(compute_fp.dilate(1).shape)
            nodata_mask = arr == self._nodata
            nodata_mask = ndi.binary_dilation(nodata_mask)
            kernel = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
            arru = ndi.maximum_filter(arr, None, kernel) - arr
            arru = np.arctan(arru / self.pxsizex)
            arru = arru / np.pi * 180.
            arru[nodata_mask] = self._nodata
            arru = arru[1:-1, 1:-1]

            arrd = arr - ndi.minimum_filter(arr, None, kernel)
            arrd = np.arctan(arrd / self.pxsizex)
            arrd = arrd / np.pi * 180.
            arrd[nodata_mask] = self._nodata
            arrd = arrd[1:-1, 1:-1]

            arr = np.dstack([arrd, arru])
            return arr


        def to_collect_of_to_compute(fp):
            """
            computes to collect from to compute (dilation of 1)
            """
            return {"dsm": fp.dilate(1)}

        full_fp = dsm.fp
        primitives = {"dsm": dsm}
        nodata = dsm.nodata
        num_bands = 2
        dtype = "float32"
        computation_pool = mp.pool.ThreadPool()
        io_pool = mp.pool.ThreadPool()

        super().__init__(full_fp,
                         dtype,
                         num_bands,
                         nodata,
                         None, # dsm.wkt_origin
                         compute_data,
                         io_pool,
                         computation_pool,
                         primitives,
                         to_collect_of_to_compute
                        )






class HeatmapRaster(CachedRaster):
    """
    heatmap raster with primitives: ortho + slopes
    """

    def __init__(self, model, resampled_rgba, slopes, cache_dir, cache_fps):

        def to_collect_of_to_compute(fp):
            """
            Computes the to_collect data from model
            """
            rgba_tile = output_fp_to_input_fp(fp, 0.64, model.get_layer("rgb").input_shape[1])
            slope_tile = output_fp_to_input_fp(fp, 1.28, model.get_layer("slopes").input_shape[1])
            return {"rgba": rgba_tile, "slopes": slope_tile}

        def compute_data(compute_fp, *data):
            """
            predicts data using model
            """
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            rgba_data, slope_data = data

            rgb_data = np.where((rgba_data[..., 3] == 255)[..., np.newaxis], rgba_data, 0)[..., 0:3]
            rgb = (rgb_data.astype('float32') - 127.5) / 127.5

            slopes = slope_data / 45 - 1

            prediction = model.predict([rgb[np.newaxis], slopes[np.newaxis]])[0]
            assert prediction.shape[0:2] == tuple(compute_fp.shape)
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            return prediction


        dtype = "float32"

        num_bands = LABEL_COUNT

        primitives = {"rgba": resampled_rgba, "slopes": slopes}

        max_scale = max(resampled_rgba.fp.scale[0], slopes.fp.scale[0])
        min_scale = min(resampled_rgba.fp.scale[0], slopes.fp.scale[0])

        full_fp = resampled_rgba.fp.intersection(slopes.fp, scale=max_scale, alignment=(0, 0))
        full_fp = full_fp.intersection(full_fp, scale=min_scale)

        computation_tiles = full_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)

        io_pool = mp.pool.ThreadPool()
        computation_pool = mp.pool.ThreadPool(1)

        super().__init__(full_fp, dtype, num_bands, None, None,
                         compute_data, cache_dir, cache_fps, io_pool, computation_pool,
                         primitives, to_collect_of_to_compute, computation_tiles
                        )





def raster_factory(footprint,
                   dtype,
                   nbands,
                   nodata,
                   srs,
                   computation_function,
                   cached,
                   cache_dir,
                   cache_fps,
                   io_pool,
                   computation_pool,
                   primitives,
                   to_collect_of_to_compute,
                   to_compute_fps):#clé -> remplir dict
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
                              cache_dir,
                              cache_fps,
                              io_pool,
                              computation_pool,
                              primitives,
                              to_collect_of_to_compute,
                              to_compute_fps
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
                        to_collect_of_to_compute
                       )

    return raster





def main():
    """
    main program, used for tests
    """
    rgb_path = "./ortho_8.00cm.tif"
    dsm_path = "./dsm_8.00cm.tif"
    model_path = "./18-01-25-15-38-19_1078_1.00000000_0.07799472_aracena.hdf5"

    # Path(CACHE_DIR) / DIR_NAMES[frozenset({"ortho"})]
    # Path(CACHE_DIR) / DIR_NAMES[frozenset({"dsm"})]
    # Path(CACHE_DIR) / DIR_NAMES[frozenset({"ortho", "dsm"})]

    for path in DIR_NAMES.values():
        os.makedirs(str(Path(CACHE_DIR) / path), exist_ok=True)

    datasrc = buzz.DataSource(allow_interpolation=True)

    print("model...")

    model = load_model(model_path)
    model._make_predict_function()
    print("")

    with datasrc.open_araster(rgb_path).close as raster:
        out_fp = raster.fp.intersection(raster.fp, scale=1.28, alignment=(0, 0))

    tile_count128 = np.ceil(out_fp.rsize / 500)
    cache_tiles128 = out_fp.tile_count(*tile_count128, boundary_effect='shrink')

    out_fp = out_fp.intersection(out_fp, scale=0.64)

    tile_count64 = np.ceil(out_fp.rsize / 500)
    cache_tiles64 = out_fp.tile_count(*tile_count64, boundary_effect='shrink')

    initial_rgba = datasrc.open_araster(rgb_path)
    initial_dsm = datasrc.open_araster(dsm_path)

    resampled_rgba = ResampledRaster(initial_rgba, 0.64, str(Path(CACHE_DIR) / DIR_NAMES[frozenset({"ortho"})]), cache_tiles64)
    resampled_dsm = ResampledRaster(initial_dsm, 1.28, str(Path(CACHE_DIR) / DIR_NAMES[frozenset({"dsm"})]), cache_tiles128)

    slopes = Slopes(resampled_dsm)

    hmr = HeatmapRaster(model, resampled_rgba, slopes, str(Path(CACHE_DIR) / DIR_NAMES[frozenset({"ortho", "dsm"})]), cache_tiles64)

    big_display_fp = out_fp
    big_dsm_disp_fp = big_display_fp.intersection(big_display_fp, scale=1.28, alignment=(0, 0))

    tile_count64 = np.ceil(out_fp.rsize / 100)
    display_tiles = big_display_fp.tile_count(*tile_count64, boundary_effect='shrink')
    dsm_display_tiles = big_dsm_disp_fp.tile_count(5, 5, boundary_effect='shrink')

    # rgba_out = resampled_rgba.get_multi_data(list(cache_tiles64.flat), 1)
    # slopes_out = slopes.get_multi_data(list(cache_tiles128.flat), 1)
    hm_out = hmr.get_multi_data(cache_tiles64.flat, 1)

    for display_fp in cache_tiles64.flat:
        try:
            show_many_images(
                [np.argmax(next(hm_out), axis=-1)],
                extents=[display_fp.extent]
            )
        except StopIteration:
            print("ended")

    initial_rgba.close()
    initial_dsm.close()

if __name__ == "__main__":
    main()
