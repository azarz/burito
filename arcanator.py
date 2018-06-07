from pathlib import Path
import multiprocessing as mp
import multiprocessing.pool
import os
import queue
import threading
import hashlib
import sys
import time
from collections import defaultdict

from keras.models import load_model

import scipy.ndimage as ndi
import numpy as np
import shapely.geometry as sg
import networkx as nx
import buzzard as buzz
import matplotlib.pyplot as plt

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

STEP_NAME_INDEX = {
    0: "to_produce",
    1: "to_read",
    2: "to_write",
    3: "to_compute",
    4: "to_collect"
}


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


class AbstractRaster(object):
    """
    Abstract class defining the raster behaviour
    """
    def __init__(self, full_fp):
        self._queries = []
        self._new_queries = []
        self._primitives = {}

        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._scheduler_thread.start()

        self._computation_pool = mp.pool.ThreadPool()
        self._io_pool = mp.pool.ThreadPool()
        self._produce_pool = mp.pool.ThreadPool()

        self._thread_storage = threading.local()

        self._full_fp = full_fp

        self._num_bands = None  # to implement in subclasses
        self._nodata = None     # to implement in subclasses
        self._wkt_origin = None # to implement in subclasses
        self._dtype = None      # to implement in subclasses

        self._graph = nx.DiGraph()

        # Used to keep duplicates in to_produce
        self._to_produce_in_occurencies_dict = defaultdict(int)
        self._to_produce_out_occurencies_dict = defaultdict(int)


    def _pressure_ratio(self, query):
        """
        defines a pressure ration of a query: lesser values -> emptier query
        """
        num = query.produce.verbed.qsize() + len(query.produce.staging)
        den = query.produce.verbed.maxsize
        return num/den

    def _prepare_query(self, query):
        return

    @property
    def fp(self):
        return self._full_fp

    @property
    def nodata(self):
        return self._nodata

    @property
    def wkt_origin(self):
        return self._wkt_origin

    @property
    def dtype(self):
        return self._dtype

    @property
    def pxsizex(self):
        return self._full_fp.pxsizex



    def __len__(self):
        return int(self._num_bands)


    def _burn_data(self, big_fp, to_fill_data, to_burn_fp, to_burn_data):
        to_fill_data[to_burn_fp.slice_in(big_fp, clip=True)] = to_burn_data[big_fp.slice_in(to_burn_fp, clip=True)]

    def _compute_data(self, compute_fp, data):
        raise NotImplementedError('Should be implemented by all subclasses')

    def _get_graph_uid(self, fp, _type):
        return hash(repr(fp) + _type)

    def _scheduler(self):
        print(self.__class__.__name__, " scheduler in ", threading.currentThread().getName())

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
                self._queries.remove(query)
                continue

            # if the emptiest query is full, wainting
            if query.produce.verbed.full():
                continue

            # testing if at least 1 of the collected queues is empty (1 queue per primitive)
            one_is_empty = False
            for primitive in self._primitives.keys():
                collected_primitive = query.collect.verbed[primitive]
                if collected_primitive.empty():
                    one_is_empty = True


            # if they are all full
            if not one_is_empty:
                # getting all the collected data
                collected_data = []
                for collected_primitive in query.collect.verbed.keys():
                    collected_data.append(query.collect.verbed[collected_primitive].get())

                # for each graph edge out of the collected, applying the asyncresult to the out node
                for prim in self._primitives.keys():
                    collect_out_edges = (self._graph.copy().out_edges(self._get_graph_uid(query.collect.to_verb[prim][0], "to_collect")))

                    for edge in collect_out_edges:
                        self._graph.nodes[edge[1]]["future"] = self._computation_pool.apply_async(
                            self._compute_data,
                            (
                                self._graph.nodes[edge[1]]["footprint"],
                                *collected_data
                            )
                        )
                        self._graph.remove_edge(*edge)

                    query.collect.to_verb[prim].pop(0)
                continue

            # iterating through the graph
            for index, to_produce in enumerate(query.produce.to_verb):
                # beginning at to_produce
                node_id = self._get_graph_uid(to_produce, "to_produce" + str(self._to_produce_out_occurencies_dict[to_produce]))
                # going as deep as possible (upstream the edges)
                while len(self._graph.in_edges(node_id)) > 0:
                    node_id = list(self._graph.in_edges(node_id))[0][0]

                node = self._graph.nodes[node_id]

                if node["type"] == "to_collect":
                    continue

                # if the deepest is to_produce, updating produced
                if index == 0 and node["type"] == "to_produce":
                    not_ready_list = [future for future in node["futures"] if not future.ready()]
                    if not not_ready_list:
                        try:
                            if len(node["data"].shape) == 3 and node["data"].shape[2] == 1:
                                node["data"] = node["data"].squeeze(axis=-1)
                            query.produce.verbed.put(node["data"].astype(self._dtype), timeout=1e-2)
                            query.produce.to_verb.pop(0)
                            self._to_produce_out_occurencies_dict[to_produce] += 1
                            self._graph.remove_node(node_id)
                        except queue.Full:
                            continue
                    continue

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


                out_edges = self._graph.copy().out_edges(node_id)

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

                    for out_edge in out_edges:
                        if self._graph.nodes[out_edge[1]]["type"] in ("to_produce", "to_write"):
                            self._graph.nodes[out_edge[1]]["futures"].append(self._graph.nodes[out_edge[1]]["pool"].apply_async(
                                self._burn_data,
                                (
                                    self._graph.nodes[out_edge[1]]["footprint"],
                                    self._graph.nodes[out_edge[1]]["data"],
                                    node["footprint"],
                                    in_data
                                )
                            ))
                        else:
                            self._graph.nodes[out_edge[1]]["in_data"] = in_data
                        self._graph.remove_edge(*out_edge)

                break

            self._clean_graph()



    def _clean_graph(self):
        # Used to keep duploicates in to_produce
        to_produce_occurencies_dict = self._to_produce_out_occurencies_dict.copy()

        to_remove = list(nx.isolates(self._graph))

        for query in self._queries:
            for to_produce in query.produce.to_verb:
                to_produce_uid = self._get_graph_uid(to_produce, "to_produce" + str(to_produce_occurencies_dict[to_produce]))
                to_produce_occurencies_dict[to_produce] += 1
                while to_produce_uid in to_remove:
                    to_remove.remove(to_produce_uid)

        self._graph.remove_nodes_from(to_remove)



    def _collect_data(self, to_collect):
        # in: [
        #    [to_collect_p1_1, ..., to_collect_p1_n],
        #    ...,
        #    [to_collect_pp_1, ..., to_collect_pp_n]
        #]
        # out: [queue_1, queue_2, ..., queue_p]
        results = {}
        for primitive in self._primitives.keys():
            results[primitive] = self._primitives[primitive].get_multi_data_queue(to_collect[primitive])
        return results


    def _update_graph_from_queries(self):
        raise NotImplementedError()


    def get_multi_data_queue(self, fp_iterable, queue_size=5):
        """
        returns a queue (could be generator) from a fp_iterable
        """
        query = FullQuery(queue_size)
        to_produce = list(fp_iterable.copy())

        query.produce.to_verb = to_produce

        self._queries.append(query)
        self._new_queries.append(query)

        return query.produce.verbed


    def get_multi_data(self, fp_iterable, queue_size=5):
        """
        returns a  generator from a fp_iterable
        """
        out_queue = self.get_multi_data_queue(fp_iterable, queue_size)
        def out_generator():
            for fp in fp_iterable:
                assert fp.same_grid(self.fp)
                result = out_queue.get()
                assert np.array_equal(fp.shape, result.shape[0:2])
                yield result

        return out_generator()


    def get_data(self, fp):
        return next(self.get_multi_data([fp]))





class AbstractCachedRaster(AbstractRaster):
    """
    Cached implementation of abstract raster
    """
    def __init__(self, full_fp, rtype):
        super().__init__(full_fp)
        self._rtype = rtype

        tile_count = np.ceil(self._full_fp.rsize / 500)
        self._cache_tiles = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')
        self._computation_tiles = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        # Used to keep duplicates in to_read
        self._to_read_in_occurencies_dict = defaultdict(int)




    def _get_cache_tile_path(self, cache_tile):
        """
        Returns a string, which is a path to a cache tile from its fp
        """
        path = str(
            Path(CACHE_DIR) /
            DIR_NAMES[frozenset({*self._rtype})] /
            "{:.2f}_{:.2f}_{:.2f}_{}".format(*cache_tile.tl, cache_tile.pxsizex, cache_tile.rsizex)
        )
        return path


    def _read_cache_data(self, cache_tile, _placeholder=None):
        print(self.__class__.__name__, " reading in ", threading.currentThread().getName())
        filepath = self._get_cache_tile_path(cache_tile)

        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds
        else:
            ds = self._thread_storage.ds

        with ds.open_araster(filepath).close as src:
            data = src.get_data(band=-1, fp=cache_tile)
        print(self.__class__.__name__, " reading out ", threading.currentThread().getName())
        return data


    def _write_cache_data(self, cache_tile, data):
        print(self.__class__.__name__, " writing in ", threading.currentThread().getName())
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
        print(self.__class__.__name__, " writing out ", threading.currentThread().getName())


    def _update_graph_from_queries(self):
        """
        Updates the dependency graph from the new queries
        """

        while self._new_queries:
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
                to_produce_uid = self._get_graph_uid(to_produce, "to_produce" + str(self._to_produce_in_occurencies_dict[to_produce]))
                self._to_produce_in_occurencies_dict[to_produce] += 1
                self._graph.add_node(
                    to_produce_uid,
                    footprint=to_produce,
                    futures=[],
                    data=np.zeros(tuple(to_produce.shape) + (self._num_bands,)),
                    type="to_produce",
                    pool=self._produce_pool,
                    in_data=None
                )
                to_read_tiles = self._to_read_of_to_produce(to_produce)
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
                    self._graph.add_edge(to_read_uid, to_produce_uid)

                    # if the tile is not written, writing it
                    if not self._is_written(to_read):
                        to_write = to_read

                        new_query.write.to_verb.append(to_write)

                        to_write_uid = self._get_graph_uid(to_write, "to_write")
                        if to_write_uid in self._graph.nodes():
                            self._graph.add_edge(to_write_uid, to_read_uid)
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
                            self._graph.add_edge(to_write_uid, to_read_uid)
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
                                self._graph.add_edge(to_compute_uid, to_write_uid)
                                multi_to_collect = self._to_collect_of_to_compute(to_compute)

                                for key, to_collect_primitive in zip(self._primitives.keys(), multi_to_collect):
                                    if to_collect_primitive not in new_query.collect.to_verb[key]:
                                        new_query.collect.to_verb[key].append(to_collect_primitive)

                                for to_collect in multi_to_collect:
                                    to_collect_uid = self._get_graph_uid(to_collect, "to_collect")
                                    self._graph.add_node(
                                        to_collect_uid,
                                        footprints=to_collect,
                                        future=None,
                                        type="to_collect"
                                    )
                                    self._graph.add_edge(to_collect_uid, to_compute_uid)

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


    def _to_collect_of_to_compute(self, fp):
        raise NotImplementedError()






class AbstractNotCachedRaster(AbstractRaster):
    def __init__(self, full_fp):
        super().__init__(full_fp)


    def _update_graph_from_queries(self):
        """
        Updates the dependency graph from the new queries
        """

        while self._new_queries:
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
                to_produce_uid = self._get_graph_uid(to_produce, "to_produce" + str(self._to_produce_in_occurencies_dict[to_produce]))
                self._to_produce_in_occurencies_dict[to_produce] += 1
                self._graph.add_node(
                    to_produce_uid,
                    futures=[],
                    footprint=to_produce,
                    data=np.zeros(tuple(to_produce.shape) + (self._num_bands,)),
                    type="to_produce",
                    pool=self._produce_pool
                )

                to_compute = to_produce

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
                self._graph.add_edge(to_compute_uid, to_produce_uid)
                multi_to_collect = self._to_collect_of_to_compute(to_compute)

                for key, to_collect_primitive in zip(self._primitives.keys(), multi_to_collect):
                    if to_collect_primitive not in new_query.collect.to_verb[key]:
                        new_query.collect.to_verb[key].append(to_collect_primitive)

                for to_collect in multi_to_collect:
                    to_collect_uid = self._get_graph_uid(to_collect, "to_collect")
                    self._graph.add_node(
                        to_collect_uid,
                        footprints=to_collect,
                        future=None,
                        type="to_collect"
                    )
                    self._graph.add_edge(to_collect_uid, to_compute_uid)

            new_query.collect.verbed = self._collect_data(new_query.collect.to_verb)


    def _to_collect_of_to_compute(self, fp):
        raise NotImplementedError()

    def _compute_data(self, compute_fp, data):
        raise NotImplementedError('Should be implemented by all subclasses')






class ResampledRaster(AbstractCachedRaster):

    def __init__(self, raster, scale, rtype):

        full_fp = raster.fp.intersection(raster.fp, scale=scale, alignment=(0, 0))

        super().__init__(full_fp, rtype)

        tile_count = np.ceil(self._full_fp.rsize / 500)
        self._cache_tiles = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')
        self._num_bands = len(raster)
        self._nodata = raster.nodata
        self._wkt_origin = raster.wkt_origin
        self._dtype = raster.dtype

        self._primitives = {"primitive": raster}



    def _compute_data(self, compute_fp, data):
        print(self.__class__.__name__, " computing data ", threading.currentThread().getName())

        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds
        else:
            ds = self._thread_storage.ds

        with ds.open_araster(self._primitives["primitive"].path).close as prim:
            data = prim.get_data(compute_fp, band=-1)

        print(self.__class__.__name__, " computed data ", threading.currentThread().getName())
        return data

    def _collect_data(self, to_collect):
        """
        mocks the behaviour of a primitive so the general function works
        """
        result = queue.Queue()
        for _ in to_collect["primitive"]:
            result.put([])
        return {"primitive": result}

    def _to_collect_of_to_compute(self, fp):
        return [fp]






class ResampledOrthoimage(ResampledRaster):

    def __init__(self, raster, scale):
        super().__init__(raster, scale, ("ortho",))




class ResampledDSM(ResampledRaster):

    def __init__(self, raster, scale):
        super().__init__(raster, scale, ("dsm",))
        self._dtype = "float32"




class Slopes(AbstractNotCachedRaster):
    def __init__(self, dsm):
        super().__init__(dsm.fp)
        self._primitives = {"dsm": dsm}
        self._nodata = dsm.nodata
        self._num_bands = 2
        self._dtype = "float32"


    def _compute_data(self, _compute_fp, data):
        print(self.__class__.__name__, " computing data ", threading.currentThread().getName())
        arr = data
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
        arru[nodata_mask] = 0
        arru = arru[1:-1, 1:-1]

        arrd = arr - ndi.minimum_filter(arr, None, kernel)
        arrd = np.arctan(arrd / self.pxsizex)
        arrd = arrd / np.pi * 180.
        arrd[nodata_mask] = 0
        arrd = arrd[1:-1, 1:-1]

        arr = np.dstack([arrd, arru])
        return arr


    def _to_collect_of_to_compute(self, fp):
        return [fp.dilate(1)]




class HeatmapRaster(AbstractCachedRaster):

    def __init__(self, model, resampled_rgba, slopes):

        max_scale = max(resampled_rgba.fp.scale[0], slopes.fp.scale[0])
        min_scale = min(resampled_rgba.fp.scale[0], slopes.fp.scale[0])

        full_fp = resampled_rgba.fp.intersection(slopes.fp, scale=max_scale, alignment=(0, 0))

        super().__init__(full_fp, ("dsm", "ortho"))
        self._computation_pool = mp.pool.ThreadPool(5)

        self._dtype = "float32"

        self._model = model
        self._num_bands = LABEL_COUNT

        self._primitives = {"rgba": resampled_rgba, "slopes": slopes}

        self._full_fp = self._full_fp.intersection(self._full_fp, scale=min_scale)

        self._computation_tiles = self._full_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)

        tile_count = np.ceil(self._full_fp.rsize / 500)
        self._cache_tiles = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._computation_pool = mp.pool.ThreadPool(1)


    def _to_collect_of_to_compute(self, fp):
        rgba_tile = output_fp_to_input_fp(fp, 0.64, self._model.get_layer("rgb").input_shape[1])
        dsm_tile = output_fp_to_input_fp(fp, 1.28, self._model.get_layer("slopes").input_shape[1])
        return [rgba_tile, dsm_tile]


    def _compute_data(self, _compute_fp, rgba_data, slope_data):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", self.__class__.__name__, " computing data ", threading.currentThread().getName())
        rgb_data = np.where((rgba_data[..., 3] == 255)[..., np.newaxis], rgba_data, 0)[..., 0:3]
        rgb = (rgb_data.astype('float32') - 127.5) / 127.5

        slopes = slope_data / 45 - 1

        prediction = self._model.predict([rgb[np.newaxis], slopes[np.newaxis]])[0]
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", self.__class__.__name__, " computed data ", threading.currentThread().getName())
        return prediction




def main():
    rgb_path = "./ortho_8.00cm.tif"
    dsm_path = "./dsm_8.00cm.tif"
    model_path = "./18-01-25-15-38-19_1078_1.00000000_0.07799472_aracena.hdf5"

    CACHE_DIR = "./.cache"

    for path in DIR_NAMES.values():
        os.makedirs(str(Path(CACHE_DIR) / path), exist_ok=True)

    datasrc = buzz.DataSource(allow_interpolation=True)

    print("model...")

    model = load_model(model_path)
    model._make_predict_function()
    print("")

    with datasrc.open_araster(rgb_path).close as raster:
        out_fp = raster.fp.intersection(raster.fp, scale=1.28, alignment=(0, 0))

    out_fp = out_fp.intersection(out_fp, scale=0.64)


    initial_rgba = datasrc.open_araster(rgb_path)
    initial_dsm = datasrc.open_araster(dsm_path)

    resampled_rgba = ResampledOrthoimage(initial_rgba, 0.64)
    resampled_dsm = ResampledDSM(initial_dsm, 1.28)

    slopes = Slopes(resampled_dsm)

    hmr = HeatmapRaster(model, resampled_rgba, slopes)

    big_display_fp = out_fp
    big_dsm_disp_fp = big_display_fp.intersection(big_display_fp, scale=1.28, alignment=(0, 0))

    display_tiles = big_display_fp.tile_count(5, 5, boundary_effect='shrink')
    dsm_display_tiles = big_dsm_disp_fp.tile_count(5, 5, boundary_effect='shrink')

    # rgba_out = resampled_rgba.get_multi_data(list(display_tiles.flat), 5)
    dsm_out = resampled_dsm.get_multi_data(dsm_display_tiles.flat, 5)
    # slopes_out = slopes.get_multi_data(list(dsm_display_tiles.flat), 5)
    # hm_out = hmr.get_multi_data(display_tiles.flat, 5)

    for display_fp, dsm_disp_fp in zip(display_tiles.flat, dsm_display_tiles.flat):
        try:
            next(dsm_out)
            # show_many_images(
            #     [next(rgba_out), next(slopes_out)[...,0], np.argmax(next(hm_out), axis=-1)], 
            #     extents=[display_fp.extent, dsm_disp_fp.extent, display_fp.extent]
            # )
        except StopIteration:
            print("ended")

    initial_rgba.close()
    initial_dsm.close()

if __name__ == "__main__":
    main()