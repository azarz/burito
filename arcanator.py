from pathlib import Path
from collections import defaultdict
import functools
import multiprocessing as mp
import multiprocessing.pool
import os
import pickle
import queue
import threading
import hashlib
import sys
import time

from keras.models import load_model
import buzzard as buzz
import scipy.ndimage as ndi
import numpy as np
import shapely.geometry as sg
import networkx as nx
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
    frozenset(["ortho","dsm"]): "both"
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
    out = buzz.Footprint(tl=fp.tl, size=fp.size, rsize=fp.size/scale)
    padding = (rsize - out.rsizex) / 2
    assert padding == int(padding)
    out = out.dilate(padding)
    return out

class DummyFuture(object):
    def ready():
        return False

    def get():
        raise NotImplementedError()

class AbstractRaster(object):

    def __init__(self, full_fp, cached):
        self._queries = []
        self._new_queries = []
        self._primitives = []

        self._cached = cached

        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)

        self._scheduler_thread.start()
        self._cv = threading.Condition()

        self._computation_pool = mp.pool.ThreadPool()
        self._io_pool = mp.pool.ThreadPool()
        self._produce_pool = mp.pool.ThreadPool()

        self._thread_storage = threading.local()
    
        ds = buzz.DataSource(allow_interpolation=True)

        self._full_fp = full_fp
    
        self._num_bands = None # to implement in subclasses
        self._nodata = None # to implement in subclasses
        self._wkt_origin = None # to implement in subclasses
        self._dtype = None # to implement in subclasses


        comp_tile_count = np.ceil(self._full_fp.rsize / 500)
        self._computation_tiles = self._full_fp.tile_count(*comp_tile_count, boundary_effect='shrink')            


    def _pressure_ratio(self, query):
        num = query.produce.verbed.qsize() + len(query.produce.staging)
        den = query.produce.verbed.maxsize
        return num/den


    def _merge_out_tiles(self, tiles, data, out_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

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
        return self._num_bands
    

    def _produce_data(self, input_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    def _compute_data(self, input_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    def get_multi_data(self, fp_iterable, queue_size=5):
        query = FullQuery(queue_size)
        to_produce = list(fp_iterable.copy())

        query.produce.to_verb = to_produce

        self._prepare_query(query)
 
        self._queries.append(query)
        self._new_queries.append(query)

        # def out_generator():
        #     for fp in fp_iterable:
        #         assert fp.same_grid(self.fp)
        #         result = query.produce.verbed.get()
        #         assert np.array_equal(fp.shape, result.shape[0:2])
        #         yield result

        # return out_generator()
        return query.produce.verbed


    def get_data(self, fp):
        # return next(self.get_multi_data([fp]))
        return self.get_multi_data([fp]).get()
    




class AbstractCachedRaster(AbstractRaster):
    def __init__(self, full_fp, rtype):
        super().__init__(full_fp, True)
        self._rtype = rtype

        tile_count = np.ceil(self._full_fp.rsize / 500) 
        self._cache_tiles = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._graph = nx.DiGraph()

        
    def _get_cache_tile_path(self, cache_tile):
        path = str(
            Path(CACHE_DIR) / 
            DIR_NAMES[frozenset({*self._rtype})] / 
            "{:.2f}_{:.2f}_{:.2f}_{}".format(*cache_tile.tl, cache_tile.pxsizex, cache_tile.rsizex)
        )
        return path

    def _read_cache_data(self, cache_tile, placeholder=None):
        ds = buzz.DataSource(allow_interpolation=True)
        filepath = self._get_cache_tile_path(cache_tile)

        with ds.open_araster(filepath).close as src:
            data = src.get_data(band=-1, fp=cache_tile)
        return data


    def _write_cache_data(self, cache_tile, data):
        filepath = self._get_cache_tile_path(cache_tile)
        ds = buzz.DataSource(allow_interpolation=True)

        out_proxy = ds.create_araster(filepath, cache_tile, data.dtype, self._num_bands, driver="GTiff", sr=self._primitives[0].wkt_origin)
        out_proxy.set_data(data, band=-1)
        out_proxy.close()


    def _get_graph_uid(self, fp, _type):
        return hash(repr(fp) + _type)


    def _scheduler(self):
        print(self.__class__.__name__, " scheduler in ", threading.currentThread().getName())

        while True:
            time.sleep(1e-2)
            if not self._queries:
                continue      

            print(self.__class__.__name__, " yes queries ", threading.currentThread().getName())
            self._update_graph_from_queries()

            ordered_queries = sorted(self._queries, key=self._pressure_ratio)
            query = ordered_queries[0]

            one_is_empty = False
            for collected_primitive in query.collect.verbed:
                if collected_primitive.empty():
                    one_is_empty = True


            if not one_is_empty:
                for collected_primitive in range(len(query.collect.verbed)):
                    out_data.append(query.collect.verbed[collected_primitive].get())

                collect_out_edges = self._graph.out_edges(query.collect.to_verb[0])
                for edge in collect_out_edges:
                    self._graph.nodes[edge[1]]["future"] = self._computation_pool.apply_async(
                        self._compute_data, 
                        (self._graph.nodes[edge[1]]["footprint"], *out_data)
                    )
                    self._graph.remove_edge(edge)

                continue


            for index, to_produce in enumerate(query.produce.to_verb):
                node = self._get_graph_uid(to_produce, "to_produce")
                while len(self._graph.in_edges(node)) > 0:
                    node = list(self._graph.in_edges(node))[0][0]

                if not node.future.ready():
                    continue

                if index == 0 and node == self._get_graph_uid(to_produce, "to_produce"):
                    query.produce.verbed.put(node["future"].get())
                    query.produce.to_verb.pop(0)

                out_edges = self._graph.out_edges(node)
                for out_edge in out_edges:
                    self._graph.nodes[out_edge[1]]["future"] = out_edge["pool"].apply_async(
                        out_edge["function"], 
                        (
                            self._graph.nodes[out_edge[1]]["footprint"], 
                            node.future.get()
                        )
                    )
                    self._graph.remove_edge(out_edge)

                break

            self._clean_graph()


    def _update_graph_from_queries(self):
        while self._new_queries:
            new_query = self._new_queries.pop(0)
            new_query.collect.to_verb = [[] for p in self._primitives]

            for to_produce in new_query.produce.to_verb:
                to_produce_uid = self._get_graph_uid(to_produce, "to_produce")
                self._graph.add_node(to_produce_uid, footprint=to_produce, future=DummyFuture())
                to_read_tiles = self._to_read_of_to_produce(to_produce)
                new_query.read.to_verb.append(to_read_tiles)

                for to_read in to_read_tiles:
                    to_read_uid = self._get_graph_uid(to_read, "to_read")

                    if self._is_written(to_read):
                        self._graph.add_node(
                            to_read_uid, 
                            footprint=to_read, 
                            future=self._io_pool.apply_async(
                                self._read_cache_data, 
                                to_read
                            )
                        )

                        self._graph.add_edge(to_read_uid, to_produce_uid, pool=self._produce_pool, function=self._produce_data)

                    else:
                        to_write = to_read

                        self._graph.add_node(to_read_uid, footprint=to_read, future=DummyFuture())
                        self._graph.add_edge(to_read_uid, to_produce_uid, pool=self._produce_pool, function=self._produce_data)

                        new_query.write.to_verb.append(to_write)

                        to_write_uid = self._get_graph_uid(to_write, "to_write")

                        self._graph.add_node(to_write_uid, footprint=to_write, future=DummyFuture())
                        self._graph.add_edge(to_write_uid, to_read_uid, pool=self._io_pool, function=self._read_cache_data)
                        new_query.compute.to_verb.append(self._to_compute_of_to_write(to_write))

                        for to_compute in new_query.compute.to_verb:
                            print(len(new_query.compute.to_verb))
                            to_compute_uid = self._get_graph_uid(to_compute, "to_compute")

                            self._graph.add_node(to_compute_uid, footprint=to_compute, future=DummyFuture())
                            self._graph.add_edge(to_compute_uid, to_write_uid, pool=self._io_pool, function=self._write_cache_data)
                            multi_to_collect = self._to_collect_of_to_compute(to_compute)

                            for index, to_collect_primitive in enumerate(multi_to_collect):
                                new_query.collect.to_verb[index].append(to_collect_primitive)

                            for to_collect in new_query.collect.to_verb:
                                to_collect_uid = self._get_graph_uid(to_collect, "to_collect")
                                self._graph.add_node(to_collect_uid, footprints=to_collect, future=DummyFuture())
                                self._graph.add_edge(to_collect_uid, to_compute_uid, pool=self._io_pool)

            new_query.collect.verbed = self._collect_data(new_query.collect.to_verb)


    def _collect_data(self, to_collect):
        results = []
        for primitive, to_collect_batch in zip(self._primitives, to_collect):
            results.append(primitive.get_data(to_collect_batch))
        return results

    def _clean_graph(self):
        self._graph.remove_nodes_from(nx.isolates(self._graph))

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


    def _to_collect_of_to_compute(self, unique_fp):
        raise NotImplementedError()









class AbstractNotCachedRaster(AbstractRaster):
    def __init__(self, full_fp):
        super().__init__(full_fp, False)


    def _scheduler(self):
        print(self.__class__.__name__, " scheduler in ", threading.currentThread().getName())
        while True:
            ordered_queries = sorted(self._queries, key=self._pressure_ratio)

            for query in ordered_queries:
                # print(self.__class__.__name__, "  ", query,  " queues:   ", threading.currentThread().getName())
                # print("       produced:  ", query.produce.verbed.qsize())
                # print("       computed:  ", query.compute.verbed.qsize())
                # print("       collected:  ", query.collect.verbed.qsize())

                while query.compute.to_verb:
                    to_compute_fp = query.compute.to_verb.pop(0)
                    query.collect.to_verb.append(to_compute_fp)
                    query.collect.staging.append(self._io_pool.apply_async(self._collect_data, (to_compute_fp,)))

                # If all to_produce was consumed, the query has ended
                if not query.produce.to_verb:
                    self._queries.remove(query)
                    del query
                    continue

                if query.collect.staging and query.collect.staging[0].ready():
                    collected_data = query.collect.staging.pop(0).get()
                    query.compute.staging.append(self._computation_pool.apply_async(self._compute_data, (collected_data,)))

                if query.compute.staging and query.compute.staging[0].ready() and not query.produce.verbed.full():
                    query.produce.verbed.put(query.compute.staging.pop(0).get())
                    del query.produce.to_verb[0]


            time.sleep(1e-2)



    def _to_produce_to_to_compute(self, query):
        if not query.produce.to_verb:
            return
        elif self._cached:
            raise RuntimeError()
        else:
            for to_produce in query.produce.to_verb:
                query.compute.to_verb.append(to_produce)


    def _prepare_query(self, query):
        self._to_produce_to_to_compute(query)






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

        self._primitives = [raster]
    
    
    def _computation_method(self, input_fp):
        print(self.__class__.__name__, " computing data ", threading.currentThread().getName())
        return self._collect_data(input_fp)

    def _collect_data(self, input_fp):
        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds
        else:
            ds = self._thread_storage.ds

        with ds.open_araster(self._primitives[0].path).close as prim:
            data = prim.get_data(input_fp)

        assert np.array_equal(input_fp.shape, data.shape[0:2])
        return data

    def _to_collect_of_to_compute(self, unique_fp):
        return [unique_fp]






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
        self._primitives = [dsm]
        self._nodata = dsm.nodata
        self._num_bands = 2
        self._dtype = "float32"    


    def _compute_data(self, data):
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


    def _collect_data(self, input_fp):
        data = self._primitives[0].get_data(input_fp.dilate(1))
        return data





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

        self._primitives = [resampled_rgba, slopes]
      
        self._full_fp = self._full_fp.intersection(self._full_fp, scale=min_scale)

        self._computation_tiles = self._full_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)

        tile_count = np.ceil(self._full_fp.rsize / 500) 
        self._cache_tiles = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._lock = threading.Lock()

        self._computation_pool = mp.pool.ThreadPool(1)


    def _to_collect_of_to_compute(self, unique_fp):
        rgba_tile = output_fp_to_input_fp(unique_fp, 0.64, self._model.get_layer("rgb").input_shape[1])
        dsm_tile = output_fp_to_input_fp(unique_fp, 1.28, self._model.get_layer("slopes").input_shape[1])
        return [rgba_tile, dsm_tile]


    def _computation_method(self, computation_tile, rgba_data, slope_data):
        rgb_data = np.where((rgba_data[...,3] == 255)[...,np.newaxis], rgba_data, 0)[...,0:3]
        rgb = (rgb_data.astype('float32') - 127.5) / 127.5

        slopes = slope_data / 45 - 1

        with self._lock:
            prediction = self._model.predict([rgb[np.newaxis], slopes[np.newaxis]])[0]

        return prediction


    def _collect_rgba_data(self, input_fp):
        self._primitives[0].get_data(input_fp)

    def _collect_slope_data(self, input_fp):
        self._primitives[0].get_data(input_fp)





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

    # hmr = HeatmapRaster(model, resampled_rgba, slopes)

    big_display_fp = out_fp
    big_dsm_disp_fp = big_display_fp.intersection(big_display_fp, scale=1.28, alignment=(0, 0))

    display_tiles = big_display_fp.tile_count(5, 5, boundary_effect='shrink')
    dsm_display_tiles = big_dsm_disp_fp.tile_count(5, 5, boundary_effect='shrink')


    # hm_out = hmr.get_multi_data(display_tiles.flat, 5)
    # rgba_out = resampled_rgba.get_multi_data(display_tiles.flat, 5)
    dsm_out = resampled_dsm.get_multi_data(dsm_display_tiles.flat, 5)
    slopes_out = slopes.get_multi_data(dsm_display_tiles.flat, 5)

    def out_generator(tiles, out_q):
        for fp in tiles.flat:
            result = out_q.get()
            assert np.array_equal(fp.shape, result.shape[0:2])
            yield result

    slopes_out_gen = out_generator(dsm_display_tiles, slopes_out)
    dsm_out_gen = out_generator(dsm_display_tiles, dsm_out)

    for display_fp, dsm_disp_fp in zip(display_tiles.flat, dsm_display_tiles.flat):
        try:
            next(slopes_out_gen)
            # next(slopes_out)
            # show_many_images(
            #     [next(rgba_out), next(slopes_out)[...,0],np.argmax(next(hm_out), axis=-1)], 
            #     extents=[display_fp.extent, dsm_disp_fp.extent, display_fp.extent]
            # )
        except StopIteration:
            print("ended")

if __name__ == "__main__":
    main()