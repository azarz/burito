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

from show_many_images import show_many_images
from uids_of_paths import uids_of_paths
from watcher import Watcher

from Query import FullQuery
from DoubleTiledStructure import DoubleTiledStructure

CATEGORIES= (
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
#         "ortho": rgb_path,
#         "dsm": dsm_path
#     })
DIR_NAMES = {
        frozenset(["ortho"]): "rgb",
        frozenset(["dsm"]): "dsm",
        frozenset(["ortho","dsm"]): "both"
    }

CACHE_DIR = "./.cache"



def output_fp_to_input_fp(fp, scale, rsize):
    out = buzz.Footprint(tl=fp.tl, size=fp.size, rsize=fp.size/scale)
    padding = (rsize - out.rsizex) / 2
    assert padding == int(padding)
    out = out.dilate(padding)
    return out



class AbstractRaster(object):

    def __init__(self, full_fp, cached):
        self._queries = []
        self._primitives = []

        self._cached = cached

        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)

        self._scheduler_thread.start()
        self._cv = threading.Condition()

        self._computation_pool = mp.pool.ThreadPool()
    
        ds = buzz.DataSource(allow_interpolation=True)

        self._full_fp = full_fp

        self._num_bands = None # to implement in subclasses
        self._nodata = None # to implement in subclasses
        self._wkt_origin = None # to implement in subclasses
        self._dtype = None # to implement in subclasses


        comp_tile_count = np.ceil(self._full_fp.rsize / 500)
        self._computation_tiles = self._full_fp.tile_count(*comp_tile_count, boundary_effect='shrink')            


    def _emptiest_query(self, query):
        num = query.produce.verbed.qsize() + len(query.produce.staging)
        den = query.produce.verbed.maxsize
        return num/den


    def _merge_out_tiles(self, tiles, data, out_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

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
    

    def __len__(self):
        return self._num_bands
    

    def _produce_data(self, input_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    def _compute_data(self, input_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    def get_multi_data(self, fp_iterable, queue_size=5):
        query = FullQuery(queue_size)
        to_produce_unscaled = list(fp_iterable.copy())
        to_produce = []

        for fp in to_produce_unscaled:
            rescaled_fp = fp.intersection(self.fp, scale=self.fp.scale, alignment=(0, 0))
            try:
                assert np.array_equal(fp.shape, rescaled_fp.shape)
            except AssertionError:
                print(fp.shape, rescaled_fp.shape)
                print(fp.scale, rescaled_fp.scale)
                raise RuntimeError
            to_produce.append(rescaled_fp)

        query.produce.to_verb = to_produce

        self._prepare_query(query)
 
        self._queries.append(query)

        def out_generator():
            for fp in fp_iterable:
                result = query.produce.verbed.get()
                assert np.array_equal(fp.shape, result.shape[0:2])
                yield result

        return out_generator()


    def get_data(self, fp):
        return next(self.get_multi_data([fp]))
    




class AbstractCachedRaster(AbstractRaster):
    def __init__(self, full_fp, rtype):
        super().__init__(full_fp, True)
        self._io_pool = mp.pool.ThreadPool()
        self._rtype = rtype

        tile_count = np.ceil(self._full_fp.rsize / 500) 
        self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._double_tiled_structure = DoubleTiledStructure(list(self._cache_tiles_fps.flat), list(self._computation_tiles.flat), self._computation_method)
        self._cache_tiles_met = set()

        self._produce_cache_dict = defaultdict(set)
        self._produce_cache_data_dict = defaultdict(lambda: defaultdict(functools.partial(np.ndarray, 0)))


    def _scheduler(self):
        print(self.__class__.__name__, " scheduler in ", threading.currentThread().getName())
        while True:                
            ordered_queries = sorted(self._queries, key=self._emptiest_query)

            for query in ordered_queries:
                # print(self.__class__.__name__, "  ", query,  " queues:   ", threading.currentThread().getName())
                # print("       produced:  ", query.produce.verbed.qsize())
                # print("       cached:  ", query.cache_out.verbed.qsize())
                # print("       computed:  ", query.compute.verbed.qsize())
                # print("       collected:  ", query.collect.verbed.qsize())


                # If all to_produce was consumed, the query has ended
                if not query.produce.to_verb:
                    self._queries.remove(query)
                    del query
                    continue

                if not query.produce.staging:
                    query.produce.staging.append(self._computation_pool.apply_async(self._produce_data, (query.produce.to_verb[0],)))

                elif query.produce.staging[0].ready():
                    query.produce.verbed.put(query.produce.staging.pop(0).get())
                    del query.produce.to_verb[0]

                if query.cache_out.staging and query.cache_out.staging[0].ready():
                    query.cache_out.verbed.put(query.cache_out.staging.pop(0).get())

                if not query.cache_out.verbed.empty():
                    self._cache_out_to_produce_staging(query)

            time.sleep(1e-2)


    def _cache_out_to_produce_staging(self, query):
        cache_data = query.cache_out.verbed.get()
        cache_fp = query.cache_out.to_verb.pop(0)

        for produce_fp in self._produce_cache_dict.keys():
            if cache_fp in self._produce_cache_dict[produce_fp]:
                self._produce_cache_data_dict[produce_fp][cache_fp] = cache_data




    def _produce_data(self, out_fp):
        if self._num_bands == 1:
            out = np.empty(tuple(out_fp.shape), dtype=self._dtype)
        else:
            out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype=self._dtype)

        for cache_fp in  self._produce_cache_dict[out_fp].copy():
            while self._produce_cache_data_dict[out_fp][cache_fp].size == 0:
                pass

            data = self._produce_cache_data_dict[out_fp][cache_fp]
            del self._produce_cache_data_dict[out_fp][cache_fp]

            self._produce_cache_dict[out_fp].discard(cache_fp)
            assert out_fp.same_grid(cache_fp)
            out[cache_fp.slice_in(out_fp, clip=True)] = data[out_fp.slice_in(cache_fp, clip=True)]

        return out



    def _get_cache_tile_path(self, cache_tile):
        path = str(
            Path(CACHE_DIR) / 
            DIR_NAMES[frozenset({*self._rtype})] / 
            "{:.2f}_{:.2f}_{:.2f}_{}".format(*cache_tile.tl, cache_tile.pxsizex, cache_tile.rsizex)
        )

        return path


    def _to_produce_to_to_cache(self, query):
        for to_produce in query.produce.to_verb:
            for cache_tile in self._cache_tiles_fps.flat:
                if cache_tile.share_area(to_produce):
                    if not cache_tile in query.cache_out.to_verb:
                        query.cache_out.to_verb.append(cache_tile)
                    self._produce_cache_dict[to_produce].add(cache_tile)


    def _build_cache_staging(self, query):
        for to_cache_out in query.cache_out.to_verb:
            if to_cache_out in self._cache_tiles_met:
                query.cache_out.staging.append(self._io_pool.apply_async(self._wait_for_computing, (to_cache_out,)))

            else:
                self._cache_tiles_met.add(to_cache_out)

                if os.path.isfile(self._get_cache_tile_path(to_cache_out)):
                    query.cache_out.staging.append(self._io_pool.apply_async(self._read_cache_data, (to_cache_out,)))
                else:
                    query.cache_out.staging.append(self._computation_pool.apply_async(self._compute_cache_data, (to_cache_out,)))



    def _read_cache_data(self, cache_tile):
        ds = buzz.DataSource(allow_interpolation=True)
        filepath = self._get_cache_tile_path(cache_tile)

        with ds.open_araster(filepath).close as src:
            data = src.get_data(band=-1, fp=cache_tile)
        return data


    def _write_cache_data(self, cache_tile, data):
        print(self.__class__.__name__, " writing in ", threading.currentThread().getName())
        filepath = self._get_cache_tile_path(cache_tile)
        ds = buzz.DataSource(allow_interpolation=True)
        print(self.__class__.__name__, " filepath ", filepath, threading.currentThread().getName())
        out_proxy = ds.create_araster(filepath, cache_tile, data.dtype, self._num_bands, driver="GTiff", sr=self._primitives[0].wkt_origin)
        out_proxy.set_data(data, band=-1)
        out_proxy.close()

        with self._cv:
            self._cv.notify_all()


    def _compute_cache_data(self, cache_tile):
        data = self._double_tiled_structure.compute_cache_data(cache_tile)
        print(self.__class__.__name__, " writing data ", threading.currentThread().getName())
        self._io_pool.apply_async(self._write_cache_data, (cache_tile, data))
        return data


    def _wait_for_computing(self, cache_tile):
        with self._cv:
            while not os.path.isfile(self._get_cache_tile_path(cache_tile)):
                self._cv.wait()
        return self._read_cache_data(cache_tile)


    def _prepare_query(self, query):
        self._to_produce_to_to_cache(query)
        self._build_cache_staging(query)






class AbstractNotCachedRaster(AbstractRaster):
    def __init__(self, full_fp):
        super().__init__(full_fp, False)
        self._produce_compute_dict = defaultdict(set)
        self._produce_compute_data_dict = defaultdict(lambda: defaultdict(functools.partial(np.ndarray, 0)))
        self._consumed_fps = []


    def _scheduler(self):
        print(self.__class__.__name__, " scheduler in ", threading.currentThread().getName())
        while True:
            ordered_queries = sorted(self._queries, key=self._emptiest_query)

            for query in ordered_queries:
                # print(self.__class__.__name__, "  ", query,  " queues:   ", threading.currentThread().getName())
                # print("       produced:  ", query.produce.verbed.qsize())
                # print("       cached:  ", query.cache_out.verbed.qsize())
                # print("       computed:  ", query.compute.verbed.qsize())
                # print("       collected:  ", query.collect.verbed.qsize())
                self._consume_compute_fp(query)

                # If all to_produce was consumed, the query has ended
                if not query.produce.to_verb:
                    self._queries.remove(query)
                    del query
                    continue

                if not query.produce.staging:
                    query.produce.staging.append(self._computation_pool.apply_async(self._produce_data, (query.produce.to_verb[0],)))

                elif query.produce.staging[0].ready():
                    query.produce.verbed.put(query.produce.staging.pop(0).get())
                    del query.produce.to_verb[0]

                if query.compute.staging and query.compute.staging[0].ready():
                    query.compute.verbed.put(query.compute.staging.pop(0).get())

                if not query.compute.verbed.empty():
                    self._computed_to_produce_staging(query)

            time.sleep(1e-2)



    def _computed_to_produce_staging(self, query):
        computed_data = query.compute.verbed.get()
        computed_fp = self._consumed_fps.pop(0)

        assert np.array_equal(computed_fp.shape, computed_data.shape[0:2])

        for produce_fp in self._produce_compute_dict.keys():
            if computed_fp in self._produce_compute_dict[produce_fp]:
                self._produce_compute_data_dict[produce_fp][computed_fp] = computed_data



    def _produce_data(self, out_fp):
        if self._num_bands == 1:
            out = np.empty(tuple(out_fp.shape), dtype=self._dtype)
        else:
            out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype=self._dtype)


        for computed_fp in  self._produce_compute_dict[out_fp].copy():
            while self._produce_compute_data_dict[out_fp][computed_fp].size == 0:
                time.sleep(1e-2)

            data = self._produce_compute_data_dict[out_fp][computed_fp]
            del self._produce_compute_data_dict[out_fp][computed_fp]

            self._produce_compute_dict[out_fp].discard(computed_fp)
            assert out_fp.same_grid(computed_fp)
            out[computed_fp.slice_in(out_fp, clip=True)] = data[out_fp.slice_in(computed_fp, clip=True)]

        assert np.array_equal(out_fp.shape, out.shape[0:2])
        return out



    def _to_produce_to_to_compute(self, query):
        if not query.produce.to_verb:
            return
        elif self._cached:
            raise RuntimeError()
        else:
            for to_produce in query.produce.to_verb:
                for computation_tile in self._computation_tiles.flat:
                    if computation_tile.share_area(to_produce):
                        query.compute.to_verb.append(computation_tile)
                        self._produce_compute_dict[to_produce].add(computation_tile)


    def _consume_compute_fp(self, query):
        if not query.compute.to_verb:
            return
        to_compute_fp = query.compute.to_verb.pop(0)
        self._consumed_fps.append(to_compute_fp)
        query.compute.staging.append(self._computation_pool.apply_async(self._compute_data, (to_compute_fp,)))


    def _prepare_query(self, query):
        self._to_produce_to_to_compute(query)






class ResampledRaster(AbstractCachedRaster):

    def __init__(self, raster, scale, rtype):

        full_fp = raster.fp.intersection(raster.fp, scale=scale, alignment=(0, 0))
        self._thread_storage = threading.local()
    
        super().__init__(full_fp, rtype)

        tile_count = np.ceil(self._full_fp.rsize / 500) 
        self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')
        self._num_bands = len(raster)
        self._nodata = raster.nodata
        self._wkt_origin = raster.wkt_origin
        self._dtype = raster.dtype

        self._double_tiled_structure = DoubleTiledStructure(list(self._cache_tiles_fps.flat), list(self._computation_tiles.flat), self._computation_method)

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

        if isinstance(self._primitives[0], ResampledRaster):
            data = self._primitives[0].get_data(input_fp)
        else:
            with ds.open_araster(self._primitives[0].path).close as prim:
                data = prim.get_data(input_fp, band=-1)

        assert np.array_equal(input_fp.shape, data.shape[0:2])
        return data






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


    def _to_produce_to_to_compute(self, query):
        if not query.produce.to_verb:
            return
        elif self._cached:
            raise RuntimeError()
        else:
            for to_produce in query.produce.to_verb:
                query.compute.to_verb.append(to_produce)
                self._produce_compute_dict[to_produce].add(to_produce)


    def _compute_data(self, input_fp):
        print(self.__class__.__name__, " computing data ", threading.currentThread().getName())
        arr = self._collect_data(input_fp.dilate(1))
        nodata_mask = arr == self._nodata
        nodata_mask = ndi.binary_dilation(nodata_mask)
        kernel = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        arru = ndi.maximum_filter(arr, None, kernel) - arr
        arru = np.arctan(arru / input_fp.pxsizex)
        arru = arru / np.pi * 180.
        arru[nodata_mask] = 0
        arru = arru[1:-1, 1:-1]

        arrd = arr - ndi.minimum_filter(arr, None, kernel)
        arrd = np.arctan(arrd / input_fp.pxsizex)
        arrd = arrd / np.pi * 180.
        arrd[nodata_mask] = 0
        arrd = arrd[1:-1, 1:-1]

        arr = np.dstack([arrd, arru])
        return arr


    def _collect_data(self, input_fp):
        return self._primitives[0].get_data(input_fp)





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
        self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._double_tiled_structure = DoubleTiledStructure(list(self._cache_tiles_fps.flat), list(self._computation_tiles.flat), self._computation_method)
        self._lock = threading.Lock()


    def _computation_method(self, computation_tile):
        rgba_tile = output_fp_to_input_fp(computation_tile, 0.64, self._model.get_layer("rgb").input_shape[1])
        dsm_tile = output_fp_to_input_fp(computation_tile, 1.28, self._model.get_layer("slopes").input_shape[1])

        rgba_data = self._collect_rgba_data(rgba_tile)
        slope_data = self._collect_slope_data(dsm_tile)

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
    rgba_out = resampled_rgba.get_multi_data(display_tiles.flat, 5)
    # slopes_out = slopes.get_multi_data(dsm_display_tiles.flat, 5)


    for display_fp, dsm_disp_fp in zip(display_tiles.flat, dsm_display_tiles.flat):
        try:
            next(rgba_out)
            # next(slopes_out)
            # show_many_images(
            #     [next(rgba_out), next(slopes_out)[...,0],np.argmax(next(hm_out), axis=-1)], 
            #     extents=[display_fp.extent, dsm_disp_fp.extent, display_fp.extent]
            # )
        except StopIteration:
            print("ended")

if __name__ == "__main__":
    main()