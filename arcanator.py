from pathlib import Path
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
        self._io_pool = mp.pool.ThreadPool()

        ds = buzz.DataSource(allow_interpolation=True)

        self._full_fp = full_fp
        if cached:
            tile_count = np.ceil(self._full_fp.rsize / 500) 
            self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._num_bands = None # to implement in subclasses
        self._nodata = None # to implement in subclasses
        self._wkt_origin = None # to implement in subclasses


        comp_tile_count = np.ceil(self._full_fp.rsize / 500)
        self._computation_tiles = self._full_fp.tile_count(*comp_tile_count, boundary_effect='shrink')

        if cached:
            self._double_tiled_structure = DoubleTiledStructure(self._cache_tiles_fps, self._computation_tiles, self._computation_method)
            self._cache_tiles_met = set()


    def _emptiest_query(self, query):
        num = query.produce.verbed.qsize() + len(query.produce.staging)
        den =query.verbed.maxsize
        return num/dem


    def _merge_out_tiles(self, tiles, data, out_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    @property
    def fp(self):
        return self._full_fp

    def _produce_data(self, input_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    def _compute_data(self, input_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    def get_multi_data(self, fp_iterable, queue_size=5):
        query = FullQuery(queue_size)
        query.produce.to_verb = fp_iterable

        self._queries.append(query)
        return query.produce.verbed


    def get_data(self, fp):
        return get_multi_data([fp]).get()
    




class AbstractCachedRaster(AbstractRaster):
    def __init__(self, full_fp, rtype):
        super().__init__(full_fp, True)
        self._rtype = rtype
        self._produce_cache_dict = defaultdict(set)


    def _scheduler(self):

        self._cache_fp_queue = queue.Queue(5)
        self._cache_data_queue = queue.Queue(5)

        cache_fp_list = []

        def _cache_to_produce_staging(query):
            cache_fp = cache_fp_list.pop(0)

            self._cache_fp_queue.put(cache_fp)
            self._cache_data_queue.put(query.compute.verbed.get())

            self._produce_cache_dict[query.produce.to_verb[0]].discard(cache_fp)


        while True:
            for query in self._queries:
                self._to_produce_to_to_cache(query)
                self._build_cache_staging(query)

            ordered_queries = sorted(self._queries, key=self._emptiest_query)

            for query in ordered_queries:
                # If all to_produce was consumed, the query has ended
                if not query.produce.to_verb:
                    del query
                    continue

                if not query.produce.staging:
                    query.produce.staging.append(self._computation_pool.apply_async(self._produce_data, (query.produce.to_verb[0],)))

                elif query.produce.staging[0].ready():
                    query.produce.verbed.put(query.produce.staging.pop(0).get())
                    del query.produce.to_verb[0]

                if query.cache_out.staging[0].ready():
                    query.cache_out.verbed.put(query.compute.cache_out.pop(0).get())

                if not query.cache_out.verbed.empty():
                    _cache_to_produce_staging(query)

                if query.cache.to_verb:
                    to_cache_fp = query.cache.to_verb.pop(0)
                    cache_fp_list.append(to_cache_fp)

            time.sleep(1e-2)



    def _produce_data(self, out_fp):
        out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype="float32")

        while self._produce_cache_dict[out_fp]:
            cache_fp = self._cache_fp_queue.get()
            data = self._cache_data_queue.get()

            out[cache_fp.slice_in(out_fp, clip=True)] = data[out_fp.slice_in(cache_fp, clip=True)]

        return out



    def _get_cache_tile_path(self, cache_tile):
        path = str(
            Path(cache_dir) / 
            DIR_NAMES[frozenset({*self._rtype})] / 
            "{:.2f}_{:.2f}_{:.2f}_{}".format(*cache_tile.tl, cache_tile.pxsizex, cache_tile.rsizex)
        )

        return path


    def _to_produce_to_to_cache(self, query):
        if not query.produce.to_verb:
            return
        elif not self._cached:
            return
        else:
            for to_produce in query.produce.to_verb:
                for cache_tile in self._cache_tiles_fps.flat:
                    if cache_tile.share_area(to_produce):
                        query.cache_out.to_verb.append(cache_tile)
                        self._produce_cache_dict[to_produce].add(cache_tile)


    def _build_cache_staging(self, query):
        for to_cache_out in query.cache_out.to_verb:
            if to_cache_out in self._cache_tiles_met:
                query.cache_out.staging.append(self._io_pool.apply_async(self._wait_for_computing, (to_cache_out,)))

            else:
                self._cache_tiles_met.add(to_cache_out)

                if os.isfile(self._get_cache_tile_path(to_cache_out)):
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
        filepath = self._get_cache_tile_path(cache_tile)
        out_proxy = ds.create_araster(filepath, cache_tile, data.dtype, self._num_bands, driver="GTiff", sr=self._resampled_rgba.wkt_origin)
        out_proxy.set_data(data, band=-1)
        out_proxy.close()

        with self._cv:
            self._cv.notify_all()


    def _compute_cache_data(self, cache_tile):
        data = self._double_tiled_structure.compute_cache_data(to_cache_out)
        self._io_pool.apply_async(self._write_cache_data, (cache_tile, data))
        return data


    def _wait_for_computing(self, cache_tile):
        with self._cv:
            while not os.isfile(self._get_cache_tile_path(cache_tile)):
                self._cv.wait()
        return self._read_cache_data(cache_tile)






class AbstractNotCachedRaster(AbstractRaster):
    def __init__(self, full_fp):
        super().__init__(full_fp, False)
        self._produce_compute_dict = defaultdict(set)


    def _scheduler(self):
        self._computed_fp_queue = queue.Queue(5)
        self._computed_data_queue = queue.Queue(5)

        computed_fp_list = []

        def _computed_to_produce_staging(query):
            computed_fp = computed_fp_list.pop(0)

            self._computed_fp_queue.put(computed_fp)
            self._computed_data_queue.put(query.compute.verbed.get())

            self._produce_compute_dict[query.produce.to_verb[0]].discard(computed_fp)


        while True:
            for query in self._queries:
                self._to_produce_to_to_compute(query)

            ordered_queries = sorted(self._queries, key=self._emptiest_query)

            for query in ordered_queries:
                # If all to_produce was consumed, the query has ended
                if not query.produce.to_verb:
                    del query
                    continue

                if not query.produce.staging:
                    query.produce.staging.append(self._computation_pool.apply_async(self._produce_data, (query.produce.to_verb[0],)))

                elif query.produce.staging[0].ready():
                    query.produce.verbed.put(query.produce.staging.pop(0).get())
                    del query.produce.to_verb[0]

                if query.compute.staging[0].ready():
                    query.compute.verbed.put(query.compute.staging.pop(0).get())

                if not query.compute.verbed.empty():
                    _computed_to_produce_staging(query)

                if query.compute.to_verb:
                    to_compute_fp = query.compute.to_verb.pop(0)
                    computed_fp_list.append(to_compute_fp)
                    query.compute.staging.append(self._computation_pool.apply_async(self._compute_data, (to_compute_fp,)))

            time.sleep(1e-2)




    def _produce_data(self, out_fp):
        out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype="float32")

        while self._produce_compute_dict[out_fp]:
            computed_fp = self._computed_fp_queue.get()
            data = self._computed_data_queue.get()

            out[computed_fp.slice_in(out_fp, clip=True)] = data[out_fp.slice_in(computed_fp, clip=True)]

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






class ResampledRaster(AbstractCachedRaster):

    def __init__(self, path, scale, rtype, cache_dir="./.cache"):
        super().__init__(path, rtype)
    
    







class ResampledOrthoimage(ResampledRaster):
    
    def __init__(self, path, scale, cache_dir="./.cache"):
        super().__init__(path, scale, ("ortho",), cache_dir)


    def _merge_out_tiles(self, tiles, data, out_fp):

        out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype="uint8")

        for tile, dat  in zip(tiles, data):
            assert tile.same_grid(out_fp)
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]

        return out









class ResampledDSM(ResampledRaster):
    
    def __init__(self, path, scale, cache_dir="./.cache"):
        super().__init__(path, scale, ("dsm",), cache_dir)

    def _merge_out_tiles(self, tiles, data, out_fp):
        out = np.empty(tuple(out_fp.shape), dtype="float32")

        for tile, dat  in zip(tiles, data):
            assert tile.same_grid(out_fp)
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]

        return out







class Slopes(AbstractNotCachedRaster):
    def __init__(self, dsm):
        super().__init__(dsm.fp)
        self._primitives = [dsm]
        self._nodata = dsm.nodata
        self._num_bands = 2


    def _to_produce_to_to_compute(self, query):
        if not query.produce.to_verb:
            return
        elif self._cached:
            raise RuntimeError()
        else:
            to_produce = query.produce.to_verb.pop(0)
            query.compute.to_verb.append(to_produce)


    def _compute_data(self, input_fp):
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

    def __init__(self, model, resampled_rgba, slopes, DIR_NAMES, cache_dir="./.cache"):

        self._model = model
        self._num_bands = LABEL_COUNT

        self._resampled_rgba = resampled_rgba
        self._slopes = slopes

        max_scale = max(resampled_rgba.fp.scale[0], slopes.fp.scale[0])
        min_scale = min(resampled_rgba.fp.scale[0], slopes.fp.scale[0])
      
        self._full_fp = resampled_rgba.fp.intersection(slopes.fp, scale=max_scale, alignment=(0, 0))

        self._full_fp = self._full_fp.intersection(self._full_fp, scale=min_scale)

        self._computation_tiles = self._full_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)

        tile_count = np.ceil(self._full_fp.rsize / 500) 
        self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._double_tiled_structure = DoubleTiledStructure(list(self._cache_tiles_fps.flat), list(self._computation_tiles.flat), self._computation_method)

        self._cache_tile_paths = [
            str(Path(cache_dir) / DIR_NAMES[frozenset({"dsm", "ortho"})] / str(str(np.around(fp.tlx, 2)) + "_" + 
                                                                        str(np.around(fp.tly, 2)) + "_" + 
                                                                        str(np.around(fp.pxsizex, 2)) + "_" + 
                                                                        str(np.around(fp.rsizex, 2)))
            )
            for fp in self._cache_tiles_fps.flat
        ]


        self._lock = threading.Lock()
        self._thread_storage = threading.local()


    def _computation_method(self, computation_tile):

        rgba_tile = output_fp_to_input_fp(computation_tile, 0.64, self._model.get_layer("rgb").input_shape[1])
        dsm_tile = output_fp_to_input_fp(computation_tile, 1.28, self._model.get_layer("slopes").input_shape[1])

        rgba_data = self._resampled_rgba.get_data(rgba_tile)
        slope_data = self._slopes.get_data(dsm_tile)

        rgb_data = np.where((rgba_data[...,3] == 255)[...,np.newaxis], rgba_data, 0)[...,0:3]
        rgb = (rgb_data.astype('float32') - 127.5) / 127.5

        slopes = slope_data / 45 - 1

        prediction = self._model.predict([rgb[np.newaxis], slopes[np.newaxis]])[0]

        return prediction



    def _merge_out_tiles(self, tiles, data, out_fp):

        out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype="float32")

        for tile, dat  in zip(tiles, data):
            assert tile.same_grid(out_fp)
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]
        return out



    def get_data(self, input_fp):

        output_data = []
        intersecting_tiles = []

        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds

        else:
            ds = self._thread_storage.ds

        def tile_info_gen():
            for cache_tile, filename in zip(self._cache_tiles_fps.flat, self._cache_tile_paths):
                if cache_tile.share_area(input_fp):
                    yield cache_tile, filename

        for cache_tile, filepath in tile_info_gen():

            file_exists = os.path.isfile(filepath)
            
            if not file_exists:

                with self._lock:
                    # checking again if the file exists (happens when entering the lock after waiting)
                    file_exists = os.path.isfile(filepath)

                    if not file_exists:
                        print("--> using GPU...")
                        prediction = self._double_tiled_structure.get_cache_data(cache_tile)
                        print("<-- no more GPU")

                        out_proxy = ds.create_araster(filepath, cache_tile, "float32", LABEL_COUNT, driver="GTiff", sr=self._resampled_rgba.wkt_origin)
                        out_proxy.set_data(prediction, band=-1)
                        out_proxy.close()

                if file_exists:
                    print("!! cache was calculated when i was waiting")
                    with ds.open_araster(filepath).close as src:
                        prediction = src.get_data(band=-1)

            else:
                with ds.open_araster(filepath).close as src:
                    prediction = src.get_data(band=-1)

            intersecting_tiles.append(cache_tile)
            output_data.append(prediction)

        return self._merge_out_tiles(intersecting_tiles, output_data, input_fp)






def main():
    print("hello")

    rgb_path = "./ortho_8.00cm.tif"
    dsm_path = "./dsm_8.00cm.tif"
    model_path = "./18-01-25-15-38-19_1078_1.00000000_0.07799472_aracena.hdf5"

    cache_dir = "./.cache"

    for path in DIR_NAMES.values():
        os.makedirs(str(Path(cache_dir) / path), exist_ok=True)

    datasrc = buzz.DataSource(allow_interpolation=True)

    print("model...")

    model = load_model(model_path)
    model._make_predict_function()


    with datasrc.open_araster(rgb_path).close as raster:
        out_fp = raster.fp.intersection(raster.fp, scale=1.28, alignment=(0, 0))

    out_fp = out_fp.intersection(out_fp, scale=0.64)



    initial_rgba = datasrc.open_araster(rgb_path)
    initial_dsm = datasrc.open_araster(dsm_path)

    resampled_rgba = ResampledOrthoimage(rgb_path, 0.64, DIR_NAMES, cache_dir)
    resampled_dsm = ResampledDSM(dsm_path, 1.28, DIR_NAMES, cache_dir)

    slopes = Slopes(resampled_dsm)

    hmr = HeatmapRaster(model, resampled_rgba, slopes, DIR_NAMES)


    big_display_fp = out_fp
    big_dsm_disp_fp = big_display_fp.intersection(big_display_fp, scale=1.28, alignment=(0, 0))

    display_tiles = big_display_fp.tile_count(5, 5, boundary_effect='shrink')
    dsm_display_tiles = big_dsm_disp_fp.tile_count(5, 5, boundary_effect='shrink')


    out_queue1 = hmr.get_multi_data(display_tiles.flat, 5)
    out_queue2 = resampled_rgba.get_multi_data(display_tiles.flat, 5)
    out_queue3 = slopes.get_multi_data(dsm_display_tiles.flat, 5)


    hm_thread = threading.Thread(target=hm_worker)
    hm_thread.start()    
    rgb_thread = threading.Thread(target=rgb_worker)
    rgb_thread.start()    
    slope_thread = threading.Thread(target=slope_worker)
    slope_thread.start()

    for display_fp, dsm_disp_fp in zip(display_tiles.flat, dsm_display_tiles.flat):
        # show_many_images(
        hey =    [out_queue2.get().get(), out_queue3.get().get()[...,0], np.argmax(out_queue1.get().get(), axis=-1)]#, 
            # extents=[display_fp.extent, dsm_disp_fp.extent, display_fp.extent]
        # )

if __name__ == "__main__":
    main()