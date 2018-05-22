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

from keras.models import load_model
import buzzard as buzz
import scipy.ndimage as ndi
import numpy as np
import shapely.geometry as sg

from show_many_images import show_many_images
from uids_of_paths import uids_of_paths
from watcher import Watcher

from DoubleTiledStructure import DoubleTiledStructure
from Query import FullQuery


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



def output_fp_to_input_fp(fp, scale, rsize):
    out = buzz.Footprint(tl=fp.tl, size=fp.size, rsize=fp.size/scale)
    padding = (rsize - out.rsizex) / 2
    assert padding == int(padding)
    out = out.dilate(padding)
    return out



class AbstractRaster(object):

    def __init__(self, raster_path):
        self._queries = []
        self._primitives = []
        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._scheduler_thread.start()

        ds = buzz.DataSource(allow_interpolation=True)

        with ds.open_araster(raster_path).close as raster:
            self._full_fp = raster.fp.intersection(raster.fp, scale=scale, alignment=(0, 0))
            tile_count = np.ceil(self._full_fp.rsize / 500) 
            self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')
            self._num_bands = len(raster)
            self._nodata = raster.nodata
            self._wkt_origin = raster.wkt_origin


        self._cache_tile_paths = [
            str(Path(cache_dir) / dir_names[frozenset({rtype})] / str(str(np.around(fp.tlx, 2)) + "_" + 
                                                                        str(np.around(fp.tly, 2)) + "_" + 
                                                                        str(np.around(fp.pxsizex, 2)) + "_" + 
                                                                        str(np.around(fp.rsizex, 2)))
            )
            for fp in self._cache_tiles_fps.flat
        ]


        self._computation_tiles = self._cache_tiles_fps

        self._double_tiled_structure = DoubleTiledStructure(self._cache_tiles_fps, self._computation_tiles, self._computation_method)


    def _scheduler(self):
        while True:
            pass





    def _merge_out_tiles(self, tiles, data, out_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    @property
    def fp(self):
        return self._full_fp

    def _produce_data(self, input_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    def get_multi_data(self, fp_iterable, queue_size=5):
        query = FullQuery(queue_size)
        query.produce.to_verb = fp_iterable

        self._queries.append(query)
        return query.produce.verbed


    def get_data(self, fp):
        return get_multi_data([fp]).get()
    



class ResampledRaster(AbstractRaster):

    def __init__(self, path, scale, rtype, dir_names, cache_dir="./.cache"):

    
    







class ResampledOrthoimage(ResampledRaster):
    
    def __init__(self, path, scale, dir_names, cache_dir="./.cache"):
        super().__init__(path, scale, "ortho", dir_names, cache_dir)


    def _merge_out_tiles(self, tiles, data, out_fp):

        out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype="uint8")

        for tile, dat  in zip(tiles, data):
            assert tile.same_grid(out_fp)
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]

        return out









class ResampledDSM(ResampledRaster):
    
    def __init__(self, path, scale, dir_names, cache_dir="./.cache"):
        super().__init__(path, scale, "dsm", dir_names, cache_dir)

    def _merge_out_tiles(self, tiles, data, out_fp):
        out = np.empty(tuple(out_fp.shape), dtype="float32")

        for tile, dat  in zip(tiles, data):
            assert tile.same_grid(out_fp)
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]

        return out







class Slopes(AbstractRaster):
    def __init__(self, dsm):
        self._full_fp = dsm.fp
        self._parent_dsm = dsm
        self._nodata = dsm.nodata


    def get_data(self, input_fp):
        arr = self._parent_dsm.get_data(input_fp.dilate(1))
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





class HeatmapRaster(AbstractRaster):

    def __init__(self, model, resampled_rgba, slopes, dir_names, cache_dir="./.cache"):

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
            str(Path(cache_dir) / dir_names[frozenset({"dsm", "ortho"})] / str(str(np.around(fp.tlx, 2)) + "_" + 
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

    print("uids...")
    # dir_names = uids_of_paths({
    #         "ortho": rgb_path,
    #         "dsm": dsm_path
    #     })
    dir_names = {
            frozenset(["ortho"]): "rgb",
            frozenset(["dsm"]): "dsm",
            frozenset(["ortho","dsm"]): "both"
        }

    cache_dir = "./.cache"

    for path in dir_names.values():
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

    resampled_rgba = ResampledOrthoimage(rgb_path, 0.64, dir_names, cache_dir)
    resampled_dsm = ResampledDSM(dsm_path, 1.28, dir_names, cache_dir)

    slopes = Slopes(resampled_dsm)

    hmr = HeatmapRaster(model, resampled_rgba, slopes, dir_names)


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