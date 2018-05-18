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

    def __init__(self, full_fp):
        self._full_fp = full_fp

    def _merge_out_tiles(self, tiles, data, out_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    @property
    def fp(self):
        return self._full_fp

    def get_data(self, input_fp):
        raise NotImplementedError('Should be implemented by all subclasses')

    def get_multi_data(self, fp_iterable, out_queue):
        out_pool = mp.pool.ThreadPool(5)
        for fp in fp_iterable:
            out_queue.put(out_pool.apply_async(self.get_data, (fp,)))
    


class ResampledRaster(AbstractRaster):

    def __init__(self, path, scale, rtype, dir_names, cache_dir="./.cache"):

        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._req_q = queue.Queue(5)
        self._thread_storage = threading.local()

        self._computation_pool = mp.pool.ThreadPool()
        self._io_pool = mp.pool.ThreadPool()

        self._dispatcher_thread = threading.Thread(target=self._dispatcher)
        self._dispatcher_thread.daemon = True
        self._dispatcher_thread.start()

        self._dico = {}

        self._raster_path = path

        ds = buzz.DataSource(allow_interpolation=True)

        with ds.open_araster(self._raster_path).close as raster:
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



    def _callback_dico(self, args, res):
        with self._lock:
            self._dico[args[1]] = res
        with self._cv:
            self._cv.notify_all()
        self._req_q.task_done()

    def _dispatcher(self):
        while True:
            args = self._req_q.get()

            with self._lock:
                met = args[1] in self._dico.keys()
                file_exists = os.path.isfile(args[1])

                # met
                if met:
                    value = self._dico[args[1]]

                    # Computing
                    if isinstance(value, int):
                        # In both fileExists and !fileExists cases, we wait
                        self._computation_pool.apply_async(self._wait_for_resampling, (args[1],))
                        
                    # Computed
                    else:
                        # fileExists
                        if os.path.isfile(args[1]):
                            # ReadingFile
                            self._io_pool.apply_async(self._get_tile_data, (args[1],), callback=functools.partial(self._callback_dico, args))
                        # !fileExists
                        else:
                            # Should not happen
                            raise RuntimeError("There should be a file")

                # !met
                else:
                    # fileExists
                    if file_exists:
                        print("reading_file")
                        # Reading file
                        self._dico[args[1]] = 0
                        self._io_pool.apply_async(self._get_tile_data, (args[1],), callback=functools.partial(self._callback_dico, args))
                    # !fileExists
                    else:
                        print("writing_file")
                        # Writing file
                        self._dico[args[1]] = 0
                        self._computation_pool.apply_async(self._resample_tile, args, callback=functools.partial(self._callback_dico, args))



    def _wait_for_resampling(self, dummy_value):
        with self._cv:
            while isinstance(self._dico[dummy_value], int):
                self._cv.wait()
            self._req_q.task_done()


    def _get_tile_data(self, tile_path):
        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds
        else:
            ds = self._thread_storage.ds

        with ds.open_araster(tile_path).close as raster:
            out = raster.get_data(band=-1)
        return out

    def _resample_tile(self, tile_fp, tile_path):
        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds

        else:
            ds = self._thread_storage.ds

        with ds.open_araster(self._raster_path).close as src:
            out = src.get_data(band=-1, fp=tile_fp)
            if len(src) == 4:
                out = np.where((out[...,3] == 255)[...,np.newaxis], out, 0)

            out_proxy = ds.create_araster(tile_path, tile_fp, src.dtype, len(src), driver="GTiff", band_schema={"nodata": src.nodata}, sr=src.wkt_origin)

        out_proxy.set_data(out, band=-1)
        out_proxy.close()
        return out


    @property
    def nodata(self):
        return self._nodata

    @property
    def wkt_origin(self):
        return self._wkt_origin
    
    



    def get_data(self, input_fp):
        if not hasattr(self._thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self._thread_storage.ds = ds

        else:
            ds = self._thread_storage.ds

        input_data = []
        intersecting_tiles = []

        def tile_info_gen():
            for cache_tile, filename in zip(self._cache_tiles_fps.flat, self._cache_tile_paths):
                if cache_tile.share_area(input_fp):
                    yield cache_tile, filename

        tile_info = list(tile_info_gen())

        for fp, filename in tile_info:
            self._req_q.put((fp, filename))
            
        self._req_q.join()

        for fp, filename in tile_info:
            with self._lock:
                input_data.append(self._dico[filename].copy())
                self._dico[filename][input_fp.slice_in(fp, clip=True)] = -1
                if (self._dico[filename] == -1).all():
                    print(filename, "   deleting (Resampler)  ", len(self._dico.keys()),)
                    del self._dico[filename]
                    
            intersecting_tiles.append(fp)

        return self._merge_out_tiles(intersecting_tiles, input_data, input_fp)






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

        rgba_data = np.where((rgba_data[...,3] == 255)[...,np.newaxis], rgba_data, 0)[...,0:3]
        rgb = (rgba_data.astype('float32') - 127.5) / 127.5

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
                        prediction = self._double_tiled_structure.compute_cache_data(cache_tile)
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


    # for display_fp in display_tiles.flat:
    #     # display_fp = out_fp.intersection(sg.Point(348264,50978)).dilate(200)

    #     dsm_disp_fp = display_fp.intersection(display_fp, scale=1.28, alignment=(0,0))

    #     ini_rgb_fp = initial_rgba.fp.intersection(display_fp)
    #     ini_dsm_fp = initial_dsm.fp.intersection(display_fp)


    #     ini_rgb_data = initial_rgba.get_data(fp=ini_rgb_fp, band=-1)    
    #     ini_dsm_data = initial_dsm.get_data(fp=ini_dsm_fp)
     
    #     hm_data = hmr.get_data(display_fp)

    #     r_rgb_data = resampled_rgba.get_data(display_fp)

    #     r_dsm_data = resampled_dsm.get_data(dsm_disp_fp)
    #     r_slope_data = slopes.get_data(dsm_disp_fp)

    #     show_many_images(
    #         [ini_rgb_data, ini_dsm_data, r_rgb_data, r_dsm_data, r_slope_data[...,0], np.argmax(hm_data, axis=-1)], 
    #         extents=[ini_rgb_fp.extent, ini_dsm_fp.extent, display_fp.extent, dsm_disp_fp.extent, dsm_disp_fp.extent, display_fp.extent]
    #     )

    out_queue1 = queue.Queue(5)
    out_queue2 = queue.Queue(5)
    out_queue3 = queue.Queue(5)

    def hm_worker():
        hmr.get_multi_data(display_tiles.flat, out_queue1)
    def rgb_worker():
        resampled_rgba.get_multi_data(display_tiles.flat, out_queue2)
    def slope_worker():
        slopes.get_multi_data(dsm_display_tiles.flat, out_queue3)

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