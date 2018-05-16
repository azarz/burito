from pathlib import Path
import functools
import multiprocessing as mp
import multiprocessing.pool
import os
import pickle
import queue
import threading
import hashlib

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

    def __init__(self, scale):
        self._scale = scale        

    def _merge_out_tiles(self, tiles, data, out_fp):
        return None

    def get_data(self, input_fp):
        return None



class ResamplableRaster(AbstractRaster):

    def __init__(self, path, scale, rtype, dir_names, cache_dir="./.cache"):

        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._req_q = queue.Queue(5)
        self._thread_storage = threading.local()

        self._computation_pool = mp.pool.ThreadPool()
        self._io_pool = mp.pool.ThreadPool()

        self._dispatcher_thread = threading.Thread(target=self._dispatcher)
        self._dispatcher_thread.start()

        self._dico = {}

        self._raster_path = path
        self._scale = scale

        ds = buzz.DataSource(allow_interpolation=True)

        with ds.open_araster(self._raster_path).close as raster:
            self._full_fp = raster.fp.intersection(raster.fp, scale=scale, alignment=(0,0))
            tile_count = np.ceil(self._full_fp.rsize / 500) 
            self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')
            self._num_bands = len(raster)
            self._nodata = raster.nodata


        self._cache_tile_paths = [
            str(Path(cache_dir) / dir_names[frozenset({rtype})] / str(hashlib.md5(repr(fp).encode()).hexdigest()))
            for fp in self._cache_tiles_fps.flat
        ]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self._req_q.put(None)
        self._dispatcher_thread.join()    


    def _callback_dico(self, args, res):
        with self._lock:
            self._dico[args[1]] = res
        with self._cv:
            self._cv.notify_all()
        self._req_q.task_done()

    def _dispatcher(self):
        while True:
            args = self._req_q.get()
            # Stopping the thread
            if args == None:
                return

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
                input_data.append(self._dico[filename])
                # del self._dico[filename]
            intersecting_tiles.append(fp)

        print(len(self._dico.values()))

        return self._merge_out_tiles(intersecting_tiles, input_data, input_fp)


    def _get_multi(self, get_function, fp_iterable):

        out_pool = mp.pool.ThreadPool()

        def async_result_gen():
            for fp in fp_iterable:
                yield out_pool.apply_async(get_function, (fp,))

        return async_result_gen()


    def get_multi_data(self, fp_iterable):
        return self._get_multi(self.get_data, fp_iterable)






class ResamplableOrthoimage(ResamplableRaster):
    
    def __init__(self, path, scale, dir_names, cache_dir="./.cache"):
        super().__init__(path, scale, "ortho", dir_names, cache_dir)


    def _merge_out_tiles(self, tiles, data, out_fp):

        out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype="uint8")

        for tile, dat  in zip(tiles, data):
            assert tile.same_grid(out_fp)
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]
        return out









class ResamplableDSM(ResamplableRaster):
    
    def __init__(self, path, scale, dir_names, cache_dir="./.cache"):
        super().__init__(path, scale, "dsm", dir_names, cache_dir)

    def get_slopes(self, fp):
        arr = self.get_data(fp.dilate(1))
        nodata_mask = arr == self._nodata
        nodata_mask = ndi.binary_dilation(nodata_mask)
        kernel = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        arru = ndi.maximum_filter(arr, None, kernel) - arr
        arru = np.arctan(arru / fp.pxsizex)
        arru = arru / np.pi * 180.
        arru[nodata_mask] = 0
        arru = arru[1:-1, 1:-1]

        arrd = arr - ndi.minimum_filter(arr, None, kernel)
        arrd = np.arctan(arrd / fp.pxsizex)
        arrd = arrd / np.pi * 180.
        arrd[nodata_mask] = 0
        arrd = arrd[1:-1, 1:-1]

        arr = np.dstack([arrd, arru])
        return arr


    def get_multi_slopes(self, fp_iterable):
        return self._get_multi(self.get_slopes, fp_iterable)


    def _merge_out_tiles(self, tiles, data, out_fp):
        out = np.empty(tuple(out_fp.shape), dtype="float32")

        for tile, dat  in zip(tiles, data):
            assert tile.same_grid(out_fp)
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]
        return out





class HeatmapRaster(AbstractRaster):

    def __init__(self, model, scale, rgb_path, dsm_path, dir_names, cache_dir="./.cache"):

        self._scale = scale

        self._model = model
        self._num_bands = LABEL_COUNT

        self._rgb_path = rgb_path
        self._dsm_path = dsm_path

        with ds.open_araster(self._rgb_path).close as rgba:
            self._full_fp = rgba.fp

        out_tiles = self._full_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)
        
        tile_count = np.ceil(self._full_fp.rsize / 500) 
        self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._computation_tiles = out_tiles

        self._double_tiled_structure = DoubleTiledStructure(list(self._cache_tiles_fps.flat), list(out_tiles.flat), self._computation_method)

        self._cache_tile_paths = [
            str(Path(cache_dir) / dir_names[frozenset({'dsm', "ortho"})] / str(hashlib.md5(repr(fp).encode()).hexdigest()))
            for fp in self._cache_tiles_fps.flat
        ]


    def _computation_method(self, computation_tile):

        rgba_tile = output_fp_to_input_fp(computation_tile, 0.64, self._model.get_layer("rgb").input_shape[1])
        dsm_tile = output_fp_to_input_fp(computation_tile, 1.28, self._model.get_layer("slopes").input_shape[1])

        with ResamplableOrthoimage(self._rgb_path, 0.64, dir_names, cache_dir) as rgba_resampler:
            with ResamplableDSM(self._dsm_path, 1.28, dir_names, cache_dir) as dsm_resampler:

                model_input = (rgba_resampler.get_data(rgba_tile)[...,0:3], dsm_resampler.get_slopes(dsm_tile))

        rgb = (model_input[0].astype('float32') - 127.5) / 127.5
        slopes = model_input[1] / 45 - 1
        prediction = model.predict([rgb[np.newaxis], slopes[np.newaxis]])[0]

        return prediction



    def _merge_out_tiles(self, tiles, data, out_fp):

        out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype="float32")

        for tile, dat  in zip(tiles, data):
            assert tile.same_grid(out_fp)
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]
        return out



    def get_data(self, input_fp):

        ds = buzz.DataSource(allow_interpolation=True)

        input_data = []
        intersecting_tiles = []

        def tile_info_gen():
            for cache_tile, filename in zip(self._cache_tiles_fps.flat, self._cache_tile_paths):
                if cache_tile.share_area(input_fp):
                    yield cache_tile, filename

        for cache_tile, filepath in tile_info_gen():

            file_exists = os.path.isfile(filepath)
            
            if not file_exists:
                prediction = self._double_tiled_structure.compute_cache_data(cache_tile)

                with datasrc.open_araster(rgb_path).close as src:
                    out_proxy = datasrc.create_araster(filepath, cache_tile, "float32", LABEL_COUNT, driver="GTiff", sr=src.wkt_origin)

                out_proxy.set_data(prediction, band=-1)
                out_proxy.close()

            else:
                with datasrc.open_araster(filepath).close as src:
                    prediction = src.get_data(band=-1)

            intersecting_tiles.append(cache_tile)
            input_data.append(prediction)

        return self._merge_out_tiles(intersecting_tiles, input_data, input_fp)






if __name__ == "__main__":
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
        out_fp = raster.fp.intersection(raster.fp, scale=1.28, alignment=(0,0))

    out_fp = out_fp.intersection(out_fp, scale=0.64)

    hmr = HeatmapRaster(model, 0.64, rgb_path, dsm_path, dir_names)

    display_fp = out_fp.intersection(sg.Point(348264,50978)).dilate(200)
    print(display_fp, out_fp)
    
    data = hmr.get_data(display_fp)

    show_many_images(
        [np.argmax(data, axis=-1)], 
        extents=[display_fp.extent]
    )
