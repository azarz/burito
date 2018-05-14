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



class MultiThreadedRasterResampler(object):

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


    def _merge_out_tiles(self, tiles, data, out_fp):
        if self._num_bands > 1:
            out = np.empty(tuple(out_fp.shape) + (self._num_bands,), dtype="uint8")
        else:
            out = np.empty(tuple(out_fp.shape), dtype="float32")

        for tile, dat  in zip(tiles, data):
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]
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



    def _get_multi(self, get_function, fp_iterable):

        out_pool = mp.pool.ThreadPool()

        def async_result_gen():
            for fp in fp_iterable:
                yield out_pool.apply_async(get_function, (fp,))

        return async_result_gen()


    def get_multi_data(self, fp_iterable):
        return self._get_multi(self.get_data, fp_iterable)


    def get_multi_slopes(self, fp_iterable):
        return self._get_multi(self.get_slopes, fp_iterable)




class HeatmapRaster(object):

    def __init__(self, model, scale, out_tiles, dir_names, cache_dir="./.cache"):

        self._scale = scale

        self._model = model

        ds = buzz.DataSource(allow_interpolation=True)

        with ds.open_araster(self._raster_path).close as raster:
            self._full_fp = raster.fp.intersection(raster.fp, scale=scale, alignment=(0,0))
            tile_count = np.ceil(self._full_fp.rsize / 500) 
            self._cache_tiles_fps = self._full_fp.tile_count(*tile_count, boundary_effect='shrink')

        self._computation_tiles = out_tiles

        self._double_tiled_structure = DoubleTiledStructure(self._cache_tiles_fps, out_tiles, self._computation_method)

        self._cache_tile_paths = [
            str(Path(cache_dir) / dir_names[frozenset({rtype})] / str(hashlib.md5(repr(fp).encode()).hexdigest()))
            for fp in self._cache_tiles_fps.flat
        ]


    def _computation_method(self, computation_tile):

        rgba_tile = output_fp_to_input_fp(computation_tile, 0.64, self._model.get_layer("rgb").input_shape[1])
        dsm_tile = output_fp_to_input_fp(computation_tile, 1.28, self._model.get_layer("slopes").input_shape[1])

        model_input = (rgba_resampler.get_data(rgba_tile), dsm_resampler.get_slopes(dsm_tile))

        rgb = (model_input[0].astype('float32') - 127.5) / 127.5
        slopes = model_input[1] / 45 - 1
        prediction = model.predict([rgb[np.newaxis], slopes[np.newaxis]])[0]

        return prediction



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
                prediction = self._double_tiled_structure.compute_cache_tile(cache_tile)

                with datasrc.open_araster(rgb_path).close as src:
                    out_proxy = datasrc.create_araster(filepath, fp, "float32", LABEL_COUNT, driver="GTiff", sr=src.wkt_origin)

                out_proxy.set_data(prediction, band=-1)
                out_proxy.close()

            else:
                with datasrc.open_araster(filepath).close as src:
                    prediction = src.get_data(band=-1)

            prediction_q.put((prediction, fp))






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

    out_tiles = out_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)

    rgba_tiles = np.asarray([
        output_fp_to_input_fp(tile, 0.64, model.get_layer("rgb").input_shape[1]) 
        for tile in out_tiles.flatten()
    ]).reshape(out_tiles.shape)

    dsm_tiles = np.asarray([
        output_fp_to_input_fp(tile, 1.28, model.get_layer("slopes").input_shape[1]) 
        for tile in out_tiles.flatten()
    ]).reshape(out_tiles.shape)



    model_input_q = queue.Queue(5)
    display_input_q = queue.Queue(5)
    prediction_q = queue.Queue(5)

    rgb_results_q = queue.Queue(5)
    slopes_results_q = queue.Queue(5)



    def keras_worker():
        hmRaster = HeatmapRaster()

        while True:
            inputs = model_input_q.get()

            if inputs == None:
                prediction_q.put([None])
                return

            model_input = inputs[0]
            fp = inputs[1]

            filepath = str(Path(cache_dir) / dir_names[frozenset({'dsm', "ortho"})] / str(hashlib.md5(repr(fp).encode()).hexdigest()))

            file_exists = os.path.isfile(filepath)

            if not file_exists:

                rgb = (model_input[0].astype('float32') - 127.5) / 127.5
                slopes = model_input[1] / 45 - 1
                prediction = model.predict([rgb[np.newaxis], slopes[np.newaxis]])[0]

                with datasrc.open_araster(rgb_path).close as src:
                    out_proxy = datasrc.create_araster(filepath, fp, "float32", LABEL_COUNT, driver="GTiff", sr=src.wkt_origin)

                out_proxy.set_data(prediction, band=-1)
                out_proxy.close()

            else:
                with datasrc.open_araster(filepath).close as src:
                    prediction = src.get_data(band=-1)

            prediction_q.put((prediction, fp))
            model_input_q.task_done()



    def resample_results_filler(rgb_results_gen, slopes_results_gen):
        rgb_res = next(rgb_results_gen, None)
        slope_res = next(slopes_results_gen, None)

        while rgb_res != None:
            assert slope_res != None

            rgb_results_q.put(rgb_res)
            slopes_results_q.put(slope_res)

            rgb_res = next(rgb_results_gen, None)
            slope_res = next(slopes_results_gen, None)
        

    def resampler_worker():
        with MultiThreadedRasterResampler(rgb_path, 0.64, 'ortho', dir_names, cache_dir) as rgb_resampler:
            with MultiThreadedRasterResampler(dsm_path, 1.28, 'dsm', dir_names, cache_dir) as dsm_resampler:

                rgb_results_gen = rgb_resampler.get_multi_data(rgba_tiles.flat)
                slopes_results_gen = dsm_resampler.get_multi_slopes(dsm_tiles.flat)

                filler_thread = threading.Thread(target=resample_results_filler, args=(rgb_results_gen, slopes_results_gen))
                filler_thread.start()

                for result_index in range(len(out_tiles.flat)):
                    inputs = (rgb_results_q.get().get()[...,0:3], slopes_results_q.get().get())
                    fp = out_tiles.flat[result_index]

                    model_input_q.put((inputs, fp))
                    display_input_q.put((inputs, fp))


                model_input_q.put(None)


    print("overhead done!")

    keras_thread = threading.Thread(target=keras_worker)
    keras_thread.start()

    resampler_thread = threading.Thread(target=resampler_worker)
    resampler_thread.start()

    pred = prediction_q.get()
    model_in = display_input_q.get()[0]

    i = 0

    while not np.array_equal(pred, [None]):
        data = pred[0]
        fp = pred[1]

        in_rgb_fp = rgba_tiles.flat[i]
        in_dsm_fp = dsm_tiles.flat[i]

        print(fp.extent)
        print(in_rgb_fp.extent)

        with datasrc.open_araster(dsm_path).close as dsm:
            in_dsm_dat = dsm.get_data(band=-1, fp=in_dsm_fp)
            in_dsm_dat[in_dsm_dat < 0] = 999999
            in_dsm_dat[in_dsm_dat == 999999] = np.amin(in_dsm_dat)

        show_many_images(
            [model_in[0], in_dsm_dat, model_in[1][:,:,0], np.argmax(data, axis=-1)], 
            extents=[in_rgb_fp.extent, in_dsm_fp.extent, in_rgb_fp.extent, in_dsm_fp.extent, fp.extent]
        )
        pred = prediction_q.get()
        model_in = display_input_q.get()[0]
        i += 1


    resampler_thread.join()
    keras_thread.join()