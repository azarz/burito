from pathlib import Path
import functools
import multiprocessing as mp
import multiprocessing.pool
import os
import pickle
import queue
import threading

from keras.models import load_model
import buzzard as buzz
import numpy as np

from show_many_images import show_many_images
from uids_of_paths import uids_of_paths
from watcher import Watcher


class MultiThreadedRasterResampler(object):

    def __init__(self, path, scale, rtype, cache_dir="./.cache", get_slopes=False):

        self.get_slopes = get_slopes

        self.lock = threading.Lock()
        self.cv = threading.Condition()
        self.req_q = queue.Queue()
        self.thread_storage = threading.local()

        self.computation_pool = mp.pool.ThreadPool()
        self.io_pool = mp.pool.ThreadPool()

        self.dispatcher_thread = threading.Thread(target=self._dispatcher)
        self.dispatcher_thread.start()

        self.dico = {}

        self.raster_path = path
        self.scale = scale

        ds = buzz.DataSource(allow_interpolation=True)

        with ds.open_araster(self.raster_path).close as raster:
            self.full_fp = raster.fp.intersection(raster.fp, scale=scale, alignment=(0,0))
            tile_count = np.ceil(self.full_fp.rsize / 500) 
            self.cache_tiles_fps = self.full_fp.tile_count(*tile_count, boundary_effect='shrink')
            self.num_bands = len(raster)

        self.cache_tile_paths = [
            str(Path(cache_dir) / dir_names[frozenset({rtype})] / str(str(fp.tlx) + "_" + str(fp.tly) + ".tif"))
            for fp in self.cache_tiles_fps.flat
        ]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.req_q.put(None)


    def _callback_dico(self, args, res):
        with self.lock:
            self.dico[args[1]] = res
        with self.cv:
            self.cv.notify_all()
        self.req_q.task_done()

    def _dispatcher(self):
        while True:
            args = self.req_q.get()
            # Stopping the thread
            if args == None:
                return

            with self.lock:
                started = args[1] in self.dico.keys()
                file_exists = os.path.isfile(args[1])

                # Starded
                if started:
                    value = self.dico[args[1]]

                    # Computing
                    if isinstance(value, int):
                        # In both fileExists and !fileExists cases, we wait
                        self.computation_pool.apply_async(self._wait_for_resampling, (args[1],))
                        
                    # Computed
                    else:
                        # fileExists
                        if os.path.isfile(args[1]):
                            # ReadingFile
                            self.io_pool.apply_async(self._get_tile_data, (args[1],), callback=functools.partial(self._callback_dico, args))
                        # !fileExists
                        else:
                            # Should not happen
                            raise RuntimeError("There should be a file")

                # NotStarted
                else:
                    # fileExists
                    if file_exists:
                        # Reading file
                        self.dico[args[1]] = 0
                        self.io_pool.apply_async(self._get_tile_data, (args[1],), callback=functools.partial(self._callback_dico, args))
                    # !fileExists
                    else:
                        # Writing file
                        self.dico[args[1]] = 0
                        self.computation_pool.apply_async(self._resample_tile, args, callback=functools.partial(self._callback_dico, args))



    def _get_slopes(self, proxy):
        fp = proxy.fp
        arr = proxy.get_data(fp=fp.dilate(1))
        nodata_mask = arr == proxy.nodata
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

    def _wait_for_resampling(self, dummy_value):
        with self.cv:
            while isinstance(self.dico[dummy_value], int):
                self.cv.wait()
            self.req_q.task_done()


    def _get_tile_data(self, tile_path):
        if not hasattr(self.thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self.thread_storage.ds = ds
        else:
            ds = self.thread_storage.ds

        with ds.open_araster(tile_path).close as raster:
            if self.get_slopes:
                out = self._get_slopes(raster)
            else:
                out = raster.get_data(band=-1)
        return out


    def _merge_out_tiles(self, tiles, data, out_fp):
        if self.num_bands > 1:
            out = np.empty(tuple(out_fp.shape) + (self.num_bands,), dtype="uint8")
        else:
            out = np.empty(tuple(out_fp.shape), dtype="float32")

        for tile, dat  in zip(tiles, data):
            out[tile.slice_in(out_fp, clip=True)] = dat[out_fp.slice_in(tile, clip=True)]
        return out


    def _resample_tile(self, tile_fp, tile_path):
        print("resample in")
        if not hasattr(self.thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self.thread_storage.ds = ds

        else:
            ds = self.thread_storage.ds

        with ds.open_araster(self.raster_path).close as src:
            out = src.get_data(band=-1, fp=tile_fp)
            if len(src) == 4:
                out = np.where((out[...,3] == 255)[...,np.newaxis], out, 0)

        out_proxy = ds.create_araster(tile_path, tile_fp, src.dtype, len(src), driver="GTiff", band_schema={"nodata": src.nodata}, sr=src.wkt_origin)
        out_proxy.set_data(out, band=-1)
        if self.get_slopes:
            out = self._get_slopes(out_proxy)
        out_proxy.close()
        print("resample out")
        return out


    def get_data(self, input_fp):
        if not hasattr(self.thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self.thread_storage.ds = ds

        else:
            ds = self.thread_storage.ds

        input_data = []
        intersecting_tiles = []

        def tile_info_gen():
            for cache_tile, filename in zip(self.cache_tiles_fps.flat, self.cache_tile_paths):
                if cache_tile.share_area(input_fp):
                    yield cache_tile, filename


        for fp, filename in tile_info_gen():
            self.req_q.put((fp, filename))
            
        self.req_q.join()

        for fp, filename in tile_info_gen():
            with self.lock:
                input_data.append(self.dico[filename])
            intersecting_tiles.append(fp)

        return self._merge_out_tiles(intersecting_tiles, input_data, input_fp)










def output_fp_to_input_fp(fp, scale, rsize):
    out = buzz.Footprint(tl=fp.tl, size=fp.size, rsize=fp.size/scale)
    padding = (rsize - out.rsizex) / 2
    assert padding == int(padding)
    out = out.dilate(padding)
    return out








if __name__ == "__main__":
    print("hello")

    rgb_path = "./ortho_8.00cm.tif"
    dsm_path = "./dsm_8.00cm.tif"
    model_path = "./18-01-25-15-38-19_1078_1.00000000_0.07799472_aracena.hdf5"

    print("uids...")
    dir_names = uids_of_paths({
            "ortho": rgb_path,
            "dsm": dsm_path
        })

    cache_dir = "./.cache"

    for path in dir_names.values():
        os.makedirs(str(Path(cache_dir) / path), exist_ok=True)

    datasrc = buzz.DataSource(allow_interpolation=True)

    print("model...")
    model = load_model(model_path)


    with datasrc.open_araster(rgb_path).close as raster:
        out_fp = raster.fp.intersection(raster.fp, scale=1.28, alignment=(0,0))

    out_fp = out_fp.intersection(out_fp, scale=0.64)

    out_tiles = out_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)
    print(out_tiles[0,0])

    rgba_tiles = np.asarray([
        output_fp_to_input_fp(tile, 0.64, model.get_layer("rgb").input_shape[1]) 
        for tile in out_tiles.flatten()
    ]).reshape(out_tiles.shape)

    dsm_tiles = np.asarray([
        output_fp_to_input_fp(tile, 1.28, model.get_layer("slopes").input_shape[1]) 
        for tile in out_tiles.flatten()
    ]).reshape(out_tiles.shape)

    print("overhead done!")

    with MultiThreadedRasterResampler(rgb_path, 0.64, 'ortho', cache_dir) as rgb_resampler:
        with MultiThreadedRasterResampler(dsm_path, 1.28, 'dsm', cache_dir, get_slopes=False) as dsm_resampler:

            rgb_results = []
            slopes_results = []

            for tile_index in range(13):
                rgb_results.append(rgb_resampler.computation_pool.apply_async(rgb_resampler.get_data, (rgba_tiles.flat[tile_index],)))
                # slopes_results.append(rgb_resampler.computation_pool.apply_async(dsm_resampler.get_data, (dsm_tiles.flat[tile_index],)))

            for result_index in range(13):
                rgb_results[result_index].get()
                # slopes_results[result_index].get()


            # show_many_images(
            #     [out_arrays[0]], 
            #     extents=[input_tiles[0, 0].extent]
            # )