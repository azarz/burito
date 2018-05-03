import multiprocessing as mp
import multiprocessing.pool
import threading
import os
import queue
from pathlib import Path
import functools
import pickle

import numpy as np
import buzzard as buzz

from show_many_images import show_many_images
from uids_of_paths import uids_of_paths
from watcher import Watcher

rgb_path = "./ortho_8.00cm.tif"
dsm_path = "./dsm_8.00cm.tif"

print("uids...")
dir_names = uids_of_paths({
        "ortho": rgb_path,
        "dsm": dsm_path
    })

cache_dir = "./.cache/"

overwrite = False

class MultiThreadedRasterResampler(object):

    def __init__(self):
        self.lock = threading.Lock()
        self.dico = {}

        self.cv = threading.Condition()

        self.req_q = queue.Queue()

        self.thread_storage = threading.local()

        self.raster_path = rgb_path

        ds = buzz.DataSource(allow_interpolation=True)

        with ds.open_araster(rgb_path).close as raster:
            self.file_fp = raster.fp.intersection(raster.fp, scale=0.64, alignment=(0,0))
            tile_count = np.ceil(self.file_fp.rsize / 500) 
            self.cache_tiles_fps = self.file_fp.tile_count(*tile_count, boundary_effect='shrink')
            self.num_bands = len(raster)

        self.cache_tile_paths = [
            str(Path(cache_dir) / dir_names[frozenset({'ortho'})] / str(str(fp.tlx) + "_" + str(fp.tly) + ".tif"))
            for fp in self.cache_tiles_fps.flat
        ]

        self.computation_pool = mp.pool.ThreadPool()
        self.io_pool = mp.pool.ThreadPool()

    def _callback_dico(self, args,res):
        with self.lock:
            self.dico[args[1]] = res
        with self.cv:
            self.cv.notify_all()
        self.req_q.task_done()

    def dispatcher(self):
        # TODO
        # (file_exists, !file_exist) && (NotStarted, Computing, Computed)
        while True:
            args = self.req_q.get()
            if os.path.isfile(args[1]):
                self.io_pool.apply_async(self._get_raster_data, (args[1],), callback=functools.partial(self._callback_dico, args))
            else:
                try:
                    with lock:
                        if isinstance(self.dico[args[1]], int):
                            self.computation_pool.apply_async(wait_for_resampling, (args[1],))
                        
                except KeyError:
                    with lock:
                        self.dico[args[1]] = 0
                    self.computation_pool.apply_async(resample_tile, args, callback=functools.partial(callback_dico, args))


    def wait_for_resampling(dummy_value):
        with self.cv:
            while isinstance(self.dico[dummy_value], int):
                self.cv.wait()
            self.req_q.task_done()


    def _get_raster_data(self, raster_path):
        if not hasattr(self.thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self.thread_storage.ds = ds
        else:
            ds = self.thread_storage.ds

        with ds.open_araster(raster_path).close as raster:
            out = raster.get_data(band=-1)
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



    def _merge_out_tiles(self, tiles, results, out_fp):

        if self.num_bands > 1:
            out = np.empty(tuple(out_fp.shape) + (self.num_bands,), dtype="uint8")
        else:
            out = np.empty(tuple(out_fp.shape), dtype="float32")

        for new, out_tile in zip(results, tiles):
            out[out_tile.slice_in(out_fp, clip=True)] = new[out_fp.slice_in(out_tile, clip=True)]
        return out


    def resample_tile(tile_fp, tile_path):
        if not hasattr(self.thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            self.thread_storage.ds = ds

        else:
            ds = self.thread_storage.ds

        with ds.open_araster(raster_path).close as src:
            out = src.get_data(band=-1, fp=tile_fp)
            if len(src) == 4:
                out = np.where((out[...,3] == 255)[...,np.newaxis], out, 0)

        out_proxy = ds.create_araster(tile_path, tile_fp, src.dtype, len(src), driver="GTiff", band_schema={"nodata": src.nodata}, sr=src.wkt_origin)
        out_proxy.set_data(out, band=-1)
        out_proxy.close()
        return out


if __name__ == "__main__":
    print("hello")
    test_obj = MultiThreadedRasterResampler()
    dispatcher_t = threading.Thread(target=test_obj.dispatcher)
    dispatcher_t.daemon = True
    dispatcher_t.start()

    input_tiles = test_obj.file_fp.tile(np.asarray((1030, 1030)))

    # out_array = get_data(input_tiles[0, 0])
    test = test_obj.computation_pool.apply_async(test_obj.get_data, (input_tiles[0, 0],))
    test2 = test_obj.computation_pool.apply_async(test_obj.get_data, (input_tiles[0, 0],))

    # show_many_images(
    #     [out_array], 
    #     extents=[input_tiles[0, 0].extent]
    # )
    test.get()
    test2.get()