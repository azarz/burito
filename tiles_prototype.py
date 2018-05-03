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

lock = threading.Lock() 
dico = {}

cv = threading.Condition()

req_q = queue.Queue()
resample_q = queue.Queue()

thread_storage = threading.local()

raster_path = rgb_path

ds = buzz.DataSource(allow_interpolation=True)

with ds.open_araster(rgb_path).close as raster:
    file_fp = raster.fp.intersection(raster.fp, scale=0.64, alignment=(0,0))
    tile_count = np.ceil(file_fp.rsize / 500) 
    cache_tiles_fps = file_fp.tile_count(*tile_count, boundary_effect='shrink')
    num_bands = len(raster)

cache_tile_paths = [
    str(Path(cache_dir) / dir_names[frozenset({'ortho'})] / str(str(fp.tlx) + "_" + str(fp.tly) + ".tif"))
    for fp in cache_tiles_fps.flat
]

computation_pool = mp.pool.ThreadPool()
io_pool = mp.pool.ThreadPool()

def callback_dico(args,res):
    with lock:
        dico[args[1]] = res
    with cv:
        cv.notify_all()
    req_q.task_done()

def dispatcher():
    while True:
        args = req_q.get()
        if os.path.isfile(args[1]):
            io_pool.apply_async(get_raster_data, (args[1],), callback=functools.partial(callback_dico, args))
        else:
            try:
                if isinstance(dico[args[1]], int):
                    computation_pool.apply_async(wait_for_resampling, (args[1],))
                    
            except KeyError:
                dico[args[1]] = 0
                computation_pool.apply_async(resample_tile, args, callback=functools.partial(callback_dico, args))


def wait_for_resampling(dummy_value):
    with cv:
        while isinstance(dico[dummy_value], int):
            cv.wait()
        req_q.task_done()


def get_raster_data(raster_path):
    with ds.open_araster(raster_path).close as raster:
        out = raster.get_data(band=-1)
    return out

def get_data(input_fp):
    if not hasattr(thread_storage, "ds"):
        ds = buzz.DataSource(allow_interpolation=True)
        ds_obj = pickle.dumps(ds)
        ds = pickle.loads(ds_obj)
        thread_storage.ds = ds

    else:
        ds = thread_storage.ds

    input_data = []
    intersecting_tiles = []

    def tile_info_gen():
        for cache_tile, filename in zip(cache_tiles_fps.flat, cache_tile_paths):
            if cache_tile.share_area(input_fp):
                yield cache_tile, filename


    for fp, filename in tile_info_gen():
        req_q.put((fp, filename))
        
    req_q.join()

    for fp, filename in tile_info_gen():
        with lock:
            input_data.append(dico[filename])
        intersecting_tiles.append(fp)

    return merge_out_tiles(intersecting_tiles, input_data, input_fp)



def merge_out_tiles(tiles, results, out_fp):

    if num_bands > 1:
        out = np.empty(tuple(out_fp.shape) + (num_bands,), dtype="uint8")
    else:
        out = np.empty(tuple(out_fp.shape), dtype="float32")

    for new, out_tile in zip(results, tiles):
        out[out_tile.slice_in(out_fp, clip=True)] = new[out_fp.slice_in(out_tile, clip=True)]
    return out


def resample_tile(tile_fp, tile_path):
    if not hasattr(thread_storage, "ds"):
        ds = buzz.DataSource(allow_interpolation=True)
        thread_storage.ds = ds

    else:
        ds = thread_storage.ds

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
    dispatcher_t = threading.Thread(target=dispatcher)
    dispatcher_t.daemon = True
    dispatcher_t.start()

    input_tiles = file_fp.tile(np.asarray((1030, 1030)))

    # out_array = get_data(input_tiles[0, 0])
    test = computation_pool.apply_async(get_data, (input_tiles[0, 0],))
    test2 = computation_pool.apply_async(get_data, (input_tiles[0, 0],))

    # show_many_images(
    #     [out_array], 
    #     extents=[input_tiles[0, 0].extent]
    # )
    test.get()
    test2.get()