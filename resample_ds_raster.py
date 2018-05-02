import multiprocessing as mp
import os
import threading

# if __name__ == "__main__":
#     mp.set_start_method("fork")

import os
import pickle
import threading
import concurrent.futures as cf
import functools
from pathlib import Path
import queue

import buzzard as buzz
import joblib
import numpy as np
import cloudpickle as cldpckl

from watcher import Watcher
from uids_of_paths import uids_of_paths

NUM_CORES = mp.cpu_count()
# NUM_CORES = 2

def resample_ds_raster(ds, res, in_key, out_key, cache_path, overwrite, parallel_mode="", tile_shape=(100,100)):

    if os.path.isfile(cache_path) and not overwrite:
        ds.open_raster(out_key, cache_path)
        return

    src = ds[in_key]
    out_fp = src.fp.intersection(src.fp, scale=res, alignment=(0,0))

    if parallel_mode == "":
        out = src.get_data(band=-1, fp=out_fp)

    else :

        thread_storage = threading.local()

        def _resample_tile(tile):
            return resample_tile(tile, ds, in_key)

        def _resample_tile_thread(tile):
            return resample_tile_thread(tile, ds, in_key, thread_storage)

        tiles = out_fp.tile(np.asarray(tile_shape).T)

        if parallel_mode == "joblib":
            results = joblib.Parallel(n_jobs=NUM_CORES)(joblib.delayed(resample_tile)(tile, ds, in_key) for tile in tiles.flat)

        elif parallel_mode == "multiprocessing_map":
            with mp.Pool(NUM_CORES) as p:
                results = p.map(functools.partial(resample_tile, ds=ds, key=in_key), tiles.flat)

        elif parallel_mode == "multiprocessing_map_closure":
            with mp.Pool(NUM_CORES) as p:
                results = p.map(functools.partial(uncloudpickler, cldpckl.dumps(_resample_tile)), tiles.flat)

        elif parallel_mode == "multiprocessing_imap":
            with mp.Pool(NUM_CORES) as p:
                results = p.imap(functools.partial(resample_tile, ds=ds, key=in_key), tiles.flat)
                results = list(results)

        elif parallel_mode == "multiprocessing_map_async":
            with mp.Pool(NUM_CORES) as p:
                results = p.map_async(functools.partial(resample_tile, ds=ds, key=in_key), tiles.flat)
                results = results.get()

        elif parallel_mode == "multiprocessing_apply":
            with mp.Pool(NUM_CORES) as p:
                results = [p.apply(resample_tile, (tile, ds, in_key)) for tile in tiles.flat]

        elif parallel_mode == "multiprocessing_apply_async":
            with mp.Pool(NUM_CORES) as p:
                results = [p.apply_async(resample_tile, (tile, ds, in_key)) for tile in tiles.flat]
                results = [result.get() for result in results]

        elif parallel_mode == "multiprocessing_threadpool":
            with mp.pool.ThreadPool(NUM_CORES) as p:
                results = p.map(functools.partial(resample_tile_thread, ds=ds, key=in_key, thread_storage=thread_storage), tiles.flat)

        elif parallel_mode == "multiprocessing_threadpool_async":
            with mp.pool.ThreadPool(NUM_CORES) as p:
                results = p.map_async(functools.partial(resample_tile_thread, ds=ds, key=in_key, thread_storage=thread_storage), tiles.flat)
                results = results.get()

        elif parallel_mode == "multiprocessing_threadpool_closure":
            with mp.pool.ThreadPool(NUM_CORES) as p:
                results = p.map(_resample_tile_thread, tiles.flat)

        elif parallel_mode == "cf_threadpool":
            with cf.ThreadPoolExecutor(max_workers=NUM_CORES) as ex:
                results = ex.map(functools.partial(resample_tile_thread, ds=ds, key=in_key, thread_storage=thread_storage), tiles.flat)

        elif parallel_mode == "cf_threadpool_as_completed":
            results = []
            with cf.ThreadPoolExecutor(max_workers=NUM_CORES) as ex:
                futures = {ex.submit(functools.partial(resample_tile_thread, ds=ds, key=in_key, thread_storage=thread_storage), tile): res for tile in tiles.flat}
                for future in cf.as_completed(futures):
                    results.append(future.result())                

        elif parallel_mode == "cf_processpool":
            with cf.ProcessPoolExecutor(max_workers=NUM_CORES) as ex:
                results = ex.map(functools.partial(resample_tile, ds=ds, key=in_key), tiles.flat)

        elif parallel_mode == "multiprocessing_process":
            ctx = mp.get_context("spawn")

            # if __name__ == "__main__":
            #     ctx = mp.get_context("spawn")

            output = ctx.Queue()
            input_queue = ctx.Queue()
            processes = []

            for i in range(NUM_CORES):
                p = ctx.Process(target=process_worker, args=(input_queue, output, ds, in_key))
                p.start()
                processes.append(p)

            for tile in tiles.flat:    
                input_queue.put(tile)

            results_unsorted = [output.get() for tile in tiles.flat]
            results = []

            footprint_index = {}
            for i, tile in enumerate(tiles.flat):
                footprint_index[repr(tile)] = i

            results = np.asarray(sorted(results_unsorted, key=lambda tuple: footprint_index[repr(tuple[0])]))[:,1]

            for i in range(NUM_CORES):
                input_queue.put(None)

            for p in processes:
                p.join()

        elif parallel_mode == "threading_thread":

            new_watcher = Watcher()

            def worker():
                while True:
                    tile = q.get()
                    if tile is None:
                        break
                    output.put((tile, resample_tile_thread(tile, ds, in_key, thread_storage)))
                    q.task_done()

            q = queue.Queue()
            output = queue.Queue()
            threads = []
            for i in range(NUM_CORES):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            for tile in tiles.flat:
                q.put(tile)

            results_unsorted = [output.get() for tile in tiles.flat]

            footprint_index = {}
            for i, tile in enumerate(tiles.flat):
                footprint_index[repr(tile)] = i

            results = np.asarray(sorted(results_unsorted, key=lambda tuple: footprint_index[repr(tuple[0])]))[:,1]

            # block until all tasks are done
            q.join()

            # stop workers
            for i in range(NUM_CORES):
                q.put(None)
            for t in threads:
                t.join()

        else:
            raise ValueError()

        out = merge_out_tiles(tiles, results, out_fp, src)

    ds.create_raster(out_key, cache_path, out_fp, src.dtype, len(src), driver="GTiff", band_schema={"nodata": src.nodata}, sr=src.wkt_origin)
    ds[out_key].set_data(out, band=-1)


def resample_tile(tile, ds, key):
    print('thread id:', threading.current_thread().ident, 'process id:', os.getpid())
    src = ds[key]
    out = src.get_data(band=-1, fp=tile)

    if len(src) == 4:
        out = np.where((out[...,3] == 255)[...,np.newaxis], out, 0)

    return out

def resample_tile_thread(tile, ds, key, thread_storage):

    if not hasattr(thread_storage, "ds"):
        ds_obj = pickle.dumps(ds)
        ds = pickle.loads(ds_obj)
        thread_storage.ds = ds

    else:
        ds = thread_storage.ds
    return resample_tile(tile, ds, key)

def resample_tile_queue(tile, ds, key, queue):
    ds.deactivate_all()
    queue.put((tile, resample_tile(tile, ds, key)))

def merge_out_tiles(tiles, results, out_fp, src):

    if len(src) > 1:
        out = np.empty(tuple(out_fp.shape) + (len(src),), dtype="uint8")
    else:
        out = np.empty(tuple(out_fp.shape), dtype="float32")

    for new, out_tile in zip(results, tiles.flat):
        out[out_tile.slice_in(out_fp, clip=True)] = new[out_fp.slice_in(out_tile, clip=True)]
    return out

def uncloudpickler(cldpickled_function, *args):
    return cldpckl.loads(cldpickled_function)(*args)


def process_worker(in_q, out_q, ds, key):
    while True:
        tile = in_q.get()
        if tile is None:
            break
        resample_tile_queue(tile, ds, key, out_q)

if __name__ == "__main__":
    # https://bugs.python.org/issue6721
    methods = [
                # "joblib", "multiprocessing_map", 
                # "multiprocessing_map_closure", "multiprocessing_imap", 
                # "multiprocessing_map_async", "multiprocessing_apply", "multiprocessing_apply_async","cf_processpool", 
                # "multiprocessing_map_async",
                # "multiprocessing_process",
                # "cf_threadpool", 
                # "cf_threadpool_as_completed", 
                "threading_thread",
                # "multiprocessing_threadpool", 
                # "multiprocessing_threadpool_async", 
                # "multiprocessing_threadpool_closure",
                ""]

    rgb_path = './ortho_8.00cm.tif'
    dsm_path = './dsm_8.00cm.tif'

    model_path = "./18-01-25-15-38-19_1078_1.00000000_0.07799472_aracena.hdf5"
    cache_dir = "./.cache/"

    watcher = Watcher()

    dir_names = uids_of_paths({
        "ortho": rgb_path,
        "dsm": dsm_path
    })

    for path in dir_names.values():
        os.makedirs(str(Path(cache_dir) / path), exist_ok=True)

    for method in methods:
        ds = buzz.DataSource(allow_interpolation=True)
        ds.open_raster('rgba', rgb_path)
        ds.open_raster('dsm', dsm_path)
        with watcher(method):
            resample_ds_raster(ds, 0.64, "rgba", "rgba64", str(Path(cache_dir) / dir_names[frozenset({'ortho'})] / "rgba64.tif"), True, method)
        ds.deactivate_all()
