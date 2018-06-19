from pathlib import Path
import multiprocessing as mp
import multiprocessing.pool
import os
import queue
import threading

from keras.models import load_model

import scipy.ndimage as ndi
import numpy as np
import buzzard as buzz

from show_many_images import show_many_images
from uids_of_paths import uids_of_paths

from Raster import raster_factory
from output_fp_to_input_fp import output_fp_to_input_fp

CATEGORIES = (
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
#     "ortho": rgb_path,
#     "dsm": dsm_path
# })
DIR_NAMES = {
    frozenset(["ortho"]): "rgb",
    frozenset(["dsm"]): "dsm",
    frozenset(["ortho", "dsm"]): "both"
}

CACHE_DIR = "./.cache"


g_io_pool = mp.pool.ThreadPool()
g_cpu_pool = mp.pool.ThreadPool()
g_merge_pool = mp.pool.ThreadPool()
g_gpu_pool = mp.pool.ThreadPool(1)




def resampled_raster(raster, scale, cache_dir, cache_fps):
    """
    resampled raster from buzzard raster
    """

    full_fp = raster.fp.intersection(raster.fp, scale=scale, alignment=(0, 0))

    num_bands = len(raster)
    nodata = raster.nodata
    wkt_origin = raster.wkt_origin
    dtype = raster.dtype

    thread_storage = threading.local()

    def compute_data(compute_fp, *data): #*prim_footprints?
        """
        resampled raster compted data when collecting. this is a particular case
        """
        print("resample computing ", threading.currentThread().getName())
        if not hasattr(thread_storage, "ds"):
            ds = buzz.DataSource(allow_interpolation=True)
            thread_storage.ds = ds
        else:
            ds = thread_storage.ds

        with ds.open_araster(raster.path).close as prim:
            got_data = prim.get_data(compute_fp, band=-1)

        assert len(data) == 1

        return got_data


    primitives = {}

    return raster_factory(
        footprint=full_fp, 
        dtype=dtype, 
        nbands=num_bands, 
        nodata=nodata, 
        srs=wkt_origin,
        computation_function=compute_data, 
        cached=True, 
        cache_dir=cache_dir, 
        cache_fps=cache_fps, 
        io_pool=g_io_pool, 
        computation_pool=g_cpu_pool,
        primitives=primitives, 
        computation_fps=cache_fps, 
        merge_pool=g_merge_pool
        )




def slopes_raster(dsm):
    """
    slopes from a raster (abstract raster)
    """

    nodata = dsm.nodata
    full_fp = dsm.fp

    def compute_data(compute_fp, *data):
        """
        computes up and down slopes
        """
        print("slopes computing", threading.currentThread().getName())
        arr, = data
        assert arr.shape == tuple(compute_fp.dilate(1).shape)
        nodata_mask = arr == nodata
        nodata_mask = ndi.binary_dilation(nodata_mask)
        kernel = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        arru = ndi.maximum_filter(arr, None, kernel) - arr
        arru = np.arctan(arru / full_fp.pxsizex)
        arru = arru / np.pi * 180.
        arru[nodata_mask] = 0
        arru = arru[1:-1, 1:-1]

        arrd = arr - ndi.minimum_filter(arr, None, kernel)
        arrd = np.arctan(arrd / full_fp.pxsizex)
        arrd = arrd / np.pi * 180.
        arrd[nodata_mask] = 0
        arrd = arrd[1:-1, 1:-1]

        arr = np.dstack([arrd, arru])
        return arr


    def to_collect_of_to_compute(fp):
        """
        computes to collect from to compute (dilation of 1)
        """
        return {"dsm": fp.dilate(1)}

    
    primitives = {"dsm": dsm.get_multi_data_queue}
    nodata = dsm.nodata
    num_bands = 2
    dtype = "float32"

    return raster_factory(footprint=full_fp,
                     dtype=dtype,
                     nbands=num_bands,
                     nodata=nodata,
                     srs=None, # dsm.wkt_origin
                     computation_function=compute_data,
                     io_pool=g_io_pool,
                     computation_pool=g_cpu_pool,
                     primitives=primitives,
                     to_collect_of_to_compute=to_collect_of_to_compute,
                     merge_pool=g_merge_pool
                    )






def heatmap_raster(model, resampled_rgba, slopes, cache_dir, cache_fps):
    """
    heatmap raster with primitives: ortho + slopes
    """
    def to_collect_of_to_compute(fp):
        """
        Computes the to_collect data from model
        """
        rgba_tile = output_fp_to_input_fp(fp, 0.64, model.get_layer("rgb").input_shape[1])
        slope_tile = output_fp_to_input_fp(fp, 1.28, model.get_layer("slopes").input_shape[1])
        return {"rgba": rgba_tile, "slopes": slope_tile}

    def compute_data(compute_fp, *data):
        """
        predicts data using model
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        rgba_data, slope_data = data

        rgb_data = np.where((rgba_data[..., 3] == 255)[..., np.newaxis], rgba_data, 0)[..., 0:3]
        rgb = (rgb_data.astype('float32') - 127.5) / 127.5

        slopes = slope_data / 45 - 1

        prediction = model.predict([rgb[np.newaxis], slopes[np.newaxis]])[0]
        assert prediction.shape[0:2] == tuple(compute_fp.shape)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return prediction


    dtype = "float32"

    num_bands = LABEL_COUNT

    primitives = {"rgba": resampled_rgba.get_multi_data_queue, "slopes": slopes.get_multi_data_queue}

    max_scale = max(resampled_rgba.fp.scale[0], slopes.fp.scale[0])
    min_scale = min(resampled_rgba.fp.scale[0], slopes.fp.scale[0])

    full_fp = resampled_rgba.fp.intersection(slopes.fp, scale=max_scale, alignment=(0, 0))
    full_fp = full_fp.intersection(full_fp, scale=min_scale)

    computation_tiles = full_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)

    return raster_factory(
        footprint=full_fp, 
        dtype=dtype, 
        nbands=num_bands, 
        computation_function=compute_data,
        cached=True, 
        cache_dir=cache_dir, 
        cache_fps=cache_fps, 
        io_pool=g_io_pool, 
        computation_pool=g_gpu_pool,
        primitives=primitives, 
        to_collect_of_to_compute=to_collect_of_to_compute, 
        computation_fps=computation_tiles, 
        merge_pool=g_merge_pool
    )



def main():
    """
    main program, used for tests
    """
    rgb_path = "./ortho_8.00cm.tif"
    dsm_path = "./dsm_8.00cm.tif"
    model_path = "./18-01-25-15-38-19_1078_1.00000000_0.07799472_aracena.hdf5"

    # Path(CACHE_DIR) / DIR_NAMES[frozenset({"ortho"})]
    # Path(CACHE_DIR) / DIR_NAMES[frozenset({"dsm"})]
    # Path(CACHE_DIR) / DIR_NAMES[frozenset({"ortho", "dsm"})]

    for path in DIR_NAMES.values():
        os.makedirs(str(Path(CACHE_DIR) / path), exist_ok=True)

    datasrc = buzz.DataSource(allow_interpolation=True)

    print("model...")

    model = load_model(model_path)
    model._make_predict_function()
    print("")

    with datasrc.open_araster(rgb_path).close as raster:
        out_fp = raster.fp.intersection(raster.fp, scale=1.28, alignment=(0, 0))

    tile_count128 = np.ceil(out_fp.rsize / 500)
    cache_tiles128 = out_fp.tile_count(*tile_count128, boundary_effect='shrink')

    out_fp = out_fp.intersection(out_fp, scale=0.64)

    tile_count64 = np.ceil(out_fp.rsize / 500)
    cache_tiles64 = out_fp.tile_count(*tile_count64, boundary_effect='shrink')

    initial_rgba = datasrc.open_araster(rgb_path)
    initial_dsm = datasrc.open_araster(dsm_path)

    resampled_rgba = resampled_raster(initial_rgba, 0.64, str(Path(CACHE_DIR) / DIR_NAMES[frozenset({"ortho"})]), cache_tiles64)
    resampled_dsm = resampled_raster(initial_dsm, 1.28, str(Path(CACHE_DIR) / DIR_NAMES[frozenset({"dsm"})]), cache_tiles128)

    slopes = slopes_raster(resampled_dsm)

    hmr = heatmap_raster(model, resampled_rgba, slopes, str(Path(CACHE_DIR) / DIR_NAMES[frozenset({"ortho", "dsm"})]), cache_tiles64)

    big_display_fp = out_fp
    big_dsm_disp_fp = big_display_fp.intersection(big_display_fp, scale=1.28, alignment=(0, 0))

    tile_count64 = np.ceil(out_fp.rsize / 100)
    display_tiles = big_display_fp.tile_count(*tile_count64, boundary_effect='shrink')
    dsm_display_tiles = big_dsm_disp_fp.tile_count(5, 5, boundary_effect='shrink')

    # rgba_out = resampled_rgba.get_multi_data(list(cache_tiles64.flat), 1)
    # slopes_out = slopes.get_multi_data(list(cache_tiles128.flat), 1)
    hm_out = hmr.get_multi_data(cache_tiles64.flat, 1)

    for display_fp in cache_tiles64.flat:
        try:
            show_many_images(
                [np.argmax(next(hm_out), axis=-1)],
                extents=[display_fp.extent]
            )
        except StopIteration:
            print("ended")

    initial_rgba.close()
    initial_dsm.close()

if __name__ == "__main__":
    main()
