import multiprocessing as mp
import os
import threading

# if __name__ == "__main__":
#     mp.set_start_method("fork")

print(__name__, 'thread id:', threading.current_thread().ident, 'process id:', os.getpid())

import argparse
import os
from pathlib import Path

from keras.models import load_model
import buzzard as buzz
import descartes
import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndi
import skimage.morphology as morph

from show_many_images import show_many_images
from uids_of_paths import uids_of_paths
from watcher import Watcher

from resample_ds_raster import resample_ds_raster

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


def get_slopes(proxy):
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


def output_fp_to_input_fp(fp, scale, rsize):
    out = buzz.Footprint(tl=fp.tl, size=fp.size, rsize=fp.size/scale)
    padding = (rsize - out.rsizex) / 2
    assert padding == int(padding)
    out = out.dilate(padding)
    return out


def main(rgb_path, dsm_path, overwrite):
    model_path = "./18-01-25-15-38-19_1078_1.00000000_0.07799472_aracena.hdf5"
    cache_dir = "./.cache/"

    watcher = Watcher()

    dir_names = uids_of_paths({
        "ortho": rgb_path,
        "dsm": dsm_path
    })

    for path in dir_names.values():
        os.makedirs(str(Path(cache_dir) / path), exist_ok=True)

    ds = buzz.DataSource(allow_interpolation=True)
    ds.open_raster('rgba', rgb_path)
    ds.open_raster('dsm', dsm_path)

    with watcher("rgb64"):
        resample_ds_raster(ds, 0.64, "rgba", "rgba64", str(Path(cache_dir) / dir_names[frozenset({'ortho'})] / "rgba64.tif"), overwrite, "multiprocessing_map")

    with watcher("dsm128"):
        resample_ds_raster(ds, 1.28, "dsm", "dsm128", str(Path(cache_dir) / dir_names[frozenset({'dsm'})] / "dsm128.tif"), overwrite, "multiprocessing_map")
    with watcher("slopes128"):    
        add_slopes_to_ds(ds, "dsm128", "slopes128", str(Path(cache_dir) / dir_names[frozenset({'dsm'})] / "slopes.tif"), overwrite)

    with watcher("hm"):
        add_prediction_to_ds(ds, "hm", model_path, str(Path(cache_dir) / dir_names[frozenset({'dsm', "ortho"})] / "hm.tif"), overwrite)

    with watcher("roads"):
        add_roads_poly_to_ds(ds, "roads", str(Path(cache_dir) / dir_names[frozenset({'dsm', "ortho"})] / "roads.shp"), overwrite)

    show_many_images(
        [ds.rgba64.get_data(band=-1), ds.dsm128.get_data(), ds.hm.get_data(band=INDEX_ROADS + 1), ds.hm.get_data(band=INDEX_VEHICLES + 1)], 
        extents=[ds.rgba.fp.extent, ds.dsm128.fp.extent, ds.hm.fp.extent, ds.hm.fp.extent],
        patchess=[[descartes.PolygonPatch(poly, fill=False, ec='#ff0000', lw=3, ls='--') for poly in ds.roads.iter_data(None)]]
    )


"""
def resample_ds_raster(ds, res, in_key, out_key, cache_path, overwrite):

    if os.path.isfile(cache_path) and not overwrite:
        ds.open_raster(out_key, cache_path)
        return

    src = ds[in_key]
    out_fp = src.fp.intersection(src.fp, scale=res, alignment=(0,0))

    ds.create_raster(out_key, cache_path, out_fp, src.dtype, len(src), driver="GTiff", band_schema={"nodata": src.nodata}, sr=src.wkt_origin)
    out = src.get_data(band=-1, fp=out_fp)

    if len(src) == 4:
        out = np.where((out[...,3] == 255)[...,np.newaxis], out, 0)

    ds[out_key].set_data(out, band=-1)
"""

def add_slopes_to_ds(ds, in_key, out_key, cache_path, overwrite):

    # if os.path.isfile(cache_path) and not overwrite:
    #     ds.open_raster(out_key, cache_path)
    #     return

    src = ds[in_key]
    slopes = get_slopes(src)
    ds.create_raster(out_key, cache_path, src.fp, src.dtype, 2, driver="GTiff", band_schema={"nodata": src.nodata}, sr=src.wkt_origin)
    ds[out_key].set_data(slopes, band=-1)



def add_prediction_to_ds(ds, out_key, model_path, cache_path, overwrite):

    if os.path.isfile(cache_path) and not overwrite:
        ds.open_raster(out_key, cache_path)
        return

    model = load_model(model_path)

    out_fp = ds.rgba64.fp
    out_tiles = out_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)

    rgba_tiles = np.asarray([
        output_fp_to_input_fp(tile, 0.64, model.get_layer("rgb").input_shape[1]) 
        for tile in out_tiles.flatten()
    ]).reshape(out_tiles.shape)

    dsm_tiles = np.asarray([
        output_fp_to_input_fp(tile, 1.28, model.get_layer("slopes").input_shape[1]) 
        for tile in out_tiles.flatten()
    ]).reshape(out_tiles.shape)

    out = np.empty(tuple(out_fp.shape) + (LABEL_COUNT,), dtype="float32")

    for out_tile, rgba_tile, dsm_tile in zip(out_tiles.flat, rgba_tiles.flat, dsm_tiles.flat):

        assert out_tile.poly.within(rgba_tile.poly)
        assert out_tile.poly.within(dsm_tile.poly)

        rgb = ds.rgba64.get_data(band=(1, 2, 3), fp=rgba_tile).astype('uint8')
        # rgb = np.where((rgba[...,3] == 255)[...,np.newaxis], rgba[...,0:3], 0)
        assert abs(rgba_tile.cx - dsm_tile.cx) < 0.00001
        assert abs(rgba_tile.cy - dsm_tile.cy) < 0.00001

        slopes = ds.slopes128.get_data(fp=dsm_tile, band=-1)
        slopes[slopes == ds.slopes128.nodata] = 0

        rgb_float = (rgb.astype('float32') - 127.5) / 127.5
        slopes = slopes / 45 - 1
        new = model.predict([rgb_float[np.newaxis], slopes[np.newaxis]])
        out[out_tile.slice_in(out_fp, clip=True)] = new[0][out_fp.slice_in(out_tile, clip=True)];

    ds.create_raster(out_key, cache_path, out_fp, "float32", LABEL_COUNT, driver="GTiff", sr=ds.rgba64.wkt_origin)
    ds[out_key].set_data(out, band=-1)


def add_roads_poly_to_ds(ds, out_key, cache_path, overwrite):
    if os.path.isfile(cache_path) and not overwrite:
        ds.open_vector(out_key, cache_path)
        return

    poly_list = roads_poly_from_hm(ds.hm.get_data(band=-1), ds.hm.fp)
    ds.create_vector(out_key, cache_path, "polygon", driver="ESRI Shapefile")
    for poly in poly_list:
        ds[out_key].insert_data(poly)



def roads_poly_from_hm(heatmap, 
                        fp,
                        binary_threshold=0.5,
                        max_water_area=250,
                        max_vehicle_area=50,
                        water_dilation_radius=0.5,
                        vehicle_dilation_radius=0.5,
                        min_hole_area=250,
                        min_road_area=2000,
                        road_close_radius=2,
                        simplify_slack_distance=1):

    roads = heatmap[...,INDEX_ROADS] >= binary_threshold
    water = heatmap[...,INDEX_WATER] >= binary_threshold
    vehicles = heatmap[...,INDEX_VEHICLES] >= binary_threshold

    pixel_surface = np.prod(fp.pxsizex)

    water_nb_px_thresh = np.ceil(max_water_area/pixel_surface)
    vehic_nb_px_thresh = np.ceil(max_vehicle_area/pixel_surface)

    labels, nlabels = ndi.label(water)
    objects = ndi.find_objects(labels, nlabels)

    for current_label, obj in enumerate(objects, 1):
        if (labels[obj] == current_label).sum() > water_nb_px_thresh:
            water[obj][labels[obj] == current_label] = False

    labels, nlabels = ndi.label(vehicles)
    objects = ndi.find_objects(*ndi.label(vehicles))

    for current_label, obj in enumerate(objects, 1):
        if (labels[obj] == current_label).sum() > vehic_nb_px_thresh:
            vehicles[obj] = np.where(labels[obj] == current_label, vehicles[obj], False)

    water = ndi.morphology.binary_dilation(water, iterations=np.ceil(water_dilation_radius/fp.pxsizex))
    vehicles = ndi.morphology.binary_dilation(vehicles, iterations=np.ceil(vehicle_dilation_radius/fp.pxsizex))

    merged = roads | water | vehicles

    nb_px_thresh = np.ceil(min_road_area/pixel_surface)
    hole_nb_px_thresh = np.ceil(min_hole_area/pixel_surface)
    morph.remove_small_holes(merged, hole_nb_px_thresh, in_place=True)
    morph.remove_small_objects(merged, nb_px_thresh, in_place=True)

    merged = ndi.morphology.binary_closing(merged, iterations=int(np.ceil(road_close_radius/fp.pxsizex)))

    poly_list = fp.find_polygons(merged)

    poly_list = [poly.simplify(simplify_slack_distance) for poly in poly_list]

    return poly_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ortho_path', type=str,
                    help='orthoimage path', default='./ortho_8.00cm.tif')
    parser.add_argument('--dsm_path', type=str, default='./dsm_8.00cm.tif',
                    help='dsm path')
    parser.add_argument('-o', action="store_true")

    args = parser.parse_args()

    main(args.ortho_path, args.dsm_path, args.o)