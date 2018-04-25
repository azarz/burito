import buzzard as buzz
import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndi
import skimage.morphology as morph
import descartes

from keras.models import load_model
from show_many_images import show_many_images

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

def get_slopes(dsm_path, fp, normalize=True, intersect=False):
    ds = buzz.DataSource(allow_interpolation=True)
    with ds.open_araster(dsm_path).close as r:
        if intersect:
            fp = r.fp.dilate(fp.rlength // 2) & fp
        arr = r.get_data(fp=fp.dilate(1))
        nodata_mask = arr == r.nodata
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
    if normalize:
        arr = arr / 45 - 1
    return arr


def output_fp_to_input_fp(fp, scale, rsize):
    out = buzz.Footprint(tl=fp.tl, size=fp.size, rsize=fp.size/scale)
    padding = (rsize - out.rsizex) / 2
    assert padding == int(padding)
    out = out.dilate(padding)
    return out


model = load_model("./18-01-25-15-38-19_1078_1.00000000_0.07799472_aracena.hdf5")
rgb_path = "./ortho_8.00cm.tif"
dsm_path = "./dsm_8.00cm.tif"

ds = buzz.DataSource(allow_interpolation=True)

ds.open_raster('rgba', rgb_path)
ds.open_raster('dsm', dsm_path)

out_fp = ds.rgba.fp.intersection(ds.dsm.fp, scale=0.64)

out_tiles = out_fp.tile(np.asarray(model.outputs[0].shape[1:3]).T)

rgba_tiles = np.asarray([
    output_fp_to_input_fp(tile, 0.64, model.get_layer("rgb").input_shape[1]) 
    for tile in out_tiles.flatten()
]).reshape(out_tiles.shape)

dsm_tiles  = np.asarray([
    output_fp_to_input_fp(tile, 1.28, model.get_layer("slopes").input_shape[1]) 
    for tile in out_tiles.flatten()
]).reshape(out_tiles.shape)

out = np.empty(tuple(out_fp.shape) + (13,), dtype="float32")

for out_tile, rgba_tile, dsm_tile in zip(out_tiles.flat, rgba_tiles.flat, dsm_tiles.flat):

    assert out_tile.poly.within(rgba_tile.poly)
    assert out_tile.poly.within(dsm_tile.poly)

    rgba = ds.rgba.get_data(band=(-1), fp=rgba_tile).astype('uint8')
    rgb = np.where((rgba[...,3] == 255)[...,np.newaxis], rgba[...,0:3], 0)
    assert abs(rgba_tile.cx - dsm_tile.cx) < 0.00001
    assert abs(rgba_tile.cy - dsm_tile.cy) < 0.00001

    slopes = get_slopes(dsm_path, dsm_tile)

    rgb_float = (rgb.astype('float32') - 127.5) / 127.5
    new = model.predict([rgb_float[np.newaxis], slopes[np.newaxis]])
    out[out_tile.slice_in(out_fp, clip=True)] = new[0][out_fp.slice_in(out_tile, clip=True)];

roads = out[...,INDEX_ROADS] >= 0.5
water = out[...,INDEX_WATER] >= 0.5
vehicles = out[...,INDEX_VEHICLES] >= 0.5

pixel_surface = np.prod(out_tile.pxsize)

water_nb_px_thresh = np.ceil(250/pixel_surface)
vehic_nb_px_thresh = np.ceil(50/pixel_surface)

labels, nlabels = ndi.label(water)
objects = ndi.find_objects(labels, nlabels)

for current_label, obj in enumerate(objects, 1):
    if (labels[obj] == current_label).sum() > water_nb_px_thresh:
        water[obj][labels[obj] == current_label] =  False

labels, nlabels = ndi.label(vehicles)
objects = ndi.find_objects(*ndi.label(vehicles))

for current_label, obj in enumerate(objects, 1):
    if (labels[obj] == current_label).sum() > vehic_nb_px_thresh:
        vehicles[obj] = np.where(labels[obj] == current_label, vehicles[obj], False)

water = ndi.morphology.binary_dilation(water, iterations=np.ceil(0.5/out_tile.pxsizex))
vehicles = ndi.morphology.binary_dilation(vehicles, iterations=np.ceil(0.5/out_tile.pxsizex))

merged = roads | water | vehicles

nb_px_thresh = np.ceil(2000/pixel_surface)
morph.remove_small_holes(merged, water_nb_px_thresh, in_place=True)
morph.remove_small_objects(merged, nb_px_thresh, in_place=True)

merged = ndi.morphology.binary_closing(merged, iterations=int(np.ceil(2/out_tile.pxsizex)))

poly_list = out_fp.find_polygons(merged)

poly_list = [poly.simplify(1) for poly in poly_list]

show_many_images(
    [ds.rgba.get_data(band=(1,2,3), fp=out_fp).astype('uint8'), water, merged], 
    extents=[out_fp.extent] * 3,
    patchess=[[descartes.PolygonPatch(poly_list[0], fill=False, ec='#ff0000', lw=3, ls='--')]]
)


