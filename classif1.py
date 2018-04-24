import buzzard as buzz
import numpy as np
import scipy.ndimage as ndi

from keras.models import load_model
from show_many_images import show_many_images

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

out_tiles = out_fp.tile(model.outputs[0].shape[1:3])

rgba_tiles = np.asarray([
    output_fp_to_input_fp(tile, 0.64, model.get_layer("rgb").input_shape[1]) 
    for tile in out_tiles.flatten()
]).reshape(out_tiles.shape)

dsm_tiles  = np.asarray([
    output_fp_to_input_fp(tile, 1.28, model.get_layer("slopes").input_shape[1]) 
    for tile in out_tiles.flatten()
]).reshape(out_tiles.shape)

for out_tile, rgba_tile, dsm_tile in zip(out_tiles.flat, rgba_tiles.flat, dsm_tiles.flat):

    rgba = ds.rgba.get_data(band=(-1), fp=rgba_tile).astype('uint8')
    rgb = ds.rgba.get_data(band=(1,2,3), fp=rgba_tile).astype('uint8')

    assert abs(rgba_tile.cx - dsm_tile.cx) < 0.00001
    assert abs(rgba_tile.cy - dsm_tile.cy) < 0.00001

    slopes = get_slopes(dsm_path, dsm_tile)

    rgb_float = (rgb.astype('float32') - 127.5) / 127.5

    out = model.predict([rgb_float[np.newaxis], slopes[np.newaxis]])

    show_many_images(
        [rgb, slopes[..., 0], out[0, ..., 1]], 
        extents=[rgba_tile.extent, dsm_tile.extent, out_tile.extent]
    )
