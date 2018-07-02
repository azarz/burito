import os
import collections

from pytilia.visorbearer import *
from pytilia.watcher import *
import buzzard as buzz
import scipy.ndimage as ndi
import numpy as np
import shapely.geometry as sg

from custom_rasters import *

buzz.Footprint.__hash__ = lambda fp: hash(repr(fp))

def main():

    # ******************************************************************************************* **
    # Local constants
    q = 'A2C-Frecul-20160330'

    pref = '/media/ngoguey/Donnees1/ngoguey/gannet_v3'
    dsm_suff = 'SRC/3_DSM/dsm.tif'
    ortho_suff = 'SRC/2_Ortho/ortho.tif'

    # ******************************************************************************************* **
    # Constants transformation
    dsm_path = os.path.join(pref, q, dsm_suff)
    ortho_path = os.path.join(pref, q, ortho_suff)

    # ******************************************************************************************* **
    # Open base data
    ds = buzz.DataSource()
    with ds.open_araster(dsm_path).close as r:
        dsm = wrap_buzzard_raster(r)
    with ds.open_araster(ortho_path).close as r:
        ortho = wrap_buzzard_raster(r)

    # ******************************************************************************************* **
    # Open resampled rasters
    scales = [0.16, 0.32]
    # scales = [0.16, 0.32, 0.64, 1.28]


    dsms = collections.OrderedDict()
    r = dsm
    for scale in scales:
        rr = derive_raster(r, fp=r.fp.intersection(r.fp, scale=scale, alignment=(0, 0)))
        dsms[scale] = rr
        r = rr

    orthos = collections.OrderedDict()
    r = ortho
    for scale in scales:
        rr = derive_raster(r, fp=r.fp.intersection(r.fp, scale=scale, alignment=(0, 0)))
        orthos[scale] = rr
        r = rr


    poly = sg.Point(*dsm.fp.c).buffer(10)
    imgs = [
        # ortho.get_multi_data([ortho.fp]),
        # *[r.get_multi_data([r.fp]) for r in orthos.values()],


        # *[r.get_multi_data([r.fp]) for r in dsms.values()],

        # dsm.get_multi_data([dsm.fp & poly]),
        dsms[0.16].get_multi_data([dsms[0.16].fp & poly]),
        dsms[0.32].get_multi_data([dsms[0.32].fp & poly]),
        # dsms[0.64].get_multi_data([dsms[0.64].fp & poly]),
        # dsms[1.28].get_multi_data([dsms[1.28].fp & poly]),

    ]
    extents = [dsm.fp.extent]  * len(imgs)

        # dsm16.fp.extent,
        # dsm32.fp.extent,
        # ortho.fp.extent,
    # ]
    # + [r.fp.extent for r in dsms.values()]

    imgs = list(map(next, imgs))


    for img in imgs:
        if img.ndim == 2:
            nodata_mask = img == dsm.nodata
            img[nodata_mask] = img[~nodata_mask].min()
    # show_many_images(imgs, patchess=(), extents=(), batches=1, subtitles=(), cmaps=(), title='', ratio=1.7777777777777777, mode='display', offset=True, visible_axes=True, share_scale=False)

    show_many_images(imgs, extents=extents)


if __name__ == "__main__":
    with buzz.Env(significant=9):
        print('World Hello')
        main()
        print('World Bye')
