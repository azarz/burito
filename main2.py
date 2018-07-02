import os

from pytilia.visorbearer import *
from pytilia.watcher import *
import buzzard as buzz
import scipy.ndimage as ndi
import numpy as np

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
    ds = buzz.DataSource()
    with ds.open_araster(dsm_path).close as r:
        dsm = wrap_buzzard_raster(r)
    with ds.open_araster(ortho_path).close as r:
        ortho = wrap_buzzard_raster(r)


    imgs = [
        dsm.get_multi_data([dsm.fp]),
        ortho.get_multi_data([ortho.fp]),
    ]
    extents = [
        dsm.fp.extent,
        ortho.fp.extent,
    ]

    imgs = list(map(next, imgs))


    nodata_mask = imgs[0] == dsm.nodata
    imgs[0][nodata_mask] = imgs[0][~nodata_mask].min()
    # show_many_images(imgs, patchess=(), extents=(), batches=1, subtitles=(), cmaps=(), title='', ratio=1.7777777777777777, mode='display', offset=True, visible_axes=True, share_scale=False)

    show_many_images(imgs, extents=extents)


if __name__ == "__main__":
    with buzz.Env(significant=9):
        print('World Hello')
        main()
        print('World Bye')
