"""
For testing purposes only
"""

import numpy as np
import buzzard as buzz

def create_work_tiles(fp, work_overlap_fraction):
    """Create the tiling of `fp` that is compatible with the model `m` and all its inputs.
    Constraint 1: Should be aligned with all model's input files
    Constraint 2: Two neighboring tiles should have the same global aligment in their poolings
      to avoid huge boundary effect when stitching tiles.

    Parameters
    ----------
    fp: Footprint
        The heatmap footprint to compute
    m: Model
    work_overlap_fraction: float
        Fraction of a tile to overlap with its neighbor

    Returns
    -------
    work_tiles: ndarray of Footprint of shape (H, W)
        Tiling with overlap and modulo alignment

    """

    # Phase 0 - Recreate `fp` such that it can serve as a basis for tiling
    fp0 = fp
    fp1 = fp0.intersection(fp0, scale=1.28, alignment=(0, 0))
    fp2 = buzz.Footprint(gt=fp1.gt, rsize=(512, 512))

    fp3 = fp2.intersection(fp2, scale=0.64)
    fp3 = fp3.erode((fp3.rw - 500) / 2)
    assert np.allclose(fp2.c, fp3.c)
    fp4 = fp3.move(
        tl=fp3.tl - np.ceil((fp3.tl - fp0.tl) / fp2.scale) * fp2.scale
    )
    fp5 = buzz.Footprint(
        gt=fp4.gt,
        rsize=fp4.rsize + np.around((fp0.br - fp4.br) / fp0.pxvec),
    )
    assert np.allclose(fp5.br, fp0.br)

    fp = fp5

    # Phase 1 - Calculate the stride between neighboring tiles ********************************** **
    work_rsize = np.array([500, 500])
    work_modulo = 8.

    x = work_rsize * (1 - work_overlap_fraction)
    x = x / work_modulo
    x = np.around(x)
    x = x * work_modulo
    x = x.clip(work_modulo, work_rsize // work_modulo * work_modulo)
    x = x.astype(int)
    aligned_rsize = x

    # Phase 2 - Instanciate tiles *************************************************************** **
    aligned_tiles = fp.tile(aligned_rsize)
    work_tiles = np.vectorize(
        lambda tile: buzz.Footprint(gt=tile.gt, rsize=work_rsize)
    )(aligned_tiles)

    return work_tiles
