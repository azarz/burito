"""
Creates a footprint of rsize from fp using dilation and scale
Used to compute a keras model input footprint from output
(e.g.: deduce the big RGB and slopes extents from the smaller output heatmap)
"""
import numpy as np
import buzzard as buzz

def output_fp_to_input_fp(fp, scale, rsize):
    """
    Creates a footprint of rsize from fp using dilation and scale
    Used to compute a keras model input footprint from output
    (e.g.: deduce the big RGB and slopes extents from the smaller output heatmap)
    """
    out = buzz.Footprint(tl=fp.tl, size=fp.size, rsize=np.around(fp.size/scale))
    padding = (rsize - out.rsizex) / 2
    assert padding == int(padding)
    out = out.dilate(padding)
    return out
