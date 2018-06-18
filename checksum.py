"""
Computes the checksum of an array
"""
import numpy as np
import buzzard as buzz

def checksum(array):
    """
    Computes the checksum of an array
    """
    # Reinterpret array as unsigned integers
    b = array.view({
        np.float32: np.uint32,
        np.float64: np.uint64,
        np.uint8: np.uint8
    }[array.dtype.type])

    # Convert to 64 bit
    b = b.astype('uint64')

    # Sum
    cs = int(b.sum())

    # Merge the first and second half
    cs = ((cs >> 32) ^ (cs >> 0)) & (2 ** 32 - 1)
    return cs


def checksum_file(path):
    """
    Computes the checksum of a file
    """
    ds = buzz.DataSource()
    with ds.open_araster(path).close as raster:
        cs = checksum(raster.get_data(band=-1))
    return cs
