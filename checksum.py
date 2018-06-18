"""
Computes the checksum of an array
"""
import numpy as np
import buzzard as buzz

def checksum(array):
    """
    Computes the checksum of an array
    """
    array.dtype.bytes
    # Reinterpret array as unsigned integers
    b = array.view({
        4: np.uint32,
        8: np.uint64,
        1: np.uint8,
        2: np.uint16
    }[array.dtype.num])

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
