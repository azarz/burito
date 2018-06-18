from Raster import raster_factory
import buzzard as buzz
import queue
import numpy as np
import multiprocessing as mp
import multiprocessing.pool

def main():
    g_cpu_pool = mp.pool.ThreadPool()
    g_io_pool = mp.pool.ThreadPool()
    full_fp = buzz.Footprint(tl=(0, 0), size=(12691, 12691), rsize=(126910, 126910))
    print(full_fp, full_fp.scale, full_fp.rarea*4/(1024**3))
    print()
    print(full_fp._significant_min)

    def compute_data(fp, *args):
        print(args) 
        return np.random.rand(fp.shape[0], fp.shape[1])


    raster = raster_factory(footprint=full_fp,
                         dtype='float32',
                         nbands=1,
                         nodata=-99999,
                         srs=None, # dsm.wkt_origin
                         computation_function=compute_data,
                         cached=False,
                         cache_dir=None,
                         cache_fps=None,
                         io_pool=g_io_pool,
                         computation_pool=g_cpu_pool,
                         primitives={},
                         to_collect_of_to_compute=None,
                         to_compute_fps=None,
                         merge_pool=None,
                         merge_function=None
                        )

    raster.get_data(full_fp)


if __name__ == "__main__":
    main()
