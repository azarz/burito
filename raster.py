"""
Multi-threaded, back pressure management, caching
"""

import threading
import datetime
import queue
import uuid
import multiprocessing as mp

import numpy as np
from buzzard import _tools
from buzzard._tools import Query, GetDataWithPrimitive
from buzzard._footprint import Footprint

# import get_uname, qeinfo, qrinfo

from buzzard._backend_raster import BackendRaster
from buzzard._backend_cached_raster import BackendCachedRaster


class Raster(object):
    def __init__(self,
                 ds,
                 footprint=None,
                 dtype='float32',
                 nbands=1,
                 nodata=None,
                 sr=None,
                 computation_function=None,
                 cached=False,
                 overwrite=False,
                 cache_dir=None,
                 cache_tiles=None,
                 io_pool=None,
                 computation_pool=None,
                 primitives=None,
                 to_collect_of_to_compute=None,
                 max_computation_size=None,
                 computation_tiles=None,
                 merge_pool=None,
                 merge_function=None,
                 debug_callbacks=None):
        """
        creates a raster from arguments
        """
        if footprint is None:
            raise ValueError("footprint must be provided")

        if computation_function is None:
            raise ValueError("computation function must be provided")


        if io_pool is None:
            io_pool = mp.pool.ThreadPool()
        elif isinstance(io_pool, int):
            io_pool = mp.pool.ThreadPool(io_pool)
        else:
            pass

        if computation_pool is None:
            computation_pool = mp.pool.ThreadPool()
        elif isinstance(computation_pool, int):
            computation_pool = mp.pool.ThreadPool(computation_pool)
        else:
            pass

        if merge_pool is None:
            merge_pool = mp.pool.ThreadPool()
        elif isinstance(merge_pool, int):
            merge_pool = mp.pool.ThreadPool(merge_pool)
        else:
            pass


        if primitives is None:
            primitives = {}

        if not isinstance(primitives, dict):
            raise ValueError("primitives must be dict or None")

        if primitives.keys() and to_collect_of_to_compute is None:
            raise ValueError("must provide to_collect_of_to_compute when having primitives")

        
        if max_computation_size is not None:
            max_computation_size = np.asarray(max_computation_size).astype(int)
            max_computation_size = np.atleast_1d(max_computation_size)
            if max_computation_size.shape == (2,):
                pass
            elif max_computation_size.shape == (1,):
                max_computation_size = np.r_[max_computation_size, max_computation_size]
            else:
                raise ValueError("max_computation_size invalid")


        def default_merge_data(out_fp, in_fps, in_arrays):
            """
            Default merge function: burning
            """
            out_data = np.full(
                tuple(out_fp.shape) + (self._num_bands,),
                self.nodata or 0,
                dtype=self.dtype
            )
            for to_burn_fp, to_burn_data in zip(in_fps, in_arrays):
                out_data[to_burn_fp.slice_in(out_fp, clip=True)] = to_burn_data[out_fp.slice_in(to_burn_fp, clip=True)]
            return out_data


        if merge_function is None:
            self._merge_data = default_merge_data
        else:
            self._merge_data = merge_function


        if cached:
            if cache_dir is None:
                raise ValueError("cache_dir must be provided when cached")

            if cache_tiles is None:
                raise ValueError("cache tiles must be provided")
            if not isinstance(cache_tiles, np.ndarray):
                raise ValueError("cache tiles must be in np array")
            cache_fps = list(cache_tiles.flat)
            if not isinstance(cache_fps[0], Footprint):
                raise ValueError("cache tiles must be footprints")

            backend_raster = BackendCachedRaster(ds,
                                                 footprint,
                                                 dtype,
                                                 nbands,
                                                 nodata,
                                                 sr,
                                                 computation_function,
                                                 overwrite,
                                                 cache_dir,
                                                 np.asarray(cache_tiles),
                                                 io_pool,
                                                 computation_pool,
                                                 primitives,
                                                 to_collect_of_to_compute,
                                                 computation_tiles,
                                                 merge_pool,
                                                 merge_function,
                                                 debug_callbacks
                                                )
        else:
            backend_raster = BackendRaster(ds,
                                           footprint,
                                           dtype,
                                           nbands,
                                           nodata,
                                           sr,
                                           computation_function,
                                           io_pool,
                                           computation_pool,
                                           primitives,
                                           to_collect_of_to_compute,
                                           max_computation_size,
                                           merge_pool,
                                           merge_function,
                                           debug_callbacks
                                          )

        self._backend = backend_raster

        self._scheduler_thread = threading.Thread(target=self._backend._scheduler, daemon=True)
        self._scheduler_thread.start()

    def __del__(self):
        self._backend._stop_scheduler = True



    @property
    def fp(self):
        """
        returns the raster footprint
        """
        return self._backend.fp

    @property
    def nbands(self):
        """
        returns the raster's number of bands'
        """
        return self._backend.nbands

    @property
    def nodata(self):
        """
        returns the raster nodata
        """
        return self._backend.nodata

    @property
    def wkt_origin(self):
        """
        returns the raster wkt origin
        """
        return self._backend.wkt_origin

    @property
    def dtype(self):
        """
        returns the raster dtype
        """
        return self._backend.dtype

    @property
    def pxsizex(self):
        """
        returns the raster 1D pixel size
        """
        return self._backend.pxsizex

    @property
    def primitives(self):
        """
        Returns dictionnary of raster primitives
        """
        return self._backend.primitives


    def __len__(self):
        return len(self._backend)



    @property
    def get_multi_data_queue(self):
        """
        returns a queue from a fp_iterable and can manage the context of the method
        """

        def _get_multi_data_queue(fp_iterable, band=-1, queue_size=5):
            """
            returns a queue from a fp_iterable
            """

            # Normalize and check band parameter
            bands, is_flat = _tools.normalize_band_parameter(band, len(self), None)

            fp_iterable = list(fp_iterable)
            for fp in fp_iterable:
                assert fp.same_grid(self.fp)
            q = queue.Queue(queue_size)
            query = Query(q, bands, is_flat)
            to_produce = [(fp, "sleeping", str(uuid.uuid4())) for fp in fp_iterable]
            # to_produce = [(fp, "sleeping", get_uname()) for fp in fp_iterable]

            query.to_produce += to_produce

            # self._backend._queries.append(query)
            self._backend._new_queries.append(query)

            return q

        return GetDataWithPrimitive(self, _get_multi_data_queue)


    def get_multi_data(self, fp_iterable, band=-1, queue_size=5):
        """
        returns a  generator from a fp_iterable
        """
        fp_iterable = list(fp_iterable)
        out_queue = self.get_multi_data_queue(fp_iterable, band, queue_size)

        def out_generator():
            """
            output generator from the queue. next() waits the .get() to end
            """
            for fp in fp_iterable:
                assert fp.same_grid(self.fp)
                result = out_queue.get()
                assert np.array_equal(fp.shape, result.shape[0:2])
                yield result

        return out_generator()


    def get_data(self, fp, band=-1):
        """
        returns a np array
        """
        return next(self.get_multi_data([fp], band))


