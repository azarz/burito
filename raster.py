"""
Multi-threaded, back pressure management, caching
"""

from pathlib import Path
import multiprocessing as mp
import multiprocessing.pool
import os
import threading
import time
import datetime
from collections import defaultdict
import glob
import sys
import queue
import itertools
import collections
import uuid

import numpy as np
import networkx as nx
import buzzard as buzz
from buzzard import _tools
import rtree.index
from osgeo import gdal, osr
from buzzard._tools import conv
import names

from burito.uids_of_paths import md5
from burito.query import Query
from burito.backend_raster import BackendRaster
from burito.backend_cached_raster import BackendCachedRaster
from burito.tools import SingletonCounter, GetDataWithPrimitive, get_uname, qeinfo, qrinfo


class Raster(object):
    def __init__(self,
                 footprint=None,
                 dtype='float32',
                 nbands=1,
                 nodata=None,
                 srs=None,
                 computation_function=None,
                 cached=False,
                 overwrite=False,
                 cache_dir=None,
                 cache_fps=None,
                 io_pool=None,
                 computation_pool=None,
                 primitives=None,
                 to_collect_of_to_compute=None,
                 max_computation_size=None,
                 computation_fps=None,
                 merge_pool=None,
                 merge_function=None,
                 debug_callbacks=None):
        """
        creates a raster from arguments
        """
        if footprint is None:
            raise ValueError("footprint must be provided")
        if cached:
            if cache_dir is None:
                raise ValueError("cache_dir must be provided when cached")
            backend_raster = BackendCachedRaster(footprint,
                                                 dtype,
                                                 nbands,
                                                 nodata,
                                                 srs,
                                                 computation_function,
                                                 overwrite,
                                                 cache_dir,
                                                 np.asarray(cache_fps),
                                                 io_pool,
                                                 computation_pool,
                                                 primitives,
                                                 to_collect_of_to_compute,
                                                 computation_fps,
                                                 merge_pool,
                                                 merge_function,
                                                 debug_callbacks
                                                )
        else:
            backend_raster = BackendRaster(footprint,
                                           dtype,
                                           nbands,
                                           nodata,
                                           srs,
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
            # to_produce = [(fp, "sleeping", str(uuid.uuid4())) for fp in fp_iterable]
            to_produce = [(fp, "sleeping", get_uname()) for fp in fp_iterable]

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
