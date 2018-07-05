"""
Query class used by the raster to manage concurrent requests
"""
import weakref

class Query(object):
    """
    Query used by the Raster class
    """
    def __init__(self, queue, bands, is_flat):
        self.to_produce = []
        self.produced = weakref.ref(queue)

        self.to_collect = {}
        self.collected = {}

        self.to_compute = []

        self.to_discard = {}

        self.bands = bands

        self.is_flat = is_flat
