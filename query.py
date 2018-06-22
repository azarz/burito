"""
Query class used by the raster to manage concurrent requests
"""
import weakref

class Query(object):
    """
    Query used by the Raster class
    """
    def __init__(self, queue):
        self.to_produce = []
        self.produced = weakref.ref(queue)

        self.to_collect = {}
        self.collected = {}
