"""
FullQuery class used by the raster to manage concurrent requests
"""

import queue

class Query(object):
    """
    Query used by the Raster class
    """
    def __init__(self, qsize=5):
        self.to_produce = []
        self.produced = queue.Queue(qsize)

        self.to_collect = {}
        self.collected = []
