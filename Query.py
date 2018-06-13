"""
FullQuery class used by the raster to manage concurrent requests
"""

import queue

class QueryPart(object):
    """
    Component of a query
    """
    def __init__(self, qsize=5):
        self.to_verb = []
        self.verbed = queue.Queue(qsize)


class FullQuery(object):
    """
    Query used by the Raster class
    """
    def __init__(self, queue_size=5):
        self.produce = QueryPart(qsize=queue_size)
        self.read = QueryPart(qsize=queue_size)
        self.write = QueryPart(qsize=queue_size)
        self.compute = QueryPart(qsize=queue_size)
        self.collect = QueryPart(qsize=queue_size)
