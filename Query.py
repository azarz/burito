import queue

class QueryPart(object):

	def __init__(self, qsize=5):
		self.to_verb = []
		self.verbed = queue.Queue(qsize)
		self.staging = []


class FullQuery(object):

	def __init__(self, queue_size=5):
		self.produce = QueryPart(qsize=queue_size)
		self.uncache = QueryPart(qsize=queue_size)
		self.compute = QueryPart(qsize=queue_size)
		self.collect = QueryPart(qsize=queue_size)