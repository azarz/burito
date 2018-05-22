import queue

class QueryPart(object):

	def __init__(self, qsize=5):
		self._to_verb = []
		self._verbed = queue.Queue(qsize)
		self._staging = []


class FullQuery(object):

	def __init__(self):
		self._produce = QueryPart()
		self._cache_out = QueryPart()
		self._compute = QueryPart()
		self._collect = QueryPart()