class DoubleTiledStructure(object):

	def __init__(self, cache_tiles, computation_tiles):

		self._to_compute_dict = {}
		self._computed_dict = {}

		self._cache_tiles = cache_tiles
		self._computation_tiles = computation_tiles

		for cache_tile in self._cache_tiles:
			intersect_list = []
			for computation_tile in self._computation_tiles:
				if cache_tile.share_area(computation_tile):
					intersect_list.append(computation_tile)

			self._to_compute_dict[cache_tile] = set(intersect_list)
			self._computed_dict[cache_tile] = set([])



	def _remember(self, cache_tile, computation_tile):

		if computation_tile in self._to_compute_dict[cache_tile]:
			self._computed_dict[cache_tile].add(computation_tile)
			self._to_compute_dict[cache_tile].discard(computation_tile)



	def _update_all_sets(self, computation_tile):

		for cache_tile in self._cache_tiles:
			self._remember(cache_tile, computation_tile)