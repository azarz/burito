from collections import defaultdict

import numpy as np

class DoubleTiledStructure(object):

    def __init__(self, cache_tiles, computation_tiles, computation_method):

        self._computation_method = computation_method

        self._to_compute_dict = defaultdict(set)
        self._computed_dict = defaultdict(set)

        self._computed_data = {}  

        self._cache_tiles = cache_tiles
        self._computation_tiles = computation_tiles

        for cache_tile in self._cache_tiles:
            for computation_tile in self._computation_tiles:
                if cache_tile.share_area(computation_tile):
                    self._to_compute_dict[cache_tile].add(computation_tile)



    def _remember(self, cache_tile_hash, computation_tile):
        if computation_tile in self._to_compute_dict[cache_tile_hash]:
            self._computed_dict[cache_tile_hash].add(computation_tile)
            self._to_compute_dict[cache_tile_hash].remove(computation_tile)



    def _update_all_sets(self, computation_tile):

        for cache_tile in self._cache_tiles:
            self._remember(cache_tile, computation_tile)


    def compute_cache_data(self, cache_tile):

        assert cache_tile in self._cache_tiles

        for computation_tile in self._to_compute_dict[cache_tile].copy():
            self._compute_tile(computation_tile)

        first_array = list(self._computed_data.values())[0]

        out = np.empty(tuple(cache_tile.shape) + (first_array.shape[-1],), dtype=first_array.dtype)

        del first_array

        for computation_tile in self._computed_dict[cache_tile]:
            dat = self._computed_data[computation_tile]
            out[computation_tile.slice_in(cache_tile, clip=True)] = dat[cache_tile.slice_in(computation_tile, clip=True)]

            del dat
            self._computed_data[computation_tile][cache_tile.slice_in(computation_tile, clip=True)] = -1
            print(len(np.unique(self._computed_data[computation_tile])))
            if (self._computed_data[computation_tile] == -1).all():
                print("deleting (DTS)  ", len(self._computed_data.keys()))
                del self._computed_data[computation_tile]

        return out


    def _compute_tile(self, computation_tile):
        self._computed_data[computation_tile] = self._computation_method(computation_tile)
        self._update_all_sets(computation_tile)