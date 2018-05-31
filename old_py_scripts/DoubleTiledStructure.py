from collections import defaultdict
import multiprocessing as mp
import multiprocessing.pool

import numpy as np

class DoubleTiledStructure(object):

    def __init__(self, out_tiles, computation_tiles, computation_method, parallel_tasks=5):

        self._computation_method = computation_method

        self._computation_pool = mp.pool.ThreadPool(parallel_tasks)

        self._to_compute_dict = defaultdict(set)
        self._computed_dict = defaultdict(set)

        self._computed_data = {}

        self._out_tiles = out_tiles
        self._computation_tiles = computation_tiles

        for out_tile in self._out_tiles:
            for computation_tile in self._computation_tiles:
                if out_tile.share_area(computation_tile):
                    self._to_compute_dict[out_tile].add(computation_tile)



    def _remember(self, out_tile_hash, computation_tile):
        if computation_tile in self._to_compute_dict[out_tile_hash]:
            self._computed_dict[out_tile_hash].add(computation_tile)
            self._to_compute_dict[out_tile_hash].remove(computation_tile)



    def _update_all_sets(self, computation_tile):
        for out_tile in self._out_tiles:
            self._remember(out_tile, computation_tile)


    def compute_out_data(self, out_tile):

        assert out_tile in self._out_tiles

        self._computation_pool.map(self._compute_tile, self._to_compute_dict[out_tile])

        first_array = list(self._computed_data.values())[0]

        if len(first_array.shape) > 2:
            out = np.empty(tuple(out_tile.shape) + (first_array.shape[-1],), dtype=first_array.dtype)
        else:
            out = np.empty(tuple(out_tile.shape), dtype=first_array.dtype)

        del first_array

        for computation_tile in self._computed_dict[out_tile].copy():
            dat = self._computed_data[computation_tile]
            out[computation_tile.slice_in(out_tile, clip=True)] = dat[out_tile.slice_in(computation_tile, clip=True)]

            del dat
            self._computed_data[computation_tile][out_tile.slice_in(computation_tile, clip=True)] = -1
            if (self._computed_data[computation_tile] == -1).all():
                del self._computed_data[computation_tile]

        return out


    def _compute_tile(self, computation_tile):
        self._computed_data[computation_tile] = self._computation_method(computation_tile)
        self._update_all_sets(computation_tile)