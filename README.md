# BuRITo, Buzzard's Raster Inception Tool (name subject to change)

## Vocabulary:
For a given request to a raster:

	Requested fps: to_produce (list)

	Returned queue: produced (queue)

	Cache_tiles to read: to_read

	Cache_tiles to write: to_write

	Computation tile to compute: to_compute

	Primitive tiles to collect: to_collect (dict of lists)

	Primitive tiles returned: collected (dict of queues)
	

## Specifications:

- Finish jobs as fast as possible
	+ Do not compute expensive pixels twice
		- between 2 executions: disk caching
		- inside the execution: caching: disk caching + OS RAM caching
	+ Optimize hardware usage
		- Split the computations: tiling
		- Order the tasks
			+ between tasks
			+ inside raster
		- Use pools
	+ Minimize overhead

- Limited RAM usage
	+ see split computations above
	+ use disk caching
