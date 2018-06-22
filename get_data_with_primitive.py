"""Used to retrieve the context of a get_multi_data_queue function"""

class GetDataWithPrimitive(object):
    """Used to retrieve the context of a get_multi_data_queue function"""
    def __init__(self, obj, function):
        self._primitive = obj
        self._function = function

    def __call__(self, fp_iterable, queue_size=5):
        return self._function(fp_iterable, queue_size)

    @property
    def primitive(self):
        """
        Returns the primitive raster
        """
        return self._primitive
