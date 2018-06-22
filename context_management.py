"""Used to retrieve the context of a get_multi_data_queue function"""

class GetDataWithPrimitive(object):
    """Used to retrieve the context of a get_multi_data_queue function"""
    def __init__(self, obj, routine):
        self._primitive = obj
        self._routine = routine

    def __call__(self):
        self._routine()

    @property
    def primitive(self):
        """
        Returns the primitive raster
        """
        return self._primitive