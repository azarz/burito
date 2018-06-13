"""
threadpool task counter using singleton design pattern
"""

from collections import defaultdict

class Singleton(type):
    """
    implementation of the singleton design pattern
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonCounter(defaultdict, metaclass=Singleton):
    """
    the threadpool task counter is a defaultdict(int) applying the singleton pattern
    """
    def __init__(self, *args):
        if args:
            super().__init__(*args)
        else:
            super().__init__(int)
