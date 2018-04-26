import hashlib
import itertools
import functools
import operator

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def uids_of_paths(paths):
    """Deterministically create uuids for all combinations of input files.
    Avoid small files.

    Exemple
    -------
    >>> paths = {'ortho': 'path/to/ortho.tif', 'dsm': 'path/to/dsm.tif'}'
    >>> uids = uids_of_paths(paths)
    >>> uids
    {frozenset({'ortho'}): 'aa5a7fa915d3c7661544761623779835', frozenset({'dsm'}): '78deb3ad6718b2c4eed579a371340c43', frozenset({'dsm', 'ortho'}): 'd284cc0472cb75a2fb910fb552439476'}
    """
    md5s = {
        k: int(md5(path), base=16)
        for k, path in paths.items()
    }
    combos = [
        frozenset(keys)
        for size in range(1, len(paths) + 1)
        for keys in itertools.combinations(paths.keys(), size)
    ]
    res = {}
    for keys in combos:
        numbers = [
            md5s[key]
            for key in keys
        ]
        nbr = functools.reduce(operator.xor, numbers)
        nbr = hex(nbr)[2:]
        res[keys] = nbr
    return res

if __name__ == '__main__':
    # Test
    paths = {
        'ortho': '/media/ngoguey/4b16cd43-a81f-4aef-b59e-6de3120df13b/more_quarry/Lucara_diamond-Karowe-20170717/SRC/ORTHO.tif',
        'dsm': '/media/ngoguey/4b16cd43-a81f-4aef-b59e-6de3120df13b/more_quarry/Lucara_diamond-Karowe-20170717/SRC/DSM.tif',
    }
    uids = uids_of_paths(paths)
    print(uids)

