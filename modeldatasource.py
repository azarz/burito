# print('Import phase datasource')

import collections
import contextlib
import os
import functools
import multiprocessing
import multiprocessing.pool
import threading
from pprint import pprint
import re

import pandas
import buzzard as buzz
import numpy as np
import pandas as pd
import shapely.geometry as sg
from attrdict import AttrDict
import scipy.ndimage as ndi

from constants import *
from async_tools import iter_parallel

class ModelDataSource(object):

    def __init__(self, minfo, shapes, acquisitions_prexix):
        print("Instanciating a <{}> with out.reso:{} and sources:{}".format(
            self.__class__.__name__, minfo.resos.out, minfo.inputs,
        ))
        self.minfo = minfo
        self.resos = minfo.resos
        self.shapes = shapes
        self.acquisitions_prexix = acquisitions_prexix
        maxreso = max(self.resos.values())
        self.shifts = [
            (dx * maxreso, dy * maxreso)
            for dx in [0, 1, -1, 2, -2, 3, -3, 4, -4]
            for dy in [0, 1, -1, 2, -2, 3, -3, 4, -4]
        ]

    @staticmethod
    def _shapes_of_km(km, minfo):
        shapes = AttrDict()
        for key in minfo.resos.keys():
            lyr = km.get_layer(key)
            shapes[key] = lyr.output_shape[1:3]
            assert shapes[key][0] == shapes[key][1]
        return shapes

    @classmethod
    def from_km(cls, km, minfo, acquisitions_prexix):
        return cls(minfo, cls._shapes_of_km(km, minfo), acquisitions_prexix)

    # Informations extraction ******************************************************************* **
    @property
    @functools.lru_cache()
    def out_tiles(self):
        def _tiles_infos_of_name2(n):
            ds = buzz.DataSource()
            ortho = ds.open_araster(self.ortho_path(self.resos.out, n))
            data_bounds = self.get_bounds(n).simplify(self.resos.out * 1 * 2 ** 0.5)
            tiles = ortho.fp.tile(self.shapes.out, 0, 0)
            shape = tiles.shape
            tiles = [
                tile
                for tile in tiles.flatten()
                for tile_coverate in [(tile.poly & data_bounds).area / tile.area]
                if tile_coverate > 0.01 # TODO sieve
            ]

            return collections.OrderedDict([
                ('name', n),
                ('test', n in TEST_NAMES),
                ('extra', n not in NAMES),
                ('count', len(tiles)),
                ('count_nodata', np.prod(shape) - len(tiles)),
                ('shape', shape),
                ('fp', ortho.fp),
                ('tiles', tiles),
            ])

        def _tiles_infos_of_name(n):
            try:
                return _tiles_infos_of_name2(n)
            except Exception as e:
                print(e)
                raise

        print("Building out_tiles".format())
        pool = multiprocessing.pool.ThreadPool(multiprocessing.cpu_count())
        # Small leak, but computed once
        requests = [
            pool.apply_async(_tiles_infos_of_name, [n])
            for n in NAMES + [n for n in EXTRA_NAMES if os.path.isfile(self.dsm_path(0.64, n))]
        ]
        for i, r in enumerate(requests, 1):
            r.wait()
            if not r.successful():
                raise ValueError('Could not compute machin')

        df = pd.DataFrame(
            [r.get() for r in requests]
        ).reset_index(drop=True)
        pool.terminate()
        pool.join()
        return df

    @staticmethod
    def acquisition_labels_stats(dat):
        reso, n, path, test, tiles = dat

        with buzz.Env(significant=9):
            ds = buzz.DataSource()
            ds.open_raster('labs', path)

            numerators_acq = np.zeros(LABEL_COUNT, int)
            denominators_acq = np.zeros(LABEL_COUNT, int)
            for fp in tiles:
                counts_tile = np.bincount(ds.labs.get_data(fp=fp).flatten(), None, LABEL_COUNT)
                numerators_acq += counts_tile
                denominators_acq += (counts_tile > 0) * fp.rarea

            numerators_acq = numerators_acq * reso ** 2 / 1e4
            denominators_acq = denominators_acq * reso ** 2 / 1e4

            numerators_acq = [
                (('nume', catname), nbr)
                for catname, nbr in zip(CATEGORIES, numerators_acq)
            ]
            denominators_acq = [
                (('deno', catname), nbr)
                for catname, nbr in zip(CATEGORIES, denominators_acq)
            ]

            return collections.OrderedDict([
                ('name', n),
                ('test', test),
            ] + numerators_acq + denominators_acq)

    @property
    @functools.lru_cache()
    def acquisitions_labels_stats(self):
        rows = []
        def _cb(row):
            rows.append(row)

        works = [
            (self.resos.out, n, self.labels_path(n), n in TEST_NAMES,
             self.out_tiles.set_index('name').loc[n, 'tiles'])
            for n in NAMES
        ]
        # iter_parallel(self.acquisition_labels_stats, works, _cb)
        print('DISABLED ITER_PARALLEL DISABLED ITER_PARALLEL DISABLED ITER_PARALLEL DISABLED ITER_PARALLEL DISABLED ITER_PARALLEL')
        for dat in works:
            _cb(self.acquisition_labels_stats(dat))

        df = pd.DataFrame(rows).reset_index(drop=True)
        return df

    @property
    def mfb(self):
        if not hasattr(self, '_mfb') or self._mfb is None:
            print('Computing mfb...')
            if self.resos.out == 0.08:
                df = pandas.Series({
                    'nolab': 0.4145,
                    'vege': 0.5889,
                    'bank': 0.7786,
                    'water': 0.7844,
                    'stocks': 0.7857,
                    'roads': 0.8759,
                    'aggregate': 1.0000,
                    'faces': 1.2140,
                    'building': 1.5282,
                    'berms': 2.0917,
                    'tapis': 3.3218,
                    'vehicles': 5.6403,
                    'blocks': 107.3331,
                })
            elif self.resos.out == 0.16:
                df = pandas.Series({
                    'nolab': 0.1017,
                    'vege': 0.3531,
                    'roads': 0.4735,
                    'bank': 0.5898,
                    'stocks': 0.7365,
                    'faces': 0.8128,
                    'water': 1.0000,
                    'aggregate': 1.1078,
                    'berms': 1.6204,
                    'building': 1.8406,
                    'tapis': 3.7637,
                    'vehicles': 10.7039,
                    'blocks': 188.6138,
                })
            elif self.resos.out == 0.64:
                df = pandas.Series({
                    'blocks': 0.0000,
                    'nolab': 0.0546,
                    'vege': 0.2924,
                    'roads': 0.3640,
                    'bank': 0.5556,
                    'faces': 0.7496,
                    'stocks': 0.8542,
                    'water': 1.2059,
                    'berms': 1.3711,
                    'aggregate': 1.3789,
                    'building': 2.1227,
                    'tapis': 4.4432,
                    'vehicles': 11.5983,
                })
            else:
                # Areas per lab per acquisitions
                df = self.acquisitions_labels_stats
                df = df.query('test == False')
                df = df.set_index('name', drop=True)
                df = df.drop('test', axis=1)
                df.columns = pd.MultiIndex.from_tuples(df.columns)

                # Areas per lab
                df = df.sum(axis=0)

                # Frequencies per lab
                df = df.nume / df.deno

                # Mfb per lab
                df = df.median() / df

                # Fix absent classes
                df[~np.isfinite(np.asarray(df))] = 0
            self._mfb = df
            print(df.sort_values())

        return self._mfb

    # Image getters ***************************************************************************** **
    def get_bounds(self, n):
        ds = buzz.DataSource()
        with ds.open_avector(self.data_bounds_path(n)).close as v:
            polys = [p.buffer(0) for p in v.iter_data(None)]
        return sg.MultiPolygon(polys)

    # def get_viewables(self, n, outfp):
    #     fps = {}
    #     for name in self.resos.keys():
    #         reso = self.resos[name]
    #         shape = self.shapes[name]

    #         resofactor = reso / self.resos.out
    #         assert (self.shapes.out[0] * resofactor).is_integer()
    #         fp = outfp.intersection(outfp, scale=reso)
    #         dilate_count = (shape[0] * resofactor - self.shapes.out[0]) / resofactor / 2
    #         assert dilate_count.is_integer()
    #         fp = fp.dilate(dilate_count)
    #         assert tuple(fp.shape) == shape
    #         fps[name] = fp

    #     maxfootprint = max(fps.values(), key=lambda fp: fp.area)

    #     for name, fpsmall in fps.items():
    #         half_width_diff = (maxfootprint.sizex - fpsmall.sizex) / 2
    #         dilate_count = np.around(half_width_diff / fpsmall.pxsizex, 4)
    #         if not dilate_count.is_integer():
    #             print('Dilate count is not an integer!! please fix this one day', dilate_count)
    #             dilate_count = dilate_count // 1
    #         fpbig = fpsmall.dilate(dilate_count)
    #         # assert (np.abs(fpbig.size - maxfootprint.size) < 0.1).all()
    #         fps[name] = AttrDict(fpbig=fpbig, fpsmall=fpsmall)

    #     tensors = self.get_in_tensors(n, outfp)
    #     if n in NAMES:
    #         tensors['out'] = self.get_out_tensors(n, outfp)
    #     else:
    #         tensors['out'] = np.zeros(np.r_[self.shapes['out'], LABEL_COUNT], 'float32')

    #     for name, arr in tensors.items():
    #         arr = arr[0]
    #         if 'rgb' in name:
    #             arr = ((arr + 1) / 2 * 255).astype('uint8')
    #         elif 'out' in name:
    #             arr = arr.argmax(-1).astype('uint8')
    #         elif 'slope' in name:
    #             arr = (arr + 1) / 2 * 90
    #         else:
    #             assert False

    #         shape = list(arr.shape)
    #         shape[:2] = fps[name].fpbig.shape
    #         arr2 = np.zeros(shape, arr.dtype)
    #         arr2[fps[name].fpsmall.slice_in(fps[name].fpbig)] = arr

    #         tensors[name] = arr2

    #     tensors['truth'] = tensors['out']
    #     del tensors['out']
    #     tensors['slopes down'] = tensors['slopes'][..., 0]
    #     tensors['slopes up'] = tensors['slopes'][..., 1]
    #     del tensors['slopes']


    #     def _get_rank(k):
    #         if 'rgb' in k:
    #             return 0, k
    #         elif 'down' in k:
    #             return 1, k
    #         elif 'up' in k:
    #             return 2, k
    #         else:
    #             return 3, k

    #     tensors = collections.OrderedDict([
    #         (k, tensors[k])
    #         for k in sorted(tensors.keys(), key=_get_rank)
    #     ])

    #     # pprint({
    #     #     name: t.shape + (t.dtype,)
    #     #     for name, t in tensors.items()
    #     # })
    #     return tensors

    def get_rgb(self, reso, n, fp, normalize=True, intersect=False):
        ds = buzz.DataSource()
        with ds.open_araster(self.ortho_path(reso, n)).close as r:
            if intersect:
                fp = r.fp.dilate(fp.rlength // 2) & fp
            arr = r.get_data(band=(1, 2, 3), fp=fp, nodata=0).astype('uint8')
        if normalize:
            arr = arr.astype('float32') * 2 / 255 - 1
        return arr

    def get_cl(self, reso, n, fp, dist, mod, vshift, intersect=False):
        ds = buzz.DataSource()
        with ds.open_araster(self.dsm_path(reso, n)).close as r:
            if intersect:
                fp = r.fp.dilate(fp.rlength // 2) & fp
            dsm = r.get_data(fp=fp)
            nodata_mask = dsm == r.nodata

        dsm = dsm + vshift * mod * dist

        # Smooth dsm
        kernel = ndi.gaussian_filter(np.pad([[1]], 3, 'constant').astype(float), 1.5)
        kernel[kernel < 0.0005] = 0
        dsm_smooth = ndi.convolve(dsm, kernel, mode='constant', cval=0)

        # Undo smoothing near nodata
        nodata_mask_smooth = ndi.convolve(nodata_mask.astype('float32'), kernel, mode='constant', cval=1) != 0
        near_nodata_mask = nodata_mask ^ nodata_mask_smooth
        dsm_smooth[near_nodata_mask] = dsm[near_nodata_mask]

        arr = dsm_smooth // dist % mod

        arr = np.dstack([
            arr == i
            for i in range(mod)
        ])
        arr[nodata_mask] = 0

        return arr

    def get_slopes(self, reso, n, fp, normalize=True, intersect=False):
        ds = buzz.DataSource()
        with ds.open_araster(self.dsm_path(reso, n)).close as r:
            if intersect:
                fp = r.fp.dilate(fp.rlength // 2) & fp
            arr = r.get_data(fp=fp.dilate(1))
            nodata_mask = arr == r.nodata
        nodata_mask = ndi.binary_dilation(nodata_mask)
        kernel = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        arru = ndi.maximum_filter(arr, None, kernel) - arr
        arru = np.arctan(arru / fp.pxsizex)
        arru = arru / np.pi * 180.
        arru[nodata_mask] = 0
        arru = arru[1:-1, 1:-1]

        arrd = arr - ndi.minimum_filter(arr, None, kernel)
        arrd = np.arctan(arrd / fp.pxsizex)
        arrd = arrd / np.pi * 180.
        arrd[nodata_mask] = 0
        arrd = arrd[1:-1, 1:-1]

        arr = np.dstack([arrd, arru])
        if normalize:
            arr = arr / 45 - 1
        return arr

    def get_labels(self, n, fp, intersect=False):
        if n in NAMES:
            ds = buzz.DataSource()
            with ds.open_araster(self.labels_path(n)).close as r:
                if intersect:
                    fp = r.fp.dilate(fp.rlength // 2) & fp
                labs = r.get_data(fp=fp, nodata=0).astype('uint8')
        else:
            labs = np.zeros(fp.shape, 'uint8')
        return labs

    def get_edt(self, n, fp, intersect=False):
        if n in NAMES:
            ds = buzz.DataSource()
            with ds.open_araster(self.edt_path(n)).close as r:
                if intersect:
                    fp = r.fp.dilate(fp.rlength // 2) & fp
                edt = r.get_data(fp=fp, nodata=0).astype('float32')
                assert (edt >= 0).all()
        else:
            edt = np.zeros(fp.shape, 'float32')
        return edt

    def get_in_tensors(self, n, outfp, mirror=False, rotate=0, shift=0., vshift=0., dilate=True):
        """See ModelDataSource.get_tensors"""
        rotate = int(rotate)
        assert 0 <= rotate <= 3
        shift = float(shift)
        assert 0. <= shift < 1.

        shift = self.shifts[int(shift * len(self.shifts))]
        outfp = outfp.move(outfp.tl + (shift))

        ret = {}
        for name in self.resos.keys():
            if name == 'out':
                continue
            reso = self.resos[name]
            shape = self.shapes[name]

            resofactor = reso / self.resos.out
            assert (self.shapes.out[0] * resofactor).is_integer()

            fp = outfp.intersection(outfp, scale=reso)
            assert abs(fp.pxsizex - reso) < 0.00001

            if dilate:
                dilate_count = (shape[0] * resofactor - self.shapes.out[0]) / resofactor / 2
                if not dilate_count.is_integer():
                    print("Hey ! attempting to dilate {} by {}".format(name, dilate_count))
                    assert dilate_count.is_integer()

                fp = fp.dilate(dilate_count)
                assert tuple(fp.shape) == shape

            if 'rgb' in name:
                arr = self.get_rgb(reso, n, fp)
            elif 'cl' in name:
                dist = float(re.match('.*_dist(\d*\.\d+|\d+)_.*', name).groups()[0])
                mod = int(re.match('.*_mod(\d+)_.*', name).groups()[0])
                arr = self.get_cl(reso, n, fp, dist, mod, vshift)
            elif 'slopes' in name:
                arr = self.get_slopes(reso, n, fp)
            else:
                assert False

            if mirror:
                arr = np.flipud(arr)
            arr = np.rot90(arr, rotate)
            arr = arr[None, ...]

            ret[name] = arr
        return ret

    def get_out_tensors(self, n, fp, mirror=False, rotate=0, shift=0., vshift=0.):
        """See ModelDataSource.get_tensors"""
        rotate = int(rotate)
        assert 0 <= rotate <= 3
        shift = float(shift)
        assert 0. <= shift < 1.

        shift = self.shifts[int(shift * len(self.shifts))]
        fp = fp.move(fp.tl + (shift))

        labs = self.get_labels(n, fp)
        labs = np.dstack([
            labs == i
            for i in range(LABEL_COUNT)
        ])
        if mirror:
            labs = np.flipud(labs)
        labs = np.rot90(labs, rotate)
        return labs[np.newaxis, ...]

    def get_w_tensors(self, labs1h, n, fp, mirror=False, rotate=0, shift=0., vshift=0.):
        """See ModelDataSource.get_tensors"""
        rotate = int(rotate)
        assert 0 <= rotate <= 3
        shift = float(shift)
        assert 0. <= shift < 1.

        shift = self.shifts[int(shift * len(self.shifts))]
        fp = fp.move(fp.tl + (shift))

        w = np.ones(fp.shape, 'float32')

        for options in self.minfo.weights.split(';'):
            if 'edt' in options:
                edt, low, up = options.split('_')
                assert edt == 'edt'
                low_m, low_w = low.split('@')
                up_m, up_w = up.split('@')
                low_m = float(low_m)
                low_w = float(low_w)
                up_m = float(up_m)
                up_w = float(up_w)

                edtw = self.get_edt(n, fp)
                edtw = (edtw.clip(low_m, up_m) - low_m) / up_m
                edtw = edtw * (up_w - low_w) + low_w
                assert (edtw >= low_w).all()
                assert (edtw <= up_w).all()

                if mirror:
                    edtw = np.flipud(edtw)
                edtw = np.rot90(edtw, rotate)
                w *= edtw
            elif options == 'mfb':
                classw = np.asarray([self.mfb[cname] for cname in CATEGORIES])[None, None, :]
                assert classw.ndim == labs1h.ndim
                assert labs1h.shape[-1] == LABEL_COUNT
                w *= (labs1h * classw).sum(axis=-1)
            elif options[:3] == 'no@':
                labi = INDEX_OF_LABNAME[options[3:]]
                w *= (labs1h[..., labi] + 1.) % 2.
            else:
                assert False

        return w[np.newaxis, ...]

    def get_tensors(self, n, fp, mirror=False, rotate=0, shift=0., vshift=0., dilate=True, monitor=None):
        """
        Parameters
        ----------
        n: str
            Acquisition name
        fp: Footprint
            Tile's Footprint without shifting. (Usually comes from `ModelDataSource.out_tiles`)
        mirror: bool
            Should mirror or not
        rotate: int
            Number of time to rotate 90 degree ccw
        shift: float
            Indicator to which shift to pick
            >>> (shiftx, shifty) = self.shifts[int(shift * len(self.shifts))]
        dilate: bool
            Should increase context for input tensors of not. (False may be used for debug)
        monitor: None or ThreadsMonitor
            Activity monitor (debug / log)

        Shift augmentation
        ------------------
        When using shift augmentation, output tensors are shifted from the requested Footprint.
        The requested tiles are shifted by `kx * Ex` and `ky * Ey` meters in the x/y dimensions.
            where `kx` is a random integer
            where `ky` is a random integer
            where `-(Ix / Ex / 2) < kx <= Ix / Ex / 2`
            where `-(Iy / Ey / 2) < ky <= Iy / Ey / 2`
            where `Ex` is the maximum pixel width  in meter in all input/output layers
            where `Ey` is the maximum pixel height in meter in all input/output layers
            where `Ix` is the maximum pixel width  in meter in all layers
            where `Iy` is the maximum pixel height in meter in all layers

        """
        res = ()

        with contextlib.ExitStack() as es:
            if monitor:
                es.enter_context(monitor('read/build input tensors'))
            res += (self.get_in_tensors(n, fp, mirror, rotate, shift, vshift, dilate),)

        with contextlib.ExitStack() as es:
            if monitor:
                es.enter_context(monitor('read/build output tensor'))
            res += (self.get_out_tensors(n, fp, mirror, rotate, shift),)

        if self.minfo.weights is not None:
            with contextlib.ExitStack() as es:
                if monitor:
                    es.enter_context(monitor('read/build w tensor'))
                res += (self.get_w_tensors(res[1][0], n, fp, mirror, rotate, shift),)

        return res

    def show_tensors(self, n, fp, mirror=False, rotate=0, shift=0., vshift=0., dilate=True):
        import main_display

        tin, tout, *tw = self.get_tensors(n, fp, mirror=mirror, rotate=rotate, shift=shift, vshift=vshift, dilate=True)

        tensors = collections.OrderedDict()
        resos = collections.OrderedDict()

        tensors['out'] = tout[0].argmax(-1)
        resos['out'] = self.minfo.resos.out

        if tw:
            tw = tw[0]
            tensors['w'] = tw[0]
            resos['w'] = self.minfo.resos.out

        for k, v in tin.items():
            reso = self.minfo.resos[k]

            if 'rgb' in k:
                v = v[0]
                v = ((v + 1) / 2 * 255).astype('uint8')
                tensors[k] = v
                resos[k] = reso
            elif 'cl' in k:
                v = v[0]
                v = v.argmax(-1).astype('uint8')
                tensors[k] = v
                resos[k] = reso
            elif 'slo' in k:
                v = v[0]
                down = (v[..., 0] + 1) / 2 * 90
                up = (v[..., 1] + 1) / 2 * 90

                tensors[k + 'down'] = down
                resos[k + 'down'] = reso
                tensors[k + 'up'] = up
                resos[k + 'up'] = reso
            else:
                assert False

        imgs = []
        subtitles = []
        extents = []
        for k, t, reso in zip(tensors.keys(), tensors.values(), resos.values()):

            w = t.shape[0]
            dist = (w // 2) * reso
            ex = (dist, -dist, dist, -dist)

            print(k, t.shape, reso, ex)
            imgs.append(t)
            subtitles.append(k)
            extents.append(ex)

        main_display.show_many_images(
            imgs, subtitles, extents=extents, title='{} {!s}'.format(n, fp.c)
        )

    # Paths ************************************************************************************* **

    def dir_path(self, n):
        if n in NAMES:
            return os.path.join(self.acquisitions_prexix, n)
        for dir in EXTRA_DIRECTORIES:
            dir = os.path.join(dir, n)
            if os.path.isdir(dir):
                return dir

    def dsm_path(self, reso, n):
        dir = self.dir_path(n)
        return os.path.join(dir, 'dsm_{:.2f}cm.tif'.format(reso * 100))

    def ortho_path(self, reso, n):
        dir = self.dir_path(n)
        return os.path.join(dir, 'ortho_{:.2f}cm.tif'.format(reso * 100))

    def labels_path(self, n):
        assert n in NAMES
        dir = os.path.join(self.acquisitions_prexix, n)
        return os.path.join(dir, 'labels_{:.2f}cm.tif'.format(self.resos.out * 100))

    def edt_path(self, n):
        assert n in NAMES
        dir = os.path.join(self.acquisitions_prexix, n)
        return os.path.join(dir, 'edt_{:.2f}cm.tif'.format(self.resos.out * 100))

    def data_bounds_path(self, n):
        dir = self.dir_path(n)
        return os.path.join(dir, 'data_bounds.shp')