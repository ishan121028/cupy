import pytest

import numpy

import cupy

from cupy import testing

import cupyx
import cupyx.scipy.stats  # NOQA

import scipy.stats  # NOQA


atol = {'default': 1e-6, cupy.float64: 1e-14}
rtol = {'default': 1e-6, cupy.float64: 1e-14}


@testing.with_requires('scipy')
class TestZmap:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_1dim(self, xp, scp, dtype):
        x = testing.shaped_random((5,), xp, dtype=dtype)
        y = testing.shaped_random((4,), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_2dim(self, xp, scp, dtype):
        x = testing.shaped_random((2, 6), xp, dtype=dtype)
        y = testing.shaped_random((2, 1), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_multi_dim(self, xp, scp, dtype):
        x = testing.shaped_random((3, 4, 5, 7), xp, dtype=dtype)
        y = testing.shaped_random((3, 4, 1, 1), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_with_axis(self, xp, scp, dtype):
        x = testing.shaped_random((2, 3), xp, dtype=dtype)
        y = testing.shaped_random((1, 3), xp, dtype=dtype)
        return scp.stats.zmap(x, y, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_with_axis_ddof(self, xp, scp, dtype):
        x = testing.shaped_random((2, 3), xp, dtype=dtype)
        y = testing.shaped_random((1, 3), xp, dtype=dtype)
        return scp.stats.zmap(x, y, axis=1, ddof=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_empty(self, xp, scp, dtype):
        x = xp.array([], dtype=dtype)
        y = xp.array([1, 3, 5], dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_nan_pocily_omit(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, cupy.nan], dtype=dtype)
        y = xp.array([cupy.nan, -4.0, -1.0, -5.0], dtype=dtype)
        return scp.stats.zmap(x, y, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_nan_policy_omit_axis_ddof(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, cupy.nan], dtype=dtype)
        y = xp.array([cupy.nan, -4.0, -1.0, -5.0], dtype=dtype)
        return scp.stats.zmap(x, y, axis=0, ddof=1, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    def test_zmap_nan_policy_raise(self, dtype):
        x = cupy.array([1, 2, 3], dtype=dtype)
        y = cupy.array([8, -4, cupy.nan, 4], dtype=dtype)
        with pytest.raises(ValueError):
            cupyx.scipy.stats.zmap(x, y, nan_policy='raise')
