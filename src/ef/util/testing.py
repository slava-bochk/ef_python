__all__ = ['assert_array_equal', 'assert_array_almost_equal']

from numpy.testing import assert_array_equal, assert_array_almost_equal

try:
    from cupy.testing import assert_array_equal, assert_array_almost_equal
except ImportError:
    pass
