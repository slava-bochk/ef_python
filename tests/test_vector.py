import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from ef.util.vector import vector


class TestVector:
    def test_vector(self):
        assert type(vector(0)) == np.ndarray
        assert type(vector([1, 2, 3])) == np.ndarray
        assert_array_equal(vector(0), [0, 0, 0])
        assert_array_equal(vector(3.14), [3.14, 3.14, 3.14])
        v = [1, 2, 3]
        assert_array_equal(vector(v), v)

    def test_copy(self):
        a = np.array([1., 2., 3.])
        assert a is a
        assert vector(a) is not a

    def test_dtype(self):
        assert vector(0).dtype == np.int
        assert vector(0, np.float).dtype == np.float
        assert vector(0, np.int).dtype == np.int
        assert vector(3.14).dtype == np.float
        assert vector(3.14, np.float).dtype == np.float
        with raises(TypeError):
            vector(3.14, np.int)
        assert vector([1, 2, 3.14]).dtype == np.float
        with raises(TypeError):
            vector([1, 2, 3.14], np.int)
        assert_array_equal(vector({1: 2, 3: 4}), [{1: 2, 3: 4}] * 3)  # maybe unexpected, but works
        with raises(TypeError):
            vector({1: 2, 3: 4}, np.int)

    def test_wrong_shape(self):
        with raises(ValueError):
            vector((1, 2))
        with raises(ValueError):
            vector([])
        with raises(ValueError):
            vector((1, 2, 3, 4))
        with raises(ValueError):
            vector([[1], [2], [3]])
