from typing import Type

import inject
import numpy as np
import pytest
from pytest import raises
from scipy.interpolate import RegularGridInterpolator

from ef.meshgrid import MeshGrid
from ef.util.array_on_grid import ArrayOnGrid
from ef.util.testing import assert_array_equal, assert_array_almost_equal, assert_dataclass_eq


@pytest.mark.usefixtures("backend")
class TestArrayOnGrid:
    Array: Type[ArrayOnGrid] = inject.attr(ArrayOnGrid)
    xp = inject.attr(np)

    def test_init(self):
        a = self.Array(MeshGrid(10, 11))
        assert a.value_shape == ()
        assert_array_equal(a.data, np.zeros([11] * 3))
        assert type(a._data) == self.xp.ndarray
        data = self.xp.ones((5, 5, 5, 3))
        a = self.Array(MeshGrid(1, 5), 3, data)
        assert a.value_shape == (3,)
        assert_array_equal(a.data, np.ones((5, 5, 5, 3)))
        assert type(a._data) == self.xp.ndarray
        assert a._data is not data  # data gets copied don't create ArrayOnGrid every cycle
        with raises(ValueError):
            self.Array(MeshGrid(1, 5), data=self.xp.ones((5, 5, 5, 3)))

    def test_n_nodes(self):
        assert self.Array(MeshGrid(10, 11)).n_nodes == (11, 11, 11)
        assert self.Array(MeshGrid(10, 11), None).n_nodes == (11, 11, 11)
        assert self.Array(MeshGrid(10, 11), ()).n_nodes == (11, 11, 11)
        assert self.Array(MeshGrid(10, 11), 1).n_nodes == (11, 11, 11, 1)  # Should this be true?
        assert self.Array(MeshGrid(10, 11), 3).n_nodes == (11, 11, 11, 3)
        assert self.Array(MeshGrid(10, 11), (2, 5)).n_nodes == (11, 11, 11, 2, 5)

    def test_zero(self):
        z = self.Array(MeshGrid(10, 5), (2, 5)).zero
        assert type(z) == self.xp.ndarray
        assert z.shape == (5, 5, 5, 2, 5)
        assert (z == 0).all()

    def test_reset(self):
        a = self.Array(MeshGrid(1, 5), (), self.xp.ones((5, 5, 5)))
        assert_array_equal(a.data, np.ones((5, 5, 5)))
        a.reset()
        assert_array_equal(a.data, np.zeros((5, 5, 5)))

    def test_distribute_scalar(self):
        a = self.Array(MeshGrid(1, 2))
        a.distribute_at_positions(8, self.xp.full((1000, 3), 0.5))
        assert_array_equal(a.data, np.full((2, 2, 2), 1000))
        a.distribute_at_positions(8, self.xp.full((1000, 3), 0.5))
        assert_array_equal(a.data, np.full((2, 2, 2), 2000))
        a.reset()
        a.distribute_at_positions(1, self.xp.full((1000, 3), 0.1))
        assert_array_almost_equal(a.data, np.array([[[729, 81], [81, 9]],
                                                    [[81, 9], [9, 1]]]))
        a = self.Array(MeshGrid((2, 4, 8), (3, 3, 3)))
        a.distribute_at_positions(-2, [(1, 1, 3)])
        assert_array_equal(a.data, [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[-0.25 / 8, -0.75 / 8, 0], [-0.25 / 8, -0.75 / 8, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        a.reset()
        a.distribute_at_positions(-2, [(1, 1, 3), (1, 1, 3)])
        assert_array_equal(a.data, [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[-0.25 / 4, -0.75 / 4, 0], [-0.25 / 4, -0.75 / 4, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        a.reset()
        a.distribute_at_positions(-2, [(2, 4, 8)])
        assert_array_equal(a.data, [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, -0.25]]])
        a.reset()
        # with raises(IndexError):
        #     a.distribute_at_positions(-2, [(1, 2, 8.1)])

    def test_distribute_vector(self):
        a = self.Array(MeshGrid(1, 2))
        with raises(ValueError, match="operands could not be broadcast together with shapes"):
            a.distribute_at_positions(self.xp.array([8, 8, 8]), self.xp.full((1000, 3), 0.5))

    def test_interpolate_scalar(self):
        a = self.Array(MeshGrid(1, 2), None, [[[0, 1], [0, -1]],
                                              [[0, 0], [0, 0]]])
        positions = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                              (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
                              (0.5, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 1, 0.5), (0, 0, 2)], dtype=float)
        assert_array_equal(a.interpolate_at_positions(positions), [0, 1, 0, -1,
                                                                   0, 0, 0, 0,
                                                                   0, 0.25, -0.25, 0])
        assert_array_equal(a.interpolate_at_positions(positions[1:2]), [1.])

        a = self.Array(MeshGrid((100, 20, 100), (101, 44, 101)), None, np.arange(101 * 44 * 101, dtype=float).reshape((101, 44, 101)))
        positions = np.linspace((0, 13, 100.1), (100.1, 0, 0), 314)
        o, s = np.zeros(3), np.array([100, 20, 100])
        xyz = tuple(np.linspace(o[i], o[i] + s[i], a.n_nodes[i]) for i in (0, 1, 2))
        interpolator = RegularGridInterpolator(xyz, a.data, bounds_error=False, fill_value=0)
        r1 = a.interpolate_at_positions(self.xp.asarray(positions))
        assert_array_almost_equal(r1, interpolator(positions))

    def test_interpolate_vector(self):
        a = self.Array(MeshGrid((2, 4, 8), (3, 3, 3)), 3, self.xp.full((3, 3, 3, 3), 100))
        a._data[1:2, 0:2, 0:2] = self.xp.array([[[2, 1, 0], [-3, 1, 0]],
                                                [[0, -1, 0], [-1, 0, 0]]])
        assert_array_equal(a.interpolate_at_positions(np.array([(1, 1, 3)], dtype=float)),
                       [(-1.25, 0.375, 0)])
        assert_array_equal(a.interpolate_at_positions(self.xp.array([(1, 1, 3), (2, 1, 3), (1.5, 1, 3)], dtype=float)),
                           [(-1.25, 0.375, 0),
                            (100, 100, 100),
                            (49.375, 50.1875, 50)])

    def test_gradient(self):
        m = MeshGrid((1.5, 2, 1), (4, 3, 2))
        potential = self.Array(m)
        potential._data = self.xp.stack([self.xp.array([[0., 0, 0],
                                                        [1, 2, 3],
                                                        [4, 3, 2],
                                                        [4, 4, 4]]), self.xp.zeros((4, 3))], -1)
        expected = self.Array(m, 3, [[[[-2, 0, 0], [0, 0, 0]], [[-4, 0, 0], [0, 0, 0]], [[-6, 0, 0], [0, 0, 0]]],
                                     [[[-4, -1, 1], [0, 0, 1]], [[-3, -1, 2], [0, 0, 2]], [[-2, -1, 3], [0, 0, 3]]],
                                     [[[-3, 1, 4], [0, 0, 4]], [[-2, 1, 3], [0, 0, 3]], [[-1, 1, 2], [0, 0, 2]]],
                                     [[[0, 0, 4], [0, 0, 4]], [[-2, 0, 4], [0, 0, 4]], [[-4, 0, 4], [0, 0, 4]]]])
        field = potential.gradient()
        assert_dataclass_eq(field, expected)
        with raises(ValueError, match="Trying got compute gradient for a non-scalar field: ambiguous"):
            field.gradient()

    def test_is_the_same_on_all_boundaries(self):
        mesh = MeshGrid(12, (4, 4, 3))
        a = self.Array(mesh)
        assert a.is_the_same_on_all_boundaries
        for x, y, z in np.ndindex(4, 4, 3):
            a._data[x, y, z] = 2.
            if 0 < x < 3 and 0 < y < 3 and 0 < z < 2:
                assert a.is_the_same_on_all_boundaries
            else:
                assert not a.is_the_same_on_all_boundaries
            a.reset()
            assert a.is_the_same_on_all_boundaries

        a = self.Array(mesh, 3)
        assert a.is_the_same_on_all_boundaries
        for x, y, z, t in np.ndindex(4, 4, 3, 3):
            a._data[x, y, z, t] = 2.
            if 0 < x < 3 and 0 < y < 3 and 0 < z < 2:
                assert a.is_the_same_on_all_boundaries
            else:
                assert not a.is_the_same_on_all_boundaries
            a.reset()
            assert a.is_the_same_on_all_boundaries
