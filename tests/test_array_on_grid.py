import inject
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pytest import raises

from ef.meshgrid import MeshGrid
from ef.util.array_on_grid import ArrayOnGrid


@pytest.mark.usefixtures("backend")
class TestArrayOnGrid:
    Array = inject.attr(ArrayOnGrid)
    xp = inject.attr(np)
    assert_ae = inject.attr(assert_array_equal)
    assert_almost_ae = inject.attr(assert_array_almost_equal)

    def test_init(self):
        self.assert_ae(self.Array(MeshGrid(10, 11)).data, self.xp.zeros((11, 11, 11)))
        self.assert_ae(self.Array(MeshGrid(1, 5), 3, self.xp.ones((5, 5, 5, 3))).data, self.xp.ones((5, 5, 5, 3)))
        with raises(ValueError):
            self.Array(MeshGrid(1, 5), data=self.xp.ones((5, 5, 5, 3)))

    def test_n_nodes(self):
        self.assert_ae(self.Array(MeshGrid(10, 11)).n_nodes, (11, 11, 11))
        assert self.Array(MeshGrid(10, 11), None).n_nodes == (11, 11, 11)
        assert self.Array(MeshGrid(10, 11), ()).n_nodes == (11, 11, 11)
        assert self.Array(MeshGrid(10, 11), 1).n_nodes == (11, 11, 11, 1)  # Should this be true?
        assert self.Array(MeshGrid(10, 11), 3).n_nodes == (11, 11, 11, 3)
        assert self.Array(MeshGrid(10, 11), (2, 5)).n_nodes == (11, 11, 11, 2, 5)

    def test_zero(self):
        self.assert_ae(self.Array(MeshGrid(10, 5)).zero, self.xp.zeros((5, 5, 5)))
        self.assert_ae(self.Array(MeshGrid(10, 5), 3).zero, self.xp.zeros((5, 5, 5, 3)))
        self.assert_ae(self.Array(MeshGrid(10, 5), (2, 5)).zero, self.xp.zeros((5, 5, 5, 2, 5)))

    def test_reset(self):
        a = self.Array(MeshGrid(1, 5), (), self.xp.ones((5, 5, 5)))
        self.assert_ae(a.data, self.xp.ones((5, 5, 5)))
        a.reset()
        self.assert_ae(a.data, self.xp.zeros((5, 5, 5)))

    def test_distribute_scalar(self):
        a = self.Array(MeshGrid(1, 2))
        a.distribute_at_positions(8, self.xp.full((1000, 3), 0.5))
        self.assert_ae(a.data, self.xp.full((2, 2, 2), 1000))
        a.distribute_at_positions(8, self.xp.full((1000, 3), 0.5))
        self.assert_ae(a.data, self.xp.full((2, 2, 2), 2000))
        a.reset()
        a.distribute_at_positions(1, self.xp.full((1000, 3), 0.1))
        self.assert_almost_ae(a.data, [[[729, 81], [81, 9]],
                                       [[81, 9], [9, 1]]])
        a = self.Array(MeshGrid((2, 4, 8), (3, 3, 3)))
        a.distribute_at_positions(-2, [(1, 1, 3)])
        self.assert_ae(a.data,
                       [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[-0.25 / 8, -0.75 / 8, 0], [-0.25 / 8, -0.75 / 8, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        a.reset()
        a.distribute_at_positions(-2, [(1, 1, 3), (1, 1, 3)])
        self.assert_ae(a.data,
                       [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[-0.25 / 4, -0.75 / 4, 0], [-0.25 / 4, -0.75 / 4, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        a.reset()
        a.distribute_at_positions(-2, [(2, 4, 8)])
        self.assert_ae(a.data,
                       [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, -0.25]]])
        a.reset()
        with raises(ValueError, match="Position is out of meshgrid bounds"):
            a.distribute_at_positions(-2, [(1, 2, 8.1)])

    def test_distribute_vector(self):
        a = self.Array(MeshGrid(1, 2))
        with raises(ValueError, match="operands could not be broadcast together with shapes"):
            self.assert_ae(a.distribute_at_positions(self.xp.array([8, 8, 8]), self.xp.full((1000, 3), 0.5)),
                           self.xp.full((2, 2, 2, 3), 1000))

    def test_interpolate_scalar(self):
        a = self.Array(MeshGrid(1, 2), None, [[[0, 1], [0, -1]],
                                              [[0, 0], [0, 0]]])
        positions = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                     (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
                     (0.5, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 1, 0.5), (0, 0, 2)]
        self.assert_ae(a.interpolate_at_positions(positions), [0, 1, 0, -1,
                                                               0, 0, 0, 0,
                                                               0, 0.25, -0.25, 0])
        self.assert_ae(a.interpolate_at_positions([positions[1]]), [1])

    def test_interpolate_vector(self):
        a = self.Array(MeshGrid((2, 4, 8), (3, 3, 3)), 3, self.xp.full((3, 3, 3, 3), 100))
        a._data[1:2, 0:2, 0:2] = self.xp.array([[[2, 1, 0], [-3, 1, 0]],
                                               [[0, -1, 0], [-1, 0, 0]]])
        self.assert_ae(a.interpolate_at_positions([(1, 1, 3)]), [(-1.25, 0.375, 0)])
        self.assert_ae(a.interpolate_at_positions([(1, 1, 3), (2, 1, 3), (1.5, 1, 3)]),
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
        assert field == expected
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
