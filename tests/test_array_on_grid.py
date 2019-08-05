import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pytest import raises

from ef.meshgrid import MeshGrid
from ef.util.array_on_grid import ArrayOnGrid


class TestArrayOnGrid:
    def test_init(self):
        assert_array_equal(ArrayOnGrid(MeshGrid(10, 11)).data, np.zeros((11, 11, 11)))
        assert_array_equal(ArrayOnGrid(MeshGrid(1, 5), 3, np.ones((5, 5, 5, 3))).data, np.ones((5, 5, 5, 3)))
        with raises(ValueError):
            ArrayOnGrid(MeshGrid(1, 5), data=np.ones((5, 5, 5, 3)))

    def test_n_nodes(self):
        assert ArrayOnGrid(MeshGrid(10, 11)).n_nodes == (11, 11, 11)
        assert ArrayOnGrid(MeshGrid(10, 11), None).n_nodes == (11, 11, 11)
        assert ArrayOnGrid(MeshGrid(10, 11), ()).n_nodes == (11, 11, 11)
        assert ArrayOnGrid(MeshGrid(10, 11), 1).n_nodes == (11, 11, 11, 1)  # Should this be true?
        assert ArrayOnGrid(MeshGrid(10, 11), 3).n_nodes == (11, 11, 11, 3)
        assert ArrayOnGrid(MeshGrid(10, 11), (2, 5)).n_nodes == (11, 11, 11, 2, 5)

    def test_zero(self):
        assert_array_equal(ArrayOnGrid(MeshGrid(10, 5)).zero, np.zeros((5, 5, 5)))
        assert_array_equal(ArrayOnGrid(MeshGrid(10, 5), 3).zero, np.zeros((5, 5, 5, 3)))
        assert_array_equal(ArrayOnGrid(MeshGrid(10, 5), (2, 5)).zero, np.zeros((5, 5, 5, 2, 5)))

    def test_reset(self):
        a = ArrayOnGrid(MeshGrid(1, 5), (), np.ones((5, 5, 5)))
        assert_array_equal(a.data, np.ones((5, 5, 5)))
        a.reset()
        assert_array_equal(a.data, np.zeros((5, 5, 5)))

    def test_distribute_scalar(self):
        a = ArrayOnGrid(MeshGrid(1, 2))
        a.distribute_at_positions(8, np.full((1000, 3), 0.5))
        assert_array_equal(a.data, np.full((2, 2, 2), 1000))
        a.distribute_at_positions(8, np.full((1000, 3), 0.5))
        assert_array_equal(a.data, np.full((2, 2, 2), 2000))
        a.reset()
        a.distribute_at_positions(1, np.full((1000, 3), 0.1))
        assert_array_almost_equal(a.data, [[[729, 81], [81, 9]],
                                           [[81, 9], [9, 1]]])
        a = ArrayOnGrid(MeshGrid((2, 4, 8), (3, 3, 3)))
        a.distribute_at_positions(-2, [(1, 1, 3)])
        assert_array_equal(a.data,
                           [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[-0.25 / 8, -0.75 / 8, 0], [-0.25 / 8, -0.75 / 8, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        a.reset()
        a.distribute_at_positions(-2, [(1, 1, 3), (1, 1, 3)])
        assert_array_equal(a.data,
                           [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[-0.25 / 4, -0.75 / 4, 0], [-0.25 / 4, -0.75 / 4, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        a.reset()
        a.distribute_at_positions(-2, [(2, 4, 8)])
        assert_array_equal(a.data,
                           [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, -0.25]]])
        a.reset()
        with raises(ValueError, match="Position is out of meshgrid bounds"):
            a.distribute_at_positions(-2, [(1, 2, 8.1)])

    def test_distribute_vector(self):
        a = ArrayOnGrid(MeshGrid(1, 2))
        with raises(ValueError, match="operands could not be broadcast together with shapes"):
            assert_array_equal(a.distribute_at_positions(np.array([8, 8, 8]), np.full((1000, 3), 0.5)),
                               np.full((2, 2, 2, 3), 1000))

    def test_interpolate_scalar(self):
        a = ArrayOnGrid(MeshGrid(1, 2), None, [[[0, 1], [0, -1]],
                                               [[0, 0], [0, 0]]])
        positions = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                     (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
                     (0.5, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 1, 0.5), (0, 0, 2)]
        assert_array_equal(a.interpolate_at_positions(positions), [0, 1, 0, -1,
                                                                   0, 0, 0, 0,
                                                                   0, 0.25, -0.25, 0])
        assert_array_equal(a.interpolate_at_positions(positions[1]), [1])

    def test_interpolate_vector(self):
        a = ArrayOnGrid(MeshGrid((2, 4, 8), (3, 3, 3)), 3, np.full((3, 3, 3, 3), 100))
        a.data[1:2, 0:2, 0:2] = np.array([[[2, 1, 0], [-3, 1, 0]],
                                          [[0, -1, 0], [-1, 0, 0]]])
        assert_array_equal(a.interpolate_at_positions([(1, 1, 3)]), [(-1.25, 0.375, 0)])
        assert_array_equal(a.interpolate_at_positions([(1, 1, 3), (2, 1, 3), (1.5, 1, 3)]),
                           [(-1.25, 0.375, 0),
                            (100, 100, 100),
                            (49.375, 50.1875, 50)])

    def test_gradient(self):
        m = MeshGrid((1.5, 2, 1), (4, 3, 2))
        potential = ArrayOnGrid(m)
        potential.data = np.stack([np.array([[0., 0, 0],
                                             [1, 2, 3],
                                             [4, 3, 2],
                                             [4, 4, 4]]), np.zeros((4, 3))], -1)
        expected = ArrayOnGrid(m, 3, [[[[-2, 0, 0], [0, 0, 0]], [[-4, 0, 0], [0, 0, 0]], [[-6, 0, 0], [0, 0, 0]]],
                                      [[[-4, -1, 1], [0, 0, 1]], [[-3, -1, 2], [0, 0, 2]], [[-2, -1, 3], [0, 0, 3]]],
                                      [[[-3, 1, 4], [0, 0, 4]], [[-2, 1, 3], [0, 0, 3]], [[-1, 1, 2], [0, 0, 2]]],
                                      [[[0, 0, 4], [0, 0, 4]], [[-2, 0, 4], [0, 0, 4]], [[-4, 0, 4], [0, 0, 4]]]])
        field = potential.gradient()
        assert field == expected
        with raises(ValueError, match="Trying got compute gradient for a non-scalar field: ambiguous"):
            field.gradient()

    def test_is_the_same_on_all_boundaries(self):
        mesh = MeshGrid(12, (4, 4, 3))
        a = ArrayOnGrid(mesh)
        assert a.is_the_same_on_all_boundaries
        for x, y, z in np.ndindex(4, 4, 3):
            a.data[x, y, z] = 2.
            if 0 < x < 3 and 0 < y < 3 and 0 < z < 2:
                assert a.is_the_same_on_all_boundaries
            else:
                assert not a.is_the_same_on_all_boundaries
            a.reset()
            assert a.is_the_same_on_all_boundaries

        a = ArrayOnGrid(mesh, 3)
        assert a.is_the_same_on_all_boundaries
        for x, y, z, t in np.ndindex(4, 4, 3, 3):
            a.data[x, y, z, t] = 2.
            if 0 < x < 3 and 0 < y < 3 and 0 < z < 2:
                assert a.is_the_same_on_all_boundaries
            else:
                assert not a.is_the_same_on_all_boundaries
            a.reset()
            assert a.is_the_same_on_all_boundaries

