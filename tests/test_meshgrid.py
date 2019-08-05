import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pytest import raises

from ef.meshgrid import MeshGrid


class TestMeshGrid:
    def test_init(self):
        m = MeshGrid(10, 11)
        assert_array_equal(m.size, (10, 10, 10))
        assert_array_equal(m.n_nodes, (11, 11, 11))
        assert_array_equal(m.origin, (0, 0, 0))

    def test_from_step(self):
        assert MeshGrid.from_step(10, 1, 7) == MeshGrid(10, 11, 7)
        assert MeshGrid.from_step([10, 20, 5], [1, 4, 1]) == MeshGrid([10, 20, 5], [11, 6, 6])
        assert MeshGrid.from_step(10, [1, 2, 4]) == MeshGrid(10, [11, 6, 4])

    def test_cell(self):
        assert_array_equal(MeshGrid(10, 11, 7).cell, (1, 1, 1))
        assert_array_equal(MeshGrid.from_step(10, [1, 2, 4]).cell, [1, 2, 10 / 3])

    def test_node_coordinates(self):
        coords = np.array([[[[0., 0., 0.], [0., 0., 3.]], [[0., 1., 0.], [0., 1., 3.]],
                            [[0., 2., 0.], [0., 2., 3.]]],
                           [[[2., 0., 0.], [2., 0., 3.]], [[2., 1., 0.], [2., 1., 3.]],
                            [[2., 2., 0.], [2., 2., 3.]]],
                           [[[4., 0., 0.], [4., 0., 3.]], [[4., 1., 0.], [4., 1., 3.]],
                            [[4., 2., 0.], [4., 2., 3.]]]])
        assert_array_equal(MeshGrid.from_step((4, 2, 3), (2, 1, 3)).node_coordinates, coords)
        assert_array_equal(MeshGrid.from_step((4, 2, 3), (2, 1, 3), (1, 2, 3.14)).node_coordinates,
                           coords + [1, 2, 3.14])
