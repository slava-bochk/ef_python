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

    def test_distribute_scalar(self):
        mesh = MeshGrid(1, 2)
        assert_array_equal(mesh.distribute_scalar_at_positions(8, np.full((1000, 3), 0.5)), np.full((2, 2, 2), 1000))
        assert_array_almost_equal(mesh.distribute_scalar_at_positions(1, np.full((1000, 3), 0.1)),
                                  [[[729, 81], [81, 9]],
                                   [[81, 9], [9, 1]]])

        mesh = MeshGrid((2, 4, 8), (3, 3, 3))
        assert_array_equal(mesh.distribute_scalar_at_positions(-2, [(1, 1, 3)]),
                           [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[-0.25 / 8, -0.75 / 8, 0], [-0.25 / 8, -0.75 / 8, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        assert_array_equal(mesh.distribute_scalar_at_positions(-2, [(1, 1, 3), (1, 1, 3)]),
                           [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[-0.25 / 4, -0.75 / 4, 0], [-0.25 / 4, -0.75 / 4, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        assert_array_equal(mesh.distribute_scalar_at_positions(-2, [(2, 4, 8)]),
                           [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, -0.25]]])
        with raises(ValueError, match="Position is out of meshgrid bounds"):
            mesh.distribute_scalar_at_positions(-2, [(1, 2, 8.1)])

    def test_distribute_vector(self):
        mesh = MeshGrid(1, 2)
        with raises(ValueError, match="operands could not be broadcast together with shapes"):
            assert_array_equal(mesh.distribute_scalar_at_positions(np.array([8, 8, 8]), np.full((1000, 3), 0.5)),
                               np.full((2, 2, 2, 3), 1000))

    def test_interpolate_field_scalar(self):
        mesh = MeshGrid(1, 2)
        potential = [[[0, 1], [0, -1]],
                     [[0, 0], [0, 0]]]
        positions = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                     (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
                     (0.5, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 1, 0.5), (0, 0, 2)]
        assert_array_equal(mesh.interpolate_field_at_positions(potential, positions), [0, 1, 0, -1,
                                                                                       0, 0, 0, 0,
                                                                                       0, 0.25, -0.25, 0])
        assert_array_equal(mesh.interpolate_field_at_positions(potential, positions[1]), [1])

    def test_interpolate_field_vector(self):
        mesh = MeshGrid((2, 4, 8), (3, 3, 3))
        field = np.full((3, 3, 3, 3), 100)
        field[1:2, 0:2, 0:2] = np.array([[[2, 1, 0], [-3, 1, 0]],
                                         [[0, -1, 0], [-1, 0, 0]]])
        assert_array_equal(mesh.interpolate_field_at_positions(field, [(1, 1, 3)]), [(-1.25, 0.375, 0)])
        assert_array_equal(mesh.interpolate_field_at_positions(field, [(1, 1, 3), (2, 1, 3), (1.5, 1, 3)]),
                           [(-1.25, 0.375, 0),
                            (100, 100, 100),
                            (49.375, 50.1875, 50)])
