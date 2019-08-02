import numpy as np
from numpy.testing import assert_array_equal

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
