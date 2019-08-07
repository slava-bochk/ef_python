import logging

import h5py
import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from ef.config.components import SpatialMeshConf, BoundaryConditionsConf
from ef.meshgrid import MeshGrid
from ef.util.array_on_grid import ArrayOnGrid


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

    def test_config(self, capsys, backend):
        mesh, charge, potential, field = SpatialMeshConf((4, 2, 3), (2, 1, 3)).make()
        assert mesh == MeshGrid((4, 2, 3), (3, 3, 2))
        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""

    def test_do_init_warnings(self, capsys, caplog):
        MeshGrid.from_step((12, 12, 12), (5, 5, 7))
        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""
        assert caplog.record_tuples == [
            ('root', logging.WARNING,
             "X step on spatial grid was reduced to 4.000 from 5.000 to fit in a round number of cells."),
            ('root', logging.WARNING,
             "Y step on spatial grid was reduced to 4.000 from 5.000 to fit in a round number of cells."),
            ('root', logging.WARNING,
             "Z step on spatial grid was reduced to 6.000 from 7.000 to fit in a round number of cells."),
        ]

    def test_do_init(self):
        a = ArrayOnGrid(MeshGrid.from_step((12, 12, 12), (4, 4, 6)))
        a.apply_boundary_values(BoundaryConditionsConf(1, 2, 3, 4, 5, 6))
        expected = np.array([[[5., 1., 6.], [5., 1., 6.], [5., 1., 6.], [5., 1., 6.]],
                             [[5., 3., 6.], [5., 0., 6.], [5., 0., 6.], [5., 4., 6.]],
                             [[5., 3., 6.], [5., 0., 6.], [5., 0., 6.], [5., 4., 6.]],
                             [[5., 2., 6.], [5., 2., 6.], [5., 2., 6.], [5., 2., 6.]]])
        assert_array_equal(a.data, expected)

    def test_do_init_ranges(self):
        with raises(ValueError):
            MeshGrid.from_step((10, 20), (2, 1, 3))
        with raises(ValueError):
            MeshGrid.from_step(((1, 2), 3), (1, 1, 1))
        with raises(ValueError):
            MeshGrid.from_step((10, 10, 10), [[2, 1, 3], [4, 5, 6], [7, 8, 9]],
                                BoundaryConditionsConf(3.14))
        with raises(ValueError):
            MeshGrid.from_step((10, 10, -30), (2, 1, 3))
        with raises(ValueError):
            MeshGrid.from_step((10, 10, 10), (2, -2, 3))
        mesh = MeshGrid.from_step((10, 10, 10), (17, 2, 3))
        assert tuple(mesh.cell) == (10, 2, 2.5)

    def test_init_h5(self, tmpdir):
        fname = tmpdir.join('test_spatialmesh_init.h5')

        mesh1 = MeshGrid.from_step((10, 20, 30), (2, 1, 3))
        with h5py.File(fname, mode="w") as h5file:
            mesh1.save_h5(h5file.create_group("/meshgroup"))
        with h5py.File(fname, mode="r") as h5file:
            mesh2 = MeshGrid.load_h5(h5file["/meshgroup"])
        assert mesh1 == mesh2
