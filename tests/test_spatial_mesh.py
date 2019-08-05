import logging

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ef.config.components import SpatialMeshConf, BoundaryConditionsConf
from ef.meshgrid import MeshGrid
from ef.spatial_mesh import SpatialMesh
from ef.util.array_on_grid import ArrayOnGrid


class TestDefaultSpatialMesh:
    def test_config(self, capsys):
        mesh = SpatialMeshConf((4, 2, 3), (2, 1, 3)).make(BoundaryConditionsConf(3.14))
        assert mesh == SpatialMesh.do_init((4, 2, 3), (2, 1, 3), BoundaryConditionsConf(3.14))
        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""

    def test_do_init_warnings(self, capsys, caplog):
        SpatialMesh.do_init((12, 12, 12), (5, 5, 7), BoundaryConditionsConf(0))
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
        mesh = SpatialMesh.do_init((4, 2, 3), (2, 1, 3), BoundaryConditionsConf(3.14))
        assert mesh.mesh == MeshGrid((4, 2, 3), (3, 3, 2))
        assert mesh.charge_density == ArrayOnGrid(mesh.mesh)
        assert mesh.potential == ArrayOnGrid(mesh.mesh, (), np.full((3, 3, 2), 3.14))
        assert mesh.electric_field == ArrayOnGrid(mesh.mesh, 3)
        mesh = SpatialMesh.do_init((12, 12, 12), (4, 4, 6), BoundaryConditionsConf(1, 2, 3, 4, 5, 6))
        potential = np.array([[[5., 1., 6.], [5., 1., 6.], [5., 1., 6.], [5., 1., 6.]],
                              [[5., 3., 6.], [5., 0., 6.], [5., 0., 6.], [5., 4., 6.]],
                              [[5., 3., 6.], [5., 0., 6.], [5., 0., 6.], [5., 4., 6.]],
                              [[5., 2., 6.], [5., 2., 6.], [5., 2., 6.], [5., 2., 6.]]])
        assert mesh.potential == ArrayOnGrid(mesh.mesh, (), potential)

    def test_is_potential_equal_on_boundaries(self):
        for x, y, z in np.ndindex(4, 4, 3):
            mesh = SpatialMesh.do_init((12, 12, 12), (4, 4, 6), BoundaryConditionsConf(3.14))
            assert mesh.is_potential_equal_on_boundaries()
            mesh.potential.data[x, y, z] = 2.
            if np.all([x > 0, y > 0, z > 0]) and np.all([x < 3, y < 3, z < 2]):
                assert mesh.is_potential_equal_on_boundaries()
            else:
                assert not mesh.is_potential_equal_on_boundaries()

    def test_do_init_ranges(self):
        with pytest.raises(ValueError) as excinfo:
            SpatialMesh.do_init((10, 20), (2, 1, 3), BoundaryConditionsConf(3.14))
        assert excinfo.value.args == ('grid_size must be a flat triple', (10, 20))
        with pytest.raises(ValueError) as excinfo:
            SpatialMesh.do_init(((1, 2), 3), (1, 1, 1), BoundaryConditionsConf(3.14))
        assert excinfo.value.args == ('grid_size must be a flat triple', ((1, 2), 3))
        with pytest.raises(ValueError) as excinfo:
            SpatialMesh.do_init((10, 10, 10), [[2, 1, 3], [4, 5, 6], [7, 8, 9]],
                                BoundaryConditionsConf(3.14))
        assert excinfo.value.args == ('step_size must be a flat triple', [[2, 1, 3], [4, 5, 6], [7, 8, 9]],)

        with pytest.raises(ValueError) as excinfo:
            SpatialMesh.do_init((10, 10, -30), (2, 1, 3), BoundaryConditionsConf(3.14))
        assert excinfo.value.args == ('grid_size must be positive', (10, 10, -30))
        with pytest.raises(ValueError) as excinfo:
            SpatialMesh.do_init((10, 10, 10), (2, -2, 3), BoundaryConditionsConf(3.14))
        assert excinfo.value.args == ('step_size must be positive', (2, -2, 3))
        with pytest.raises(ValueError) as excinfo:
            SpatialMesh.do_init((10, 10, 10), (17, 2, 3), BoundaryConditionsConf(3.14))
        assert excinfo.value.args == ('step_size cannot be bigger than grid_size',)

    def test_init_h5(self, tmpdir):
        fname = tmpdir.join('test_spatialmesh_init.h5')

        mesh1 = SpatialMesh.do_init((10, 20, 30), (2, 1, 3), BoundaryConditionsConf(3.14))
        with h5py.File(fname, mode="w") as h5file:
            mesh1.save_h5(h5file.create_group("/meshgroup"))
        with h5py.File(fname, mode="r") as h5file:
            mesh2 = SpatialMesh.load_h5(h5file["/meshgroup"])
        assert mesh1 == mesh2

    def test_dict(self):
        mesh = SpatialMesh.do_init((4, 2, 3), (2, 1, 3), BoundaryConditionsConf())
        d = mesh.dict
        assert d.keys() == {"mesh", "electric_field", "potential", "charge_density"}
        mesh = MeshGrid((4, 2, 3), (3, 3, 2))
        assert d["mesh"] == mesh
        assert d["electric_field"] == ArrayOnGrid(mesh, 3)
        assert d["potential"] == ArrayOnGrid(mesh)
        assert d["charge_density"] == ArrayOnGrid(mesh)

