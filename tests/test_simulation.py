from configparser import ConfigParser
from math import sqrt

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ef.config.components import *
from ef.config.config import Config
from ef.field import FieldZero
from ef.field.expression import FieldExpression
from ef.field.solvers.field_solver import FieldSolver
from ef.field.uniform import FieldUniform
from ef.inner_region import InnerRegion
from ef.particle_array import ParticleArray
from ef.particle_interaction_model import ParticleInteractionModel
from ef.simulation import Simulation
from ef.spatial_mesh import SpatialMesh
from ef.time_grid import TimeGrid


class TestSimulation:
    def test_init_from_config(self):
        efconf = Config()
        parser = ConfigParser()
        parser.read_string(efconf.export_to_string())
        sim = Config.from_configparser(parser).make()
        assert sim.time_grid == TimeGrid(100, 1, 10)
        assert sim.spat_mesh == SpatialMesh.do_init((10, 10, 10), (1, 1, 1), BoundaryConditionsConf(0))
        assert sim.inner_regions == []
        assert type(sim._field_solver) == FieldSolver
        assert sim.particle_sources == []
        assert sim.electric_fields == FieldZero('ZeroSum', 'electric')
        assert sim.magnetic_fields == FieldZero('ZeroSum', 'magnetic')
        assert sim.particle_interaction_model == ParticleInteractionModel("PIC")
        assert sim._output_filename_prefix == "out_"
        assert sim._output_filename_suffix == ".h5"

    @pytest.mark.slowish
    def test_all_config(self):
        efconf = Config(TimeGridConf(200, 20, 2), SpatialMeshConf((5, 5, 5), (.1, .1, .1)),
                        sources=[ParticleSourceConf('a', Box()),
                                 ParticleSourceConf('c', Cylinder()),
                                 ParticleSourceConf('d', Tube())],
                        inner_regions=[InnerRegionConf('1', Box(), 1),
                                       InnerRegionConf('2', Sphere(), -2),
                                       InnerRegionConf('3', Cylinder(), 0),
                                       InnerRegionConf('4', Tube(), 4)],
                        output_file=OutputFileConf(), boundary_conditions=BoundaryConditionsConf(-2.7),
                        particle_interaction_model=ParticleInteractionModelConf('binary'),
                        external_fields=[ExternalFieldUniformConf('x', 'electric', (-2, -2, 1)),
                                         ExternalFieldExpressionConf('y', 'magnetic',
                                                                     ('0', '0', '3*x + sqrt(y) - z**2'))])

        parser = ConfigParser()
        parser.read_string(efconf.export_to_string())
        sim = Config.from_configparser(parser).make()
        assert sim.time_grid == TimeGrid(200, 2, 20)
        assert sim.spat_mesh == SpatialMesh.do_init((5, 5, 5), (.1, .1, .1), BoundaryConditionsConf(-2.7))
        assert sim.inner_regions == [InnerRegion('1', Box(), 1),
                                     InnerRegion('2', Sphere(), -2),
                                     InnerRegion('3', Cylinder(), 0),
                                     InnerRegion('4', Tube(), 4)]
        assert type(sim._field_solver) == FieldSolver
        assert sim.particle_sources == [ParticleSourceConf('a', Box()).make(),
                                        ParticleSourceConf('c', Cylinder()).make(),
                                        ParticleSourceConf('d', Tube()).make()]
        assert sim.electric_fields == FieldUniform('x', 'electric', np.array((-2, -2, 1)))
        assert sim.magnetic_fields == FieldExpression('y', 'magnetic', '0', '0', '3*x + sqrt(y) - z**2')
        assert sim.particle_interaction_model == ParticleInteractionModel("binary")
        assert sim._output_filename_prefix == "out_"
        assert sim._output_filename_suffix == ".h5"

    def test_binary_field(self):
        d = Config().make()
        d.particle_arrays = [ParticleArray(1, -1, 1, [(1, 2, 3)], [(-2, 2, 0)], False)]
        assert_array_almost_equal(d.binary_electric_field_at_positions((1, 2, 3)), [(0, 0, 0)])
        assert_array_almost_equal(d.binary_electric_field_at_positions((1, 2, 4)), [(0, 0, -1)])
        assert_array_almost_equal(d.binary_electric_field_at_positions((0, 2, 3)), [(1, 0, 0)])
        assert_array_almost_equal(d.binary_electric_field_at_positions((0, 1, 2)),
                                  [(1 / sqrt(27), 1 / sqrt(27), 1 / sqrt(27))])
        d.particle_arrays = [ParticleArray(2, -1, 1, [(1, 2, 3), (1, 2, 3)], [(-2, 2, 0), (0, 0, 0)], False),
                             ParticleArray(2, -1, 1, [(1, 2, 3), (1, 2, 3)], [(-2, 2, 0), (0, 0, 0)], False)]
        assert_array_almost_equal(d.binary_electric_field_at_positions(
            [(1, 2, 3), (1, 2, 4), (0, 2, 3), (0, 1, 2)]),
            [(0, 0, 0), (0, 0, -4), (4, 0, 0), (4 / sqrt(27), 4 / sqrt(27), 4 / sqrt(27))])

    @pytest.mark.parametrize('model', ['noninteracting', 'PIC', 'binary'])
    def test_cube_of_gas(self, model, monkeypatch, tmpdir):
        monkeypatch.chdir(tmpdir)
        Config(TimeGridConf(1.0, save_step=.5, step=.1), SpatialMeshConf((10, 10, 10), (1, 1, 1)),
               [ParticleSourceConf('gas', Box(size=(10, 10, 10)), 50, 0, np.zeros(3), 300)],
               particle_interaction_model=ParticleInteractionModelConf(model)
               ).make().start_pic_simulation()

    @pytest.mark.parametrize('model', ['noninteracting', 'PIC', 'binary'])
    def test_cube_of_gas_with_hole(self, model, monkeypatch, tmpdir):
        monkeypatch.chdir(tmpdir)
        Config(TimeGridConf(1.0, save_step=.5, step=.1), SpatialMeshConf((10, 10, 10), (1, 1, 1)),
               [ParticleSourceConf('gas', Box(size=(10, 10, 10)), 50, 0, np.zeros(3), 300)],
               [InnerRegionConf('hole', Box(origin=(4, 4, 4), size=(2, 2, 2)))],
               particle_interaction_model=ParticleInteractionModelConf(model)
               ).make().start_pic_simulation()

    def test_id_generation(self, monkeypatch, tmpdir):
        monkeypatch.chdir(tmpdir)
        conf = Config(TimeGridConf(0.001, save_step=.0005, step=0.0001), SpatialMeshConf((10, 10, 10), (1, 1, 1)),
                      sources=[ParticleSourceConf('gas', Box((4, 4, 4), size=(1, 1, 1)), 50, 0, np.zeros(3), 0.00),
                               ParticleSourceConf('gas2', Box((5, 5, 5), size=(1, 1, 1)), 50, 0, np.zeros(3), 0.00)],
                      particle_interaction_model=ParticleInteractionModelConf('noninteracting')
                      )
        assert len(conf.sources) == 2
        sim = conf.make()
        assert len(sim.particle_sources) == 2
        assert len(sim.particle_arrays) == 0
        sim.start_pic_simulation()
        assert len(sim.particle_sources) == 2
        assert len(sim.particle_arrays) == 1
        assert_array_equal(sim.particle_arrays[0].ids, range(100))

    def test_write(self, monkeypatch, tmpdir, mocker):
        monkeypatch.chdir(tmpdir)
        conf = Config(TimeGridConf(1, 1, .01), SpatialMeshConf((5, 5, 5), (.1, .1, .1)),
                      sources=[ParticleSourceConf('a', Box()),
                               ParticleSourceConf('c', Cylinder()),
                               ParticleSourceConf('d', Tube((0, 0, 0), (0, 0, 1)))],
                      inner_regions=[InnerRegionConf('1', Box(), 1),
                                     InnerRegionConf('2', Sphere(), -2),
                                     InnerRegionConf('3', Cylinder(), 0),
                                     InnerRegionConf('4', Tube((0, 0, 0), (0, 0, 1)), 4)],
                      output_file=OutputFileConf(), boundary_conditions=BoundaryConditionsConf(-2.7),
                      particle_interaction_model=ParticleInteractionModelConf('binary'),
                      external_fields=[ExternalFieldUniformConf('x', 'electric', (-2, -2, 1)),
                                       ExternalFieldExpressionConf('y', 'magnetic',
                                                                   ('0', '0', '3*x + sqrt(y) - z**2'))])
        sim = conf.make()
        sim.write()
        assert tmpdir.join('out_0000000.h5').exists()
        assert tmpdir.join('out_0000000_new.h5').exists()
        sim.time_grid.update_to_next_step()
        sim.time_grid.update_to_next_step()
        sim.write()
        assert not tmpdir.join('out_0000001.h5').exists()
        assert not tmpdir.join('out_0000001_new.h5').exists()
        assert tmpdir.join('out_0000002.h5').exists()
        assert tmpdir.join('out_0000002_new.h5').exists()
        with h5py.File('out_0000002_new.h5', 'r') as h5file:
            sim2 = Simulation.init_from_h5(h5file, 'out_', '.h5')
            assert sim2 == sim
        with h5py.File('out_0000002.h5', 'r') as h5file:
            sim3 = Simulation.import_from_h5(h5file, 'out_', '.h5')
            assert sim3 == sim

    def test_particle_generation(self, mocker):
        conf = Config(TimeGridConf(2, 1, 1), SpatialMeshConf((5, 5, 5), (.1, .1, .1)),
                      sources=[ParticleSourceConf('a', Box((2, 2, 2), (0, 0, 0)), 2, 1, (0, 0, 0), 0, charge=0),
                               ParticleSourceConf('b', Box((1, 1, 1), (0, 0, 0)), 7, 5, (0, 0, 0), 0, charge=0)],
                      inner_regions=[InnerRegionConf('1', Box((.5, .5, .5), (1, 1, 1)))],
                      output_file=OutputFileConf(), boundary_conditions=BoundaryConditionsConf(),
                      particle_interaction_model=ParticleInteractionModelConf('noninteracting'),
                      external_fields=[])
        sim = conf.make()
        sim.start_pic_simulation()
        assert [len(a.ids) for a in sim.particle_arrays] == [4]
