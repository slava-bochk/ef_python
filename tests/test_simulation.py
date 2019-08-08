from configparser import ConfigParser
from typing import Type

import inject
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ef.config.components import *
from ef.config.config import Config
from ef.field import FieldZero
from ef.field.expression import FieldExpression
from ef.field.on_grid import FieldOnGrid
from ef.field.uniform import FieldUniform
from ef.inner_region import InnerRegion
from ef.meshgrid import MeshGrid
from ef.particle_interaction_model import Model
from ef.runner import Runner
from ef.time_grid import TimeGrid
from ef.util.array_on_grid import ArrayOnGrid


class TestSimulation:
    Array: Type[ArrayOnGrid] = inject.attr(ArrayOnGrid)

    def test_init_from_config(self, backend):
        efconf = Config()
        parser = ConfigParser()
        parser.read_string(efconf.export_to_string())
        sim = Config.from_configparser(parser).make()
        assert sim.time_grid == TimeGrid(100, 1, 10)
        g = MeshGrid(10, 11)
        assert sim.mesh == g
        assert sim.potential == self.Array(g)
        assert sim.charge_density == self.Array(g)
        assert sim.electric_field == FieldOnGrid('spatial_mesh', 'electric', self.Array(g, 3))
        assert sim.inner_regions == []
        assert sim.particle_sources == []
        assert sim.electric_fields == FieldZero('ZeroSum', 'electric')
        assert sim.magnetic_fields == FieldZero('ZeroSum', 'magnetic')
        assert sim.particle_interaction_model == Model.PIC

    def test_all_config(self, backend):
        efconf = Config(TimeGridConf(200, 20, 2), SpatialMeshConf((5, 5, 5), (.1, .1, .1)),
                        sources=[ParticleSourceConf('a', Box()),
                                 ParticleSourceConf('c', Cylinder()),
                                 ParticleSourceConf('d', Tube(start=(0, 0, 0), end=(0, 0, 1)))],
                        inner_regions=[InnerRegionConf('1', Box(), 1),
                                       InnerRegionConf('2', Sphere(), -2),
                                       InnerRegionConf('3', Cylinder(), 0),
                                       InnerRegionConf('4', Tube(), 4)],
                        output_file=OutputFileConf(), boundary_conditions=BoundaryConditionsConf(-2.7),
                        particle_interaction_model=ParticleInteractionModelConf('binary'),
                        external_fields=[ExternalElectricFieldUniformConf('x', (-2, -2, 1)),
                                         ExternalMagneticFieldExpressionConf('y',
                                                                             ('0', '0',
                                                                              '3*x + sqrt(y) - z**2'))])

        parser = ConfigParser()
        parser.read_string(efconf.export_to_string())
        conf = Config.from_configparser(parser)
        sim = conf.make()
        assert sim.time_grid == TimeGrid(200, 2, 20)
        assert sim.mesh == MeshGrid(5, 51)
        assert sim.electric_field == FieldOnGrid('spatial_mesh', 'electric', self.Array(sim.mesh, 3))
        assert sim.charge_density == self.Array(sim.mesh)
        expected = np.full((51, 51, 51), -2.7)
        expected[1:-1, 1:-1, 1:-1] = 0
        assert sim.potential == self.Array(sim.mesh, (), expected)
        assert sim.inner_regions == [InnerRegion('1', Box(), 1),
                                     InnerRegion('2', Sphere(), -2),
                                     InnerRegion('3', Cylinder(), 0),
                                     InnerRegion('4', Tube(), 4)]
        assert sim.particle_sources == [ParticleSourceConf('a', Box()).make(),
                                        ParticleSourceConf('c', Cylinder()).make(),
                                        ParticleSourceConf('d', Tube(start=(0, 0, 0), end=(0, 0, 1))).make()]
        assert sim.electric_fields == FieldUniform('x', 'electric', np.array((-2, -2, 1)))
        assert sim.magnetic_fields == FieldExpression('y', 'magnetic', '0', '0', '3*x + sqrt(y) - z**2')
        assert sim.particle_interaction_model == Model.binary

    @pytest.mark.parametrize('model', ['noninteracting', 'PIC', 'binary'])
    def test_cube_of_gas(self, model, backend_and_solver):
        sim = Config(TimeGridConf(1.0, save_step=.5, step=.1), SpatialMeshConf((10, 10, 10), (1, 1, 1)),
                     [ParticleSourceConf('gas', Box(size=(10, 10, 10)), 50, 0, np.zeros(3), 300)],
                     particle_interaction_model=ParticleInteractionModelConf(model)
                     ).make()
        Runner(sim).start()

    @pytest.mark.parametrize('model', ['noninteracting', 'PIC', 'binary'])
    def test_cube_of_gas_with_hole(self, model, backend_and_solver):
        sim = Config(TimeGridConf(1.0, save_step=.5, step=.1), SpatialMeshConf((10, 10, 10), (1, 1, 1)),
                     [ParticleSourceConf('gas', Box(size=(10, 10, 10)), 50, 0, np.zeros(3), 300)],
                     [InnerRegionConf('hole', Box(origin=(4, 4, 4), size=(2, 2, 2)))],
                     particle_interaction_model=ParticleInteractionModelConf(model)
                     ).make()
        Runner(sim).start()

    def test_id_generation(self, backend_and_solver):
        conf = Config(TimeGridConf(0.001, save_step=.0005, step=0.0001), SpatialMeshConf((10, 10, 10), (1, 1, 1)),
                      sources=[ParticleSourceConf('gas', Box((4, 4, 4), size=(1, 1, 1)), 50, 0, np.zeros(3), 0.00),
                               ParticleSourceConf('gas2', Box((5, 5, 5), size=(1, 1, 1)), 50, 0, np.zeros(3), 0.00)],
                      particle_interaction_model=ParticleInteractionModelConf('noninteracting'))
        assert len(conf.sources) == 2
        sim = conf.make()
        assert len(sim.particle_sources) == 2
        assert len(sim.particle_arrays) == 0
        Runner(sim).start()
        assert len(sim.particle_sources) == 2
        assert len(sim.particle_arrays) == 1
        sim.particle_arrays[0].xp.testing.assert_array_equal(sim.particle_arrays[0].ids, range(100))

    def test_particle_generation(self, monkeypatch, tmpdir, backend_and_solver):
        monkeypatch.chdir(tmpdir)
        conf = Config(TimeGridConf(2, 1, 1), SpatialMeshConf((5, 5, 5), (.1, .1, .1)),
                      sources=[ParticleSourceConf('a', Box((2, 2, 2), (0, 0, 0)), 2, 1, (0, 0, 0), 0, charge=0),
                               ParticleSourceConf('b', Box((1, 1, 1), (0, 0, 0)), 7, 5, (0, 0, 0), 0, charge=0)],
                      inner_regions=[InnerRegionConf('1', Box((.5, .5, .5), (1, 1, 1)))],
                      output_file=OutputFileConf(), boundary_conditions=BoundaryConditionsConf(),
                      particle_interaction_model=ParticleInteractionModelConf('noninteracting'),
                      external_fields=[])
        sim = conf.make()
        Runner(sim).start()
        assert [len(a.ids) for a in sim.particle_arrays] == [4]
