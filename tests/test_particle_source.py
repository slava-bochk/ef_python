from io import BytesIO
from math import sqrt

import h5py
import numpy as np
from numpy.random.mtrand import RandomState
from numpy.testing import assert_array_equal

from ef.config.components import Box
from ef.particle_array import ParticleArray
from ef.particle_source import ParticleSource


class TestParticleSource:
    def test_init(self):
        p = ParticleSource()
        assert p.name == "particle_source"
        assert p.shape == Box()
        assert p.initial_number_of_particles == 0
        assert p.particles_to_generate_each_step == 0
        assert_array_equal(p.mean_momentum, np.zeros(3))
        assert p.temperature == 0
        assert p.charge == 0
        assert p.mass == 1

    def test_generate_for_simulation(self, backend):
        ps = ParticleSource('test', Box(6, 0), 17, 13, (4, 4, 4), 0, -2, 6)
        ps.generate_initial_particles().assert_eq(
            ParticleArray(range(17), -2, 6, np.full((17, 3), 6), np.full((17, 3), 4), False))
        ps.generate_each_step().assert_eq(
            ParticleArray(range(13), -2, 6, np.full((13, 3), 6), np.full((13, 3), 4), False))

    def test_generate_particles(self, backend):
        ps = ParticleSource('test', Box((1., 2., 3.), 0), 17, 13, (-2, 3, 1), 0, -2, 6)
        ps.generate_num_of_particles(3).assert_eq(
            ParticleArray(range(3), -2, 6, [(1, 2, 3)] * 3, [(-2, 3, 1)] * 3, False))
        ps.generate_num_of_particles(1).assert_eq(
            ParticleArray([0], -2, 6, [(1, 2, 3)], [(-2, 3, 1)], False))
        ps.generate_num_of_particles(0).assert_eq(
            ParticleArray([], -2, 6, np.empty((0, 3)), np.empty((0, 3)), False))

    def test_generate_positions(self, backend):
        ps = ParticleSource()
        ps._generator = RandomState(123)
        p = ps.generate_num_of_particles(100)
        assert_ae = p.xp.testing.assert_array_equal
        assert_ae(p.positions, Box().generate_uniform_random_posititons(RandomState(123), 100))

    def test_generate_momentums(self, backend):
        ps = ParticleSource(mean_momentum=(3, -2, 0), temperature=10, mass=.3)
        ps._generator = RandomState(123)
        p = ps.generate_num_of_particles(1000000)
        assert_almost_ae = p.xp.testing.assert_array_almost_equal
        assert_almost_ae(p.momentums.mean(axis=0), (3, -2, 0), 2)
        assert_almost_ae(((p.momentums - p.xp.array([3, -2, 0])).std(axis=0)), np.full(3, sqrt(3)), 2)

    def test_write_h5(self):
        f = BytesIO()
        p1 = ParticleSource()
        with h5py.File(f, mode="w") as h5file:
            p1.save_h5(h5file)
        with h5py.File(f, mode="r") as h5file:
            p2 = ParticleSource.load_h5(h5file)
        p1.assert_eq(p2)

    def test_export_h5(self):
        f = BytesIO()
        p1 = ParticleSource()
        with h5py.File(f, mode="w") as h5file:
            p1.export_h5(h5file.create_group(p1.name))
        with h5py.File(f, mode="r") as h5file:
            p2 = ParticleSource.import_h5(h5file[p1.name])
        p1.assert_eq(p2)
