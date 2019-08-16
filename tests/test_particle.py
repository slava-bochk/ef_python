import h5py
import inject
import numpy
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ef.particle_array import ParticleArray
from ef.util.physical_constants import speed_of_light


@pytest.mark.usefixtures('backend')
class TestParticle:
    xp = inject.attr(numpy)
    assert_ae = inject.attr(assert_array_equal)
    assert_almost_ae = inject.attr(assert_array_almost_equal)

    def test_h5(self, tmpdir):
        fname = tmpdir.join('test_particle.h5')
        p1 = ParticleArray(123, -1.0, 2.0, (0., 0., 1.), (1., 0., 2.))
        with h5py.File(fname, mode="w") as h5file:
            p1.save_h5(h5file)
        with h5py.File(fname, mode="r") as h5file:
            p2 = ParticleArray.load_h5(h5file)
        assert p1 == p2

    def test_update_positions(self):
        p = ParticleArray([123], -1.0, 2.0, [(0., 0., 1.)], [(1., 0., 3.)])
        p.update_positions(10.0)
        self.assert_ae(p.positions, [(5., 0., 16.)])
        p = ParticleArray((1, 2), -1.0, 2.0, [(0., 0., 1.), (1, 2, 3)], [(1., 0., 3.), (-1, -0.5, 0)])
        p.update_positions(10.0)
        self.assert_ae(p.positions, [(5., 0., 16.), (-4, -0.5, 3)])

    def test_field_at_point(self):
        p = ParticleArray([1], -16.0, 2.0, [(0., 0., 1.)], [(1., 0., 3.)])
        self.assert_ae(p.field_at_points((2., 0., 1.)), [(-4, 0, 0)])
        self.assert_ae(p.field_at_points((2., 0., 1.)), self.xp.array([(-4, 0, 0)]))
        self.assert_ae(p.field_at_points(self.xp.array((2., 0., 1.))), [(-4, 0, 0)])
        self.assert_ae(p.field_at_points((0., 0., 1.)), self.xp.array([[0, 0, 0]]))
        p = ParticleArray((1, 2), -16.0, 2.0, [(0, 0, 1), (0, 0, 0)], self.xp.zeros((2, 3)))
        self.assert_ae(p.field_at_points((0, 0, 0.5)), [(0, 0, 0)])
        self.assert_ae(p.field_at_points((0, 0, 2)), [(0, 0, -20)])
        self.assert_ae(p.field_at_points((0., 0., 0)), self.xp.array([[0, 0, 16]]))
        self.assert_ae(p.field_at_points([(0, 0, 0.5), (0, 0, 2), (0, 0, 2)]),
                       [(0, 0, 0), (0, 0, -20), (0, 0, -20)])

    def test_update_momentums_no_mgn(self):
        p = ParticleArray(123, -1.0, 2.0, (0., 0., 1.), (1., 0., 3.))
        p.boris_update_momentum_no_mgn(0.1, (-1.0, 2.0, 3.0))
        self.assert_ae(p.momentums, [(1.1, -0.2, 2.7)])

    def test_update_momentums(self):
        p = ParticleArray(123, -1.0, 2.0, (0., 0., 1.), (1., 0., 3.))
        p.boris_update_momentums(0.1, (-1.0, 2.0, 3.0), (0, 0, 0))
        self.assert_ae(p.momentums, [(1.1, -0.2, 2.7)])

        p = ParticleArray(123, -1.0, 2.0, (0., 0., 1.), (1., 0., 3.))
        p.boris_update_momentums(2, (-1.0, 2.0, 3.0), (2 * speed_of_light, 0, 0))
        self.assert_ae(p.momentums, [(3, -2, -5)])

    def test_update_momentums_function(self):
        self.assert_ae(ParticleArray._boris_update_momentums(-1, 2, (1, 0, 3), 0.1, (-1.0, 2.0, 3.0), (0, 0, 0)),
                       (1.1, -0.2, 2.7))
        self.assert_ae(
            ParticleArray._boris_update_momentums(-1, 2, (1, 0, 3), 2, (-1.0, 2.0, 3.0), (2 * speed_of_light, 0, 0)),

            (3, -2, -5))
        self.assert_ae(
            ParticleArray._boris_update_momentums(-1, 2, (1, 0, 3), 2, (-1.0, 2.0, 3.0), (2 * speed_of_light, 0, 0)),

            (3, -2, -5))
        self.assert_ae(
            ParticleArray._boris_update_momentums(charge=-1, mass=2, momentum_arr=self.xp.array([(1, 0, 3)] * 10), dt=2,
                                                  el_field_arr=[(-1.0, 2.0, 3.0)] * 10,
                                                  mgn_field_arr=[(2 * speed_of_light, 0, 0)] * 10),
            self.xp.array([(3, -2, -5)] * 10))
