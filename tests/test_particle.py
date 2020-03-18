import inject
import numpy
import pytest

from ef.particle_array import ParticleArray
from ef.util.physical_constants import speed_of_light
from ef.util.testing import assert_array_equal


@pytest.mark.usefixtures('backend')
class TestParticle:
    xp = inject.attr(numpy)

    def test_update_positions(self):
        p = ParticleArray([123], -1.0, 2.0, [(0., 0., 1.)], [(1., 0., 3.)])
        p.update_positions(10.0)
        assert_array_equal(p.positions, [(5., 0., 16.)])
        p = ParticleArray((1, 2), -1.0, 2.0, [(0., 0., 1.), (1, 2, 3)], [(1., 0., 3.), (-1, -0.5, 0)])
        p.update_positions(10.0)
        assert_array_equal(p.positions, [(5., 0., 16.), (-4, -0.5, 3)])

    def test_field_at_point(self):
        p = ParticleArray([1], -16.0, 2.0, [(0., 0., 1.)], [(1., 0., 3.)])
        assert_array_equal(p.field_at_points((2., 0., 1.)), [(-4, 0, 0)])
        assert_array_equal(p.field_at_points((2., 0., 1.)), self.xp.array([(-4, 0, 0)]))
        assert_array_equal(p.field_at_points(self.xp.array((2., 0., 1.))), [(-4, 0, 0)])
        assert_array_equal(p.field_at_points((0., 0., 1.)), self.xp.array([[0, 0, 0]]))
        p = ParticleArray((1, 2), -16.0, 2.0, [(0, 0, 1), (0, 0, 0)], self.xp.zeros((2, 3)))
        assert_array_equal(p.field_at_points((0, 0, 0.5)), [(0, 0, 0)])
        assert_array_equal(p.field_at_points((0, 0, 2)), [(0, 0, -20)])
        assert_array_equal(p.field_at_points((0., 0., 0)), self.xp.array([[0, 0, 16]]))
        assert_array_equal(p.field_at_points([(0, 0, 0.5), (0, 0, 2), (0, 0, 2)]),
                           [(0, 0, 0), (0, 0, -20), (0, 0, -20)])

    def test_update_momentums_no_mgn(self):
        p = ParticleArray(123, -1.0, 2.0, (0., 0., 1.), (1., 0., 3.))
        p.boris_update_momentum_no_mgn(0.1, (-1.0, 2.0, 3.0))
        assert_array_equal(p.momentums, [(1.1, -0.2, 2.7)])

    def test_update_momentums(self):
        p = ParticleArray(123, -1.0, 2.0, (0., 0., 1.), (1., 0., 3.))
        p.boris_update_momentums(0.1, (-1.0, 2.0, 3.0), (0, 0, 0))
        assert_array_equal(p.momentums, [(1.1, -0.2, 2.7)])

        p = ParticleArray(123, -1.0, 2.0, (0., 0., 1.), (1., 0., 3.))
        p.boris_update_momentums(2, (-1.0, 2.0, 3.0), (2 * speed_of_light, 0, 0))
        assert_array_equal(p.momentums, [(3, -2, -5)])

    def test_update_momentums_function(self):
        assert_array_equal(ParticleArray._boris_update_momentums(-1, 2, (1, 0, 3), 0.1, (-1.0, 2.0, 3.0), (0, 0, 0)),
                           (1.1, -0.2, 2.7))
        assert_array_equal(
            ParticleArray._boris_update_momentums(-1, 2, (1, 0, 3), 2, (-1.0, 2.0, 3.0), (2 * speed_of_light, 0, 0)),

            (3, -2, -5))
        assert_array_equal(
            ParticleArray._boris_update_momentums(-1, 2, (1, 0, 3), 2, (-1.0, 2.0, 3.0), (2 * speed_of_light, 0, 0)),

            (3, -2, -5))
        assert_array_equal(
            ParticleArray._boris_update_momentums(charge=-1, mass=2, momentum_arr=self.xp.array([(1, 0, 3)] * 10), dt=2,
                                                  el_field_arr=[(-1.0, 2.0, 3.0)] * 10,
                                                  mgn_field_arr=[(2 * speed_of_light, 0, 0)] * 10),
            self.xp.array([(3, -2, -5)] * 10))
