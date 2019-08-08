import inject
import numpy
from numpy.core._multiarray_umath import normalize_axis_index

from ef.util import safe_default_inject
from ef.util.physical_constants import speed_of_light
from ef.util.serializable_h5 import SerializableH5


class ParticleArray(SerializableH5):
    xp = inject.attr(numpy)

    @safe_default_inject
    def __init__(self, ids, charge, mass, positions, momentums, momentum_is_half_time_step_shifted=False):
        self.ids = self.xp.asarray(ids)
        self.charge = charge
        self.mass = mass
        self.positions = self.xp.asarray(positions).reshape((-1, 3))
        self.momentums = self.xp.asarray(momentums).reshape((-1, 3))
        self.momentum_is_half_time_step_shifted = momentum_is_half_time_step_shifted

    @property
    def dict(self):
        d = super().dict
        if self.xp is not numpy:
            d['ids'] = d['ids'].get()
            d['positions'] = d['positions'].get()
            d['momentums'] = d['momentums'].get()
        return d

    def keep(self, mask):
        mask = self.xp.asarray(mask)
        self.ids = self.ids[mask]
        self.positions = self.positions[mask]
        self.momentums = self.momentums[mask]

    def remove(self, mask):
        mask = self.xp.asarray(mask)
        self.keep(self.xp.logical_not(mask))

    def update_positions(self, dt):
        self.positions += dt / self.mass * self.momentums

    def field_at_points(self, points):
        diff = self.xp.asarray(points) - self.positions[:, self.xp.newaxis, :]  # (m, n, 3)
        dist = self.xp.linalg.norm(diff, axis=-1)  # (m, n)
        dist[dist == 0] = 1.
        force = diff / (dist ** 3)[..., self.xp.newaxis]  # (m, n, 3)
        return self.charge * self.xp.sum(force, axis=0)  # (n, 3)

    def boris_update_momentums(self, dt, total_el_field, total_mgn_field):
        self.momentums = self._boris_update_momentums(self.charge, self.mass, self.momentums, dt, total_el_field,
                                                     total_mgn_field)

    def boris_update_momentum_no_mgn(self, dt, total_el_field):
        self.momentums += self.charge * dt * self.xp.asarray(total_el_field)

    @classmethod
    def import_h5(cls, g):
        ga = g.attrs
        return cls(ids=g['particle_id'], charge=float(ga['charge']), mass=float(ga['mass']),
                   positions=cls.xp.moveaxis(
                       cls.xp.array([g['position_{}'.format(c)] for c in 'xyz']),
                       0, -1),
                   momentums=cls.xp.moveaxis(
                       cls.xp.array([g['momentum_{}'.format(c)] for c in 'xyz']),
                       0, -1),
                   momentum_is_half_time_step_shifted=True)

    def export_h5(self, g):
        g['particle_id'] = self.dict['ids']
        g.attrs['max_id'] = self.dict['ids'].max(initial=-1)
        for i, c in enumerate('xyz'):
            g['position_{}'.format(c)] = self.dict['positions'][:, i]
            g['momentum_{}'.format(c)] = self.dict['momentums'][:, i]

    @classmethod
    def _boris_update_momentums(cls, charge, mass, momentum_arr, dt, el_field_arr, mgn_field_arr):
        momentum_arr = cls.xp.asarray(momentum_arr)
        el_field_arr = cls.xp.asarray(el_field_arr)
        mgn_field_arr = cls.xp.asarray(mgn_field_arr)
        q_quote = dt * charge / mass / 2.0  # scalar. how easy is it to move? dt * Q / m /2
        half_el_force = cls.xp.asarray(el_field_arr) * q_quote  # (n, 3) half the dv caused by electric field
        v_current = momentum_arr / mass  # (n, 3) current velocity (at -1/2 dt already)
        u = v_current + half_el_force  # (n, 3) v_minus
        h = cls.xp.array(mgn_field_arr) * (q_quote / speed_of_light)  # (n, 3)
        # rotation vector t = qB/m * dt/2
        s = h * (2.0 / (1.0 + cls.xp.sum(h * h, -1)))[..., cls.xp.newaxis]  # (n, 3) rotation vector s = 2t / (1 + t**2)
        tmp = u + cls.cross(u, h)  # (n, 3) v_prime is v_minus rotated by t
        u_quote = u + cls.cross(tmp, s)  # (n, 3) v_plus = v_minus + v_prime * s
        return (u_quote + half_el_force) * mass  # (n, 3) finally add the other half-velocity

    @classmethod
    def cross(cls, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        if axis is not None:
            axisa, axisb, axisc = (axis,) * 3
        a = cls.xp.asarray(a)
        b = cls.xp.asarray(b)
        # Check axisa and axisb are within bounds
        axisa = numpy.core._multiarray_umath.normalize_axis_index(axisa, a.ndim, msg_prefix='axisa')
        axisb = numpy.core._multiarray_umath.normalize_axis_index(axisb, b.ndim, msg_prefix='axisb')

        # Move working axis to the end of the shape
        a = cls.xp.moveaxis(a, axisa, -1)
        b = cls.xp.moveaxis(b, axisb, -1)
        msg = ("incompatible dimensions for cross product\n"
               "(dimension must be 2 or 3)")
        if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
            raise ValueError(msg)

        # Create the output array
        shape = cls.xp.broadcast(a[..., 0], b[..., 0]).shape
        if a.shape[-1] == 3 or b.shape[-1] == 3:
            shape += (3,)
            # Check axisc is within bounds
            axisc = numpy.core._multiarray_umath.normalize_axis_index(axisc, len(shape), msg_prefix='axisc')
        dtype = cls.xp.promote_types(a.dtype, b.dtype)
        cp = cls.xp.empty(shape, dtype)

        # create local aliases for readability
        a0 = a[..., 0]
        a1 = a[..., 1]
        if a.shape[-1] == 3:
            a2 = a[..., 2]
        b0 = b[..., 0]
        b1 = b[..., 1]
        if b.shape[-1] == 3:
            b2 = b[..., 2]
        if cp.ndim != 0 and cp.shape[-1] == 3:
            cp0 = cp[..., 0]
            cp1 = cp[..., 1]
            cp2 = cp[..., 2]

        if a.shape[-1] == 2:
            if b.shape[-1] == 2:
                # a0 * b1 - a1 * b0
                cls.xp.multiply(a0, b1, out=cp)
                cp -= a1 * b0
                return cp
            else:
                assert b.shape[-1] == 3
                # cp0 = a1 * b2 - 0  (a2 = 0)
                # cp1 = 0 - a0 * b2  (a2 = 0)
                # cp2 = a0 * b1 - a1 * b0
                cls.xp.multiply(a1, b2, out=cp0)
                cls.xp.multiply(a0, b2, out=cp1)
                cls.xp.negative(cp1, out=cp1)
                cls.xp.multiply(a0, b1, out=cp2)
                cp2 -= a1 * b0
        else:
            assert a.shape[-1] == 3
            if b.shape[-1] == 3:
                # cp0 = a1 * b2 - a2 * b1
                # cp1 = a2 * b0 - a0 * b2
                # cp2 = a0 * b1 - a1 * b0
                cls.xp.multiply(a1, b2, out=cp0)
                tmp = cls.xp.array(a2 * b1)
                cp0 -= tmp
                cls.xp.multiply(a2, b0, out=cp1)
                cls.xp.multiply(a0, b2, out=tmp)
                cp1 -= tmp
                cls.xp.multiply(a0, b1, out=cp2)
                cls.xp.multiply(a1, b0, out=tmp)
                cp2 -= tmp
            else:
                assert b.shape[-1] == 2
                # cp0 = 0 - a2 * b1  (b2 = 0)
                # cp1 = a2 * b0 - 0  (b2 = 0)
                # cp2 = a0 * b1 - a1 * b0
                cls.xp.multiply(a2, b1, out=cp0)
                cls.xp.negative(cp0, out=cp0)
                cls.xp.multiply(a2, b0, out=cp1)
                cls.xp.multiply(a0, b1, out=cp2)
                cp2 -= a1 * b0

        return cls.xp.moveaxis(cp, -1, axisc)
