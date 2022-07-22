import inject
import numpy as np
import rowan
from numpy.linalg import norm

from ef.config.component import ConfigComponent
from ef.util.serializable_h5 import SerializableH5
from ef.util.vector import vector

__all__ = ['Shape', 'Box', 'Cylinder', 'Tube', 'Sphere', 'Cone']


class Shape(ConfigComponent, SerializableH5):
    @inject.params(xp=np)
    def __init__(self, xp=np):
        self.xp = xp

    @property
    def dict(self):
        d = super().dict
        del d['xp']
        return d

    def visualize(self, visualizer, **kwargs):
        raise NotImplementedError()

    def are_positions_inside(self, positions):
        raise NotImplementedError()

    def generate_uniform_random_position(self, random_state):
        return self.generate_uniform_random_posititons(random_state, 1)[0]

    def generate_uniform_random_posititons(self, random_state, n):
        raise NotImplementedError()

    @staticmethod
    def import_h5(g, region):
        ga = g.attrs
        if region:
            gt = ga['object_type']
            if gt == b'box':
                origin = np.array([ga['x_right'], ga['y_bottom'], ga['z_near']])
                size = np.array([ga['x_left'], ga['y_top'], ga['z_far']]) - origin
                return Box(origin, size)
            elif gt == b'sphere':
                return Sphere([ga['origin_{}'.format(c)] for c in 'xyz'], ga['radius'])
            elif gt == b'cylinder':
                start = [ga['axis_start_{}'.format(c)] for c in 'xyz']
                end = [ga['axis_end_{}'.format(c)] for c in 'xyz']
                return Cylinder(start, end, ga['radius'])
            elif gt == b'tube':
                start = [ga['axis_start_{}'.format(c)] for c in 'xyz']
                end = [ga['axis_end_{}'.format(c)] for c in 'xyz']
                r, R = (ga['{}_radius'.format(s)] for s in ('inner', 'outer'))
                return Tube(start, end, r, R)
        else:
            gt = ga['geometry_type']
            if gt == b'box':
                origin = np.array([ga['box_x_right'], ga['box_y_bottom'], ga['box_z_near']]).reshape(3)
                size = np.array([ga['box_x_left'], ga['box_y_top'], ga['box_z_far']]).reshape(3) - origin
                return Box(origin, size)
            elif gt == b'cylinder':
                start = np.array([ga['cylinder_axis_start_{}'.format(c)] for c in 'xyz']).reshape(3)
                end = np.array([ga['cylinder_axis_end_{}'.format(c)] for c in 'xyz']).reshape(3)
                return Cylinder(start, end, ga['cylinder_radius'])
            elif gt == b'tube_along_z':
                x, y = (ga['tube_along_z_axis_{}'.format(c)] for c in 'xy')
                sz = ga['tube_along_z_axis_start_z']
                ez = ga['tube_along_z_axis_end_z']
                r, R = (ga['tube_along_z_{}_radius'.format(s)] for s in ('inner', 'outer'))
                return Tube((x, y, sz), (x, y, ez), r, R)


def rotation_from_z(vector):
    """
    Find a quaternion that rotates z-axis into a given vector.
    :param vector: Any non-zero 3-component vector
    :return: Array of length 4 with the rotation quaternion
    """
    axis = np.cross((0, 0, 1), vector)
    if norm(axis) == 0:
        return np.array((1, 0, 0, 0))
    cos2 = (vector / norm(vector))[2]
    cos = np.sqrt((1 + cos2) / 2)
    sin = np.sqrt((1 - cos2) / 2)
    vector_component = axis / norm(axis) * sin
    return np.concatenate(([cos], vector_component))


class Box(Shape):
    def __init__(self, origin=(0, 0, 0), size=(1, 1, 1)):
        super().__init__()
        self.origin = vector(origin)
        self.size = vector(size)
        self._origin = self.xp.asarray(self.origin)
        self._size = self.xp.asarray(self.size)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_box(self.size, self.origin, **kwargs)

    def are_positions_inside(self, positions):
        positions = self.xp.asarray(positions)
        return self.xp.logical_and(self.xp.all(positions >= self._origin, axis=-1),
                                   self.xp.all(positions <= self._origin + self._size, axis=-1))

    def generate_uniform_random_posititons(self, random_state, n):
        return random_state.uniform(self.origin, self.origin + self.size, (n, 3))

    def export_h5(self, g, region):
        if region:
            g.attrs['x_right'] = self.origin[0]
            g.attrs['x_left'] = self.origin[0] + self.size[0]
            g.attrs['y_bottom'] = self.origin[1]
            g.attrs['y_top'] = self.origin[1] + self.size[1]
            g.attrs['z_near'] = self.origin[2]
            g.attrs['z_far'] = self.origin[2] + self.size[2]
            g.attrs['object_type'] = np.string_(b"box")
        else:
            g.attrs['box_x_right'] = self.origin[0]
            g.attrs['box_x_left'] = self.origin[0] + self.size[0]
            g.attrs['box_y_bottom'] = self.origin[1]
            g.attrs['box_y_top'] = self.origin[1] + self.size[1]
            g.attrs['box_z_near'] = self.origin[2]
            g.attrs['box_z_far'] = self.origin[2] + self.size[2]
            g.attrs['geometry_type'] = np.string_(b"box")


class Cylinder(Shape):
    def __init__(self, start=(0, 0, 0), end=(1, 0, 0), radius=1):
        super().__init__()
        self.start = vector(start)
        self.end = vector(end)
        self._start = self.xp.asarray(self.start)
        self._end = self.xp.asarray(self.end)
        self.radius = float(radius)
        self._rotation = rotation_from_z(self.end - self.start)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_cylinder(self.start, self.end, self.radius, **kwargs)

    def are_positions_inside(self, positions):
        positions = self.xp.asarray(positions)
        pointvec = positions - self._start
        axisvec = self._end - self._start
        axis = self.xp.linalg.norm(axisvec)
        unit_axisvec = axisvec / axis
        # for one-point case, dot would return a scalar, so it's cast to array explicitly
        projection = self.xp.asarray(self.xp.dot(pointvec, unit_axisvec))
        perp_to_axis = self.xp.linalg.norm(pointvec -
                                           unit_axisvec[self.xp.newaxis] * projection[..., self.xp.newaxis], axis=-1)
        result = self.xp.logical_and(perp_to_axis <= self.radius,
                                     self.xp.logical_and(0 <= projection, projection <= axis))
        return result

    def generate_uniform_random_posititons(self, random_state, n):
        r = np.sqrt(random_state.uniform(0.0, 1.0, n)) * self.radius
        phi = random_state.uniform(0.0, 2.0 * np.pi, n)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = random_state.uniform(0.0, norm(self.end - self.start), n)
        points = np.stack((x, y, z), -1)
        return rowan.rotate(self._rotation, points) + self.start

    def export_h5(self, g, region):
        if region:
            g.attrs['radius'] = self.radius
            for i, c in enumerate('xyz'):
                g.attrs['axis_start_{}'.format(c)] = self.start[i]
                g.attrs['axis_end_{}'.format(c)] = self.end[i]
            g.attrs['object_type'] = np.string_(b"cylinder")
        else:
            g.attrs['cylinder_radius'] = self.radius
            for i, c in enumerate('xyz'):
                g.attrs['cylinder_axis_start_{}'.format(c)] = self.start[i]
                g.attrs['cylinder_axis_end_{}'.format(c)] = self.end[i]
            g.attrs['geometry_type'] = np.string_(b"cylinder")


class Tube(Shape):
    def __init__(self, start=(0, 0, 0), end=(1, 0, 0), inner_radius=1, outer_radius=2):
        super().__init__()
        self.start = vector(start)
        self.end = vector(end)
        self._start = self.xp.asarray(self.start)
        self._end = self.xp.asarray(self.end)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self._rotation = rotation_from_z(self.end - self.start)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_tube(self.start, self.end, self.inner_radius, self.outer_radius, **kwargs)

    def are_positions_inside(self, positions):
        positions = self.xp.asarray(positions)
        pointvec = positions - self._start
        axisvec = self._end - self._start
        axis = self.xp.linalg.norm(axisvec)
        unit_axisvec = axisvec / axis
        # for one-point case, dot would return a scalar, so it's cast to array explicitly
        projection = self.xp.asarray(self.xp.dot(pointvec, unit_axisvec))
        perp_to_axis = self.xp.linalg.norm(pointvec - unit_axisvec[np.newaxis] * projection[..., np.newaxis], axis=-1)
        and_ = self.xp.logical_and
        return and_(and_(0 <= projection, projection <= axis),
                    and_(self.inner_radius <= perp_to_axis, perp_to_axis <= self.outer_radius))

    def generate_uniform_random_posititons(self, random_state, n):
        r = np.sqrt(random_state.uniform(self.inner_radius / self.outer_radius, 1.0, n)) * self.outer_radius
        phi = random_state.uniform(0.0, 2.0 * np.pi, n)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = random_state.uniform(0.0, norm(self.end - self.start), n)
        points = np.stack((x, y, z), -1)
        return rowan.rotate(self._rotation, points) + self.start

    def export_h5(self, g, region):
        if region:
            g.attrs['inner_radius'] = self.inner_radius
            g.attrs['outer_radius'] = self.outer_radius
            for i, c in enumerate('xyz'):
                g.attrs['axis_start_{}'.format(c)] = self.start[i]
                g.attrs['axis_end_{}'.format(c)] = self.end[i]
            g.attrs['object_type'] = np.string_(b"tube")
        else:
            if np.any((self.start != self.end)[:2]):
                raise ValueError('Cannot export tube particle source not along z-axis')
            g.attrs['tube_along_z_inner_radius'] = self.inner_radius
            g.attrs['tube_along_z_outer_radius'] = self.outer_radius
            g.attrs['tube_along_z_axis_x'] = self.start[0]
            g.attrs['tube_along_z_axis_y'] = self.start[1]
            g.attrs['tube_along_z_axis_start_z'] = self.start[2]
            g.attrs['tube_along_z_axis_end_z'] = self.end[2]
            g.attrs['geometry_type'] = np.string_(b"tube_along_z")


class Sphere(Shape):
    def __init__(self, origin=(0, 0, 0), radius=1):
        super().__init__()
        self.origin = vector(origin)
        self._origin = self.xp.asarray(self.origin)
        self.radius = float(radius)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_sphere(self.origin, self.radius, **kwargs)

    def are_positions_inside(self, positions):
        positions = self.xp.asarray(positions)
        return self.xp.linalg.norm(positions - self._origin, axis=-1) <= self.radius

    def generate_uniform_random_posititons(self, random_state, n):
        while True:
            p = random_state.uniform(-1, 1, (n * 2, 3)) * self.radius + self.origin
            mask = self.are_positions_inside(p)
            p = p.compress(mask.get() if hasattr(mask, 'get') else mask, 0)
            if len(p) > n:
                return p[:n]

    def export_h5(self, g, region):
        if region:
            g.attrs['radius'] = self.radius
            for i, c in enumerate('xyz'):
                g.attrs['origin_{}'.format(c)] = self.origin[i]
            g.attrs['object_type'] = np.string_(b"sphere")
        else:
            raise ValueError('Cannot export spherical particle source')


class Cone(Shape):
    def __init__(self, start=(0, 0, 0, 1),
                 start_radii=(1, 2), end_radii=(3, 4)):
        super().__init__()
        self.start = np.array(start, np.float)
        self.start_radii = np.array(start_radii, np.float)
        self.end_radii = np.array(end_radii, np.float)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_cone(self.start, self.end,
                             self.start_radii, self.end_radii, **kwargs)

# TODO: def are_positions_inside(self, point)
