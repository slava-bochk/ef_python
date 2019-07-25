import numpy as np
import rowan
from numpy.linalg import norm

from ef.config.component import ConfigComponent
from ef.util.serializable_h5 import SerializableH5

__all__ = ['Shape', 'Box', 'Cylinder', 'Tube', 'Sphere', 'Cone']


class Shape(ConfigComponent, SerializableH5):
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
        self.origin = np.array(origin, np.float)
        self.size = np.array(size, np.float)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_box(self.size, self.origin, **kwargs)

    def are_positions_inside(self, positions):
        return np.logical_and(np.all(positions >= self.origin, axis=-1),
                              np.all(positions <= self.origin + self.size, axis=-1))

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
            g.attrs['object_type'] = np.string_(b"box\x00")
        else:
            g.attrs['box_x_right'] = self.origin[0]
            g.attrs['box_x_left'] = self.origin[0] + self.size[0]
            g.attrs['box_y_bottom'] = self.origin[1]
            g.attrs['box_y_top'] = self.origin[1] + self.size[1]
            g.attrs['box_z_near'] = self.origin[2]
            g.attrs['box_z_far'] = self.origin[2] + self.size[2]
            g.attrs['geometry_type'] = np.string_(b"box\x00")


class Cylinder(Shape):
    def __init__(self, start=(0, 0, 0), end=(1, 0, 0), radius=1):
        self.start = np.array(start, np.float)
        self.end = np.array(end, np.float)
        self.radius = float(radius)
        self._rotation = rotation_from_z(self.end - self.start)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_cylinder(self.start, self.end, self.radius, **kwargs)

    def are_positions_inside(self, positions):
        pointvec = positions - self.start
        axisvec = self.end - self.start
        axis = norm(axisvec)
        unit_axisvec = axisvec / axis
        # for one-point case, dot would return a scalar, so it's cast to array explicitly
        projection = np.asarray(np.dot(pointvec, unit_axisvec))
        perp_to_axis = norm(pointvec - unit_axisvec[np.newaxis] * projection[..., np.newaxis], axis=-1)
        result = np.logical_and.reduce([0 <= projection, projection <= axis, perp_to_axis <= self.radius])
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
            g.attrs['object_type'] = np.string_(b"cylinder\x00")
        else:
            g.attrs['cylinder_radius'] = self.radius
            for i, c in enumerate('xyz'):
                g.attrs['cylinder_axis_start_{}'.format(c)] = self.start[i]
                g.attrs['cylinder_axis_end_{}'.format(c)] = self.end[i]
            g.attrs['geometry_type'] = np.string_(b"cylinder\x00")


class Tube(Shape):
    def __init__(self, start=(0, 0, 0), end=(1, 0, 0), inner_radius=1, outer_radius=2):
        self.start = np.array(start, np.float)
        self.end = np.array(end, np.float)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self._rotation = rotation_from_z(self.end - self.start)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_tube(self.start, self.end, self.inner_radius, self.outer_radius, **kwargs)

    def are_positions_inside(self, positions):
        pointvec = positions - self.start
        axisvec = self.end - self.start
        axis = norm(axisvec)
        unit_axisvec = axisvec / axis
        # for one-point case, dot would return a scalar, so it's cast to array explicitly
        projection = np.asarray(np.dot(pointvec, unit_axisvec))
        perp_to_axis = norm(pointvec - unit_axisvec[np.newaxis] * projection[..., np.newaxis], axis=-1)
        return np.logical_and.reduce(
            [0 <= projection, projection <= axis, self.inner_radius <= perp_to_axis, perp_to_axis <= self.outer_radius])

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
            g.attrs['object_type'] = np.string_(b"tube\x00")
        else:
            if np.any((self.start != self.end)[:2]):
                raise ValueError('Cannot export tube particle source not along z-axis')
            g.attrs['tube_along_z_inner_radius'] = self.inner_radius
            g.attrs['tube_along_z_outer_radius'] = self.outer_radius
            g.attrs['tube_along_z_axis_x'] = self.start[0]
            g.attrs['tube_along_z_axis_y'] = self.start[1]
            g.attrs['tube_along_z_axis_start_z'] = self.start[2]
            g.attrs['tube_along_z_axis_end_z'] = self.end[2]
            g.attrs['geometry_type'] = np.string_(b"tube_along_z\x00")


class Sphere(Shape):
    def __init__(self, origin=(0, 0, 0), radius=1):
        self.origin = np.array(origin)
        self.radius = float(radius)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_sphere(self.origin, self.radius, **kwargs)

    def are_positions_inside(self, positions):
        return norm(positions - self.origin, axis=-1) <= self.radius

    def generate_uniform_random_posititons(self, random_state, n):
        while True:
            p = random_state.uniform(-1, 1, (n * 2, 3)) * self.radius + self.origin
            p = p.compress(self.are_positions_inside(p), 0)
            if len(p) > n:
                return p[:n]

    def export_h5(self, g, region):
        if region:
            g.attrs['radius'] = self.radius
            for i, c in enumerate('xyz'):
                g.attrs['origin_{}'.format(c)] = self.origin[i]
            g.attrs['object_type'] = np.string_(b"sphere\x00")
        else:
            raise ValueError('Cannot export spherical particle source')


class Cone(Shape):
    def __init__(self, start=(0, 0, 0, 1),
                 start_radii=(1, 2), end_radii=(3, 4)):
        self.start = np.array(start, np.float)
        self.start_radii = np.array(start_radii, np.float)
        self.end_radii = np.array(end_radii, np.float)

    def visualize(self, visualizer, **kwargs):
        visualizer.draw_cone(self.start, self.end,
                             self.start_radii, self.end_radii, **kwargs)

# TODO: def are_positions_inside(self, point)
