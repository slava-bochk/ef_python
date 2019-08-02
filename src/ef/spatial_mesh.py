import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ef.util.serializable_h5 import SerializableH5


class MeshGrid(SerializableH5):
    def __init__(self, size, n_nodes, origin=(0, 0, 0)):
        self.size = size
        self.n_nodes = n_nodes
        self.origin = np.asarray(origin)

    @classmethod
    def from_step(cls, size, step, origin=(0, 0, 0)):
        n_nodes = np.ceil(size / step).astype(int) + 1
        return cls(size, n_nodes, origin)

    @property
    def cell(self):
        return self.size / (self.n_nodes - 1)

    @property
    def node_coordinates(self):
        return self.origin + \
               np.moveaxis(np.mgrid[0:self.n_nodes[0], 0:self.n_nodes[1], 0:self.n_nodes[2]], 0, -1) * self.cell

    def distribute_scalar_at_positions(self, value, positions):
        """
        Given a set of points, distribute the scalar value's density onto the grid nodes.

        :param value: scalar
        :param positions: array of shape (np, 3)
        :return: array of shape (nx, ny, nz)
        """
        volume_around_node = self.cell.prod()
        density = value / volume_around_node  # scalar
        result = np.zeros(self.n_nodes)
        pos = positions - self.origin
        if np.any((pos > self.size) | (pos < 0)):
            raise ValueError("Position is out of meshgrid bounds")
        nodes, remainders = np.divmod(pos, self.cell)  # (np, 3)
        nodes = nodes.astype(int)  # (np, 3)
        weights = remainders / self.cell  # (np, 3)
        for dx in (0, 1):
            wx = weights[:, 0] if dx else 1. - weights[:, 0]  # np
            for dy in (0, 1):
                wy = weights[:, 1] if dy else 1. - weights[:, 1]  # np
                wxy = wx * wy  # np
                for dz in (0, 1):
                    wz = weights[:, 2] if dz else 1. - weights[:, 2]  # np
                    w = wxy * wz  # np
                    dn = dx, dy, dz
                    nodes_to_update = nodes + dn  # (np, 3)
                    w_nz = w[w > 0]
                    n_nz = nodes_to_update[w > 0]
                    np.add.at(result, tuple(n_nz.transpose()), w_nz * density)
        return result

    def interpolate_field_at_positions(self, field, positions):
        """
        Given a field on this grid, interpolate it at n positions.

        :param field: array of shape (nx, ny, nz, {F})
        :param positions: array of shape (np, 3)
        :return: array of shape (np, {F})
        """
        xyz = tuple(np.linspace(self.origin[i], self.origin[i] + self.size[i], self.n_nodes[i]) for i in (0, 1, 2))
        interpolator = RegularGridInterpolator(xyz, field, bounds_error=False, fill_value=0)
        return interpolator(positions)
        #
        # node, remainder = np.divmod(positions - self.origin, self.cell)
        # node = node.astype(int)  # shape is (p, 3)
        # wx, wy, wz = (remainder / self.cell).transpose()
        # field_on_nodes = np.zeros((*field.shape[3:], len(positions)))  # (F, np)
        # for dn in product((0, 1), repeat=3):
        #     dx, dy, dz = dn
        #     nodes_to_use = node + dn  # (np, 3)
        #     out_of_bounds = np.logical_or(nodes_to_use >= self.n_nodes, nodes_to_use < 0).any(axis=-1)  # (np)
        #     field_this_cycle = np.zeros((*field.shape[3:], len(positions)))  # (F, np)
        #     field_this_cycle[..., ~out_of_bounds] = field[
        #         tuple(nodes_to_use[~out_of_bounds].transpose())].transpose()  # sorry...
        #     weight_on_nodes = (wx if dx else 1. - wx) * (wy if dy else 1. - wy) * (wz if dz else 1. - wz)
        #     field_on_nodes += field_this_cycle * weight_on_nodes
        # return field_on_nodes.transpose()


class SpatialMesh(SerializableH5):
    electric_or_magnetic = 'electric'
    name = 'spatial_mesh'

    def __init__(self, mesh, charge_density, potential, electric_field):
        self.mesh = mesh
        self.charge_density = charge_density
        self.potential = potential
        self.electric_field = electric_field

    @property
    def size(self):
        return self.mesh.size

    @property
    def cell(self):
        return self.mesh.cell

    @property
    def n_nodes(self):
        return self.mesh.n_nodes

    @property
    def node_coordinates(self):
        return self.mesh.node_coordinates

    @classmethod
    def do_init(cls, grid_size, step_size, boundary_conditions):
        try:
            size = np.array(grid_size, np.float)
        except ValueError as exception:
            raise ValueError("grid_size must be a flat triple", grid_size) from exception
        try:
            step = np.array(step_size, np.float)
        except ValueError as exception:
            raise ValueError("step_size must be a flat triple", step_size) from exception
        # Check argument ranges
        if size.shape != (3,):
            raise ValueError("grid_size must be a flat triple", grid_size)
        if step.shape != (3,):
            raise ValueError("step_size must be a flat triple", step_size)
        if np.any(size <= 0):
            raise ValueError("grid_size must be positive", grid_size)
        if np.any(step <= 0):
            raise ValueError("step_size must be positive", step_size)
        if np.any(step > size):
            raise ValueError("step_size cannot be bigger than grid_size")

        grid = MeshGrid.from_step(size, step)
        for i in np.nonzero(grid.cell != step_size)[0]:
            logging.warning(f"{('X', 'Y', 'Z')[i]} step on spatial grid was reduced to "
                            f"{grid.cell[i]:.3f} from {step_size[i]:.3f} "
                            f"to fit in a round number of cells.")
        charge_density = np.zeros(grid.n_nodes, dtype='f8')
        potential = np.zeros(grid.n_nodes, dtype='f8')
        potential[:, 0, :] = boundary_conditions.bottom
        potential[:, -1, :] = boundary_conditions.top
        potential[0, :, :] = boundary_conditions.right
        potential[-1, :, :] = boundary_conditions.left
        potential[:, :, 0] = boundary_conditions.near
        potential[:, :, -1] = boundary_conditions.far
        electric_field = np.zeros(list(grid.n_nodes) + [3], dtype='f8')
        return cls(grid, charge_density, potential, electric_field)

    def weight_particles_charge_to_mesh(self, particle_arrays):
        for p in particle_arrays:
            self.charge_density += self.mesh.distribute_scalar_at_positions(p.charge, p.positions)

    def field_at_position(self, positions):
        return self.mesh.interpolate_field_at_positions(self.electric_field, positions)

    def get_at_points(self, positions, time):
        return self.field_at_position(positions)

    def clear_old_density_values(self):
        self.charge_density.fill(0)

    def is_potential_equal_on_boundaries(self):
        p = self.potential[0, 0, 0]
        return np.all(self.potential[0] == p) and np.all(self.potential[-1] == p) and \
               np.all(self.potential[:, 0] == p) and np.all(self.potential[:, -1] == p) and \
               np.all(self.potential[:, :, 0] == p) and np.all(self.potential[:, :, -1] == p)

    def eval_field_from_potential(self):
        self.electric_field = -np.stack(np.gradient(self.potential, *self.cell), -1)

    @classmethod
    def import_h5(cls, g):
        ga = g.attrs
        size = np.array([ga['{}_volume_size'.format(c)] for c in 'xyz']).reshape(3)
        n_nodes = np.array([ga['{}_n_nodes'.format(c)] for c in 'xyz']).reshape(3)
        charge = np.reshape(g['charge_density'], n_nodes)
        potential = np.reshape(g['potential'], n_nodes)
        field = np.moveaxis(
            np.array([np.reshape(g['electric_field_{}'.format(c)], n_nodes) for c in 'xyz']),
            0, -1)
        return cls(MeshGrid(size, n_nodes), charge, potential, field)

    def export_h5(self, g):
        for i, c in enumerate('xyz'):
            g.attrs['{}_volume_size'.format(c)] = [self.size[i]]
            g.attrs['{}_cell_size'.format(c)] = [self.cell[i]]
            g.attrs['{}_n_nodes'.format(c)] = [self.n_nodes[i]]
            g['node_coordinates_{}'.format(c)] = self.node_coordinates[..., i].flatten()
            g['electric_field_{}'.format(c)] = self.electric_field[..., i].flatten()
        g['charge_density'] = self.charge_density.flatten()
        g['potential'] = self.potential.flatten()
