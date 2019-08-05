import logging

import numpy as np

from ef.meshgrid import MeshGrid
from ef.util.array_on_grid import ArrayOnGrid
from ef.util.serializable_h5 import SerializableH5


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
        charge_density = ArrayOnGrid(grid)
        potential = ArrayOnGrid(grid)
        potential.data[:, 0, :] = boundary_conditions.bottom
        potential.data[:, -1, :] = boundary_conditions.top
        potential.data[0, :, :] = boundary_conditions.right
        potential.data[-1, :, :] = boundary_conditions.left
        potential.data[:, :, 0] = boundary_conditions.near
        potential.data[:, :, -1] = boundary_conditions.far
        electric_field = ArrayOnGrid(grid, 3)
        return cls(grid, charge_density, potential, electric_field)

    def weight_particles_charge_to_mesh(self, particle_arrays):
        for p in particle_arrays:
            self.charge_density.distribute_at_positions(p.charge, p.positions)

    def get_at_points(self, positions, time):
        return self.electric_field.interpolate_at_positions(positions)

    def clear_old_density_values(self):
        self.charge_density.reset()

    def is_potential_equal_on_boundaries(self):
        p = self.potential.data[0, 0, 0]
        return np.all(self.potential.data[0] == p) and np.all(self.potential.data[-1] == p) and \
               np.all(self.potential.data[:, 0] == p) and np.all(self.potential.data[:, -1] == p) and \
               np.all(self.potential.data[:, :, 0] == p) and np.all(self.potential.data[:, :, -1] == p)

    def eval_field_from_potential(self):
        self.electric_field = ArrayOnGrid(self.mesh, 3, -np.stack(np.gradient(self.potential.data, *self.cell), -1))

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
        g = MeshGrid(size, n_nodes)
        return cls(g, ArrayOnGrid(g, (), charge), ArrayOnGrid(g, (), potential), ArrayOnGrid(g, 3, field))

    def export_h5(self, g):
        for i, c in enumerate('xyz'):
            g.attrs['{}_volume_size'.format(c)] = [self.size[i]]
            g.attrs['{}_cell_size'.format(c)] = [self.cell[i]]
            g.attrs['{}_n_nodes'.format(c)] = [self.n_nodes[i]]
            g['node_coordinates_{}'.format(c)] = self.node_coordinates[..., i].flatten()
            g['electric_field_{}'.format(c)] = self.electric_field.data[..., i].flatten()
        g['charge_density'] = self.charge_density.data.flatten()
        g['potential'] = self.potential.data.flatten()
