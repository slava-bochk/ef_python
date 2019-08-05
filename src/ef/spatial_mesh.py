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

    def get_at_points(self, positions, time):
        return self.electric_field.interpolate_at_positions(positions)

    @classmethod
    def import_h5(cls, g):
        mesh = MeshGrid.import_h5(g)
        charge = ArrayOnGrid(mesh, (), np.reshape(g['charge_density'], mesh.n_nodes))
        potential = ArrayOnGrid(mesh, (), np.reshape(g['potential'], mesh.n_nodes))
        field = ArrayOnGrid(mesh, 3, np.moveaxis(
            np.array([np.reshape(g['electric_field_{}'.format(c)], mesh.n_nodes) for c in 'xyz']),
            0, -1))
        return cls(mesh, charge, potential, field)

    def export_h5(self, g):
        self.mesh.export_h5(g)
        for i, c in enumerate('xyz'):
            g['electric_field_{}'.format(c)] = self.electric_field.data[..., i].flatten()
        g['charge_density'] = self.charge_density.data.flatten()
        g['potential'] = self.potential.data.flatten()
