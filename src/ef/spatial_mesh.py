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
        grid = MeshGrid.from_step(grid_size, step_size)
        charge_density = ArrayOnGrid(grid)
        potential = ArrayOnGrid(grid)
        potential.apply_boundary_values(boundary_conditions)
        electric_field = ArrayOnGrid(grid, 3)
        return cls(grid, charge_density, potential, electric_field)

    def get_at_points(self, positions, time):
        return self.electric_field.interpolate_at_positions(positions)
