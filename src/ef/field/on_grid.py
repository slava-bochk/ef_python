import numpy as np

from ef.field import Field


class FieldOnGrid(Field):
    def __init__(self, name, electric_or_magnetic, grid, field=None):
        super().__init__(name, electric_or_magnetic)
        self.grid = grid
        if field is None:
            self.field = np.zeros(grid.n_nodes.append(3), np.float)
        else:
            self.field = field

    def get_at_points(self, positions, time):
        return self.grid.interpolate_field_at_positions(self.field, positions)
