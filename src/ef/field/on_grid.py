import numpy as np

from ef.field import Field


class FieldOnGrid(Field):
    def __init__(self, name, electric_or_magnetic, grid, field=None):
        super().__init__(name, electric_or_magnetic)
        self.grid = grid
        if field is None:
            self.field = np.zeros(self.shape, np.float)
        else:
            field = np.array(field)
            if field.shape != self.shape:
                raise ValueError("Unexpected raw data array shape: {} for this field's shape: {}".format(
                    field.shape, self.shape
                ))
            self.field = field

    @property
    def shape(self):
        return (*self.grid.n_nodes, 3)

    def get_at_points(self, positions, time):
        return self.grid.interpolate_field_at_positions(self.field, positions)
