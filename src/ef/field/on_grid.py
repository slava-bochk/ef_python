from ef.field import Field
from ef.util.array_on_grid import ArrayOnGrid


class FieldOnGrid(Field):
    def __init__(self, name, electric_or_magnetic, grid, field=None):
        super().__init__(name, electric_or_magnetic)
        self.grid = grid
        self.field = ArrayOnGrid(grid, 3, field)

    def get_at_points(self, positions, time):
        return self.field.interpolate_at_positions(positions)
