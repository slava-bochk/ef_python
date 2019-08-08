from ef.field import Field
from ef.util.array_on_grid import ArrayOnGrid


class FieldOnGrid(Field):
    def __init__(self, name, electric_or_magnetic, array: ArrayOnGrid):
        if array.value_shape != (3,):
            raise ValueError("Can only use 3d vector array as a field")
        super().__init__(name, electric_or_magnetic)
        self.array = array

    def get_at_points(self, positions, time):
        return self.array.interpolate_at_positions(positions)
