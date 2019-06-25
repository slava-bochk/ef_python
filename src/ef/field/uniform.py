from ef.field import Field
import numpy as np


class FieldUniform(Field):
    def __init__(self, name, electric_or_magnetic, uniform_field_vector):
        super().__init__(name, electric_or_magnetic)
        self.uniform_field_vector = uniform_field_vector

    def get_at_points(self, positions, time):
        return self.uniform_field_vector


class FieldZero(FieldUniform):
    def __init__(self, name, electric_or_magnetic):
        super().__init__(name, electric_or_magnetic, np.zeros(3))