import inject
import numpy

from ef.field import Field
from ef.util.vector import vector


class FieldUniform(Field):
    xp = inject.attr(numpy)

    def __init__(self, name, electric_or_magnetic, uniform_field_vector):
        super().__init__(name, electric_or_magnetic)
        self._uniform_field_vector = self.xp.array(vector(uniform_field_vector))

    def get_at_points(self, positions, time):
        return self._uniform_field_vector

    @property
    def dict(self):
        d = super().dict
        v = self._uniform_field_vector if self.xp is numpy else self._uniform_field_vector.get()
        d['uniform_field_vector'] = v
        return d
