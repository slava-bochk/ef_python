from ef.field import Field
import numpy as np


class FieldParticles(Field):
    def __init__(self, name, particle_arrays):
        super().__init__(name, electric_or_magnetic='electric')
        self.particle_arrays=particle_arrays

    def get_at_points(self, positions, time):
        return sum(np.nan_to_num(p.field_at_points(positions)) for p in self.particle_arrays)
