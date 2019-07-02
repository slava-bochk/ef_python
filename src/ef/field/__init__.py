import numpy as np

from ef.util.serializable_h5 import SerializableH5


class Field(SerializableH5):
    def __init__(self, name, electric_or_magnetic):
        self.name = name
        self.electric_or_magnetic = electric_or_magnetic

    def __add__(self, b):
        return FieldSum.factory((self, b))

    def get_at_points(self, positions, time):
        raise NotImplementedError()


class FieldZero(Field):
    def get_at_points(self, positions, time):
        return np.zeros_like(positions)


class FieldSum(Field):
    def __init__(self, electric_or_magnetic, fields):
        super().__init__('FieldSum', electric_or_magnetic)
        self.fields = fields

    @classmethod
    def factory(cls, fields, electric_or_magnetic=None):
        fields = [f for f in fields if f is not None]
        em = (set(f.electric_or_magnetic for f in fields) | {electric_or_magnetic}) - {None}
        if len(em) > 1:
            raise ValueError('Trying to combine inconsistent fields')
        elif em:
            em = em.pop()
        else:
            raise ValueError('FieldSum type unknown')
        sums = [f for f in fields if type(f) is FieldSum]
        fields = [f for f in fields if type(f) not in (FieldZero, FieldSum)]
        for f in sums:
            fields += f.fields
        if len(fields) > 1:
            return cls(em, fields)
        elif len(fields) == 1:
            return fields[0]
        else:
            return FieldZero('ZeroSum', em)

    def get_at_points(self, positions, time):
        return sum(f.get_at_points(positions, time) for f in self.fields)
