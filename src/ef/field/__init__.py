import inject
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

    @staticmethod
    def import_h5(g):
        from ef.field.expression import FieldExpression
        from ef.field.on_grid import FieldOnGrid
        from ef.field.uniform import FieldUniform
        ga = g.attrs
        ft = ga['field_type']
        name = g.name.split('/')[-1]
        if ft == b'electric_uniform':
            return FieldUniform(name, 'electric',
                                np.array([ga['electric_uniform_field_{}'.format(c)] for c in 'xyz']).reshape(3))
        elif ft == b'electric_tinyexpr':
            return FieldExpression(name, 'electric',
                                   *[ga['electric_tinyexpr_field_{}'.format(c)].decode('utf8') for c in 'xyz'])
        elif ft == b'electric_on_regular_grid':
            return FieldOnGrid(name, 'electric', ga['electric_h5filename'].decode('utf8'))
        elif ft == b'magnetic_uniform':
            return FieldUniform(name, 'magnetic',
                                np.array([ga['magnetic_uniform_field_{}'.format(c)] for c in 'xyz']).reshape(3))
        elif ft == b'magnetic_tinyexpr':
            return FieldExpression(name, 'magnetic',
                                   *[ga['magnetic_tinyexpr_field_{}'.format(c)].decode('utf8') for c in 'xyz'])
        elif ft == b'magnetic_on_regular_grid':
            return FieldOnGrid(name, 'magnetic', ga['magnetic_h5filename'].decode('utf8'))


class FieldZero(Field):
    xp = inject.attr(np)

    def get_at_points(self, positions, time):
        positions = self.xp.array(positions)
        return self.xp.zeros_like(positions)


class FieldSum(Field):
    def __init__(self, electric_or_magnetic, fields):
        super().__init__('FieldSum', electric_or_magnetic)
        self.fields = fields

    @classmethod
    def factory(cls, fields, electric_or_magnetic=None):
        try:
            fields = [f for f in fields if f is not None]
        except TypeError:
            fields = [] if fields is None else [fields]
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
