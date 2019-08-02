import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import raises

from ef.field import Field, FieldSum, FieldZero
from ef.field.expression import FieldExpression
from ef.field.from_csv import FieldFromCSVFile
from ef.field.on_grid import FieldOnGrid
from ef.field.uniform import FieldUniform
from ef.meshgrid import MeshGrid


class TestFields:
    def test_field(self):
        f = Field('f', 'electric')
        with raises(NotImplementedError):
            f.get_at_points(f.get_at_points((1, 2, 3), 0.), (3.14, 2.7, -0.5))

    def test_sum(self):
        f = FieldSum.factory([FieldUniform('u1', 'electric', np.array((3.14, 2.7, -0.5)))])
        assert type(f) is FieldUniform
        assert_array_equal(f.get_at_points((1, 2, 3), 0.), (3.14, 2.7, -0.5))
        assert_array_equal(f.get_at_points((1, 2, 3), 5.), (3.14, 2.7, -0.5))
        assert_array_equal(f.get_at_points((3, 2, 1), 5.), (3.14, 2.7, -0.5))
        f = FieldSum.factory(
            [FieldUniform('u1', 'electric', np.array((3.14, 2.7, -0.5))), None, FieldZero('u1', 'electric')])
        assert type(f) is FieldUniform

        f = FieldSum.factory([FieldUniform('u1', 'electric', np.array((3.14, 2.7, -0.5))),
                              FieldExpression('e1', 'electric', '-1+t', 'x*y-z', 'x+y*z')])
        assert type(f) is FieldSum
        assert_array_almost_equal(f.get_at_points((1, 2, 3), 0.), (2.14, 1.7, 6.5))
        assert_array_almost_equal(f.get_at_points((1, 2, 3), 5.), (7.14, 1.7, 6.5))
        assert_array_almost_equal(f.get_at_points((3, 2, 1), 5.), (7.14, 7.7, 4.5))

        f = FieldSum.factory([], 'electric')
        assert type(f) is FieldZero
        with raises(ValueError):
            FieldSum.factory([])

        f = FieldZero('zero', 'magnetic') + None
        assert type(f) is FieldZero
        with raises(ValueError):
            FieldZero('zero', 'magnetic') + FieldZero('zero', 'electric')

        f = FieldSum('electric', [FieldUniform('u1', 'electric', np.array((3.14, 2.7, -0.5)))]) + \
            FieldExpression('e1', 'electric', '-1+t', 'x*y-z', 'x+y*z') + None
        assert type(f) is FieldSum
        assert len(f.fields) == 2
        assert_array_almost_equal(f.get_at_points((1, 2, 3), 0.), (2.14, 1.7, 6.5))
        assert_array_almost_equal(f.get_at_points((1, 2, 3), 5.), (7.14, 1.7, 6.5))
        assert_array_almost_equal(f.get_at_points((3, 2, 1), 5.), (7.14, 7.7, 4.5))

    def test_uniform(self):
        f = FieldUniform('u1', 'electric', np.array((3.14, 2.7, -0.5)))
        assert_array_equal(f.get_at_points((1, 2, 3), 0.), (3.14, 2.7, -0.5))
        assert_array_equal(f.get_at_points((1, 2, 3), 5.), (3.14, 2.7, -0.5))
        assert_array_equal(f.get_at_points((3, 2, 1), 5.), (3.14, 2.7, -0.5))

    def test_zero(self):
        f = FieldZero('z', 'magnetic')
        assert_array_equal(f.get_at_points((1, 2, 3), 0.), (0, 0, 0))
        assert_array_equal(f.get_at_points((1, 2, 3), 5.), (0, 0, 0))
        assert_array_equal(f.get_at_points((3, 2, 1), 5.), (0, 0, 0))

    def test_expression(self):
        f = FieldExpression('e1', 'electric', '-1+t', 'x*y-z', 'x+y*z')
        assert_array_equal(f.get_at_points((1, 2, 3), 0.), (-1, -1, 7))
        assert_array_equal(f.get_at_points((1, 2, 3), 5.), (4, -1, 7))
        assert_array_equal(f.get_at_points((3, 2, 1), 5.), (4, 5, 5))

    def test_on_grid(self):
        f = FieldOnGrid('f1', 'electric', MeshGrid(10, 11))
        assert_array_equal(f.field, np.zeros((11, 11, 11, 3)))
        assert f == FieldOnGrid('f1', 'electric', MeshGrid(10, 11), np.zeros((11, 11, 11, 3)))
        f = FieldOnGrid('f1', 'electric', MeshGrid(5, 6), np.full((6, 6, 6, 3), -3.14))
        assert_array_equal(f.get_at_points([(-1, 0, 0), (1, 2, 3.5)], 1), [(0, 0, 0), (-3.14, -3.14, -3.14)])
        with raises(ValueError, match="Unexpected raw data array shape: \(3,\) for this field's shape: \(6, 6, 6, 3\)"):
            FieldOnGrid('f1', 'electric', MeshGrid(5, 6), np.full(3, 3.14))

    def test_from_file(self):
        f = FieldFromCSVFile('f1', 'electric', 'tests/test_field.csv')
        assert_array_equal(f.get_at_points([(0, 0, 0), (1, 1, 1), (1, 0, 1), (.5, .5, .5)], 0),
                           [(1, 1, 1), (-1, -1, -1), (3, 2, 1), (1, 1, 1)])
        assert_array_equal(f.get_at_points([(0, 0, 0), (1, 1, 1), (1, 0, 1), (.5, .5, .5)], 10.),
                           [(1, 1, 1), (-1, -1, -1), (3, 2, 1), (1, 1, 1)])
        assert_array_almost_equal(f.get_at_points([(.5, 1., .3), (0, .5, .7)], 5), [(0., .5, 1.), (1, 1.5, 2)])
        assert_array_equal(f.get_at_points([(-1, 1., .3), (1, 1, 10)], 3), [(0, 0, 0), (0, 0, 0)])
