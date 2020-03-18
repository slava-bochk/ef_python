import numpy as np
from pytest import raises

from ef.util.data_class import DataClass
from ef.util.testing import assert_dataclass_eq, _assert_value_eq


class TestDataClass:
    class AB(DataClass):
        c = 10

        def __init__(self, a, b):
            self.a = a
            self.b = b

        @property
        def p(self):
            return self.a + self.b

        @classmethod
        def cm(cls):
            return cls(10, 20)

        @staticmethod
        def sm():
            return "foo"

    class AB2(AB):
        pass

    class ABx(AB):
        def __init__(self, a, b):
            super().__init__(a, b)
            self.x = a + b

    class AB_x(AB):
        def __init__(self, a, b):
            super().__init__(a, b)
            self._x = a + b

    def test_dict(self):
        AB = self.AB
        ab = AB(1, 2)
        assert ab.dict == {'a': 1, 'b': 2}
        assert AB(ab, []).dict == {'a': ab, 'b': []}
        assert self.ABx(1, 2).dict == {'a': 1, 'b': 2, 'x': 3}
        assert self.AB_x(1, 2).dict == {'a': 1, 'b': 2}

    def test_assert_eq(self):
        AB = self.AB
        ab = AB(1, 2)
        assert_dataclass_eq(ab, ab)
        assert_dataclass_eq(ab, AB(1, 2))
        with raises(AssertionError, match='a'):
            assert_dataclass_eq(ab, AB(2, 2))
        with raises(AssertionError, match=''):
            assert_dataclass_eq(ab, (1, 2))
        with raises(AssertionError, match='xyz'):
            assert_dataclass_eq(ab, self.AB2(1, 2), 'xyz')

    def test_assert_attr_eq(self):
        AB = self.AB
        _assert_eq = _assert_value_eq
        _assert_eq(1, 1.)
        _assert_eq([1, 2, 3], [1, 2, 3])
        _assert_eq(np.array([[4, 5, 6], [7, 8, 9]]), np.array([[4, 5, 6], [7, 8, 9]]))
        with raises(AssertionError, match='path'):
            _assert_eq(np.array([[4, 5, 6], [7, 8, 9]]), np.array([[4, 5, 7], [7, 8, 9]]), 'path')
        with raises(AssertionError, match=''):
            _assert_eq(np.array([[4, 5, 6], [7, 8, 9]]), np.array([[4, 5, 7], [7, 8, 9]]))
        _assert_eq(np.array([4, 5, 6]), np.array([4, 5, 6.]))
        _assert_eq(AB(AB(1, 2), 5), AB(AB(1, 2), 5))
        with raises(AssertionError, match=r'a\.b'):
            _assert_eq(AB(AB(1, 3), 5), AB(AB(1, 2), 5))
        with raises(AssertionError, match=r'root\.a\.b'):
            _assert_eq(AB(AB(1, 3), 5), AB(AB(1, 2), 5), 'root')

    def test_nan(self):
        assert self.AB([1, 2, 3], np.array([[4, 5, np.NaN], [7, 8, 9]])) != \
               self.AB([1, 2, 3], np.array([[4, 5, np.NaN], [7, 8, 9]]))

    def test_str(self):
        assert str(self.AB(1, 2)) == "### AB:\na = 1\nb = 2"

    def test_repr(self):
        assert repr(self.AB(1, 2)) == "AB(a=1, b=2)"

    def test_dict_init(self):
        ab = self.AB(1, 'ham')
        assert_dataclass_eq(self.AB(**ab.dict_init), ab)

    def test_repr_init(self):
        AB = self.AB
        ab = AB(2, 'spam')
        assert_dataclass_eq(eval(repr(ab)), ab)
