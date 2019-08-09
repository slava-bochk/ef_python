import numpy as np
from pytest import raises

from ef.util.data_class import DataClass


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

    def test_eq(self):
        AB = self.AB
        ab = AB(1, 2)
        assert ab == ab
        assert ab == AB(1, 2)
        assert ab != AB(2, 2)
        assert ab != (1, 2)
        assert ab != self.AB2(1, 2)

    def test_assert_eq(self):
        AB = self.AB
        ab = AB(1, 2)
        ab.assert_eq(ab)
        ab.assert_eq(AB(1, 2))
        with raises(AssertionError, match='a'):
            ab.assert_eq(AB(2, 2))
        with raises(AssertionError, match=''):
            ab.assert_eq((1, 2))
        with raises(AssertionError, match='xyz'):
            ab.assert_eq(self.AB2(1, 2), 'xyz')

    def test_attr_eq(self):
        AB = self.AB
        eq = DataClass._attr_eq
        assert eq(1, 1.)
        assert eq([1, 2, 3], [1, 2, 3])
        assert eq(np.array([[4, 5, 6], [7, 8, 9]]), np.array([[4, 5, 6], [7, 8, 9]]))
        assert not eq(np.array([[4, 5, 6], [7, 8, 9]]), np.array([[4, 5, 7], [7, 8, 9]]))
        assert eq(np.array([4, 5, 6]), np.array([4, 5, 6.]))
        assert eq(AB(AB(1, 2), 5), AB(AB(1, 2), 5))
        assert not eq(AB(AB(1, 3), 5), AB(AB(1, 2), 5))

    def test_assert_attr_eq(self):
        AB = self.AB
        _assert_eq = DataClass._assert_attr_eq
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

    def test_attr_eq_override(self):
        class ABchild(self.AB):
            @staticmethod
            def _attr_eq(a, b):
                return True

        assert ABchild(1, 13) == ABchild(3.14, 'spam')

    def test_nan(self):
        assert self.AB([1, 2, 3], np.array([[4, 5, np.NaN], [7, 8, 9]])) != \
               self.AB([1, 2, 3], np.array([[4, 5, np.NaN], [7, 8, 9]]))

    def test_str(self):
        assert str(self.AB(1, 2)) == "### AB:\na = 1\nb = 2"

    def test_repr(self):
        assert repr(self.AB(1, 2)) == "AB(a=1, b=2)"

    def test_dict_init(self):
        ab = self.AB(1, 'ham')
        assert self.AB(**ab.dict_init) == ab

    def test_repr_init(self):
        AB = self.AB
        ab = AB(2, 'spam')
        assert eval(repr(ab)) == ab
