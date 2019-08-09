import numpy as np
from numpy.testing import assert_array_equal


class DataClass:
    """ Mixin class to implement default methods for hierarchies of simple objects containing ndarrays. """

    @property
    def dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    @property
    def dict_compare(self) -> dict:
        return self.dict

    @property
    def dict_init(self) -> dict:
        return self.dict

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        if self.dict_compare.keys() != other.dict_compare.keys():
            return False
        for k, v in self.dict_compare.items():
            w = other.dict_compare[k]
            if not self._attr_eq(v, w):
                return False
        return True

    @staticmethod
    def _attr_eq(v, w):
        if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
            if np.asarray(v).shape != np.asarray(w).shape:
                return False
            if np.any(v != w):
                return False
        else:
            if v != w:
                return False
        return True

    def assert_eq(self, other, path=''):
        assert type(self) is type(other), path
        assert self.dict.keys() == other.dict.keys(), path
        for k, v in self.dict.items():
            w = other.dict[k]
            self._assert_attr_eq(v, w, f'{path}.{k}' if path else k)

    @staticmethod
    def _assert_attr_eq(v, w, path=''):
        if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
            assert np.asarray(v).shape == np.asarray(w).shape, path
            assert_array_equal(v, w, path)
        elif isinstance(v, DataClass):
            v.assert_eq(w, path)
        else:
            assert v == w, path

    def __repr__(self):
        cls = self.__class__.__name__
        args = ', '.join(f"{k}={v!r}" for k, v in self.dict_init.items())
        return f"{cls}({args})"

    def __str__(self):
        cls = self.__class__.__name__
        args = '\n'.join(f"{k} = {v!r}" for k, v in vars(self).items())
        return f"### {cls}:\n{args}"
