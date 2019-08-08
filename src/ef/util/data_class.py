import numpy as np
from numpy.testing import assert_array_equal


class DataClass:
    """ Mixin class to implement default methods for hierarchies of simple objects containing ndarrays. """
    @property
    def dict(self):
        """
        Try overriding this first to customize a child class.

        :return: A dict representation of object attributes that can be used to construct it.
        """
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        if self.dict.keys() != other.dict.keys():
            return False
        for k, v in self.dict.items():
            w = other.dict[k]
            if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
                if np.asarray(v).shape != np.asarray(w).shape:
                    return False
                if np.any(v != w):
                    return False
            else:
                if v != w:
                    return False
        return True

    def assert_eq(self, other):
        if self is other:
            return
        assert type(self) is type(other)
        assert self.dict.keys() == other.dict.keys()
        for k, v in self.dict.items():
            w = other.dict[k]
            if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
                assert np.asarray(v).shape == np.asarray(w).shape, k
                assert_array_equal(v, w, k)
            elif isinstance(v, DataClass):
                v.assert_eq(w)
            else:
                assert v == w, k

    def __repr__(self):
        cls = self.__class__.__name__
        args = ', '.join(f"{k}={v!r}" for k, v in self.dict.items())
        return f"{cls}({args})"

    def __str__(self):
        cls = self.__class__.__name__
        args = '\n'.join(f"{k} = {v!r}" for k, v in vars(self).items())
        return f"### {cls}:\n{args}"


class DataClassHashable(DataClass):
    def __hash__(self):
        return hash(tuple(sorted(self.dict.items())))
