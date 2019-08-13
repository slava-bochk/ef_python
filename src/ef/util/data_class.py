import numpy as np
from numpy.testing import assert_array_equal


class DataClass:
    """ Mixin class to implement default methods for hierarchies of simple objects containing ndarrays. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    @property
    def dict_compare(self) -> dict:
        return self.dict

    @property
    def dict_init(self) -> dict:
        return self.dict

    def __repr__(self):
        cls = self.__class__.__name__
        args = ', '.join(f"{k}={v!r}" for k, v in self.dict_init.items())
        return f"{cls}({args})"

    def __str__(self):
        cls = self.__class__.__name__
        args = '\n'.join(f"{k} = {v!r}" for k, v in vars(self).items())
        return f"### {cls}:\n{args}"
