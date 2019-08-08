from typing import Union, Sequence

import numpy as np

VectorInput = Union[float, Sequence, np.ndarray]


def vector(x, dtype: Union[type, str, None] = None):
    a = np.array(x)
    if a.shape == ():
        a = np.full(3, x)
    elif a.shape == (3,):
        pass
    else:
        raise ValueError("Unable to interpret {} as a 3d vector, it seems to have shape: {}".format(x, a.shape))
    if dtype is not None:
        return a.astype(dtype, casting='safe', copy=False)
    else:
        return a
