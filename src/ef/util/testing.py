import numpy

from ef.util.data_class import DataClass

__all__ = ['assert_array_equal', 'assert_array_almost_equal']

from numpy.testing import assert_array_equal, assert_array_almost_equal

try:
    import cupy
    from cupy.testing import assert_array_equal, assert_array_almost_equal
except ImportError:
    cupy = None


def assert_dataclass_eq(a: DataClass, b: DataClass, path: str = '') -> None:
    assert isinstance(a, DataClass), path
    assert isinstance(b, DataClass), path
    assert type(a) == type(b), path
    _assert_dict_eq(a.dict_compare, b.dict_compare, path)


def _assert_dict_eq(a, b, path: str = '') -> None:
    assert isinstance(a, dict)
    assert isinstance(b, dict)
    assert a.keys() == b.keys(), path
    for k in a.keys():
        p = f'{path}.{k}' if path else f'{k}'
        _assert_value_eq(a[k], b[k], p)


def _assert_value_eq(a, b, path: str = '') -> None:
    if isinstance(a, DataClass):
        assert_dataclass_eq(a, b, path)
    elif isinstance(a, numpy.ndarray):
        assert isinstance(b, numpy.ndarray), path
        assert_array_equal(a, b, path)
    elif cupy is not None and isinstance(a, cupy.ndarray):
        assert isinstance(b, cupy.ndarray), path
        assert_array_equal(a, b, path)
    elif isinstance(a, dict):
        _assert_dict_eq(a, b)
    elif isinstance(a, list):
        assert isinstance(b, list)
        assert len(a) == len(b)
        for i, xy in enumerate(zip(a, b)):
            _assert_value_eq(*xy, f'{path}[{i}]')
    elif isinstance(a, tuple):
        assert isinstance(b, tuple)
        assert len(a) == len(b)
        for i, xy in enumerate(zip(a, b)):
            _assert_value_eq(*xy, f'{path}[{i}]')
    else:
        assert a == b, path
