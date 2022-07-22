import io

import h5py
import numpy as np
import pytest
from pytest import raises, approx

from ef.util.serializable_h5 import SerializableH5, structure_to_tree, tree_to_structure, tree_to_hdf5, hdf5_to_tree
from ef.util.testing import assert_dataclass_eq


@pytest.fixture
def hdf5_temp():
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as f:
        yield f


class A(SerializableH5):
    def __init__(self, a, b, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b


def test_structure_to_tree():
    assert structure_to_tree(A(1, 'a')) == {'_class': 'test_serializable_h5.A', 'a': 1, 'b': 'a'}
    assert structure_to_tree({'a': 1, 'b': 'a'}) == {'_class': 'dict', 'a': 1, 'b': 'a'}
    assert structure_to_tree([13, 14, 15]) == {'_class': 'list', '0': 13, '1': 14, '2': 15}
    assert structure_to_tree((13, 14, 15)) == {'_class': 'tuple', '0': 13, '1': 14, '2': 15}
    a = np.full(10, 3.14)
    assert structure_to_tree(a) is a


def test_tree_to_structure():
    assert_dataclass_eq(tree_to_structure({'_class': 'test_serializable_h5.A', 'a': 1, 'b': 'a'}), A(1, 'a'))
    assert tree_to_structure({'_class': 'dict', 'a': 1, 'b': 'a'}) == {'a': 1, 'b': 'a'}
    assert tree_to_structure({'_class': 'list', '0': 13, '1': 14, '2': 15}) == [13, 14, 15]
    assert tree_to_structure({'_class': 'tuple', '0': 13, '1': 14, '2': 15}) == (13, 14, 15)
    a = np.full(10, 3.14)
    t = tree_to_structure({'_class': 'tuple', '0': a})
    assert type(t) is tuple and len(t) == 1 and t[0] is a
    with raises(KeyError):
        tree_to_structure({})
    with raises(KeyError):
        tree_to_structure({'class_': 'nonexistant'})
    with raises(KeyError):
        tree_to_structure({'_class': 'list', 'a': 1, 'b': 'a'})
    with raises(TypeError):
        tree_to_structure({'_class': 'test_serializable_h5.A', 'a': 1})


def test_big_example():
    obj = object()
    structure = {'a': A(3.14, ()), 'x': {'foo': [('spam', obj), 123]}}
    tree = {'_class': 'dict',
            'a': {'_class': 'test_serializable_h5.A',
                  'a': 3.14,
                  'b': {'_class': 'tuple'}},
            'x': {'_class': 'dict',
                  'foo': {'_class': 'list',
                          '0': {'_class': 'tuple', '0': 'spam', '1': obj},
                          '1': 123}
                  }
            }
    assert structure_to_tree(structure) == tree
    data = tree_to_structure(tree)
    assert type(data) == dict
    assert data.keys() == {'a', 'x'}
    assert data['x'] == structure['x']
    assert_dataclass_eq(data['a'], structure['a'])


def test_tree_to_hdf5_empty(hdf5_temp: h5py.File):
    tree_to_hdf5({}, hdf5_temp)
    assert hdf5_temp.attrs == {}
    assert hdf5_temp.keys() == set()


def test_hdf5_to_tree_empty(hdf5_temp: h5py.File):
    assert hdf5_to_tree(hdf5_temp) == {}


def test_tree_to_hdf5_and_back(hdf5_temp: h5py.File):
    data = {'a': 1, 'b': 3.14, 'c': 'asdf', 'd': np.full(4, 3.14), 'q': {'foo': 'bar', 'deep': {}}}
    tree_to_hdf5(data, hdf5_temp)
    assert hdf5_temp.attrs == {'a': 1, 'b': 3.14, 'c': 'asdf'}
    assert hdf5_temp.keys() == {'d', 'q'}
    assert np.array(hdf5_temp['d']) == approx(np.full(4, 3.14))
    assert hdf5_temp['q'].attrs == {'foo': 'bar'}
    assert hdf5_temp['q'].keys() == {'deep'}
    assert hdf5_temp['q/deep'].attrs == {}
    assert hdf5_temp['q/deep'].keys() == set()
    data['d'] = approx(data['d'])
    assert hdf5_to_tree(hdf5_temp) == data
