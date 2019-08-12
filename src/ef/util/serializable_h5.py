from typing import cast, Type, Dict, List, Union, Tuple, TypeVar

import numpy as np
from h5py import Dataset, Group

from ef.util.data_class import DataClass
from ef.util.subclasses import Registered

T = TypeVar('T')


class SerializableH5(DataClass, Registered, dont_register=True):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_tree(cls: Type[T], data: 'DataTree') -> T:
        return tree_to_structure(data)

    @property
    def tree(self):
        return structure_to_tree(self)


DataStructure = Union[
    Dict[str, 'DataStructure'], List['DataStructure'], Tuple['DataStructure', ...], SerializableH5, int,
    float, str, np.ndarray]

DataTreeLeaf = Union[int, float, str, np.ndarray]
DataTree = Dict[str, Union['DataTree', DataTreeLeaf]]


def structure_to_tree(value: DataStructure) -> Union['DataTree', DataTreeLeaf]:
    if isinstance(value, SerializableH5):
        d = value.dict_init
        _class = value.class_key()
    elif isinstance(value, list):
        d = {str(i): x for i, x in enumerate(value)}
        _class = 'list'
    elif isinstance(value, tuple):
        d = {str(i): x for i, x in enumerate(value)}
        _class = 'tuple'
    elif isinstance(value, dict):
        d = value.copy()
        _class = 'dict'
    else:
        return value
    d = {k: structure_to_tree(v) for k, v in d.items()}
    d['_class'] = _class
    return d


def tree_to_structure(d: DataTree) -> DataStructure:
    _class = d['_class']
    d = {k: tree_to_structure(v) if type(v) == dict else v for k, v in d.items() if k != "_class"}
    if _class == 'dict':
        return d
    elif _class == 'list':
        return [d[str(i)] for i in range(len(d))]
    elif _class == 'tuple':
        return tuple(d[str(i)] for i in range(len(d)))
    else:
        class_ = cast(Type[SerializableH5], SerializableH5.subclasses[_class])
        return class_(**d)


def tree_to_hdf5(tree: DataTree, group: Group):
    for key, value in tree.items():
        if isinstance(value, np.ndarray):
            group[key] = value
        elif isinstance(value, dict):
            tree_to_hdf5(value, group.create_group(key))
        elif isinstance(value, (int, float, str)):
            group.attrs[key] = value
        else:
            raise TypeError(f"Unexpected value type({type(value)}) in group {group.name}: {key}={value}")


def hdf5_to_tree(group: Group) -> DataTree:
    d = dict(group.attrs)
    for key, item in group.items():
        if isinstance(item, Dataset):
            d[key] = np.array(item)
        elif isinstance(item, Group):
            d[key] = hdf5_to_tree(item)
        else:
            raise TypeError(f"Reading {type(item)} from hdf5 is not supported.")
    return d
