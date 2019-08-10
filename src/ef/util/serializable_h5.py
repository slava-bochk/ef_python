from enum import Enum
from typing import cast, Type, TypeVar, Dict, List, Union

import numpy as np
from h5py import Dataset, Group

from ef.util.data_class import DataClass
from ef.util.subclasses import Registered


class Serializable(DataClass, Registered, dont_register=True):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


DataStructure = Union[Dict[str: 'DataStructure'], List['DataStructure'], Serializable, int,
                      float, str, np.ndarray]

DataTree = TypeVar('DataTree', Dict[str: 'DataTree'], int, float, str, np.ndarray)


def structure_to_tree(value: DataStructure) -> DataTree:
    if isinstance(value, Serializable):
        d = value.dict_init
        _class = value.class_key
    elif isinstance(value, list):
        d = {str(i): x for i, x in enumerate(value)}
        _class = 'list'
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
    d = {k: tree_to_structure(v) for k, v in d.items() if k != '_class'}
    if _class == 'dict':
        return d
    elif _class == 'list':
        return [d[str(i)] for i in range(len(d))]
    else:
        class_ = cast(Type[Serializable], Serializable.subclasses[_class])
        return class_(**d)


def tree_to_hdf5(tree: Dict[str: 'DataTree'], group: Group):
    for key, value in tree.items():
        if isinstance(value, np.ndarray):
            group[key] = value
        elif isinstance(value, dict):
            tree_to_hdf5(value, group.create_group(key))
        elif isinstance(value, Enum):
            group.attrs[key] = value.name
        else:
            group.attrs[key] = value


class SerializableH5(DataClass, Registered, dont_register=True):
    def save_h5(self, h5group):
        h5group.attrs['class'] = self.class_key
        for k, v in self.dict.items():
            self._save_value(h5group, k, v)

    @staticmethod
    def load_h5(h5group: Group) -> 'SerializableH5':
        return cast('SerializableH5', SerializableH5.subclasses[h5group.attrs['class']]).load_h5_args(h5group)

    @classmethod
    def load_h5_args(cls, h5group):
        kwargs = {key: cls._load_value(value) for key, value in h5group.items()}
        kwargs.update(h5group.attrs)
        del kwargs['class']
        return cls(**kwargs)

    @classmethod
    def _save_value(cls, group, key, value):
        if isinstance(value, np.ndarray):
            group[key] = value
        elif isinstance(value, SerializableH5):
            value.save_h5(group.create_group(key))
        elif isinstance(value, list):
            subgroup = group.create_group(key)
            for i, v in enumerate(value):
                cls._save_value(subgroup, str(i), v)
        elif isinstance(value, Enum):
            group.attrs[key] = value.name
        else:
            group.attrs[key] = value

    @classmethod
    def _load_value(cls, value):
        if isinstance(value, Dataset):
            return np.array(value)
        elif isinstance(value, Group):
            try:
                return SerializableH5.load_h5(value)
            except KeyError as err:
                d = {k: cls._load_value(v) for k, v in value.items()}
                d.update(value.attrs)
                if d.keys() != {str(i) for i in range(len(d))}:
                    raise TypeError("Could not parse hdf5 group into SerializableH5", value) from err
            return [d[str(i)] for i in range(len(d.keys()))]
        else:
            raise TypeError("hdf5 group member of unexpected type found", value)
