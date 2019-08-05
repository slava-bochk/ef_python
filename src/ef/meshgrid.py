import numpy as np

from ef.util.serializable_h5 import SerializableH5
from ef.util.vector import vector


class MeshGrid(SerializableH5):
    def __init__(self, size, n_nodes, origin=(0, 0, 0)):
        self.size = vector(size)
        self.n_nodes = vector(n_nodes, np.int)
        self.origin = vector(origin)

    @classmethod
    def from_step(cls, size, step, origin=(0, 0, 0)):
        size = vector(size)
        step = vector(step)
        n_nodes = np.ceil(size / step).astype(int) + 1
        return cls(size, n_nodes, origin)

    @property
    def cell(self):
        return self.size / (self.n_nodes - 1)

    @property
    def node_coordinates(self):
        return self.origin + \
               np.moveaxis(np.mgrid[0:self.n_nodes[0], 0:self.n_nodes[1], 0:self.n_nodes[2]], 0, -1) * self.cell
