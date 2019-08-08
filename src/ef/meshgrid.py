import logging

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
        if np.any(size <= 0):
            raise ValueError('Grid size must be positive', size)
        step = vector(step)
        if np.any(step <= 0):
            raise ValueError('Grid step must be positive', step)
        n_nodes = np.ceil(size / step).astype(int) + 1
        mesh = cls(size, n_nodes, origin)
        for i in np.nonzero(mesh.cell != step)[0]:
            logging.warning(f"{('X', 'Y', 'Z')[i]} step on spatial grid was reduced to "
                            f"{mesh.cell[i]:.3f} from {step[i]:.3f} "
                            f"to fit in a round number of cells.")
        return mesh

    @property
    def cell(self):
        return self.size / (self.n_nodes - 1)

    @property
    def node_coordinates(self):
        return self.origin + \
               np.moveaxis(np.mgrid[0:self.n_nodes[0], 0:self.n_nodes[1], 0:self.n_nodes[2]], 0, -1) * self.cell

    @classmethod
    def import_h5(cls, g):
        ga = g.attrs
        size = np.array([ga['{}_volume_size'.format(c)] for c in 'xyz']).reshape(3)
        n_nodes = np.array([ga['{}_n_nodes'.format(c)] for c in 'xyz']).reshape(3)
        return cls(size, n_nodes)

    def export_h5(self, g):
        for i, c in enumerate('xyz'):
            g.attrs['{}_volume_size'.format(c)] = [self.size[i]]
            g.attrs['{}_cell_size'.format(c)] = [self.cell[i]]
            g.attrs['{}_n_nodes'.format(c)] = [self.n_nodes[i]]
            g['node_coordinates_{}'.format(c)] = self.node_coordinates[..., i].flatten()
