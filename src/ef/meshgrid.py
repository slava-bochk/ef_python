import numpy as np
from scipy.interpolate import RegularGridInterpolator

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

    def distribute_scalar_at_positions(self, value, positions):
        """
        Given a set of points, distribute the scalar value's density onto the grid nodes.

        :param value: scalar
        :param positions: array of shape (np, 3)
        :return: array of shape (nx, ny, nz)
        """
        volume_around_node = self.cell.prod()
        density = value / volume_around_node  # scalar
        result = np.zeros(self.n_nodes)
        pos = positions - self.origin
        if np.any((pos > self.size) | (pos < 0)):
            raise ValueError("Position is out of meshgrid bounds")
        nodes, remainders = np.divmod(pos, self.cell)  # (np, 3)
        nodes = nodes.astype(int)  # (np, 3)
        weights = remainders / self.cell  # (np, 3)
        for dx in (0, 1):
            wx = weights[:, 0] if dx else 1. - weights[:, 0]  # np
            for dy in (0, 1):
                wy = weights[:, 1] if dy else 1. - weights[:, 1]  # np
                wxy = wx * wy  # np
                for dz in (0, 1):
                    wz = weights[:, 2] if dz else 1. - weights[:, 2]  # np
                    w = wxy * wz  # np
                    dn = dx, dy, dz
                    nodes_to_update = nodes + dn  # (np, 3)
                    w_nz = w[w > 0]
                    n_nz = nodes_to_update[w > 0]
                    np.add.at(result, tuple(n_nz.transpose()), w_nz * density)
        return result

    def interpolate_field_at_positions(self, field, positions):
        """
        Given a field on this grid, interpolate it at n positions.

        :param field: array of shape (nx, ny, nz, {F})
        :param positions: array of shape (np, 3)
        :return: array of shape (np, {F})
        """
        xyz = tuple(np.linspace(self.origin[i], self.origin[i] + self.size[i], self.n_nodes[i]) for i in (0, 1, 2))
        interpolator = RegularGridInterpolator(xyz, field, bounds_error=False, fill_value=0)
        return interpolator(positions)
        #
        # node, remainder = np.divmod(positions - self.origin, self.cell)
        # node = node.astype(int)  # shape is (p, 3)
        # wx, wy, wz = (remainder / self.cell).transpose()
        # field_on_nodes = np.zeros((*field.shape[3:], len(positions)))  # (F, np)
        # for dn in product((0, 1), repeat=3):
        #     dx, dy, dz = dn
        #     nodes_to_use = node + dn  # (np, 3)
        #     out_of_bounds = np.logical_or(nodes_to_use >= self.n_nodes, nodes_to_use < 0).any(axis=-1)  # (np)
        #     field_this_cycle = np.zeros((*field.shape[3:], len(positions)))  # (F, np)
        #     field_this_cycle[..., ~out_of_bounds] = field[
        #         tuple(nodes_to_use[~out_of_bounds].transpose())].transpose()  # sorry...
        #     weight_on_nodes = (wx if dx else 1. - wx) * (wy if dy else 1. - wy) * (wz if dz else 1. - wz)
        #     field_on_nodes += field_this_cycle * weight_on_nodes
        # return field_on_nodes.transpose()
