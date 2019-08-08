import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ef.util.serializable_h5 import SerializableH5


class ArrayOnGrid(SerializableH5):
    def __init__(self, grid, value_shape=None, data=None):
        self.grid = grid
        if value_shape is None:
            value_shape = ()
        self.value_shape = (value_shape,) if type(value_shape) is int else tuple(value_shape)
        if data is None:
            self.data = self.zero
        else:
            data = np.array(data)
            if data.shape != self.n_nodes:
                raise ValueError("Unexpected raw data array shape: {} for this ArrayOnGrid shape: {}".format(
                    data.shape, self.n_nodes
                ))
            self.data = data

    @property
    def n_nodes(self):
        return (*self.grid.n_nodes, *self.value_shape)

    @property
    def zero(self):
        return np.zeros(self.n_nodes, np.float)

    def reset(self):
        self.data = self.zero

    def distribute_at_positions(self, value, positions):
        """
        Given a set of points, distribute the scalar value's density onto the grid nodes.

        :param value: scalar
        :param positions: array of shape (np, 3)
        """
        volume_around_node = self.grid.cell.prod()
        density = value / volume_around_node  # scalar
        pos = positions - self.grid.origin
        if np.any((pos > self.grid.size) | (pos < 0)):
            raise ValueError("Position is out of meshgrid bounds")
        nodes, remainders = np.divmod(pos, self.grid.cell)  # (np, 3)
        nodes = nodes.astype(int)  # (np, 3)
        weights = remainders / self.grid.cell  # (np, 3)
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
                    np.add.at(self.data, tuple(n_nz.transpose()), w_nz * density)

    def interpolate_at_positions(self, positions):
        """
        Given a field on this grid, interpolate it at n positions.

        :param positions: array of shape (np, 3)
        :return: array of shape (np, {F})
        """
        o, s = self.grid.origin, self.grid.size
        xyz = tuple(np.linspace(o[i], o[i] + s[i], self.grid.n_nodes[i]) for i in (0, 1, 2))
        interpolator = RegularGridInterpolator(xyz, self.data, bounds_error=False, fill_value=0)
        return interpolator(positions)

    def gradient(self):
        if self.value_shape != ():
            raise ValueError("Trying got compute gradient for a non-scalar field: ambiguous")
        return ArrayOnGrid(self.grid, 3, -np.stack(np.gradient(self.data, *self.grid.cell), -1))

    @property
    def is_the_same_on_all_boundaries(self):
        x0 = self.data[0, 0, 0]
        r3 = range(3)
        slices = [tuple(x if i == j else slice(None) for j in r3) for i in r3 for x in (0, -1)]
        return all(np.all(self.data[s] == x0) for s in slices)

    def apply_boundary_values(self, boundary_conditions):
        self.data[:, 0, :] = boundary_conditions.bottom
        self.data[:, -1, :] = boundary_conditions.top
        self.data[0, :, :] = boundary_conditions.right
        self.data[-1, :, :] = boundary_conditions.left
        self.data[:, :, 0] = boundary_conditions.near
        self.data[:, :, -1] = boundary_conditions.far