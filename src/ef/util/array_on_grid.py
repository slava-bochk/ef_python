from typing import Optional

import numpy
from scipy.interpolate import RegularGridInterpolator

from ef.util.serializable_h5 import SerializableH5


class ArrayOnGrid(SerializableH5):
    xp = numpy

    def __init__(self, grid, value_shape=None, data=None):
        self.grid = grid
        if value_shape is None:
            value_shape = ()
        self.value_shape = (value_shape,) if type(value_shape) is int else tuple(value_shape)
        if data is None:
            self._data = self.zero
        else:
            data = self.xp.array(data, dtype=self.xp.float)
            if data.shape != self.n_nodes:
                raise ValueError("Unexpected raw data array shape: {} for this ArrayOnGrid shape: {}".format(
                    data.shape, self.n_nodes
                ))
            self._data = data

    @property
    def dict(self):
        d = super().dict
        d["data"] = self.data
        return d

    @property
    def data(self):
        return self._data

    @property
    def cell(self):
        return self.xp.asarray(self.grid.cell)

    @property
    def size(self):
        return self.xp.asarray(self.grid.size)

    @property
    def origin(self):
        return self.xp.asarray(self.grid.origin)

    @property
    def n_nodes(self):
        return (*self.grid.n_nodes, *self.value_shape)

    @property
    def zero(self):
        return self.xp.zeros(self.n_nodes, self.xp.float)

    def reset(self):
        self._data = self.zero

    def distribute_at_positions(self, value, positions):
        """
        Given a set of points, distribute the scalar value's density onto the grid nodes.

        :param value: scalar
        :param positions: array of shape (np, 3)
        """
        volume_around_node = self.cell.prod()
        density = value / volume_around_node  # scalar
        pos = self.xp.asarray(positions) - self.origin
        if self.xp.any((pos > self.size) | (pos < 0)):
            raise ValueError("Position is out of meshgrid bounds")
        nodes, remainders = self.xp.divmod(pos, self.cell)  # (np, 3)
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
                    dn = self.xp.array((dx, dy, dz))
                    nodes_to_update = nodes + dn  # (np, 3)
                    w_nz = w[w > 0]
                    n_nz = nodes_to_update[w > 0]
                    self.scatter_add(self._data, tuple(n_nz.transpose()), w_nz * density)

    def scatter_add(self, a, slices, value):
        self.xp.add.at(a, slices, value)

    def interpolate_at_positions(self, positions):
        """
        Given a field on this grid, interpolate it at n positions.

        :param positions: array of shape (np, 3)
        :return: array of shape (np, {F})
        """
        positions = self.xp.asarray(positions)
        o, s = self.origin, self.size
        xyz = tuple(self.xp.linspace(o[i], o[i] + s[i], self.n_nodes[i]) for i in (0, 1, 2))
        interpolator = RegularGridInterpolator(xyz, self._data, bounds_error=False, fill_value=0)
        return interpolator(positions)

    def gradient(self, output_array: Optional['ArrayOnGrid'] = None) -> 'ArrayOnGrid':
        # based on numpy.gradient simplified for our case
        if self.value_shape != ():
            raise ValueError("Trying got compute gradient for a non-scalar field: ambiguous")
        if any(n < 2 for n in self.n_nodes):
            raise ValueError("ArrayOnGrid too small to compute gradient")
        f = self._data
        if output_array is None:
            output_array = self.__class__(self.grid, 3)
        result = output_array._data
        internal = slice(1, -1)
        to_left = slice(None, -2)
        to_right = slice(2, None)
        for axis, dx in enumerate(self.cell):
            on_axis = lambda s: tuple(s if i == axis else slice(None) for i in range(3))
            result[(*on_axis(internal), axis)] = (f[on_axis(to_left)] - f[on_axis(to_right)]) / (2. * dx)
            result[(*on_axis(0), axis)] = (f[on_axis(0)] - f[on_axis(1)]) / dx
            result[(*on_axis(-1), axis)] = (f[on_axis(-2)] - f[on_axis(-1)]) / dx
        return output_array

    @property
    def is_the_same_on_all_boundaries(self):
        x0 = self._data[0, 0, 0]
        r3 = range(3)
        slices = [tuple(x if i == j else slice(None) for j in r3) for i in r3 for x in (0, -1)]
        return all(self.xp.all(self._data[s] == x0) for s in slices)

    def apply_boundary_values(self, boundary_conditions):
        self._data[:, 0, :] = boundary_conditions.bottom
        self._data[:, -1, :] = boundary_conditions.top
        self._data[0, :, :] = boundary_conditions.right
        self._data[-1, :, :] = boundary_conditions.left
        self._data[:, :, 0] = boundary_conditions.near
        self._data[:, :, -1] = boundary_conditions.far
