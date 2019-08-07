import itertools

from ef.util.array_on_grid import ArrayOnGrid

try:
    import cupy, cupyx
except ImportError:
    pass  # Just don't try to actually use it


class ArrayOnGridCupy(ArrayOnGrid):
    def __init__(self, grid, value_shape=None, data=None):
        self.xp = cupy
        super().__init__(grid, value_shape, data)
        self._cell = self.xp.array(grid.cell)
        self._size = self.xp.array(grid.size)
        self._origin = self.xp.array(grid.origin)

    @property
    def cell(self):
        return self._cell

    @property
    def size(self):
        return self._size

    @property
    def origin(self):
        return self._origin

    def scatter_add(self, a, slices, value):
        import cupyx
        cupyx.scatter_add(a, slices, value)

    def interpolate_at_positions(self, positions):
        positions = self.xp.asarray(positions)
        node, remainder = self.xp.divmod(positions - self.origin, self.cell)
        node = node.astype(int)  # shape is (p, 3)
        weight = remainder / self.cell  # shape is (np, 3)
        w = self.xp.stack([1. - weight, weight], axis=-2)  # shape is (np, 2, 3)
        dn = self.xp.array(list(itertools.product((0, 1), repeat=3)))  # shape is (8, 3)
        nodes_to_use = node[..., self.xp.newaxis, :] + dn  # shape is (np, 8, 3)
        out_of_bounds = self.xp.logical_or(nodes_to_use >= self.xp.asarray(self.n_nodes[:3]),
                                           nodes_to_use < 0).any(axis=-1)  # (np, 8)
        field_on_nodes = self.xp.empty((len(positions), 8, *self.value_shape))  # (np, 8, F)
        field_on_nodes[out_of_bounds] = 0  # (oob(np*8), F) interpolate out-of-bounds field as 0
        nodes_in_bounds = nodes_to_use[~out_of_bounds].transpose()  # 3, ib(np*8)
        field_on_nodes[~out_of_bounds] = self.data[tuple(nodes_in_bounds)]  # (ib(np*8), F)
        weight_on_nodes = w[..., dn[:, self.xp.array((0, 1, 2))], self.xp.array((0, 1, 2))].prod(-1)  # shape is (np, 8)
        return self.xp.moveaxis((self.xp.moveaxis(field_on_nodes, (0, 1), (-2, -1)) * weight_on_nodes)
                                .sum(axis=-1), -1, 0).get()  # shape is (np, F)

    def gradient(self):
        # based on numpy.gradient simplified for our case
        if self.value_shape != ():
            raise ValueError("Trying got compute gradient for a non-scalar field: ambiguous")
        if any(n < 2 for n in self.n_nodes):
            raise ValueError("ArrayOnGrid too small to compute gradient")
        f = self.data
        result = self.xp.empty((3, *self.n_nodes))

        internal = slice(1, -1)
        to_left = slice(None, -2)
        to_right = slice(2, None)
        for axis, dx in enumerate(self.cell):
            on_axis = lambda s: tuple(s if i == axis else slice(None) for i in range(3))
            result[axis][on_axis(internal)] = -(f[on_axis(to_right)] - f[on_axis(to_left)]) / (2. * dx)
            result[axis][on_axis(0)] = -(f[on_axis(1)] - f[on_axis(0)]) / dx
            result[axis][on_axis(-1)] = -(f[on_axis(-1)] - f[on_axis(-2)]) / dx
        return self.__class__(self.grid, 3, self.xp.moveaxis(result, 0, -1))
