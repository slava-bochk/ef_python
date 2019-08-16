import numpy
from scipy.interpolate import RegularGridInterpolator

from ef.util.array_on_grid import ArrayOnGrid

try:
    import cupy, cupyx
except ImportError:
    pass  # Just don't try to actually use it


class ArrayOnGridCupy(ArrayOnGrid):
    def __init__(self, grid, value_shape=None, data=None):
        self.xp = cupy
        super().__init__(grid, value_shape, data)
        self._interpolate_field = cupy.RawKernel(r'''
        extern "C" __global__
        void interpolate_field(int size, const double* field, const double* coords, double* result) {{
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < size) {{
                double x = coords[3 * tid]/{c[0]};
                double y = coords[3 * tid + 1]/{c[1]};
                double z = coords[3 * tid + 2]/{c[2]};
                int x0 = int(x);
                int y0 = int(y);
                int z0 = int(z);
                double dx = x - x0;
                double dy = y - y0;
                double dz = z - z0;
                int where = ((x0*{n[1]} + y0)*{n[2]} + z0) * {v};
                for (int q=0; q<{v}; q++) {{
                    int wq = where + q;
                    result[tid * {v} + q] =
                        field[wq] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz) + 
                        field[wq + {v}] * (1.0 - dx) * (1.0 - dy) * dz +
                        field[wq + {v}*{n[2]}] * (1.0 - dx) * dy * (1.0 - dz) + 
                        field[wq + {v}*{n[2]} + {v}] * (1.0 - dx) * dy * dz +
                        field[wq + {v}*{n[1]}*{n[2]}] * dx * (1.0 - dy) * (1.0 - dz) + 
                        field[wq + {v}*{n[1]}*{n[2]} + {v}] * dx * (1.0 - dy) * dz +
                        field[wq + {v}*{n[1]}*{n[2]} + {v}*{n[2]}] * dx * dy * (1.0 - dz) + 
                        field[wq + {v}*{n[1]}*{n[2]} + {v}*{n[2]} + {v}] * dx * dy * dz;
                }}
                if (x0<0) result[tid] = 0;
                if (y0<0) result[tid] = 0;
                if (z0<0) result[tid] = 0;
                if (x0>{n[0]} and dx>0) result[tid] = 0;
                if (y0>{n[1]} and dy>0) result[tid] = 0;
                if (z0>{n[2]} and dz>0) result[tid] = 0;
            }}
        }}
        '''.format(c=grid.cell, n=grid.n_nodes, s=grid.size, v=numpy.prod(self.value_shape, dtype=int)),
                                                 'interpolate_field')
        self._cell = self.xp.array(grid.cell)
        self._size = self.xp.array(grid.size)
        self._origin = self.xp.array(grid.origin)

    @property
    def dict(self):
        d = super().dict
        d.pop('xp')
        return d

    @property
    def data(self):
        return self._data.get()

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
        result = self.xp.empty((positions.shape[0], *self.value_shape))
        n = positions.shape[0]
        block = 128
        grid = (n - 1) // block + 1
        self._interpolate_field((grid,), (block,), (n, self._data, positions, result))
        o, s = self.origin.get(), self.size.get()
        xyz = tuple(numpy.linspace(o[i], o[i] + s[i], self.n_nodes[i]) for i in (0, 1, 2))
        interpolator = RegularGridInterpolator(xyz, self.data, bounds_error=False, fill_value=0)
        cupy.testing.assert_array_almost_equal(result, interpolator(positions.get()))
        return result

    def gradient(self):
        # based on numpy.gradient simplified for our case
        if self.value_shape != ():
            raise ValueError("Trying got compute gradient for a non-scalar field: ambiguous")
        if any(n < 2 for n in self.n_nodes):
            raise ValueError("ArrayOnGrid too small to compute gradient")
        f = self._data
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
