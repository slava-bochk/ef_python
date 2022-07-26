import operator
from functools import reduce

import numpy

from ef.util.array_on_grid import ArrayOnGrid

import cupy, cupyx


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
                if (x0<0 | y0<0 | z0<0 |
                    (x0>={n[0]}-1 & dx>0) | 
                    (y0>={n[1]}-1 & dy>0) | 
                    (z0>={n[2]}-1 & dz>0) ) {{
                        for (int q=0; q<{v}; q++) {{
                            result[tid * {v} + q] = 0;
                        }}
                }} else {{
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
                }}
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
        positions = self.xp.asanyarray(positions)
        result = self.xp.empty(reduce(operator.mul, self.value_shape, positions.shape[0]))
        n = positions.shape[0]
        block = 128
        grid = (n - 1) // block + 1
        self._interpolate_field((grid,), (block,), (n, self._data.ravel(order='C'), positions.ravel(order='C'), result))
        result = result.reshape((positions.shape[0], *self.value_shape))
        return result
