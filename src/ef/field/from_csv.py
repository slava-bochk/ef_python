import os.path

import numpy as np

from ef.field.on_grid import FieldOnGrid
from ef.meshgrid import MeshGrid
from ef.util.array_on_grid import ArrayOnGrid


class FieldFromCSVFile(FieldOnGrid):
    def __init__(self, name, electric_or_magnetic, field_filename):
        if not os.path.exists(field_filename):
            raise FileNotFoundError("Field file not found")
        raw = np.loadtxt(field_filename)
        # assume X Y Z Fx Fy Fz columns
        # sort by column 0, then 1, then 2
        # https://stackoverflow.com/a/38194077
        ind = raw[:, 2].argsort()  # First sort doesn't need to be stable.
        raw = raw[ind]
        ind = raw[:, 1].argsort(kind='mergesort')
        raw = raw[ind]
        ind = raw[:, 0].argsort(kind='mergesort')
        raw = raw[ind]
        size = (raw[-1, :3] - raw[0, :3])
        origin = raw[0, :3]
        dist = raw[:, :3] - raw[0, :3]
        step = np.min(dist[dist > 0], axis=0)
        grid = MeshGrid.from_step(size, step, origin)
        field = raw[:, 3:].reshape((*grid.n_nodes, 3))
        super().__init__(name, electric_or_magnetic, ArrayOnGrid(grid, 3, field))
        self.field_filename = field_filename
