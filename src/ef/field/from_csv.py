import os.path

import inject
import numpy

from ef.field.on_grid import FieldOnGrid
from ef.meshgrid import MeshGrid
from ef.util.array_on_grid import ArrayOnGrid


class FieldFromCSVFile(FieldOnGrid):
    @inject.params(xp=numpy, array_class=ArrayOnGrid)
    def __init__(self, name, electric_or_magnetic, field_filename, xp=numpy, array_class=ArrayOnGrid):
        if not os.path.exists(field_filename):
            raise FileNotFoundError("Field file not found")
        raw = numpy.loadtxt(field_filename, skiprows=1)
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
        step = dist.min(axis=0, where=dist > 0, initial=dist.max())
        grid = MeshGrid.from_step(size, step, origin)
        field = xp.asarray(raw[:, 3:].reshape((*grid.n_nodes, 3)))
        super().__init__(name, electric_or_magnetic, array_class(grid, 3, field))
        self.field_filename = field_filename
