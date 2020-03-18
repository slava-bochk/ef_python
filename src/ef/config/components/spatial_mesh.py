from ef.meshgrid import MeshGrid
from ef.util import vector
from ef.util.vector import vector

__all__ = ['SpatialMeshConf', 'SpatialMeshSection']

from collections import namedtuple

from ef.config.component import ConfigComponent
from ef.config.section import ConfigSection


class SpatialMeshConf(ConfigComponent):
    def __init__(self, size=10, step=1, **kwargs):
        super().__init__(**kwargs)
        self.size = vector(size)
        self.step = vector(step)

    def visualize(self, visualizer):
        visualizer.draw_box(self.size, wireframe=True, label='volume', colors='k', linewidths=1)

    def to_conf(self):
        return SpatialMeshSection(*self.size, *self.step)

    def make(self):
        return MeshGrid.from_step(self.size, self.step)


class SpatialMeshSection(ConfigSection):
    section = "SpatialMesh"
    ContentTuple = namedtuple("SpatialMeshTuple",
                              [f'grid_{c}_size' for c in 'xyz'] + [f'grid_{c}_step' for c in 'xyz'])
    convert = ContentTuple(*[float] * 6)

    def make(self):
        return SpatialMeshConf(self.content[:3], self.content[3:])
