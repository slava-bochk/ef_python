import numpy as np

from ef.util.serializable_h5 import SerializableH5


class InnerRegion(SerializableH5):
    def __init__(self, name, shape, potential=0.0, total_absorbed_particles=0, total_absorbed_charge=0.0,
                 inverted=False):
        self.name = name
        self.shape = shape
        self.potential = potential
        self.total_absorbed_particles = total_absorbed_particles
        self.total_absorbed_charge = total_absorbed_charge
        self.inverted = inverted

    def collide_with_particles(self, particles):
        collisions = self.check_if_points_inside(particles.dict['positions'])
        c = np.count_nonzero(collisions)
        self.total_absorbed_particles += c
        self.total_absorbed_charge += c * particles.charge
        particles.remove(collisions)

    def check_if_points_inside(self, positions):
        pos_inside = self.shape.are_positions_inside(positions)
        if self.inverted:
            pos_inside = np.logical_not(pos_inside)
        return pos_inside

    @staticmethod
    def import_h5(g):
        from ef.config.components import Shape
        ga = g.attrs
        shape = Shape.import_h5(g, region=True)
        name = g.name.split('/')[-1]
        return InnerRegion(name, shape, float(ga['potential']), int(ga['total_absorbed_particles']),
                           float(ga['total_absorbed_charge']))

    def export_h5(self, g):
        for k in 'potential', 'total_absorbed_particles', 'total_absorbed_charge':
            g.attrs[k] = [getattr(self, k)]
        self.shape.export_h5(g, region=True)
