from math import sqrt

import inject
import numpy as np
from numpy.random import RandomState

from ef.config.components.shapes import Box
from ef.particle_array import ParticleArray
from ef.util.serializable_h5 import SerializableH5
from ef.util.vector import vector, VectorInput


class ParticleSource(SerializableH5):
    @inject.params(shape=Box)
    def __init__(self, name="particle_source", shape=None, initial_number_of_particles: int = 0,
                 particles_to_generate_each_step: int = 0,
                 mean_momentum: VectorInput = 0,
                 temperature: float = 0., charge: float = 0., mass: float = 1.):
        self.name = name
        self.shape = shape
        self.initial_number_of_particles = initial_number_of_particles
        self.particles_to_generate_each_step = particles_to_generate_each_step
        self.mean_momentum = vector(mean_momentum)
        self.temperature = temperature
        self.charge = charge
        self.mass = mass
        self._generator = RandomState()

    def generate_initial_particles(self) -> ParticleArray:
        return self.generate_num_of_particles(self.initial_number_of_particles)

    def generate_each_step(self):
        return self.generate_num_of_particles(self.particles_to_generate_each_step)

    def generate_num_of_particles(self, num_of_particles):
        pos = self.shape.generate_uniform_random_posititons(self._generator, num_of_particles)
        mom = self._generator.normal(self.mean_momentum, sqrt(self.mass * self.temperature), (num_of_particles, 3))
        return ParticleArray(range(num_of_particles), self.charge, self.mass, pos, mom)

    @staticmethod
    def import_h5(g):
        from ef.config.components import Shape
        ga = g.attrs
        shape = Shape.import_h5(g, region=False)
        name = g.name.split('/')[-1]
        momentum = np.array([ga['mean_momentum_{}'.format(c)] for c in 'xyz']).reshape(3)
        return ParticleSource(name, shape, int(ga['initial_number_of_particles']),
                              int(ga['particles_to_generate_each_step']),
                              momentum, float(ga['temperature']), float(ga['charge']), float(ga['mass']))

    def export_h5(self, g):
        for k in 'temperature', 'charge', 'mass', 'initial_number_of_particles', 'particles_to_generate_each_step':
            g.attrs[k] = [getattr(self, k)]
        for i, c in enumerate('xyz'):
            g.attrs['mean_momentum_{}'.format(c)] = [self.mean_momentum[i]]
        self.shape.export_h5(g, region=False)
