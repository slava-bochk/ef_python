import numpy as np

from ef.util.serializable_h5 import SerializableH5


class ParticleTracker(SerializableH5):
    def __init__(self, last_issued_id=-1):
        self.last_issued_id = last_issued_id

    def generate_particle_ids(self, num_of_particles):
        range_of_ids = range(self.last_issued_id + 1, self.last_issued_id + num_of_particles + 1)
        self.last_issued_id += num_of_particles
        return np.array(range_of_ids)
