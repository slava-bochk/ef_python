import numpy as np

from ef.field import Field
from ef.inner_region import InnerRegion
from ef.particle_array import ParticleArray
from ef.particle_interaction_model import ParticleInteractionModel
from ef.particle_source import ParticleSource
from ef.simulation import Simulation
from ef.spatial_mesh import SpatialMesh
from ef.time_grid import TimeGrid


class Reader:
    @staticmethod
    def guess_h5_format(h5file):
        if 'SpatialMesh' in h5file:
            return 'cpp'
        elif 'history' in h5file:
            return 'history'
        elif 'spat_mesh' in h5file:
            return 'python'
        else:
            raise ValueError('Cannot guess hdf5 file format')

    @staticmethod
    def read_simulation(h5file):
        format_ = Reader.guess_h5_format(h5file)
        if format_ == 'cpp':
            return Reader.import_from_h5(h5file)
        elif format_ == 'python':
            return Simulation.load_h5(h5file)
        else:
            return Simulation.load_h5(h5file['simulation'])

    @staticmethod
    def import_from_h5(h5file):
        fields = [Field.import_h5(g) for g in h5file['ExternalFields'].values()]
        sources = [ParticleSource.import_h5(g) for g in h5file['ParticleSources'].values()]
        particles = [ParticleArray.import_h5(g) for g in h5file['ParticleSources'].values()]
        max_id = int(np.max([p.ids for p in particles], initial=-1))
        return Simulation(
            time_grid=TimeGrid.import_h5(h5file['TimeGrid']),
            spat_mesh=SpatialMesh.import_h5(h5file['SpatialMesh']),
            inner_regions=[InnerRegion.import_h5(g) for g in h5file['InnerRegions'].values()],
            electric_fields=[f for f in fields if f.electric_or_magnetic == 'electric'],
            magnetic_fields=[f for f in fields if f.electric_or_magnetic == 'magnetic'],
            particle_interaction_model=ParticleInteractionModel.import_h5(h5file['ParticleInteractionModel']),
            particle_sources=sources, particle_arrays=particles, max_id=max_id
        )
