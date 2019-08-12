from typing import Type

import inject
import numpy as np

from ef.field import Field
from ef.field.on_grid import FieldOnGrid
from ef.inner_region import InnerRegion
from ef.meshgrid import MeshGrid
from ef.particle_array import ParticleArray
from ef.particle_interaction_model import Model
from ef.particle_source import ParticleSource
from ef.particle_tracker import ParticleTracker
from ef.simulation import Simulation
from ef.time_grid import TimeGrid
from ef.util.array_on_grid import ArrayOnGrid
from ef.util.serializable_h5 import hdf5_to_tree


class Reader:
    array_class: Type[ArrayOnGrid] = inject.attr(ArrayOnGrid)

    @staticmethod
    def guess_h5_format(h5file):
        if 'SpatialMesh' in h5file:
            return 'cpp'
        elif 'history' in h5file:
            return 'history'
        elif 'time_grid' in h5file:
            return 'python'
        else:
            raise ValueError('Cannot guess hdf5 file format')

    @staticmethod
    def read_simulation(h5file):
        format_ = Reader.guess_h5_format(h5file)
        if format_ == 'cpp':
            return Reader.import_from_h5(h5file)
        elif format_ == 'python':
            sim = Simulation.from_tree(hdf5_to_tree(h5file))
        else:
            sim = Simulation.from_tree(hdf5_to_tree(h5file['simulation']))
        sim.particle_interaction_model = Model[sim.particle_interaction_model]
        return sim

    @staticmethod
    def import_from_h5(h5file):
        fields = [Field.import_h5(g) for g in h5file['ExternalFields'].values()]
        sources = [ParticleSource.import_h5(g) for g in h5file['ParticleSources'].values()]
        particles = [ParticleArray.import_h5(g) for g in h5file['ParticleSources'].values()]
        max_id = int(np.max([p.ids for p in particles], initial=-1))
        g = h5file['SpatialMesh']
        mesh = MeshGrid.import_h5(g)
        charge = Reader.array_class(mesh, (), np.reshape(g['charge_density'], mesh.n_nodes))
        potential = Reader.array_class(mesh, (), np.reshape(g['potential'], mesh.n_nodes))
        field = FieldOnGrid('spatial_mesh', 'electric', Reader.array_class(mesh, 3, np.moveaxis(
            np.array([np.reshape(g['electric_field_{}'.format(c)], mesh.n_nodes) for c in 'xyz']),
            0, -1)))
        return Simulation(
            time_grid=TimeGrid.import_h5(h5file['TimeGrid']),
            mesh=mesh, charge_density=charge, potential=potential, electric_field=field,
            inner_regions=[InnerRegion.import_h5(g) for g in h5file['InnerRegions'].values()],
            electric_fields=[f for f in fields if f.electric_or_magnetic == 'electric'],
            magnetic_fields=[f for f in fields if f.electric_or_magnetic == 'magnetic'],
            particle_interaction_model=Model[
                h5file['ParticleInteractionModel'].attrs['particle_interaction_model'].decode('utf8')
            ],
            particle_sources=sources, particle_arrays=particles, particle_tracker=ParticleTracker(max_id)
        )
