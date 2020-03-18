from typing import Optional

import h5py
import numpy as np

from ef.output import OutputWriter
from ef.particle_interaction_model import Model
from ef.util.serializable_h5 import tree_to_hdf5


class OutputWriterHistory(OutputWriter):
    def __init__(self, prefix: str, suffix: str):
        self.prefix: str = prefix
        self.suffix: str = suffix
        self.h5file: h5py.File = h5py.File(f"{prefix}history{suffix}", 'w')

    def __del__(self):
        self.h5file.close()

    def write(self, sim: 'Simulation', name: Optional[str] = None) -> None:
        if name is not None:
            return  # don't write fields_without_particles etc.
        if self.h5file.get('history') is None:
            self.init_file(sim, self.h5file)
        t = sim.time_grid.current_node // sim.time_grid.node_to_save
        h = self.h5file['history']
        for p in sim.particle_arrays:
            for i, id_ in enumerate(p.ids):
                h['particles/position'][id_, t] = p.positions[i]
                h['particles/momentum'][id_, t] = p.momentums[i]
                h['particles/mass'][id_] = p.mass
                h['particles/charge'][id_] = p.charge
        if sim.particle_interaction_model == Model.PIC:
            h['field/potential'][t] = sim.potential

    def init_file(self, sim: 'Simulation', h5file: h5py.File) -> None:
        h = h5file.create_group('history')
        n_particles = sum(s.initial_number_of_particles +
                          s.particles_to_generate_each_step * sim.time_grid.total_nodes
                          for s in sim.particle_sources)
        n_time = (sim.time_grid.total_nodes - 1) // sim.time_grid.node_to_save + 1
        h['time'] = np.linspace(0, sim.time_grid.total_time, n_time)
        h['particles/ids'] = np.arange(n_particles)
        h['particles/coordinates'] = [np.string_('x'), np.string_('y'), np.string_('z')]
        h.create_dataset('particles/position', (n_particles, n_time, 3))
        h['particles/position'].dims[0].label = 'id'
        h['particles/position'].dims.create_scale(h['particles/ids'], 'ids')
        h['particles/position'].dims[0].attach_scale(h['particles/ids'])
        h['particles/position'].dims[1].label = 'time'
        h['particles/position'].dims.create_scale(h['time'], 'time')
        h['particles/position'].dims[1].attach_scale(h['time'])
        h['particles/position'].dims[2].label = 'coordinates'
        h['particles/position'].dims.create_scale(h['particles/coordinates'], 'coordinates')
        h['particles/position'].dims[2].attach_scale(h['particles/coordinates'])
        h.create_dataset('particles/momentum', (n_particles, n_time, 3))
        h['particles/momentum'].dims[0].label = 'id'
        h['particles/momentum'].dims.create_scale(h['particles/ids'], 'ids')
        h['particles/momentum'].dims[0].attach_scale(h['particles/ids'])
        h['particles/momentum'].dims[1].label = 'time'
        h['particles/momentum'].dims.create_scale(h['time'], 'time')
        h['particles/momentum'].dims[1].attach_scale(h['time'])
        h['particles/momentum'].dims[2].label = 'coordinate'
        h['particles/momentum'].dims.create_scale(h['particles/coordinates'], 'coordinates')
        h['particles/momentum'].dims[2].attach_scale(h['particles/coordinates'])
        h.create_dataset('particles/mass', (n_particles,))
        h['particles/mass'].dims[0].label = 'id'
        h['particles/mass'].dims.create_scale(h['particles/ids'], 'ids')
        h['particles/mass'].dims[0].attach_scale(h['particles/ids'])
        h.create_dataset('particles/charge', (n_particles,))
        h['particles/charge'].dims[0].label = 'id'
        h['particles/charge'].dims.create_scale(h['particles/ids'], 'ids')
        h['particles/charge'].dims[0].attach_scale(h['particles/ids'])
        if sim.particle_interaction_model == Model.PIC:
            h.create_dataset('field/potential', (n_time, *sim.potential.n_nodes))
            h['field/potential'].dims[0].label = 'time'
            h['field/potential'].dims.create_scale(h['time'], 'time')
            h['field/potential'].dims[0].attach_scale(h['time'])
            for i, c in enumerate('xyz'):
                h[f'field/{c}'] = np.linspace(0, sim.electric_field.size[i], sim.electric_field.n_nodes[i])
                h['field/potential'].dims[i + 1].label = c
                h['field/potential'].dims.create_scale(h[f'field/{c}'], c)
                h['field/potential'].dims[i + 1].attach_scale(h[f'field/{c}'])
        tree_to_hdf5(sim.tree, self.h5file.create_group('simulation'))
