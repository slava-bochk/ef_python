import h5py
import numpy as np

from ef.output import OutputWriter


class OutputWriterHistory(OutputWriter):
    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix
        self.h5file = h5py.File(self.format_filename('history'), 'w')

    def __del__(self):
        self.h5file.close()

    def format_filename(self, specific_name):
        return "{prefix}{name}{suffix}".format(prefix=self.prefix, suffix=self.suffix, name=specific_name)

    def write(self, sim, name=None):
        if name is not None:
            return  # don't write fields_without_particles etc.
        if self.h5file.get('history') is None:
            self.init_file(sim, self.h5file)
        t = sim.time_grid.current_node // sim.time_grid.node_to_save
        h = self.h5file['history']
        for p in sim.particle_arrays:
            for i, id in enumerate(p.ids):
                h['/particles/position'][id, t] = p.positions[i]
                h['/particles/momentum'][id, t] = p.momentums[i]
                h['/particles/mass'][id] = p.mass
                h['/particles/charge'][id] = p.charge
        if sim.particle_interaction_model.pic:
            h['/field/potential'][t] = sim.spat_mesh.potential

    def init_file(self, sim, h5file):
        h = self.h5file.create_group('history')
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
        if sim.particle_interaction_model.pic:
            h.create_dataset('field/potential', (n_time, *sim.spat_mesh.n_nodes))
            h['field/potential'].dims[0].label = 'time'
            h['field/potential'].dims.create_scale(h['time'], 'time')
            h['field/potential'].dims[0].attach_scale(h['time'])
            for i, c in enumerate('xyz'):
                h['field/{}'.format(c)] = np.linspace(0, sim.spat_mesh.size[i], sim.spat_mesh.n_nodes[i])
                h['field/potential'].dims[i + 1].label = c
                h['field/potential'].dims.create_scale(h['field/{}'.format(c)], c)
                h['field/potential'].dims[i + 1].attach_scale(h['field/{}'.format(c)])
        sim.save_h5(self.h5file.create_group('simulation'))
