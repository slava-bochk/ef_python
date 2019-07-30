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
            pass  # don't write fields_without_particles etc.
        if self.h5file.get('particles') is None:
            self.init_file(sim, self.h5file)
        t = sim.time_grid.current_node // sim.time_grid.node_to_save
        for p in sim.particle_arrays:
            for i, id in enumerate(p.ids):
                self.h5file['/particles/position'][id, t] = p.positions[i]
                self.h5file['/particles/momentum'][id, t] = p.momentums[i]
                self.h5file['/particles/mass'][id] = p.mass
                self.h5file['/particles/charge'][id] = p.charge
        if sim.particle_interaction_model.pic:
            self.h5file['/field/potential'][t] = sim.spat_mesh.potential

    def init_file(self, sim, h5file):
        n_particles = sum(s.initial_number_of_particles +
                          s.particles_to_generate_each_step * sim.time_grid.total_nodes
                          for s in sim.particle_sources)
        n_time = (sim.time_grid.total_nodes - 1) // sim.time_grid.node_to_save + 1
        h5file['/time'] = np.linspace(0, sim.time_grid.total_time, n_time)
        h5file['/particles/ids'] = np.arange(n_particles)
        h5file['/particles/coordinates'] = [np.string_('x'), np.string_('y'), np.string_('z')]
        h5file.create_dataset('/particles/position', (n_particles, n_time, 3))
        h5file['/particles/position'].dims[0].label = 'id'
        h5file['/particles/position'].dims.create_scale(h5file['/particles/ids'], 'ids')
        h5file['/particles/position'].dims[0].attach_scale(h5file['/particles/ids'])
        h5file['/particles/position'].dims[1].label = 'time'
        h5file['/particles/position'].dims.create_scale(h5file['/time'], 'time')
        h5file['/particles/position'].dims[1].attach_scale(h5file['/time'])
        h5file['/particles/position'].dims[2].label = 'coordinates'
        h5file['/particles/position'].dims.create_scale(h5file['/particles/coordinates'], 'coordinates')
        h5file['/particles/position'].dims[2].attach_scale(h5file['/particles/coordinates'])
        h5file.create_dataset('/particles/momentum', (n_particles, n_time, 3))
        h5file['/particles/momentum'].dims[0].label = 'id'
        h5file['/particles/momentum'].dims.create_scale(h5file['/particles/ids'], 'ids')
        h5file['/particles/momentum'].dims[0].attach_scale(h5file['/particles/ids'])
        h5file['/particles/momentum'].dims[1].label = 'time'
        h5file['/particles/momentum'].dims.create_scale(h5file['/time'], 'time')
        h5file['/particles/momentum'].dims[1].attach_scale(h5file['/time'])
        h5file['/particles/momentum'].dims[2].label = 'coordinate'
        h5file['/particles/momentum'].dims.create_scale(h5file['/particles/coordinates'], 'coordinates')
        h5file['/particles/momentum'].dims[2].attach_scale(h5file['/particles/coordinates'])
        h5file.create_dataset('/particles/mass', (n_particles,))
        h5file['/particles/mass'].dims[0].label = 'id'
        h5file['/particles/mass'].dims.create_scale(h5file['/particles/ids'], 'ids')
        h5file['/particles/mass'].dims[0].attach_scale(h5file['/particles/ids'])
        h5file.create_dataset('/particles/charge', (n_particles,))
        h5file['/particles/charge'].dims[0].label = 'id'
        h5file['/particles/charge'].dims.create_scale(h5file['/particles/ids'], 'ids')
        h5file['/particles/charge'].dims[0].attach_scale(h5file['/particles/ids'])
        if sim.particle_interaction_model.pic:
            h5file.create_dataset('/field/potential', (n_time, *sim.spat_mesh.n_nodes))
            h5file['/field/potential'].dims[0].label = 'time'
            h5file['/field/potential'].dims.create_scale(h5file['/time'], 'time')
            h5file['/field/potential'].dims[0].attach_scale(h5file['/time'])
            for i, c in enumerate('xyz'):
                h5file['/field/{}'.format(c)] = np.linspace(0, sim.spat_mesh.size[i], sim.spat_mesh.n_nodes[i])
                h5file['/field/potential'].dims[i + 1].label = c
                h5file['/field/potential'].dims.create_scale(h5file['/field/{}'.format(c)], c)
                h5file['/field/potential'].dims[i + 1].attach_scale(h5file['/field/{}'.format(c)])
        sim.save_h5(self.h5file)
