from collections import defaultdict

import h5py
import numpy as np

from ef.config.components import Box, Cylinder, Tube, Sphere
from ef.field import FieldZero, FieldSum
from ef.field.expression import FieldExpression
from ef.field.on_grid import FieldOnGrid
from ef.field.particles import FieldParticles
from ef.field.solvers.field_solver import FieldSolver
from ef.field.uniform import FieldUniform
from ef.inner_region import InnerRegion
from ef.particle_array import ParticleArray
from ef.particle_interaction_model import ParticleInteractionModel
from ef.particle_source import ParticleSource
from ef.spatial_mesh import SpatialMesh, MeshGrid
from ef.time_grid import TimeGrid
from ef.util.physical_constants import speed_of_light
from ef.util.serializable_h5 import SerializableH5


class Simulation(SerializableH5):

    def __init__(self, time_grid, spat_mesh, inner_regions,
                 particle_sources,
                 electric_fields, magnetic_fields, particle_interaction_model,
                 output_filename_prefix='', outut_filename_suffix='.h5', max_id=-1, particle_arrays=()):
        self.time_grid = time_grid
        self.spat_mesh = spat_mesh
        self.inner_regions = inner_regions
        self._field_solver = FieldSolver(spat_mesh, inner_regions)
        self.particle_sources = particle_sources
        self.electric_fields = FieldSum.factory(electric_fields, 'electric')
        self.magnetic_fields = FieldSum.factory(magnetic_fields, 'magnetic')
        self.particle_interaction_model = particle_interaction_model
        self.particle_arrays = list(particle_arrays)
        self.consolidate_particle_arrays()

        if self.particle_interaction_model.binary:
            self._dynamic_field = FieldParticles('binary_particle_field', self.particle_arrays)
            if self.inner_regions or not self.spat_mesh.is_potential_equal_on_boundaries():
                self._dynamic_field += self.spat_mesh
        elif self.particle_interaction_model.noninteracting:
            if self.inner_regions or not self.spat_mesh.is_potential_equal_on_boundaries():
                self._dynamic_field = self.spat_mesh
            else:
                self._dynamic_field = FieldZero('Uniform_potential_zero_field', 'electric')
        else:
            self._dynamic_field = self.spat_mesh

        self._output_filename_prefix = output_filename_prefix
        self._output_filename_suffix = outut_filename_suffix
        self.max_id = max_id

    @classmethod
    def init_from_h5(cls, h5file, filename_prefix, filename_suffix):
        if 'SpatialMesh' in h5file:
            simulation = cls.import_from_h5(h5file, filename_prefix, filename_suffix)
        else:
            simulation = cls.load_h5(h5file)
            simulation._output_filename_prefix = filename_prefix
            simulation._output_filename_suffix = filename_suffix
        return simulation

    def start_pic_simulation(self):
        self.eval_and_write_fields_without_particles()
        self.create_history_file()
        self.generate_and_prepare_particles(initial=True)
        self.write_step_to_save()
        self.run_pic()

    def continue_pic_simulation(self):
        self.create_history_file()
        self.run_pic()

    def run_pic(self):
        total_time_iterations = self.time_grid.total_nodes - 1
        current_node = self.time_grid.current_node
        for i in range(current_node, total_time_iterations):
            print("Time step from {:d} to {:d} of {:d}".format(
                i, i + 1, total_time_iterations))
            self.advance_one_time_step()
            self.write_step_to_save()

    def advance_one_time_step(self):
        self.push_particles()
        self.generate_and_prepare_particles()
        self.update_time_grid()

    def eval_charge_density(self):
        self.spat_mesh.clear_old_density_values()
        self.spat_mesh.weight_particles_charge_to_mesh(self.particle_arrays)

    def eval_potential_and_fields(self):
        self._field_solver.eval_potential(self.spat_mesh, self.inner_regions)
        self._field_solver.eval_fields_from_potential(self.spat_mesh)

    def push_particles(self):
        self.boris_integration(self.time_grid.time_step_size)

    def generate_and_prepare_particles(self, initial=False):
        self.generate_valid_particles(initial)
        if self.particle_interaction_model.pic:
            self.eval_charge_density()
            self.eval_potential_and_fields()
        self.shift_new_particles_velocities_half_time_step_back()
        self.consolidate_particle_arrays()

    def generate_valid_particles(self, initial=False):
        # First generate then remove.
        # This allows for overlap of source and inner region.
        self.generate_new_particles(initial)
        self.apply_domain_boundary_conditions()
        self.remove_particles_inside_inner_regions()

    def boris_integration(self, dt):
        for particles in self.particle_arrays:
            total_el_field, total_mgn_field = \
                self.compute_total_fields_at_positions(particles.positions)
            if self.magnetic_fields != 0 and total_mgn_field.any():
                particles.boris_update_momentums(dt, total_el_field, total_mgn_field)
            else:
                particles.boris_update_momentum_no_mgn(dt, total_el_field)
            particles.update_positions(dt)

    def prepare_boris_integration(self, minus_half_dt):
        # todo: place newly generated particle_arrays into separate buffer
        for particles in self.particle_arrays:
            if not particles.momentum_is_half_time_step_shifted:
                total_el_field, total_mgn_field = \
                    self.compute_total_fields_at_positions(particles.positions)
                if self.magnetic_fields != 0 and total_mgn_field.any():
                    particles.boris_update_momentums(minus_half_dt, total_el_field, total_mgn_field)
                else:
                    particles.boris_update_momentum_no_mgn(minus_half_dt, total_el_field)
                particles.momentum_is_half_time_step_shifted = True

    def compute_total_fields_at_positions(self, positions):
        total_el_field = self.electric_fields + self._dynamic_field
        return total_el_field.get_at_points(positions, self.time_grid.current_time), \
               self.magnetic_fields.get_at_points(positions, self.time_grid.current_time)

    def binary_electric_field_at_positions(self, positions):
        return sum(
            np.nan_to_num(p.field_at_points(positions)) for p in self.particle_arrays)

    def shift_new_particles_velocities_half_time_step_back(self):
        minus_half_dt = -1.0 * self.time_grid.time_step_size / 2.0
        self.prepare_boris_integration(minus_half_dt)

    def apply_domain_boundary_conditions(self):
        for arr in self.particle_arrays:
            collisions = self.out_of_bound(arr)
            arr.remove(collisions)
        self.particle_arrays = [a for a in self.particle_arrays if len(a.ids) > 0]

    def remove_particles_inside_inner_regions(self):
        for region in self.inner_regions:
            for p in self.particle_arrays:
                region.collide_with_particles(p)
            self.particle_arrays = [a for a in self.particle_arrays if len(a.ids) > 0]

    def out_of_bound(self, particle):
        return np.logical_or(np.any(particle.positions < 0, axis=-1),
                             np.any(particle.positions > self.spat_mesh.size, axis=-1))

    def generate_new_particles(self, initial=False):
        for src in self.particle_sources:
            particles = src.generate_initial_particles() if initial else src.generate_each_step()
            if len(particles.ids):
                particles.ids = self.generate_particle_ids(len(particles.ids))
                self.particle_arrays.append(particles)

    def generate_particle_ids(self, num_of_particles):
        range_of_ids = range(self.max_id + 1, self.max_id + num_of_particles + 1)
        self.max_id += num_of_particles
        return np.array(range_of_ids)

    def update_time_grid(self):
        self.time_grid.update_to_next_step()

    def write_step_to_save(self):
        current_step = self.time_grid.current_node
        step_to_save = self.time_grid.node_to_save
        if (current_step % step_to_save) == 0:
            self.write()

    @classmethod
    def import_from_h5(self, h5file, filename_prefix, filename_suffix):
        g = h5file['SpatialMesh']
        ga = g.attrs
        size = np.array([ga['{}_volume_size'.format(c)] for c in 'xyz']).reshape(3)
        n_nodes = np.array([ga['{}_n_nodes'.format(c)] for c in 'xyz']).reshape(3)
        charge = np.reshape(g['charge_density'], n_nodes)
        potential = np.reshape(g['potential'], n_nodes)
        field = np.moveaxis(
            np.array([np.reshape(g['electric_field_{}'.format(c)], n_nodes) for c in 'xyz']),
            0, -1)
        mesh = SpatialMesh(MeshGrid(size, n_nodes), charge, potential, field)

        g = h5file['TimeGrid']
        ga = g.attrs
        time_grid = TimeGrid(float(ga['total_time']), float(ga['time_step_size']), float(ga['time_save_step']),
                             float(ga['current_time']), int(ga['current_node']))

        sources = []
        particles = []
        for name, g in h5file['ParticleSources'].items():
            ga = g.attrs
            gt = ga['geometry_type']
            if gt == b'box':
                origin = np.array([ga['box_x_right'], ga['box_y_bottom'], ga['box_z_near']]).reshape(3)
                size = np.array([ga['box_x_left'], ga['box_y_top'], ga['box_z_far']]).reshape(3) - origin
                shape = Box(origin, size)
            elif gt == b'cylinder':
                start = np.array([ga['cylinder_axis_start_{}'.format(c)] for c in 'xyz']).reshape(3)
                end = np.array([ga['cylinder_axis_end_{}'.format(c)] for c in 'xyz']).reshape(3)
                shape = Cylinder(start, end, ga['cylinder_radius'])
            elif gt == b'tube_along_z':
                x, y = (ga['tube_along_z_axis_{}'.format(c)] for c in 'xy')
                sz = ga['tube_along_z_axis_start_z']
                ez = ga['tube_along_z_axis_end_z']
                r, R = (ga['tube_along_z_{}_radius'.format(s)] for s in ('inner', 'outer'))
                shape = Tube((x, y, sz), (x, y, ez), r, R)
            momentum = np.array([ga['mean_momentum_{}'.format(c)] for c in 'xyz']).reshape(3)
            sources.append(ParticleSource(name, shape, int(ga['initial_number_of_particles']),
                                          int(ga['particles_to_generate_each_step']),
                                          momentum, float(ga['temperature']), float(ga['charge']), float(ga['mass'])))
            particles.append(ParticleArray(ids=g['particle_id'], charge=float(ga['charge']), mass=float(ga['mass']),
                                           positions=np.moveaxis(
                                               np.array([g['position_{}'.format(c)] for c in 'xyz']),
                                               0, -1),
                                           momentums=np.moveaxis(
                                               np.array([g['momentum_{}'.format(c)] for c in 'xyz']),
                                               0, -1),
                                           momentum_is_half_time_step_shifted=True))
        max_id = int(np.max([p.ids for p in particles], initial=-1))

        regions = []
        for name, g in h5file['InnerRegions'].items():
            ga = g.attrs
            gt = ga['object_type']
            if gt == b'box':
                origin = np.array([ga['x_right'], ga['y_bottom'], ga['z_near']])
                size = np.array([ga['x_left'], ga['y_top'], ga['z_far']]) - origin
                shape = Box(origin, size)
            elif gt == b'sphere':
                shape = Sphere([ga['origin_{}'.format(c)] for c in 'xyz'], ga['radius'])
            elif gt == b'cylinder':
                start = [ga['axis_start_{}'.format(c)] for c in 'xyz']
                end = [ga['axis_end_{}'.format(c)] for c in 'xyz']
                shape = Cylinder(start, end, ga['radius'])
            elif gt == b'tube':
                start = [ga['axis_start_{}'.format(c)] for c in 'xyz']
                end = [ga['axis_end_{}'.format(c)] for c in 'xyz']
                r, R = (ga['{}_radius'.format(s)] for s in ('inner', 'outer'))
                shape = Tube(start, end, r, R)
            regions.append(InnerRegion(name, shape, ga['potential'], ga['total_absorbed_particles'],
                                       ga['total_absorbed_charge']))

        ef, mf = [], []
        for name, g in h5file['ExternalFields'].items():
            ga = g.attrs
            ft = ga['field_type']
            if ft == b'electric_uniform':
                ef.append(FieldUniform(name, 'electric',
                                       np.array([ga['electric_uniform_field_{}'.format(c)] for c in 'xyz']).reshape(3)))
            elif ft == b'electric_tinyexpr':
                ef.append(
                    FieldExpression(name, 'electric',
                                    *[ga['electric_tinyexpr_field_{}'.format(c)].decode('utf8') for c in 'xyz']))
            elif ft == b'electric_on_regular_grid':
                ef.append(FieldOnGrid(name, 'electric', ga['electric_h5filename'].decode('utf8')))
            elif ft == b'magnetic_uniform':
                mf.append(FieldUniform(name, 'magnetic',
                                       np.array([ga['magnetic_uniform_field_{}'.format(c)] for c in 'xyz']).reshape(3)))
            elif ft == b'magnetic_tinyexpr':
                mf.append(FieldExpression(name, 'magnetic',
                                          *[ga['magnetic_tinyexpr_field_{}'.format(c)].decode('utf8') for c in 'xyz']))

        pim = ParticleInteractionModel(
            h5file['ParticleInteractionModel'].attrs['particle_interaction_model'].decode('utf8'))

        return Simulation(time_grid=time_grid, spat_mesh=mesh, inner_regions=regions,
                          particle_sources=sources, electric_fields=ef, magnetic_fields=mf,
                          particle_interaction_model=pim,
                          output_filename_prefix=filename_prefix, outut_filename_suffix=filename_suffix,
                          max_id=max_id, particle_arrays=particles)

    def export_h5(self, h5file):
        g = h5file.create_group('SpatialMesh')
        for i, c in enumerate('xyz'):
            g.attrs['{}_volume_size'.format(c)] = [self.spat_mesh.size[i]]
            g.attrs['{}_cell_size'.format(c)] = [self.spat_mesh.cell[i]]
            g.attrs['{}_n_nodes'.format(c)] = [self.spat_mesh.n_nodes[i]]
            g['node_coordinates_{}'.format(c)] = self.spat_mesh.node_coordinates[..., i].flatten()
            g['electric_field_{}'.format(c)] = self.spat_mesh.electric_field[..., i].flatten()
        g['charge_density'] = self.spat_mesh.charge_density.flatten()
        g['potential'] = self.spat_mesh.potential.flatten()

        g = h5file.create_group('TimeGrid')
        for k in 'total_time', 'current_time', 'time_step_size', 'time_save_step', \
                 'total_nodes', 'current_node', 'node_to_save':
            g.attrs[k] = [getattr(self.time_grid, k)]

        g = h5file.create_group('ParticleSources')
        g.attrs['number_of_sources'] = [len(self.particle_sources)]
        for i, s in enumerate(self.particle_sources):
            sg = g.create_group(s.name)
            for k in 'temperature', 'charge', 'mass', 'initial_number_of_particles', 'particles_to_generate_each_step':
                sg.attrs[k] = [getattr(s, k)]
            for i, c in enumerate('xyz'):
                sg.attrs['mean_momentum_{}'.format(c)] = [s.mean_momentum[i]]
            if s.shape.__class__ is Box:
                geom = 'box'
                sg.attrs['box_x_right'] = s.shape.origin[0]
                sg.attrs['box_x_left'] = s.shape.origin[0] + s.shape.size[0]
                sg.attrs['box_y_bottom'] = s.shape.origin[1]
                sg.attrs['box_y_top'] = s.shape.origin[1] + s.shape.size[1]
                sg.attrs['box_z_near'] = s.shape.origin[2]
                sg.attrs['box_z_far'] = s.shape.origin[2] + s.shape.size[2]
            elif s.shape.__class__ is Cylinder:
                geom = 'cylinder'
                sg.attrs['cylinder_radius'] = s.shape.radius
                for i, c in enumerate('xyz'):
                    sg.attrs['cylinder_axis_start_{}'.format(c)] = s.shape.start[i]
                    sg.attrs['cylinder_axis_end_{}'.format(c)] = s.shape.end[i]
            elif s.shape.__class__ is Tube:
                geom = 'tube_along_z'
                sg.attrs['tube_along_z_inner_radius'] = s.shape.inner_radius
                sg.attrs['tube_along_z_outer_radius'] = s.shape.outer_radius
                sg.attrs['tube_along_z_axis_x'] = s.shape.start[0]
                sg.attrs['tube_along_z_axis_y'] = s.shape.start[1]
                sg.attrs['tube_along_z_axis_start_z'] = s.shape.start[2]
                sg.attrs['tube_along_z_axis_end_z'] = s.shape.end[2]
            sg.attrs['geometry_type'] = np.string_(geom.encode('utf8') + b"\x00")

        for p in self.particle_arrays:
            s = next(s for s in self.particle_sources if s.charge == p.charge and s.mass == p.mass)
            sg = g[s.name]
            sg['particle_id'] = p.ids
            sg.attrs['max_id'] = p.ids.max()
            for i, c in enumerate('xyz'):
                sg['position_{}'.format(c)] = p.positions[:, i]
                sg['momentum_{}'.format(c)] = p.momentums[:, i]

        for i, s in enumerate(self.particle_sources):
            sg = g[s.name]
            if 'particle_id' not in sg:
                sg['particle_id'] = np.zeros((0,))
                sg.attrs['max_id'] = 0
                for c in 'xyz':
                    sg['position_{}'.format(c)] = np.zeros((0, 3))
                    sg['momentum_{}'.format(c)] = np.zeros((0, 3))

        g = h5file.create_group('InnerRegions')
        g.attrs['number_of_regions'] = len(self.inner_regions)
        for s in self.inner_regions:
            sg = g.create_group(s.name)
            for k in 'potential', 'total_absorbed_particles', 'total_absorbed_charge':
                sg.attrs[k] = getattr(s, k)
            if s.shape.__class__ is Box:
                geom = 'box'
                sg.attrs['x_right'] = s.shape.origin[0]
                sg.attrs['x_left'] = s.shape.origin[0] + s.shape.size[0]
                sg.attrs['y_bottom'] = s.shape.origin[1]
                sg.attrs['y_top'] = s.shape.origin[1] + s.shape.size[1]
                sg.attrs['z_near'] = s.shape.origin[2]
                sg.attrs['z_far'] = s.shape.origin[2] + s.shape.size[2]
            elif s.shape.__class__ is Sphere:
                geom = 'sphere'
                sg.attrs['radius'] = s.shape.radius
                for i, c in enumerate('xyz'):
                    sg.attrs['origin_{}'.format(c)] = s.shape.origin[i]
            elif s.shape.__class__ is Cylinder:
                geom = 'cylinder'
                sg.attrs['radius'] = s.shape.radius
                for i, c in enumerate('xyz'):
                    sg.attrs['axis_start_{}'.format(c)] = s.shape.start[i]
                    sg.attrs['axis_end_{}'.format(c)] = s.shape.end[i]
            elif s.shape.__class__ is Tube:
                geom = 'tube'
                sg.attrs['inner_radius'] = s.shape.inner_radius
                sg.attrs['outer_radius'] = s.shape.outer_radius
                for i, c in enumerate('xyz'):
                    sg.attrs['axis_start_{}'.format(c)] = s.shape.start[i]
                    sg.attrs['axis_end_{}'.format(c)] = s.shape.end[i]
            sg.attrs['object_type'] = np.string_(geom.encode('utf8') + b"\x00")

        g = h5file.create_group('ExternalFields')
        if self.electric_fields.__class__.__name__ == "FieldZero":
            ff = []
        elif self.electric_fields.__class__.__name__ == "FieldSum":
            ff = self.electric_fields.fields
        else:
            ff = [self.electric_fields]
        g.attrs['number_of_electric_fields'] = len(ff)
        for s in ff:
            sg = g.create_group(s.name)
            if s.__class__ is FieldUniform:
                ft = 'electric_uniform'
                for i, c in enumerate('xyz'):
                    sg.attrs['electric_uniform_field_{}'.format(c)] = s.uniform_field_vector[i]
            elif s.__class__ is FieldExpression:
                ft = 'electric_tinyexpr'
                for i, c in enumerate('xyz'):
                    expr = getattr(s, 'expression_{}'.format(c))
                    expr = np.string_(expr.encode('utf8')) + b'\x00'
                    sg.attrs['electric_tinyexpr_field_{}'.format(c)] = expr
            elif s.__class__ is FieldOnGrid:
                ft = 'electric_on_regular_grid'
                sg.attrs['h5filename'] = np.string_(s.field_filename.encode('utf8') + b'\x00')
            sg.attrs['field_type'] = np.string_(ft.encode('utf8') + b'\x00')

        if self.magnetic_fields.__class__.__name__ == "FieldZero":
            ff = []
        elif self.magnetic_fields.__class__.__name__ == "FieldSum":
            ff = self.magnetic_fields.fields
        else:
            ff = [self.magnetic_fields]
        g.attrs['number_of_magnetic_fields'] = len(ff)
        for s in ff:
            sg = g.create_group(s.name)
            if s.__class__ is FieldUniform:
                ft = 'magnetic_uniform'
                sg.attrs['speed_of_light'] = speed_of_light
                for i, c in enumerate('xyz'):
                    sg.attrs['magnetic_uniform_field_{}'.format(c)] = s.uniform_field_vector[i]
            elif s.__class__ is FieldExpression:
                ft = 'magnetic_tinyexpr'
                sg.attrs['speed_of_light'] = speed_of_light
                for i, c in enumerate('xyz'):
                    expr = getattr(s, 'expression_{}'.format(c))
                    expr = np.string_(expr.encode('utf8')) + b'\x00'
                    sg.attrs['magnetic_tinyexpr_field_{}'.format(c)] = expr
            sg.attrs['field_type'] = np.string_(ft.encode('utf8') + b'\x00')

        g = h5file.create_group('ParticleInteractionModel')
        g.attrs['particle_interaction_model'] = \
            np.string_(self.particle_interaction_model.particle_interaction_model.name.encode('utf8') + b'\x00')

    def _write(self, specific_name, export=False):
        file_name_to_write = self._output_filename_prefix + specific_name + self._output_filename_suffix
        h5file = h5py.File(file_name_to_write, mode="w")
        if not h5file:
            print("Error: can't open file " + file_name_to_write + "to save results!")
            print("Recheck \'output_filename_prefix\' key in config file.")
            print("Make sure the directory you want to save to exists.")
        print("Writing to file {}".format(file_name_to_write))
        self.export_h5(h5file) if export else self.save_h5(h5file)
        h5file.close()

    def write(self):
        print("Writing step {} to file".format(self.time_grid.current_node))
        self._write("{:07}_new".format(self.time_grid.current_node))
        self._write("{:07}".format(self.time_grid.current_node), export=True)
        self.write_history()

    def create_history_file(self):
        n_particles = sum(s.initial_number_of_particles +
                              s.particles_to_generate_each_step * self.time_grid.total_nodes
                              for s in self.particle_sources)
        n_time = (self.time_grid.total_nodes - 1) // self.time_grid.node_to_save + 1
        file_name_to_write = self._output_filename_prefix + 'history' + self._output_filename_suffix
        h5file = h5py.File(file_name_to_write, mode="w")
        if not h5file:
            print("Error: can't open file " + file_name_to_write + "to save results!")
            print("Recheck \'output_filename_prefix\' key in config file.")
            print("Make sure the directory you want to save to exists.")
        print("Creating history file {}".format(file_name_to_write))

        self.save_h5(h5file)
        h5file['/time'] = np.linspace(0, self.time_grid.total_time, n_time)
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

        if self.particle_interaction_model.pic:
            h5file.create_dataset('/field/potential', (n_time, *self.spat_mesh.n_nodes))
            h5file['/field/potential'].dims[0].label = 'time'
            h5file['/field/potential'].dims.create_scale(h5file['/time'], 'time')
            h5file['/field/potential'].dims[0].attach_scale(h5file['/time'])
            for i, c in enumerate('xyz'):
                h5file['/field/{}'.format(c)] = np.linspace(0, self.spat_mesh.size[i], self.spat_mesh.n_nodes[i])
                h5file['/field/potential'].dims[i+1].label = c
                h5file['/field/potential'].dims.create_scale(h5file['/field/{}'.format(c)], c)
                h5file['/field/potential'].dims[i+1].attach_scale(h5file['/field/{}'.format(c)])

        h5file.close()

    def write_history(self):
        file_name_to_write = self._output_filename_prefix + 'history' + self._output_filename_suffix
        h5file = h5py.File(file_name_to_write, mode="r+")
        if not h5file:
            print("Error: can't open history file " + file_name_to_write + "to append results!")
            print("Was it not created properly on simulation start?")
        t = self.time_grid.current_node // self.time_grid.node_to_save
        for p in self.particle_arrays:
            for i, id in enumerate(p.ids):
                h5file['/particles/position'][id, t] = p.positions[i]
                h5file['/particles/momentum'][id, t] = p.momentums[i]
                h5file['/particles/mass'][id] = p.mass
                h5file['/particles/charge'][id] = p.charge
        if self.particle_interaction_model.pic:
            h5file['/field/potential'][t] = self.spat_mesh.potential
        h5file.close()

    def eval_and_write_fields_without_particles(self):
        self.spat_mesh.clear_old_density_values()
        self.eval_potential_and_fields()
        print("Writing initial fields to file")
        self._write("fieldsWithoutParticles_new")
        self._write("fieldsWithoutParticles", export=True)

    def consolidate_particle_arrays(self):
        particles_by_type = defaultdict(list)
        for p in self.particle_arrays:
            particles_by_type[(p.mass, p.charge, p.momentum_is_half_time_step_shifted)].append(p)
        self.particle_arrays = []
        for k, v in particles_by_type.items():
            mass, charge, shifted = k
            ids = np.concatenate([p.ids for p in v])
            positions = np.concatenate([p.positions for p in v])
            momentums = np.concatenate([p.momentums for p in v])
            if len(ids):
                self.particle_arrays.append(ParticleArray(ids, charge, mass, positions, momentums, shifted))
