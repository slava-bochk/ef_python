from collections import defaultdict

import h5py
import numpy as np

from ef.field import FieldZero, FieldSum, Field
from ef.field.expression import FieldExpression
from ef.field.on_grid import FieldOnGrid
from ef.field.particles import FieldParticles
from ef.field.solvers.field_solver import FieldSolver
from ef.field.uniform import FieldUniform
from ef.inner_region import InnerRegion
from ef.particle_array import ParticleArray
from ef.particle_interaction_model import ParticleInteractionModel
from ef.particle_source import ParticleSource
from ef.spatial_mesh import SpatialMesh
from ef.time_grid import TimeGrid
from ef.util.physical_constants import speed_of_light
from ef.util.serializable_h5 import SerializableH5


class Simulation(SerializableH5):

    def __init__(self, time_grid, spat_mesh, inner_regions,
                 particle_sources,
                 electric_fields, magnetic_fields, particle_interaction_model,
                 output_filename_prefix='', outut_filename_suffix='.h5', output_format="cpp",
                 max_id=-1, particle_arrays=()):
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
        self._output_format = output_format
        self.max_id = max_id

    @classmethod
    def init_from_h5(cls, h5file, filename_prefix, filename_suffix, output_format):
        if 'SpatialMesh' in h5file:
            simulation = cls.import_from_h5(h5file, filename_prefix, filename_suffix, output_format)
        else:
            simulation = cls.load_h5(h5file)
            simulation._output_filename_prefix = filename_prefix
            simulation._output_filename_suffix = filename_suffix
            simulation._output_format = output_format
        return simulation

    def start_pic_simulation(self):
        self.eval_and_write_fields_without_particles()
        if self._output_format == "history":
            self.create_history_file()
        self.generate_and_prepare_particles(initial=True)
        self.write_step_to_save()
        self.run_pic()

    def continue_pic_simulation(self):
        if self._output_format == "history":
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

    @staticmethod
    def import_from_h5(h5file, filename_prefix, filename_suffix, output_format):
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
            particle_sources=sources, particle_arrays=particles,
            output_filename_prefix=filename_prefix,
            outut_filename_suffix=filename_suffix,
            output_format=output_format,
            max_id=max_id,
        )

    def export_h5(self, h5file):
        self.spat_mesh.export_h5(h5file.create_group('SpatialMesh'))
        self.time_grid.export_h5(h5file.create_group('TimeGrid'))
        g = h5file.create_group('ParticleSources')
        g.attrs['number_of_sources'] = [len(self.particle_sources)]
        for s in self.particle_sources:
            s.export_h5(g.create_group(s.name))
        for p in self.particle_arrays:
            s = next(s for s in self.particle_sources if s.charge == p.charge and s.mass == p.mass)
            p.export_h5(g[s.name])
        for s in self.particle_sources:
            if 'particle_id' not in g[s.name]:
                ParticleArray([], s.charge, s.mass, np.empty((0, 3)), np.empty((0, 3)), True).export_h5(g[s.name])

        g = h5file.create_group('InnerRegions')
        g.attrs['number_of_regions'] = [len(self.inner_regions)]
        for s in self.inner_regions:
            s.export_h5(g.create_group(s.name))

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
        if self._output_format == "python":
            self._write("{:07}".format(self.time_grid.current_node))
        elif self._output_format == "cpp":
            self._write("{:07}".format(self.time_grid.current_node), export=True)
        elif self._output_format == "history":
            self.write_history()
        elif self._output_format == "none":
            pass
        else:
            raise ValueError("Unknown simulaiton output format.")

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
                h5file['/field/potential'].dims[i + 1].label = c
                h5file['/field/potential'].dims.create_scale(h5file['/field/{}'.format(c)], c)
                h5file['/field/potential'].dims[i + 1].attach_scale(h5file['/field/{}'.format(c)])

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
        if self._output_format == "python":
            self._write("fieldsWithoutParticles")
        elif self._output_format == "cpp":
            self._write("fieldsWithoutParticles", export=True)
        elif self._output_format in ("none", "history"):
            pass
        else:
            raise ValueError("Unknown simulaiton output format.")

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
