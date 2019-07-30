from collections import defaultdict

import numpy as np

from ef.config.components import OutputFileConf
from ef.field import FieldZero, FieldSum, Field
from ef.field.particles import FieldParticles
from ef.field.solvers.pyamg import FieldSolverPyamg
from ef.inner_region import InnerRegion
from ef.output import OutputWriterNone
from ef.particle_array import ParticleArray
from ef.particle_interaction_model import ParticleInteractionModel
from ef.particle_source import ParticleSource
from ef.spatial_mesh import SpatialMesh
from ef.time_grid import TimeGrid
from ef.util.serializable_h5 import SerializableH5


class Simulation(SerializableH5):

    def __init__(self, time_grid, spat_mesh, inner_regions,
                 particle_sources,
                 electric_fields, magnetic_fields, particle_interaction_model,
                 output_writer=OutputWriterNone(),
                 max_id=-1, particle_arrays=()):
        self.time_grid = time_grid
        self.spat_mesh = spat_mesh
        self.inner_regions = inner_regions
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

        self._output_writer = output_writer
        self.max_id = max_id

    @classmethod
    def init_from_h5(cls, h5file, filename_prefix, filename_suffix, output_format):
        if 'SpatialMesh' in h5file:
            simulation = cls.import_from_h5(h5file,
                                            OutputFileConf(filename_prefix, filename_suffix, output_format).make())
        else:
            simulation = cls.load_h5(h5file)
            simulation._output_writer = OutputFileConf(filename_prefix, filename_suffix, output_format).make()
        return simulation

    def start_pic_simulation(self, solver_class=FieldSolverPyamg):
        self._field_solver = solver_class(self.spat_mesh, self.inner_regions)
        self.eval_and_write_fields_without_particles()
        self.generate_and_prepare_particles(initial=True)
        self.write()
        self.run_pic()

    def continue_pic_simulation(self, solver_class=FieldSolverPyamg):
        self._field_solver = solver_class(self.spat_mesh, self.inner_regions)
        self.run_pic()

    def run_pic(self):
        total_time_iterations = self.time_grid.total_nodes - 1
        current_node = self.time_grid.current_node
        for i in range(current_node, total_time_iterations):
            print("\rTime step from {:d} to {:d} of {:d}".format(
                i, i + 1, total_time_iterations), end='')
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
        self._field_solver.eval_potential()
        self.spat_mesh.eval_field_from_potential()

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
            print()
            self.write()

    @staticmethod
    def import_from_h5(h5file, output_writer):
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
            output_writer=output_writer,
            max_id=max_id
        )

    def write(self):
        print("Writing step {} to file".format(self.time_grid.current_node))
        self._output_writer.write(self)

    def eval_and_write_fields_without_particles(self):
        self.spat_mesh.clear_old_density_values()
        self.eval_potential_and_fields()
        print("Writing initial fields to file")
        self._output_writer.write(self, "fieldsWithoutParticles")

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
