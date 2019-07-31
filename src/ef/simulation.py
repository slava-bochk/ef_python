from collections import defaultdict

import numpy as np

from ef.field import FieldZero, FieldSum
from ef.field.particles import FieldParticles
from ef.particle_array import ParticleArray
from ef.particle_interaction_model import Model
from ef.util.serializable_h5 import SerializableH5


class Simulation(SerializableH5):
    def __init__(self, time_grid, spat_mesh, inner_regions,
                 particle_sources,
                 electric_fields, magnetic_fields, particle_interaction_model,
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
        self._field_solver = None

        if self.particle_interaction_model == Model.binary:
            self._dynamic_field = FieldParticles('binary_particle_field', self.particle_arrays)
            if self.inner_regions or not self.spat_mesh.is_potential_equal_on_boundaries():
                self._dynamic_field += self.spat_mesh
        elif self.particle_interaction_model == Model.noninteracting:
            if self.inner_regions or not self.spat_mesh.is_potential_equal_on_boundaries():
                self._dynamic_field = self.spat_mesh
            else:
                self._dynamic_field = FieldZero('Uniform_potential_zero_field', 'electric')
        else:
            self._dynamic_field = self.spat_mesh

        self.max_id = max_id

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
        if self.particle_interaction_model == Model.PIC:
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
