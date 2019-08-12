from collections import defaultdict
from typing import List

from ef.config.components import Box
from ef.field import FieldZero, FieldSum
from ef.field.on_grid import FieldOnGrid
from ef.field.particles import FieldParticles
from ef.inner_region import InnerRegion
from ef.meshgrid import MeshGrid
from ef.particle_array import ParticleArray
from ef.particle_interaction_model import Model
from ef.particle_tracker import ParticleTracker
from ef.util.array_on_grid import ArrayOnGrid
from ef.util.serializable_h5 import SerializableH5


def is_trivial(potential: ArrayOnGrid, inner_regions: List[InnerRegion]):
    if not potential.is_the_same_on_all_boundaries:
        return False
    return len({float(potential.data[0, 0, 0])} | {ir.potential for ir in inner_regions}) == 1


class Simulation(SerializableH5):
    def __init__(self, time_grid,
                 mesh: MeshGrid, charge_density: ArrayOnGrid, potential: ArrayOnGrid, electric_field: FieldOnGrid,
                 inner_regions: List[InnerRegion],
                 particle_sources,
                 electric_fields, magnetic_fields, particle_interaction_model,
                 particle_tracker=None, particle_arrays=()):
        self.time_grid = time_grid
        self.mesh = mesh
        self.charge_density = charge_density
        self.potential = potential
        self.electric_field = electric_field
        self._domain = InnerRegion('simulation_domain', Box(0, mesh.size), inverted=True)
        self.inner_regions = inner_regions
        self.particle_sources = particle_sources
        self.electric_fields = FieldSum.factory(electric_fields, 'electric')
        self.magnetic_fields = FieldSum.factory(magnetic_fields, 'magnetic')
        self.particle_interaction_model = particle_interaction_model
        self.particle_arrays = list(particle_arrays)
        self.consolidate_particle_arrays()

        if self.particle_interaction_model == Model.binary:
            self._dynamic_field = FieldParticles('binary_particle_field', self.particle_arrays)
            if not is_trivial(potential, inner_regions):
                self._dynamic_field += self.electric_field
        elif self.particle_interaction_model == Model.noninteracting:
            if not is_trivial(potential, inner_regions):
                self._dynamic_field = self.electric_field
            else:
                self._dynamic_field = FieldZero('Uniform_potential_zero_field', 'electric')
        else:
            self._dynamic_field = self.electric_field

        self.particle_tracker = ParticleTracker() if particle_tracker is None else particle_tracker

    @property
    def dict(self) -> dict:
        d = super().dict
        d['particle_interaction_model'] = d['particle_interaction_model'].name
        return d

    def advance_one_time_step(self, field_solver):
        self.push_particles()
        self.generate_and_prepare_particles(field_solver)
        self.time_grid.update_to_next_step()

    def eval_charge_density(self):
        self.charge_density.reset()
        for p in self.particle_arrays:
            self.charge_density.distribute_at_positions(p.charge, p.positions)

    def push_particles(self):
        self.boris_integration(self.time_grid.time_step_size)

    def generate_and_prepare_particles(self, field_solver, initial=False):
        self.generate_valid_particles(initial)
        if self.particle_interaction_model == Model.PIC:
            self.eval_charge_density()
            field_solver.eval_potential(self.charge_density, self.potential)
            self.electric_field.array = self.potential.gradient()
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

    def shift_new_particles_velocities_half_time_step_back(self):
        minus_half_dt = -1.0 * self.time_grid.time_step_size / 2.0
        self.prepare_boris_integration(minus_half_dt)

    def apply_domain_boundary_conditions(self):
        for arr in self.particle_arrays:
            self._domain.collide_with_particles(arr)
        self.particle_arrays = [a for a in self.particle_arrays if len(a.ids) > 0]

    def remove_particles_inside_inner_regions(self):
        for region in self.inner_regions:
            for p in self.particle_arrays:
                region.collide_with_particles(p)
            self.particle_arrays = [a for a in self.particle_arrays if len(a.ids) > 0]

    def generate_new_particles(self, initial=False):
        for src in self.particle_sources:
            particles = src.generate_initial_particles() if initial else src.generate_each_step()
            if len(particles.ids):
                particles.ids = particles.xp.asarray(self.particle_tracker.generate_particle_ids(len(particles.ids)))
                self.particle_arrays.append(particles)

    def consolidate_particle_arrays(self):
        particles_by_type = defaultdict(list)
        for p in self.particle_arrays:
            particles_by_type[(p.mass, p.charge, p.momentum_is_half_time_step_shifted)].append(p)
        self.particle_arrays = []
        for k, v in particles_by_type.items():
            mass, charge, shifted = k
            ids = v[0].xp.concatenate([p.ids for p in v])
            positions = v[0].xp.concatenate([p.positions for p in v])
            momentums = v[0].xp.concatenate([p.momentums for p in v])
            if len(ids):
                self.particle_arrays.append(ParticleArray(ids, charge, mass, positions, momentums, shifted))
