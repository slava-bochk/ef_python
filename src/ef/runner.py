from ef.field.solvers.pyamg import FieldSolverPyamg
from ef.output import OutputWriterNone


class Runner:
    def __init__(self, simulation, solver=None, output_writer=OutputWriterNone()):
        if solver is None:
            solver = FieldSolverPyamg(simulation.spat_mesh, simulation.inner_regions)
        self.output_writer = output_writer
        self.simulation = simulation
        self.solver = solver

    def start(self):
        self.eval_and_write_fields_without_particles()
        self.simulation.generate_and_prepare_particles(self.solver, initial=True)
        self.write()
        self.run()

    def continue_(self):
        self.run()

    def run(self):
        total_time_iterations = self.simulation.time_grid.total_nodes - 1
        current_node = self.simulation.time_grid.current_node
        for i in range(current_node, total_time_iterations):
            print("\rTime step from {:d} to {:d} of {:d}".format(
                i, i + 1, total_time_iterations), end='')
            self.simulation.advance_one_time_step(self.solver)
            self.write_step_to_save()

    def write_step_to_save(self):
        current_step = self.simulation.time_grid.current_node
        step_to_save = self.simulation.time_grid.node_to_save
        if (current_step % step_to_save) == 0:
            print()
            self.write()

    def write(self):
        print("Writing step {} to file".format(self.simulation.time_grid.current_node))
        self.output_writer.write(self.simulation)

    def eval_and_write_fields_without_particles(self):
        self.simulation.spat_mesh.clear_old_density_values()
        self.simulation.eval_potential_and_field(self.solver)
        print("Writing initial fields to file")
        self.output_writer.write(self.simulation, "fieldsWithoutParticles")