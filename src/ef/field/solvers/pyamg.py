from ef.field.solvers import FieldSolver


class FieldSolverPyamg(FieldSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import pyamg
        self._solver = pyamg.solver(self.A, pyamg.solver_configuration(self.A))

    def eval_potential(self, charge_density, potential):
        self.init_rhs_vector(charge_density, potential)
        self.phi_vec = self._solver.solve(self.rhs, x0=self.phi_vec, accel='bicgstab',
                                          tol=self.tolerance, maxiter=self.max_iter)
        self.transfer_solution_to_spat_mesh(potential)
