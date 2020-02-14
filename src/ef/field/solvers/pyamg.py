from ef.field.solvers import FieldSolver


class FieldSolverPyamg(FieldSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import pyamg
        self._solver = pyamg

    def eval_potential(self, charge_density, potential):
        self.init_rhs_vector(charge_density, potential)
        self.phi_vec = self._solver.solve(self.A, self.rhs, x0=self.phi_vec, tol=self.tolerance, verb=False, maxiter=self.max_iter)
        self.transfer_solution_to_spat_mesh(potential)
