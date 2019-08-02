import pyamg

from ef.field.solvers import FieldSolver


class FieldSolverPyamg(FieldSolver):
    def create_solver_and_preconditioner(self):
        self.maxiter = 1000
        self.tol = 1e-10
        self._solver = pyamg.ruge_stuben_solver(self.A)

    def solve_poisson_eqn(self):
        self.init_rhs_vector()
        self.phi_vec = self._solver.solve(self.rhs, x0=self.phi_vec, tol=self.tol, maxiter=self.maxiter)
        self.transfer_solution_to_spat_mesh()
