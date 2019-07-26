import pyamgx

from ef.field.solvers import FieldSolver


class FieldSolverPyamgx(FieldSolver):
    def __del__(self):
        self._solver.destroy()
        self._rhs.destroy()
        self._phi_vec.destroy()
        self._matrix.destroy()
        self.resources.destroy()
        self.cfg.destroy()
        pyamgx.finalize()

    def create_solver_and_preconditioner(self):
        self.maxiter = 1000
        self.tol = 1e-10
        pyamgx.initialize()
        self.cfg = pyamgx.Config()
        self.cfg.create_from_dict({
            "config_version": 2,
            "solver": {
                "solver": "CG",
                "max_iters": self.maxiter,
                "monitor_residual": 1,
                "tolerance": self.tol,
                "norm": "L2",
                # "print_solve_stats": 1
                # "obtain_timings": 1,
                # "print_grid_stats": 1,
            }
        })
        self.resources = pyamgx.Resources().create_simple(self.cfg)
        self._rhs = pyamgx.Vector().create(self.resources).upload(self.rhs)
        self._phi_vec = pyamgx.Vector().create(self.resources).upload(self.phi_vec)
        self._matrix = pyamgx.Matrix().create(self.resources).upload_CSR(self.A.tocsr())
        self._solver = pyamgx.Solver().create(self.resources, self.cfg)
        self._solver.setup(self._matrix)

    def solve_poisson_eqn(self):
        self.init_rhs_vector()
        self._rhs.upload(self.rhs)
        self._solver.solve(self._rhs, self._phi_vec)
        self.phi_vec = self._phi_vec.download()
        self.transfer_solution_to_spat_mesh()
