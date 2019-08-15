import numpy

from ef.field.solvers import FieldSolver


class FieldSolverPyamgx(FieldSolver):
    def __del__(self):
        import pyamgx
        self._solver.destroy()
        self._rhs.destroy()
        self._phi_vec.destroy()
        self._matrix.destroy()
        self.resources.destroy()
        self.cfg.destroy()
        pyamgx.finalize()

    def create_solver_and_preconditioner(self):
        import pyamgx
        self.maxiter = 1000
        self.tol = 1e-10
        pyamgx.initialize()
        self.cfg = pyamgx.Config()
        # "print_solve_stats": 1
        # "obtain_timings": 1,
        # "print_grid_stats": 1,
        conf_string = """{{
            "config_version": 2,
            "solver": {{
                "solver": "CG",
                "max_iters": {maxiter},
                "monitor_residual": 1,
                "tolerance": {tol},
                "norm": "L2"
            }}
        }}""".format(tol=self.tol, maxiter=self.maxiter)
        self.cfg.create(conf_string)
        self.resources = pyamgx.Resources().create_simple(self.cfg)
        self._rhs = pyamgx.Vector().create(self.resources).upload(self.rhs)
        self._phi_vec = pyamgx.Vector().create(self.resources).upload(self.phi_vec)
        self._matrix = pyamgx.Matrix().create(self.resources).upload_CSR(self.A.tocsr())
        self._solver = pyamgx.Solver().create(self.resources, self.cfg)
        self._solver.setup(self._matrix)

    def solve_poisson_eqn(self, charge_density, potential):
        self.init_rhs_vector(charge_density, potential)
        self._rhs.upload(self.rhs)
        self._solver.solve(self._rhs, self._phi_vec)
        self.transfer_solution_to_spat_mesh(potential)

    def transfer_solution_to_spat_mesh(self, potential):
        if potential.xp is numpy:
            self._phi_vec.download(self.phi_vec)
            super().transfer_solution_to_spat_mesh(potential)
        else:
            buf = potential.xp.empty(int((self.mesh.n_nodes - 2).prod()))
            self._phi_vec.download_raw(buf.data.ptr)
            potential._data[1:-1, 1:-1, 1:-1] = buf.reshape(self.mesh.n_nodes - 2, order='F')
