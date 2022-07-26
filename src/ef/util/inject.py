import inject
import numpy
from inject import Binder, BinderCallable

from ef.field.solvers import FieldSolver
from ef.field.solvers.pyamg import FieldSolverPyamg
from ef.util.array_on_grid import ArrayOnGrid


def make_injection_config(solver: str = 'amg', backend: str = 'numpy') -> BinderCallable:
    def conf(binder: Binder) -> None:
        if solver == 'amgx':
            from ef.field.solvers.pyamgx import FieldSolverPyamgx
            binder.bind(FieldSolver, FieldSolverPyamgx)
        else:
            binder.bind(FieldSolver, FieldSolverPyamg)
        if backend == 'cupy':
            import cupy
            from ef.util.array_on_grid_cupy import ArrayOnGridCupy
            binder.bind(ArrayOnGrid, ArrayOnGridCupy)
            binder.bind(numpy, cupy)
        else:
            binder.bind(ArrayOnGrid, ArrayOnGrid)
            binder.bind(numpy, numpy)

    return conf


def configure_application(solver: str = 'amg', backend: str = 'numpy'):
    inject.configure(make_injection_config(solver, backend))
