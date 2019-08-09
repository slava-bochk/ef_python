import inject
import numpy
from decorator import decorator
from inject import Binder, BinderCallable

from ef.field.solvers import FieldSolver
from ef.field.solvers.pyamg import FieldSolverPyamg
from ef.field.solvers.pyamgx import FieldSolverPyamgx
from ef.util.array_on_grid import ArrayOnGrid
from ef.util.array_on_grid_cupy import ArrayOnGridCupy


def make_injection_config(solver: str = 'amg', backend: str = 'numpy') -> BinderCallable:
    def conf(binder: Binder) -> None:
        binder.bind(FieldSolver, FieldSolverPyamgx if solver == 'amgx' else FieldSolverPyamg)
        if backend == 'cupy':
            import cupy
            binder.bind(ArrayOnGrid, ArrayOnGridCupy)
            binder.bind(numpy, cupy)
        else:
            binder.bind(ArrayOnGrid, ArrayOnGrid)
            binder.bind(numpy, numpy)

    return conf


def configure_application(solver: str = 'amg', backend: str = 'numpy'):
    inject.configure(make_injection_config(solver, backend))


@decorator
def safe_default_inject(foo, *args, **kwargs):
    if not inject.is_configured():
        configure_application()
    return foo(*args, **kwargs)
