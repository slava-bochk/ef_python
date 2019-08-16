import inject
import numpy
from decorator import decorator
from inject import Binder

from ef.field.solvers import FieldSolver
from ef.field.solvers.pyamg import FieldSolverPyamg
from ef.util.array_on_grid import ArrayOnGrid


def default_injection_config(binder: Binder) -> None:
    binder.bind(FieldSolver, FieldSolverPyamg)
    binder.bind(ArrayOnGrid, ArrayOnGrid)
    binder.bind(numpy, numpy)


@decorator
def safe_default_inject(foo, *args, **kwargs):
    if not inject.is_configured():
        inject.configure(default_injection_config)
    return foo(*args, **kwargs)
