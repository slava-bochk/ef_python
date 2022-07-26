import inject
import numpy
import pytest

from ef.field.solvers import FieldSolver
from ef.field.solvers.pyamg import FieldSolverPyamg
from ef.util.array_on_grid import ArrayOnGrid

@pytest.fixture(params=['numpy', pytest.param('cupy', marks=pytest.mark.cupy)], ids=['np', 'cp'])
def _backend(request):
    xp = request.param

    def conf(binder):
        if xp == 'numpy':
            binder.bind(ArrayOnGrid, ArrayOnGrid)
            binder.bind(numpy, numpy)
        elif xp == 'cupy':
            import cupy
            from ef.util.array_on_grid_cupy import ArrayOnGridCupy
            binder.bind(ArrayOnGrid, ArrayOnGridCupy)
            binder.bind(numpy, cupy)
    return conf


@pytest.fixture(params=['amg', pytest.param('amgx', marks=pytest.mark.amgx)])
def _solver(request):
    s = request.param

    def conf(binder):
        if s == 'amg':
            binder.bind(FieldSolver, FieldSolverPyamg)
        elif s == 'amgx':
            from ef.field.solvers.pyamgx import FieldSolverPyamgx
            binder.bind(FieldSolver, FieldSolverPyamgx)
    return conf


@pytest.fixture
def backend(_backend):
    inject.clear_and_configure(_backend)
    yield
    inject.clear()


@pytest.fixture
def solver(_solver):
    inject.clear_and_configure(_solver)
    yield
    inject.clear()


@pytest.fixture
def backend_and_solver(_backend, _solver):
    def conf(binder):
        binder.install(_backend)
        binder.install(_solver)

    inject.clear_and_configure(conf)
    yield
    inject.clear()
