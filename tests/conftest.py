import inject
import numpy
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ef.field.solvers import FieldSolver
from ef.field.solvers.pyamg import FieldSolverPyamg
from ef.field.solvers.pyamgx import FieldSolverPyamgx
from ef.util.array_on_grid import ArrayOnGrid
from ef.util.array_on_grid_cupy import ArrayOnGridCupy

@pytest.fixture(params=['numpy', pytest.param('cupy', marks=pytest.mark.cupy)], ids=['np', 'cp'])
def _backend(request):
    xp = request.param

    def conf(binder):
        if xp == 'numpy':
            binder.bind(ArrayOnGrid, ArrayOnGrid)
            binder.bind(numpy, numpy)
            binder.bind(assert_array_equal, assert_array_equal)
            binder.bind(assert_array_almost_equal, assert_array_almost_equal)
        elif xp == 'cupy':
            import cupy
            binder.bind(ArrayOnGrid, ArrayOnGridCupy)
            binder.bind(numpy, cupy)
            binder.bind(assert_array_equal, cupy.testing.assert_array_equal)
            binder.bind(assert_array_almost_equal, cupy.testing.assert_array_almost_equal)
    return conf


@pytest.fixture(params=['amg', pytest.param('amgx', marks=pytest.mark.amgx)])
def _solver(request):
    s = request.param

    def conf(binder):
        if s == 'amg':
            binder.bind(FieldSolver, FieldSolverPyamg)
        elif s == 'amgx':
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
