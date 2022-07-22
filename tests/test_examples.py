import os
import subprocess
from shutil import copytree

import inject
import pytest

from ef.config.config import Config
from ef.runner import Runner
from ef.util.testing import assert_dataclass_eq

_examples_conf = [("examples/axially_symmetric_beam_contour/contour.conf", pytest.mark.slow),
                  ("examples/minimal_working_example/minimal_conf.conf", ()),
                  ("examples/single_particle_in_free_space/single_particle_in_free_space.conf", pytest.mark.slowish),
                  ("examples/single_particle_in_radial_electric_field/single_particle_in_radial_electric_field.conf",
                   ()),
                  ("examples/single_particle_in_magnetic_field/single_particle_in_magnetic_field.conf",
                   pytest.mark.slow),
                  ("examples/single_particle_in_magnetic_field/large_time_step.conf", pytest.mark.slow),
                  ("examples/single_particle_in_magnetic_field/long_simulation_time.conf", pytest.mark.slow),
                  ("examples/ribbon_beam_contour/contour_bin.conf", pytest.mark.slowish),
                  ("examples/drift_tube_potential/pot.conf", pytest.mark.slow),
                  ("examples/ribbon_beam_contour/contour.conf", pytest.mark.slow),
                  ("examples/tube_source_test/contour.conf", pytest.mark.slow)
                  ]

_pytest_params_example_conf = [pytest.param(f.replace('/', os.path.sep), marks=m) for f, m in _examples_conf]


@pytest.mark.parametrize("fname", _pytest_params_example_conf)
def test_example_conf(fname, tmpdir, monkeypatch, backend_and_solver):
    sim = Config.from_fname(fname).make()
    monkeypatch.chdir(tmpdir)
    Runner(sim).start()


def run_jupyter(dir, fname, path=None, copy_dir=False):
    dir = dir.replace('/', os.path.sep)
    fname = os.path.join(dir, fname)
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
    if copy_dir:
        copytree(dir, path)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {'metadata': {'path': path}} if path is not None else {})
    inject.clear()


@pytest.mark.slowish
@pytest.mark.jupyter_examples
def test_all_examples_visualize():
    run_jupyter("examples/jupyter", "visualize_examples.ipynb", 'examples/jupyter/')


@pytest.mark.slow
@pytest.mark.jupyter_examples
def test_axially_symmetric_beam_contour(tmpdir):
    run_jupyter("examples/axially_symmetric_beam_contour", "axially_symmetric_beam_contour.ipynb", tmpdir)


@pytest.mark.slow
@pytest.mark.jupyter_examples
def test_drift_tube_potential(tmpdir):
    run_jupyter("examples/drift_tube_potential", "potential.ipynb", tmpdir.join('newdir'), True)


@pytest.mark.slow
@pytest.mark.jupyter_examples
def test_ribbon_beam_contour(tmpdir):
    run_jupyter("examples/ribbon_beam_contour", "beam.ipynb", tmpdir.join('newdir'), True)


@pytest.mark.slow
@pytest.mark.jupyter_examples
def test_tube_source(tmpdir):
    run_jupyter("examples/tube_source_test", "plot.ipynb", tmpdir.join('newdir'), True)


@pytest.mark.slowish
@pytest.mark.jupyter_examples
def test_single_particle_in_free_space(tmpdir):
    run_jupyter("examples/single_particle_in_free_space", "single_particle_in_free_space.ipynb",
                tmpdir.join('newdir'), True)
    assert_dataclass_eq(Config.from_fname(tmpdir.join('newdir').join('config.ini')),
                        Config.from_fname(tmpdir.join('newdir').join('single_particle_in_free_space.conf')))


@pytest.mark.slowish
@pytest.mark.jupyter_examples
def test_single_particle_in_uniform_electric_field(tmpdir):
    run_jupyter("examples/single_particle_in_electric_field", "single_particle_in_uniform_electric_field.ipynb",
                tmpdir)


@pytest.mark.slowish
@pytest.mark.jupyter_examples
def test_single_particle_in_radial_electric_field(tmpdir):
    run_jupyter("examples/single_particle_in_radial_electric_field", "plot.ipynb", tmpdir.join('newdir'), True)


@pytest.mark.slow
@pytest.mark.jupyter_examples
def test_particle_in_magnetic_field(tmpdir):
    run_jupyter("examples/single_particle_in_magnetic_field", "Single Particle in Uniform Magnetic Field.ipynb", tmpdir)
    run_jupyter("examples/single_particle_in_magnetic_field", "single_particle_in_magnetic_field.ipynb",
                tmpdir.join('newdir'), True)


@pytest.mark.requires_install
@pytest.mark.parametrize("fname", _pytest_params_example_conf)
def test_main_shell(fname, tmpdir, monkeypatch):
    basedir = os.path.join(os.path.dirname(__file__), '..')
    monkeypatch.chdir(tmpdir)
    result = subprocess.run(['ef', os.path.join(basedir, fname)], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    # assert result.stderr == '' TODO: actual testing
    assert result.stdout != ''
