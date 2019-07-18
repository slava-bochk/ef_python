import os
import subprocess
from os.path import basename
from shutil import copytree, copy

import pytest

from ef.config.config import Config
from ef.main import main

_examples_conf = [("examples/minimal_working_example/minimal_conf.conf", ()),
                  ("examples/single_particle_in_free_space/single_particle_in_free_space.conf", pytest.mark.slowish),
                  ("examples/single_particle_in_magnetic_field/single_particle_in_magnetic_field.conf",
                   pytest.mark.slowish),
                  ("examples/single_particle_in_magnetic_field/large_time_step.conf", pytest.mark.slowish),
                  ("examples/single_particle_in_radial_electric_field/single_particle_in_radial_electric_field.conf",
                   pytest.mark.slowish),
                  ("examples/tube_source_test/contour.conf", pytest.mark.slow),
                  ("examples/ribbon_beam_contour/contour_bin.conf", ()),
                  ("examples/ribbon_beam_contour/contour.conf", pytest.mark.slowish),
                  ("examples/drift_tube_potential/pot.conf", pytest.mark.slowish)
                  ]

_pytest_params_example_conf = [pytest.param(f.replace('/', os.path.sep), marks=m) for f, m in _examples_conf]


@pytest.mark.parametrize("fname", _pytest_params_example_conf)
def test_example_conf(fname, mocker, capsys, tmpdir, monkeypatch):
    copy(fname, tmpdir.join(basename(fname)))
    monkeypatch.chdir(tmpdir)
    mocker.patch("sys.argv", ["main.py", str(basename(fname))])
    main()
    out, err = capsys.readouterr()
    assert err == ""


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


@pytest.mark.jupyter
def test_all_examples_visualize():
    run_jupyter("examples/jupyter", "visualize_examples.ipynb", 'examples/jupyter/')


@pytest.mark.slow
@pytest.mark.jupyter
def test_axially_symmetric_beam_contour(tmpdir):
    run_jupyter("examples/axially_symmetric_beam_contour", "axially_symmetric_beam_contour.ipynb", tmpdir)


@pytest.mark.slowish
@pytest.mark.jupyter
def test_drift_tube_potential(tmpdir):
    run_jupyter("examples/drift_tube_potential", "potential.ipynb", tmpdir.join('newdir'), True)


@pytest.mark.slow
@pytest.mark.jupyter
def test_ribbon_beam_contour(tmpdir):
    run_jupyter("examples/ribbon_beam_contour", "beam.ipynb", tmpdir.join('newdir'), True)


@pytest.mark.jupyter
def test_single_particle_in_free_space(tmpdir):
    run_jupyter("examples/single_particle_in_free_space", "single_particle_in_free_space.ipynb",
                tmpdir.join('newdir'), True)
    assert Config.from_fname(tmpdir.join('newdir').join('config.ini')) == \
           Config.from_fname(tmpdir.join('newdir').join('single_particle_in_free_space.conf'))


@pytest.mark.jupyter
def test_single_particle_in_uniform_electric_field(tmpdir):
    run_jupyter("examples/single_particle_in_electric_field", "single_particle_in_uniform_electric_field.ipynb",
                tmpdir)


@pytest.mark.requires_install
@pytest.mark.parametrize("fname", _pytest_params_example_conf)
def test_main_shell(fname, tmpdir, monkeypatch):
    basedir = os.path.join(os.path.dirname(__file__), '..')
    monkeypatch.chdir(tmpdir)
    result = subprocess.run(['ef', os.path.join(basedir, fname)], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for line in result.stderr.split("\n"):
        assert line == '' or line.startswith("WARNING:")
    assert result.stdout != ""
